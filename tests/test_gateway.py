import json

import pandas as pd
import pytest
from google.api_core import exceptions as gexc

from analyst.config import User, load_settings
from analyst.sql.brands import BrandMatcher
from analyst.sql.gateway import (
    Gateway,
    GatewayFatalError,
    GatewayUnavailableError,
    SafetyStop,
    classify_bq_error,
    load_curated,
    wrap_sql,
)
from analyst.sql.scope import BRAND_PARAM, scoped_tables
from analyst.store.results import ResultNotFound, ResultStore
from analyst.tools.run_sql import make_run_sql_tool

CEO = User(id="ceo", name="CEO", role="ceo", brands=["*"])
BM = User(id="bm", name="BM", role="brand_manager", brands=["Levi's", "Canada Goose"])
CURATED = {"sales": "SELECT 1 AS a", "customers": "SELECT 2 AS b", "products": "SELECT 3 AS c"}
BRANDS = BrandMatcher(["Levi's", "Canada Goose", "Carhartt", "Calvin Klein"])


class FakeJob:
    def __init__(self, df=None, bytes_=1000, result_exc=None):
        self.total_bytes_processed = bytes_
        self._df = df
        self._exc = result_exc
        self.cancelled = False

    def result(self, timeout=None):
        if self._exc:
            raise self._exc
        return self

    def to_dataframe(self, create_bqstorage_client=True):
        return self._df

    def cancel(self):
        self.cancelled = True


class FakeClient:
    def __init__(self, df=None, bytes_=1000, dry_exc=None, result_exc=None, dry_excs=(), result_excs=()):
        self.df, self.bytes_, self.dry_exc, self.result_exc = df, bytes_, dry_exc, result_exc
        self.dry_excs, self.result_excs = list(dry_excs), list(result_excs)  # one per call, then success
        self.queries = []
        self.last_job = None

    def query(self, sql, job_config=None):
        self.queries.append((sql, job_config))
        if job_config.dry_run:
            if self.dry_excs:
                raise self.dry_excs.pop(0)
            if self.dry_exc:
                raise self.dry_exc
            return FakeJob(bytes_=self.bytes_)
        exc = self.result_excs.pop(0) if self.result_excs else self.result_exc
        self.last_job = FakeJob(df=self.df if self.df is not None else pd.DataFrame({"a": [1]}), result_exc=exc)
        return self.last_job


def gateway(client, store=None, settings=None):
    return Gateway(settings or load_settings().bigquery, client, store or ResultStore(), CURATED, BRANDS,
                   sleep=lambda s: None)


def bq_error(cls, reason, message="m"):
    return cls(message, errors=[{"reason": reason, "message": message}])


# Curated tables and wrapping

def test_curated_file_has_three_tables_and_no_pii():
    tables = load_curated("ds")
    assert set(tables) == {"sales", "customers", "products"}
    text = " ".join(tables.values()).lower()
    for col in ["email", "first_name", "last_name", "street_address", "postal_code", "latitude", "longitude", "user_geom"]:
        assert col not in text
    assert "u.age," not in text  # banded only (D1.6)


def test_wrap_plain_select():
    out = wrap_sql("SELECT a FROM sales;", CURATED)
    assert out.startswith("WITH sales AS (") and out.rstrip().endswith("SELECT a FROM sales")


def test_wrap_merges_model_with_clause_and_strips_comments():
    out = wrap_sql("-- note\nwith t AS (SELECT a FROM sales) SELECT * FROM t", CURATED)
    assert out.count("WITH") == 1
    assert "products AS (\nSELECT 3 AS c\n),\nt AS (SELECT a FROM sales)" in out


# Scope rewrite

def test_ceo_tables_unfiltered():
    tables, params = scoped_tables(CURATED, CEO)
    assert tables == CURATED and params == []


def test_scoped_tables_filter_by_brand_parameter():
    tables, params = scoped_tables(CURATED, BM)
    assert f"brand IN UNNEST(@{BRAND_PARAM})" in tables["sales"]
    assert f"brand IN UNNEST(@{BRAND_PARAM})" in tables["products"]
    assert "customer_id IN (SELECT customer_id FROM sales)" in tables["customers"]
    assert params[0].name == BRAND_PARAM and params[0].values == ["Levi's", "Canada Goose"]
    assert list(tables) == ["sales", "customers", "products"]  # customers depends on scoped sales


def test_scoped_query_executes_with_filter_and_parameter():
    client = FakeClient()
    store = ResultStore()
    gateway(client, store).run("SELECT state, COUNT(*) FROM sales GROUP BY 1", BM)
    sql, config = client.queries[-1]
    assert f"brand IN UNNEST(@{BRAND_PARAM})" in sql
    assert config.query_parameters[0].values == ["Levi's", "Canada Goose"]
    assert store.get("r1").sql_rewritten == sql  # rewritten SQL kept for the trace


# Gateway outcomes

def test_success_stores_full_result_and_metadata():
    df = pd.DataFrame({"month": [f"m{i}" for i in range(60)], "revenue": [1.5] * 59 + [None]})
    store = ResultStore()
    out = gateway(FakeClient(df=df), store).run(
        "SELECT month, SUM(sale_price) AS revenue FROM sales GROUP BY month", CEO)
    assert out["result_id"] == "r1" and out["row_count"] == 60
    assert len(out["preview"]) == 50 and out["truncated"] and out["null_flags"] == {"revenue": 1}
    assert store.get("r1").meta == {"metric": "revenue", "filters": None, "grouping": "month"}


@pytest.mark.parametrize("sql, code", [
    ("SELECT * FROM users", "UNKNOWN_TABLE"),
    ("SELECT postal_code FROM customers", "GEO_TOO_FINE"),
    ("SELECT 1; SELECT 2", "MULTI_STATEMENT"),
    ("SELEC 1", "PARSE_ERROR"),
])
def test_repairable_rejections_never_reach_bigquery(sql, code):
    client = FakeClient()
    out = gateway(client).run(sql, CEO)
    assert out["error"]["code"] == code and out["error"]["hint"] is not None
    assert client.queries == []


@pytest.mark.parametrize("sql, code", [
    ("DELETE FROM sales WHERE TRUE", "NOT_SELECT"),
    ("SELECT email FROM customers", "PII_IDENTIFIER"),
])
def test_stop_codes_raise_and_never_reach_bigquery(sql, code):
    client = FakeClient()
    with pytest.raises(SafetyStop) as info:
        gateway(client).run(sql, BM)
    assert info.value.rejection.code == code and client.queries == []


def test_brand_backstop_blocks_out_of_scope_brand():
    client = FakeClient()
    out = gateway(client).run("SELECT SUM(sale_price) FROM sales WHERE brand = 'Carhartt'", BM)
    err = out["error"]
    assert err["code"] == "NO_ACCESS_BRAND" and err["brands"] == [{"value": "Carhartt", "status": "not in your access"}]
    assert client.queries == []


def test_brand_typo_suggests_own_brand():
    out = gateway(FakeClient()).run("SELECT SUM(sale_price) FROM sales WHERE brand = 'levis'", BM)
    assert out["error"]["brands"] == [{"value": "levis", "did_you_mean": ["Levi's"]}]


def test_bytes_over_cap_is_cost_limit():
    client = FakeClient(bytes_=10**12)
    out = gateway(client).run("SELECT * FROM sales", CEO)
    assert out["error"]["code"] == "COST_LIMIT" and len(client.queries) == 1


def test_bigquery_rejection_is_bq_invalid():
    exc = gexc.BadRequest("x", errors=[{"message": "Unrecognized name: revnue at [1:8]"}])
    out = gateway(FakeClient(dry_exc=exc)).run("SELECT revnue FROM sales", CEO)
    assert out["error"] == {"code": "BQ_INVALID", "message": "Unrecognized name: revnue at [1:8]",
                            "hint": "Fix the SQL using BigQuery's message and try again."}


def test_timeout_cancels_job():
    client = FakeClient(result_exc=TimeoutError())
    out = gateway(client).run("SELECT 1 FROM sales", CEO)
    assert out["error"]["code"] == "TIMEOUT" and client.last_job.cancelled


def test_permission_error_is_fatal():
    with pytest.raises(GatewayFatalError, match="doctor"):
        gateway(FakeClient(dry_exc=gexc.Forbidden("no access"))).run("SELECT 1 FROM sales", CEO)


def test_run_sql_tool_returns_json():
    tool = make_run_sql_tool(gateway(FakeClient(df=pd.DataFrame({"a": [1]}))), CEO)
    assert json.loads(tool.invoke({"sql": "SELECT a FROM sales"}))["preview"] == [{"a": 1}]


def test_store_unknown_id():
    with pytest.raises(ResultNotFound, match="none yet"):
        ResultStore().get("r9")


# Error codes by reason, transient retries (D2.3, D2.4)

@pytest.mark.parametrize("exc, code", [
    (bq_error(gexc.InternalServerError, "bytesBilledLimitExceeded"), "COST_LIMIT"),  # a 500 that isn't transient
    (bq_error(gexc.Forbidden, "rateLimitExceeded"), "BQ_TRANSIENT"),  # a 403 that is transient
    (bq_error(gexc.Forbidden, "quotaExceeded"), "BQ_FATAL"),
    (bq_error(gexc.Forbidden, "billingNotEnabled"), "BQ_FATAL"),
    (bq_error(gexc.BadRequest, "invalidQuery"), "BQ_INVALID"),
    (bq_error(gexc.InternalServerError, "backendError"), "BQ_TRANSIENT"),
    (gexc.ServiceUnavailable("down"), "BQ_TRANSIENT"),
    (gexc.BadRequest("no reason"), "BQ_INVALID"),
    (gexc.Forbidden("no reason"), "BQ_FATAL"),
    (ConnectionError("reset"), "BQ_TRANSIENT"),
])
def test_classify_by_reason_then_status(exc, code):
    assert classify_bq_error(exc) == code


def test_billing_cap_at_execution_is_cost_limit_not_retried():
    client = FakeClient(result_excs=[bq_error(gexc.InternalServerError, "bytesBilledLimitExceeded")])
    out = gateway(client).run("SELECT * FROM sales", CEO)
    assert out["error"]["code"] == "COST_LIMIT" and out["error"]["hint"].startswith("Select fewer columns")
    assert len(client.queries) == 2  # one dry run, one execution


def test_transient_errors_are_retried_invisibly():
    client = FakeClient(dry_excs=[bq_error(gexc.Forbidden, "rateLimitExceeded")],
                        result_excs=[gexc.ServiceUnavailable("blip")])
    gw = gateway(client)
    out = gw.run("SELECT a FROM sales", CEO)
    assert out["result_id"] == "r1" and gw.transient_retries == 2


def test_transient_errors_past_the_limit_mean_bigquery_is_down():
    client = FakeClient(result_excs=[gexc.ServiceUnavailable("down")] * 5)
    with pytest.raises(GatewayUnavailableError):
        gateway(client).run("SELECT a FROM sales", CEO)
    assert len(client.queries) == 1 + load_settings().bigquery.transient_max_attempts


def test_fatal_reason_is_not_retried():
    client = FakeClient(dry_exc=bq_error(gexc.Forbidden, "quotaExceeded", "Quota exceeded"))
    with pytest.raises(GatewayFatalError, match="Quota exceeded.*doctor"):
        gateway(client).run("SELECT a FROM sales", CEO)
    assert len(client.queries) == 1


def test_lowered_processing_cap_blocks_wide_query_and_lets_narrow_one_run():
    settings = load_settings().bigquery.model_copy(update={"max_bytes_processed": 15_000_000})
    wide = gateway(FakeClient(bytes_=20_400_000), settings=settings).run("SELECT * FROM sales", CEO)
    assert wide["error"]["code"] == "COST_LIMIT"
    client = FakeClient(bytes_=10_700_000)
    assert "result_id" in gateway(client, settings=settings).run("SELECT state FROM sales", CEO)
    assert client.queries[-1][1].maximum_bytes_billed == load_settings().bigquery.maximum_bytes_billed
