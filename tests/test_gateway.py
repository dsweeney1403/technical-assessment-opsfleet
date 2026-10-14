import json

import pandas as pd
import pytest
from google.api_core import exceptions as gexc

from analyst.config import User, load_settings
from analyst.sql.gateway import (
    Gateway,
    GatewayFatalError,
    ScopeNotAvailableError,
    load_curated,
    wrap_sql,
)
from analyst.store.results import ResultNotFound, ResultStore
from analyst.tools.run_sql import make_run_sql_tool

CEO = User(id="ceo", name="CEO", role="ceo", brands=["*"])
BM = User(id="bm", name="BM", role="brand_manager", brands=["Diesel"])
CURATED = {"sales": "SELECT 1 AS a", "customers": "SELECT 2 AS b", "products": "SELECT 3 AS c"}


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
    def __init__(self, df=None, bytes_=1000, dry_exc=None, result_exc=None):
        self.df, self.bytes_, self.dry_exc, self.result_exc = df, bytes_, dry_exc, result_exc
        self.queries = []
        self.last_job = None

    def query(self, sql, job_config=None):
        self.queries.append((sql, job_config))
        if job_config.dry_run:
            if self.dry_exc:
                raise self.dry_exc
            return FakeJob(bytes_=self.bytes_)
        self.last_job = FakeJob(df=self.df, result_exc=self.result_exc)
        return self.last_job


def gateway(client, store=None):
    return Gateway(load_settings().bigquery, client, store or ResultStore(), CURATED)


def test_curated_file_has_three_tables_and_no_pii():
    tables = load_curated("ds")
    assert set(tables) == {"sales", "customers", "products"}
    text = " ".join(tables.values()).lower()
    for col in ["email", "first_name", "last_name", "street_address", "postal_code", "latitude", "longitude", "user_geom"]:
        assert col not in text
    assert "u.age," not in text  # banded only (D1.6)
    assert "`ds.order_items`" in tables["sales"]


def test_wrap_plain_select():
    out = wrap_sql("SELECT a FROM sales;", CURATED)
    assert out.startswith("WITH sales AS (")
    assert out.rstrip().endswith("SELECT a FROM sales")


def test_wrap_merges_model_with_clause_and_strips_comments():
    out = wrap_sql("-- note\nwith t AS (SELECT a FROM sales) SELECT * FROM t", CURATED)
    assert out.count("WITH") == 1
    assert "products AS (\nSELECT 3 AS c\n),\nt AS (SELECT a FROM sales)" in out


def test_successful_query_stores_full_result_and_returns_compact_preview():
    df = pd.DataFrame({"month": [f"2026-{m:02d}" for m in range(1, 61)] , "revenue": [1.5] * 59 + [None]})
    store = ResultStore()
    client = FakeClient(df=df)
    out = gateway(client, store).run("SELECT month, revenue FROM sales", CEO)
    assert out["result_id"] == "r1" and out["row_count"] == 60
    assert len(out["preview"]) == 50 and out["truncated"] is True
    assert out["null_flags"] == {"revenue": 1} and out["empty"] is False
    stored = store.get("r1")
    assert len(stored.df) == 60 and stored.sql == "SELECT month, revenue FROM sales"
    assert stored.sql_rewritten.startswith("WITH sales AS")
    _, config = client.queries[-1]
    assert config.maximum_bytes_billed == load_settings().bigquery.maximum_bytes_billed


def test_bytes_over_cap_stops_before_execution():
    client = FakeClient(bytes_=10**12)
    out = gateway(client).run("SELECT * FROM sales", CEO)
    assert "cap" in out["error"]["message"]
    assert len(client.queries) == 1  # dry run only


def test_bad_sql_returns_error_for_agent():
    exc = gexc.BadRequest("x", errors=[{"message": "Unrecognized name: revnue at [1:8]"}])
    out = gateway(FakeClient(dry_exc=exc)).run("SELECT revnue FROM sales", CEO)
    assert out["error"]["message"] == "Unrecognized name: revnue at [1:8]"


def test_timeout_cancels_job():
    client = FakeClient(df=pd.DataFrame(), result_exc=TimeoutError())
    out = gateway(client).run("SELECT 1", CEO)
    assert "cancelled" in out["error"]["message"] and client.last_job.cancelled


def test_permission_error_is_fatal():
    with pytest.raises(GatewayFatalError, match="doctor"):
        gateway(FakeClient(dry_exc=gexc.Forbidden("no access"))).run("SELECT 1", CEO)


def test_scoped_users_blocked_until_scoping_exists():
    client = FakeClient()
    with pytest.raises(ScopeNotAvailableError):
        gateway(client).run("SELECT 1", BM)
    assert client.queries == []


def test_empty_result_flag():
    out = gateway(FakeClient(df=pd.DataFrame({"a": []}))).run("SELECT a FROM sales", CEO)
    assert out["empty"] is True and out["row_count"] == 0


def test_run_sql_tool_returns_json():
    tool = make_run_sql_tool(gateway(FakeClient(df=pd.DataFrame({"a": [1]}))), CEO)
    out = json.loads(tool.invoke({"sql": "SELECT a FROM sales"}))
    assert out["preview"] == [{"a": 1}]


def test_store_unknown_id():
    with pytest.raises(ResultNotFound, match="none yet"):
        ResultStore().get("r9")
