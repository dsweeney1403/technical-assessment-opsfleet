from datetime import date

import pandas as pd
import pytest

from analyst.config import User, load_settings
from analyst.sql import diagnosis
from analyst.sql.brands import BrandMatcher
from analyst.sql.diagnosis import KnownValues, diagnose, without_filters
from analyst.sql.gateway import Gateway
from analyst.sql.validator import validate
from analyst.store.results import ResultStore
from test_gateway import CURATED, FakeClient, FakeJob

CEO = User(id="ceo", name="CEO", role="ceo", brands=["*"])
KNOWN = KnownValues(
    values={"country": ["Brasil", "China", "United States"], "state": ["California", "Texas"],
            "city": ["Austin"], "status": ["Cancelled", "Complete"], "category": ["Jeans", "Outerwear & Coats"]},
    min_date=date(2019, 1, 5), max_date=date(2026, 10, 12),
)


def problems(sql):
    return diagnose(validate(sql).tree, KNOWN)


def test_misspelled_country_suggests_the_data_spelling():
    assert problems("SELECT SUM(sale_price) FROM sales WHERE country = 'Brazil'") == [
        {"column": "country", "value": "Brazil", "closest": ["Brasil"]}]


def test_wrong_case_is_caught_unless_compared_case_insensitively():
    assert problems("SELECT 1 FROM sales WHERE state = 'texas'")[0]["closest"] == ["Texas"]
    assert problems("SELECT 1 FROM sales WHERE LOWER(state) = 'texas'") == []


def test_in_lists_subqueries_and_literal_on_the_left():
    out = problems("SELECT 1 FROM (SELECT * FROM sales WHERE 'Cancled' = status) WHERE country IN ('China', 'Chnia')")
    assert [(p["column"], p["value"]) for p in out] == [("country", "Chnia"), ("status", "Cancled")]


def test_valid_filters_and_unknown_columns_are_fine():
    assert problems("SELECT 1 FROM sales WHERE country = 'Brasil' AND category = 'Jeans' AND product_name = 'x'") == []


def test_value_with_no_close_match_has_empty_closest():
    assert problems("SELECT 1 FROM sales WHERE city = 'Zzyzx'")[0]["closest"] == []


@pytest.mark.parametrize("where, flagged", [
    ("order_created_at >= '2027-01-01'", True),
    ("DATE(order_created_at) > DATE '2027-01-01'", True),
    ("order_created_at < TIMESTAMP '2018-06-01'", True),
    ("'2027-01-01' <= order_created_at", True),
    ("DATE(order_created_at) BETWEEN '2030-01-01' AND '2030-12-31'", True),
    ("order_created_at >= '2026-01-01'", False),
    ("order_created_at < '2027-01-01'", False),
    ("DATE(order_created_at) BETWEEN '2026-01-01' AND '2026-12-31'", False),
])
def test_dates_outside_the_data_range(where, flagged):
    out = problems(f"SELECT 1 FROM sales WHERE {where}")
    assert bool(out) == flagged
    if flagged:
        assert out[0]["note"] == "Order dates run from 2019-01-05 to 2026-10-12."


def test_without_filters_strips_every_where_and_having():
    sql = ("WITH t AS (SELECT state, SUM(sale_price) AS r FROM sales WHERE status = 'Complete' GROUP BY 1 HAVING r > 5) "
           "SELECT * FROM t JOIN customers c USING (state) WHERE r > 10")
    out = without_filters(validate(sql).tree)
    assert out.startswith("SELECT * FROM (") and out.endswith(") LIMIT 1")
    assert "WHERE" not in out and "HAVING" not in out and "JOIN" in out


@pytest.mark.parametrize("df, nothing", [
    (pd.DataFrame({"revenue": pd.Series([], dtype=float)}), True),
    (pd.DataFrame({"revenue": [None]}), True),  # SUM over no rows
    (pd.DataFrame({"n": [0]}), True),  # COUNT over no rows
    (pd.DataFrame({"n": [0], "revenue": [float("nan")]}), True),
    (pd.DataFrame({"revenue": [12.5]}), False),
    (pd.DataFrame({"state": ["Texas"], "n": [0]}), False),  # a grouped row is a real answer
    (pd.DataFrame({"n": [0, 0]}), False),
])
def test_no_rows_matched(df, nothing):
    assert diagnosis.no_rows_matched(df) == nothing


# Through the gateway

UNFILTERED = ") LIMIT 1"


class CountingClient(FakeClient):
    """Main query: SUM over no rows (one NULL row). The unfiltered rerun returns `count` revenue."""

    def __init__(self, count):
        super().__init__(df=pd.DataFrame({"revenue": [None]}))
        self.count = count

    def query(self, sql, job_config=None):
        if not job_config.dry_run and sql.endswith(UNFILTERED):
            self.queries.append((sql, job_config))
            self.last_job = FakeJob(df=pd.DataFrame({"revenue": [float(self.count) or None]}))
            return self.last_job
        return super().query(sql, job_config)


def gateway(client, known=KNOWN):
    return Gateway(load_settings().bigquery, client, ResultStore(), CURATED, BrandMatcher([]),
                   sleep=lambda s: None, known=known)


def test_empty_result_with_bad_value_returns_diagnosis_and_skips_the_count():
    client = CountingClient(count=5)
    out = gateway(client).run("SELECT SUM(sale_price) AS revenue FROM sales WHERE country = 'Brazil'", CEO)
    assert out["diagnosis"]["filter_problems"][0]["closest"] == ["Brasil"]
    assert out["diagnosis"]["hint"] == diagnosis.FILTER_HINT
    assert not any(q.endswith(UNFILTERED) for q, _ in client.queries)


def test_empty_even_without_filters_points_to_joins():
    out = gateway(CountingClient(count=0)).run("SELECT SUM(sale_price) AS revenue FROM sales WHERE country = 'China'", CEO)
    assert out["diagnosis"] == {"structural": "empty_without_filters", "hint": diagnosis.JOIN_HINT}


def test_rows_without_filters_means_a_genuine_empty():
    client = CountingClient(count=12)
    out = gateway(client).run("SELECT SUM(sale_price) AS revenue FROM sales WHERE country = 'China'", CEO)
    assert out["diagnosis"]["structural"] == "genuine_empty"
    count_sql = next(q for q, _ in client.queries if q.endswith(UNFILTERED))
    assert count_sql.startswith("WITH sales AS (")  # runs through the same scope rewrite


def test_non_empty_result_has_no_diagnosis():
    out = gateway(FakeClient()).run("SELECT a FROM sales WHERE country = 'Brazil'", CEO)
    assert "diagnosis" not in out
