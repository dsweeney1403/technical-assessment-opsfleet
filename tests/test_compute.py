import pandas as pd
import pytest

from analyst.store.results import ResultStore
from analyst.tools.compute import run_compute


@pytest.fixture
def store():
    s = ResultStore()
    s.put(pd.DataFrame({"month": ["2026-01", "2026-02", "2026-03"], "revenue": [100.0, 120.0, 90.0],
                        "customers": [10, 12, 0]}), "q1", "q1", "2026-10-14T00:00:00+00:00")
    s.put(pd.DataFrame({"state": ["TX", "CA"], "revenue": [50.0, 80.0]}), "q2", "q2", "2026-10-13T00:00:00+00:00")
    s.put(pd.DataFrame({"state": ["TX", "CA", "NY"], "revenue": [60.0, 40.0, 10.0]}), "q3", "q3", "2026-10-14T00:00:00+00:00")
    return s


def rows(out):
    return out["preview"]


def test_total_and_average(store):
    assert rows(run_compute(store, "total", "r1", "revenue")) == [{"total_revenue": 310.0}]
    assert rows(run_compute(store, "average", "r1", "revenue"))[0]["average_revenue"] == pytest.approx(103.333, rel=1e-3)


def test_share(store):
    shares = [r["revenue_share_pct"] for r in rows(run_compute(store, "share", "r1", "revenue"))]
    assert shares == [32.26, 38.71, 29.03]


def test_rank(store):
    out = rows(run_compute(store, "rank", "r1", "revenue"))
    assert [(r["month"], r["revenue_rank"]) for r in out] == [("2026-02", 1), ("2026-01", 2), ("2026-03", 3)]


def test_row_over_row_percent_change(store):
    out = rows(run_compute(store, "percent_change", "r1", "revenue"))
    assert [r["revenue_pct_change"] for r in out] == [None, 20.0, -25.0]


def test_per_customer_handles_zero(store):
    out = rows(run_compute(store, "per_customer", "r1", "revenue", other_column="customers"))
    assert [r["revenue_per_customer"] for r in out] == [10.0, 10.0, None]


def test_join_two_results_on_key(store):
    out = {r["state"]: r for r in rows(run_compute(store, "percent_change", "r2", "revenue", other_result_id="r3", key="state"))}
    assert out["TX"]["revenue_pct_change"] == 20.0
    assert out["CA"]["revenue_pct_change"] == -50.0
    assert out["NY"]["revenue_pct_change"] is None  # no base value
    diff = {r["state"]: r["revenue_difference"] for r in rows(run_compute(store, "difference", "r2", "revenue", other_result_id="r3", key="state"))}
    assert diff["TX"] == 10.0


def test_output_is_stored_and_traceable(store):
    out = run_compute(store, "total", "r2", "revenue")
    stored = store.get(out["result_id"])
    assert out["derived_from"] == ["r2"]
    assert stored.sql.startswith("compute:total(r2")
    assert stored.data_ts == "2026-10-13T00:00:00+00:00"


@pytest.mark.parametrize("kwargs, msg", [
    ({"operation": "total", "result_id": "r9", "column": "revenue"}, "r9"),
    ({"operation": "total", "result_id": "r1", "column": "rev"}, "Unknown column"),
    ({"operation": "ratio", "result_id": "r1", "column": "revenue"}, "needs other_column"),
    ({"operation": "difference", "result_id": "r2", "column": "revenue", "other_result_id": "r3"}, "key"),
])
def test_errors_are_returned_not_raised(store, kwargs, msg):
    out = run_compute(store, **kwargs)
    assert out["error"]["code"] == "COMPUTE_ERROR" and msg in out["error"]["message"] + out["error"]["hint"]
