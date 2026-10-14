import pandas as pd

from analyst.store.results import ResultStore
from analyst.tools.chart import render_chart


def store_with(df):
    s = ResultStore()
    s.put(df, "SELECT 1", "SELECT 1", "t")
    return s


def test_renders_png(tmp_path):
    s = store_with(pd.DataFrame({"m": ["a", "b"], "v": [1, 2]}))
    out = render_chart(s, tmp_path, "r1", "line", "m", ["v"], "Revenue by month")
    assert out["chart_path"].endswith(".png")
    assert (tmp_path / out["chart_path"].split("/")[-1]).exists()


def test_unknown_column_lists_valid_ones(tmp_path):
    out = render_chart(store_with(pd.DataFrame({"m": [1], "v": [1]})), tmp_path, "r1", "bar", "month", ["v"], "t")
    assert "month" in out["error"]["message"] and "m, v" in out["error"]["hint"]


def test_unknown_result_id(tmp_path):
    out = render_chart(ResultStore(), tmp_path, "r7", "bar", "m", ["v"], "t")
    assert "r7" in out["error"]["message"]
