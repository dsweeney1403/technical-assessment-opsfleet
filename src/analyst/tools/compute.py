"""`compute` tool: deterministic arithmetic on stored results (D1.11, D1.17).

A fixed menu of operations, so every number in an answer comes from code. No
model-written code runs. Inputs and outputs are result IDs; each output is
stored like a query result so it can be charted, reused and grounded.
"""

from __future__ import annotations

import json
from typing import Literal

import numpy as np
import pandas as pd
from langchain_core.tools import StructuredTool

from analyst.sql.gateway import compact
from analyst.store.results import ResultNotFound, ResultStore

Operation = Literal["total", "average", "share", "percent_change", "difference", "ratio", "per_customer", "rank"]

DESCRIPTION = """Do arithmetic on stored results. Never calculate figures yourself; use this instead.
Operations:
- total / average: sum or mean of `column` over all rows.
- share: each row's percentage of the column total.
- rank: rank rows by `column`, highest first.
- percent_change, difference, ratio, per_customer: compare `column` with `other_column`.
  per_customer is column divided by a customer count in other_column.
  percent_change is (other_column - column) / column * 100, so column is the earlier value.
  With other_result_id and key, the two results are joined on key and `column` is compared across them
  (result_id is the earlier or base result).
  percent_change with no other_column and no other_result_id gives row-over-row change in the current row order.
Returns a new result_id and a preview."""


def _error(message: str, hint: str = "") -> dict:
    return {"error": {"code": "COMPUTE_ERROR", "message": message, "hint": hint}}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def run_compute(
    store: ResultStore,
    operation: Operation,
    result_id: str,
    column: str,
    other_column: str | None = None,
    other_result_id: str | None = None,
    key: str | None = None,
) -> dict:
    try:
        base = store.get(result_id)
        other = store.get(other_result_id) if other_result_id else None
    except ResultNotFound as exc:
        return _error(str(exc), "Use a result_id from the stored results.")
    df = base.df.copy()
    if column not in df.columns:
        return _error(f"Unknown column {column}.", f"Columns in {result_id}: {', '.join(df.columns)}.")

    if other is not None:
        if not key or key not in df.columns or key not in other.df.columns:
            return _error("Joining two results needs a key column present in both.",
                          f"{result_id}: {', '.join(df.columns)}; {other_result_id}: {', '.join(other.df.columns)}.")
        if column not in other.df.columns:
            return _error(f"Column {column} is not in {other_result_id}.", f"Columns: {', '.join(other.df.columns)}.")
        df = df[[key, column]].merge(other.df[[key, column]], on=key, how="outer", suffixes=("_base", "_other"))
        a, b = _num(df, f"{column}_base"), _num(df, f"{column}_other")
    elif other_column:
        if other_column not in df.columns:
            return _error(f"Unknown column {other_column}.", f"Columns in {result_id}: {', '.join(df.columns)}.")
        a, b = _num(df, column), _num(df, other_column)
    else:
        a, b = _num(df, column), None

    if operation == "total":
        out = pd.DataFrame({f"total_{column}": [a.sum()]})
    elif operation == "average":
        out = pd.DataFrame({f"average_{column}": [a.mean()]})
    elif operation == "share":
        df[f"{column}_share_pct"] = (_safe_div(a, pd.Series(a.sum(), index=a.index)) * 100).round(2)
        out = df
    elif operation == "rank":
        df[f"{column}_rank"] = a.rank(ascending=False, method="min").astype("Int64")
        out = df.sort_values(f"{column}_rank")
    elif operation == "percent_change":
        if b is None:
            df[f"{column}_pct_change"] = (a.pct_change() * 100).round(2)
        else:
            df[f"{column}_pct_change"] = (_safe_div(b - a, a) * 100).round(2)
        out = df
    elif b is None:
        return _error(f"{operation} needs other_column, or other_result_id with key.")
    elif operation == "difference":
        df[f"{column}_difference"] = b - a if other is not None else a - b
        out = df
    elif operation in ("ratio", "per_customer"):
        name = f"{column}_per_customer" if operation == "per_customer" else f"{column}_ratio"
        df[name] = _safe_div(a, b) if other is None else _safe_div(b, a)
        out = df
    else:
        return _error(f"Unknown operation {operation}.")

    out = out.reset_index(drop=True)
    label = f"compute:{operation}({result_id}{', ' + other_result_id if other_result_id else ''}; {column}"
    label += f", {other_column}" if other_column else ""
    label += f", key={key})" if key else ")"
    stored = store.put(out, label, label, min(base.data_ts, other.data_ts) if other else base.data_ts,
                       meta={"metric": operation, "filters": None, "grouping": key})
    return {**compact(stored, out, 50), "derived_from": [r for r in (result_id, other_result_id) if r]}


def make_compute_tool(store: ResultStore) -> StructuredTool:
    def compute(
        operation: Operation,
        result_id: str,
        column: str,
        other_column: str | None = None,
        other_result_id: str | None = None,
        key: str | None = None,
    ) -> str:
        return json.dumps(run_compute(store, operation, result_id, column, other_column, other_result_id, key), default=str)

    return StructuredTool.from_function(compute, name="compute", description=DESCRIPTION)
