"""Empty results: prevention values, diagnosis and the structural check (D2.5).

An empty result isn't trusted as an answer until code has checked it.
- Known values: distinct values of the filterable columns and the order date
  range, loaded once per session from the unscoped curated tables. None of it
  is brand data, so every role may see it. The short columns are listed in the
  prompt so the model spells values right first time.
- Diagnosis: on an empty result, each string filter value is checked against
  the known values ("no match for 'Brazil' in country, closest: 'Brasil'"), and
  date filters against the data's date range.
- Structural check: when diagnosis finds nothing, the query runs once more with
  every filter removed. Still nothing tells a join problem from a genuine empty.

"Nothing" includes an aggregate with no GROUP BY: SUM over no rows returns one
row of NULL, and COUNT one row of 0, never an empty result.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlglot import exp

SHORT_COLUMNS = ("department", "gender", "status", "traffic_source", "age_band", "country", "category")
LONG_COLUMNS = ("state", "city")
CLOSEST_CUTOFF = 0.6

KNOWN_VALUES_SQL = """SELECT
  (SELECT ARRAY_AGG(DISTINCT department IGNORE NULLS ORDER BY department) FROM products) AS department,
  (SELECT ARRAY_AGG(DISTINCT category IGNORE NULLS ORDER BY category) FROM products) AS category,
  (SELECT ARRAY_AGG(DISTINCT status IGNORE NULLS ORDER BY status) FROM sales) AS status,
  (SELECT ARRAY_AGG(DISTINCT gender IGNORE NULLS ORDER BY gender) FROM customers) AS gender,
  (SELECT ARRAY_AGG(DISTINCT traffic_source IGNORE NULLS ORDER BY traffic_source) FROM customers) AS traffic_source,
  (SELECT ARRAY_AGG(DISTINCT age_band IGNORE NULLS ORDER BY age_band) FROM customers) AS age_band,
  (SELECT ARRAY_AGG(DISTINCT country IGNORE NULLS ORDER BY country) FROM customers) AS country,
  (SELECT ARRAY_AGG(DISTINCT state IGNORE NULLS ORDER BY state) FROM customers) AS state,
  (SELECT ARRAY_AGG(DISTINCT city IGNORE NULLS ORDER BY city) FROM customers) AS city,
  (SELECT MIN(DATE(order_created_at)) FROM sales) AS min_date,
  (SELECT MAX(DATE(order_created_at)) FROM sales) AS max_date"""


@dataclass(frozen=True)
class KnownValues:
    values: dict[str, list[str]]
    min_date: date
    max_date: date


def load_known_values(client, curated: dict[str, str]) -> KnownValues:
    """One query over the unscoped curated tables. Code-written, so it doesn't go through the gateway."""
    from analyst.sql.gateway import wrap_sql  # avoid a circular import

    row = next(iter(client.query(wrap_sql(KNOWN_VALUES_SQL, curated)).result()))
    values = {col: list(row[col] or []) for col in (*SHORT_COLUMNS, *LONG_COLUMNS)}
    return KnownValues(values, row["min_date"], row["max_date"])


# --- Diagnosis -------------------------------------------------------------

def _column(node: exp.Expression) -> tuple[str, bool] | None:
    """(column name, compared case-insensitively) if node is a column, maybe inside LOWER, UPPER or TRIM."""
    folded = False
    while isinstance(node, (exp.Lower, exp.Upper, exp.Trim)):
        folded = folded or not isinstance(node, exp.Trim)
        node = node.this
    if isinstance(node, exp.Column):
        return node.name.lower(), folded
    return None


def _strings(nodes: list[exp.Expression]) -> list[str]:
    return [n.this for n in nodes if isinstance(n, exp.Literal) and n.is_string]


def _closest(value: str, known: list[str]) -> list[str]:
    by_lower: dict[str, list[str]] = {}
    for k in known:
        by_lower.setdefault(k.lower(), []).append(k)
    if value.lower() in by_lower:
        return by_lower[value.lower()]
    close = difflib.get_close_matches(value.lower(), list(by_lower), n=3, cutoff=CLOSEST_CUTOFF)
    return [k for c in close for k in by_lower[c]]


def _value_problems(tree: exp.Expression, known: KnownValues) -> list[dict]:
    problems = []
    for node in tree.find_all(exp.EQ, exp.In):
        pairs = [(node.this, node.expressions)] if isinstance(node, exp.In) else [
            (node.this, [node.expression]), (node.expression, [node.this])]
        for side, others in pairs:
            col = _column(side)
            if col is None or col[0] not in known.values:
                continue
            name, folded = col
            pool = known.values[name]
            have = {v.lower() for v in pool} if folded else set(pool)
            for value in _strings(others):
                if (value.lower() if folded else value) not in have:
                    problems.append({"column": name, "value": value, "closest": _closest(value, pool)})
    return problems


FLIP = {exp.GT: exp.LT, exp.GTE: exp.LTE, exp.LT: exp.GT, exp.LTE: exp.GTE, exp.EQ: exp.EQ}


def _date_of(node: exp.Expression) -> date | None:
    for lit in [node, *node.find_all(exp.Literal)]:
        if isinstance(lit, exp.Literal) and lit.is_string:
            try:
                return date.fromisoformat(lit.this[:10])
            except ValueError:
                return None
    return None


def _is_date_column(node: exp.Expression) -> str | None:
    col = node if isinstance(node, exp.Column) else node.find(exp.Column)
    if col is not None and col.name.lower().endswith("_at") and not isinstance(node, exp.Literal):
        return col.name.lower()
    return None


def _date_problems(tree: exp.Expression, known: KnownValues) -> list[dict]:
    lo, hi = known.min_date, known.max_date
    note = f"Order dates run from {lo.isoformat()} to {hi.isoformat()}."
    problems = []
    for node in tree.find_all(exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ, exp.Between):
        if isinstance(node, exp.Between):
            col, low, high = _is_date_column(node.this), _date_of(node.args["low"]), _date_of(node.args["high"])
            if col and low and high and (high < lo or low > hi):
                problems.append({"column": col, "value": f"{low} to {high}", "note": note})
            continue
        for side, other, op in ((node.this, node.expression, type(node)), (node.expression, node.this, FLIP[type(node)])):
            col, when = _is_date_column(side), _date_of(other)
            if not col or when is None or _is_date_column(other):
                continue
            outside = ((op in (exp.GT, exp.GTE) and when > hi) or (op in (exp.LT, exp.LTE) and when < lo)
                       or (op is exp.EQ and not lo <= when <= hi))
            if outside:
                problems.append({"column": col, "value": when.isoformat(), "note": note})
    return problems


def diagnose(tree: exp.Expression, known: KnownValues) -> list[dict]:
    """Filter values that can't match anything, with the closest known values."""
    return _value_problems(tree, known) + _date_problems(tree, known)


def no_rows_matched(df: pd.DataFrame) -> bool:
    """No rows, or one row of only NULLs and zeros (an aggregate over nothing)."""
    if df.empty:
        return True
    if len(df) > 1:
        return False
    return all(pd.isna(v) or (isinstance(v, (int, float)) and v == 0) for v in df.iloc[0].tolist())


def without_filters(tree: exp.Expression) -> str:
    """The query with every WHERE, HAVING and QUALIFY removed; one row is enough to tell."""
    stripped = tree.copy()
    for select in stripped.find_all(exp.Select):
        for arg in ("where", "having", "qualify"):
            select.set(arg, None)
    return f"SELECT * FROM ({stripped.sql('bigquery')}) LIMIT 1"


FILTER_HINT = "No row matches these filter values. Use the closest matches, or tell the user the value doesn't exist."
JOIN_HINT = "Returns nothing even without filters, check your joins."
GENUINE_HINT = ("The filters genuinely exclude every row. Say so in the answer, "
                "and state the filters and date range used.")
