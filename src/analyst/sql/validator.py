"""SQL validator: the gateway's first gate (D1.4, D2.1, Group 2 error codes).

The model may only run one SELECT over `sales`, `customers`, `products` and its
own CTEs. Stop codes (NOT_SELECT, PII_IDENTIFIER) end the turn as a security
event; every other code goes back to the agent with a hint so it can repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

CURATED_TABLES = ("sales", "customers", "products")
PII_COLUMNS = {"first_name", "last_name", "email", "street_address"}
GEO_COLUMNS = {"postal_code", "latitude", "longitude", "user_geom"}
STOP_CODES = {"NOT_SELECT", "PII_IDENTIFIER"}
SCRIPTING_WORDS = ("BEGIN", "DECLARE", "SET", "IF", "LOOP", "WHILE", "REPEAT", "FOR")
BLOCKED_FUNCTION_PREFIXES = ("EXTERNAL_QUERY", "ML.", "AI.", "KEYS.", "VECTOR_SEARCH")
AGGREGATES = (exp.AggFunc,)


@dataclass
class Rejection:
    code: str
    message: str
    hint: str
    brands: list[dict] = field(default_factory=list)

    @property
    def stop(self) -> bool:
        return self.code in STOP_CODES

    def as_tool_error(self) -> dict:
        err = {"code": self.code, "message": self.message, "hint": self.hint}
        if self.brands:
            err["brands"] = self.brands
        return {"error": err}


@dataclass
class Validated:
    tree: exp.Query
    brand_literals: list[str]
    cte_names: set[str]


def _function_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).upper()
    if isinstance(node, exp.Dot):
        return node.sql("bigquery").split("(")[0].upper()
    return (node.sql_name() or "").upper()


def _has_blocked_function(tree: exp.Expression) -> str | None:
    for node in tree.walk():
        if isinstance(node, exp.Dot) and isinstance(node.expression, exp.Anonymous):
            return node.sql("bigquery").split("(")[0]  # dotted call: a persistent UDF
        if isinstance(node, (exp.Func, exp.Anonymous)):
            name = _function_name(node)
            if name.startswith(BLOCKED_FUNCTION_PREFIXES):
                return name
    return None


def _brand_literals(tree: exp.Expression) -> list[str]:
    """String literals compared with the brand column via =, != or IN."""

    def is_brand(node: exp.Expression) -> bool:
        while isinstance(node, (exp.Lower, exp.Upper, exp.Trim)):
            node = node.this
        return isinstance(node, exp.Column) and node.name.lower() == "brand"

    found: list[str] = []
    for node in tree.find_all(exp.EQ, exp.NEQ, exp.In):
        if isinstance(node, exp.In):
            if is_brand(node.this):
                found += [e.this for e in node.expressions if isinstance(e, exp.Literal) and e.is_string]
        else:
            left, right = node.this, node.expression
            if is_brand(left) and isinstance(right, exp.Literal) and right.is_string:
                found.append(right.this)
            elif is_brand(right) and isinstance(left, exp.Literal) and left.is_string:
                found.append(left.this)
    return found


def validate(sql: str) -> Validated | Rejection:
    try:
        statements = [s for s in sqlglot.parse(sql, read="bigquery") if s is not None]
    except ParseError as exc:
        first = exc.errors[0]["description"] if exc.errors else str(exc).splitlines()[0]
        return Rejection("PARSE_ERROR", f"SQL could not be parsed: {first}", "Fix the syntax and try again.")
    if not statements:
        return Rejection("PARSE_ERROR", "No SQL statement found.", "Send one SELECT statement.")

    scripting = False
    for stmt in statements:
        if isinstance(stmt, (exp.Declare, exp.Set)):
            scripting = True
        elif isinstance(stmt, exp.Command):
            if str(stmt.this).upper().startswith(SCRIPTING_WORDS):
                scripting = True
            else:
                return Rejection("NOT_SELECT", "Only read-only SELECT queries are allowed.", "")
        elif not isinstance(stmt, exp.Query):
            return Rejection("NOT_SELECT", "Only read-only SELECT queries are allowed.", "")
    if scripting or len(statements) > 1:
        return Rejection(
            "MULTI_STATEMENT",
            "Only a single SELECT statement is allowed.",
            "Write it as a single SELECT. Put constants in a CTE instead of DECLARE or SET.",
        )
    tree: exp.Query = statements[0]

    blocked = _has_blocked_function(tree)
    if blocked:
        return Rejection("NOT_SELECT", f"Function {blocked} is not allowed.", "")

    for col in tree.find_all(exp.Column):
        name = col.name.lower()
        if name in PII_COLUMNS:
            return Rejection("PII_IDENTIFIER", "That query asks for personal data, which isn't available.", "")
        if name in GEO_COLUMNS:
            return Rejection(
                "GEO_TOO_FINE",
                f"Column {col.name} is finer than allowed.",
                "The finest location level allowed is city. Use country, state or city.",
            )

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    shadowed = cte_names & set(CURATED_TABLES)
    if shadowed:
        return Rejection(
            "UNKNOWN_TABLE",
            f"A CTE may not be named {', '.join(sorted(shadowed))}.",
            "sales, customers and products already exist. Give your CTE another name.",
        )
    allowed = set(CURATED_TABLES) | cte_names
    for table in tree.find_all(exp.Table):
        qualified = bool(table.args.get("db") or table.args.get("catalog"))
        if qualified or table.name.lower() not in allowed:
            return Rejection(
                "UNKNOWN_TABLE",
                f"Table {table.sql('bigquery')} is not available.",
                "Query only sales, customers and products, written without any project or dataset prefix.",
            )

    return Validated(tree=tree, brand_literals=_brand_literals(tree), cte_names=cte_names)


def describe(tree: exp.Query) -> dict:
    """Metric, filters and grouping of the outermost SELECT, for records (D1.13)."""
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None:
        return {"metric": None, "filters": None, "grouping": None}
    metrics = [p.alias_or_name for p in select.expressions if p.find(*AGGREGATES)]
    where = select.args.get("where")
    group = select.args.get("group")
    return {
        "metric": ", ".join(metrics) or None,
        "filters": where.this.sql("bigquery") if where else None,
        "grouping": ", ".join(e.sql("bigquery") for e in group.expressions) if group else None,
    }
