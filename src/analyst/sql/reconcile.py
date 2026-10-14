"""Reconciliation check (D1.20): completeness is tested, not assumed.

The unscoped curated tables must keep every row of the raw tables, including
customers who never ordered and failed orders. One query, so both sides read
the same snapshot of the live public tables.
"""

from __future__ import annotations

from dataclasses import dataclass

from google.cloud import bigquery

from analyst.sql.gateway import wrap_sql

TOLERANCE = 0.01  # revenue, in currency units


@dataclass
class ReconCheck:
    name: str
    curated: float
    raw: float

    @property
    def ok(self) -> bool:
        return abs(self.curated - self.raw) <= TOLERANCE


def reconciliation_sql(curated: dict[str, str], dataset: str) -> str:
    raw = f"`{dataset}"
    body = f"""
SELECT
  (SELECT COUNT(*) FROM sales) AS sales_rows,
  (SELECT COUNT(*) FROM {raw}.order_items`) AS raw_order_items,
  (SELECT COUNT(DISTINCT order_id) FROM sales) AS sales_orders,
  (SELECT COUNT(*) FROM {raw}.orders`) AS raw_orders,
  (SELECT COUNT(*) FROM customers) AS customers_rows,
  (SELECT COUNT(*) FROM {raw}.users`) AS raw_users,
  (SELECT COUNT(*) FROM products) AS products_rows,
  (SELECT COUNT(*) FROM {raw}.products`) AS raw_products,
  ARRAY(SELECT AS STRUCT status, ROUND(SUM(sale_price), 2) AS revenue FROM sales GROUP BY status ORDER BY status) AS sales_revenue,
  ARRAY(SELECT AS STRUCT status, ROUND(SUM(sale_price), 2) AS revenue FROM {raw}.order_items` GROUP BY status ORDER BY status) AS raw_revenue
"""
    return wrap_sql(body, curated)


def run_reconciliation(client: bigquery.Client, curated: dict[str, str], dataset: str) -> list[ReconCheck]:
    row = next(iter(client.query(reconciliation_sql(curated, dataset)).result()))
    checks = [
        ReconCheck("sales rows = order_items rows", row["sales_rows"], row["raw_order_items"]),
        ReconCheck("distinct orders in sales = orders rows", row["sales_orders"], row["raw_orders"]),
        ReconCheck("customers rows = users rows", row["customers_rows"], row["raw_users"]),
        ReconCheck("products rows = products rows", row["products_rows"], row["raw_products"]),
    ]
    raw_rev = {r["status"]: r["revenue"] for r in row["raw_revenue"]}
    cur_rev = {r["status"]: r["revenue"] for r in row["sales_revenue"]}
    for status in sorted(set(raw_rev) | set(cur_rev)):
        checks.append(ReconCheck(f"revenue for {status}", cur_rev.get(status, 0.0), raw_rev.get(status, 0.0)))
    return checks
