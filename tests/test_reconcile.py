"""Reconciliation check (D1.20): the comparison logic, without BigQuery."""

from analyst.sql.reconcile import ReconCheck, reconciliation_sql, run_reconciliation

CURATED = {"sales": "SELECT 1 AS a", "customers": "SELECT 2 AS b", "products": "SELECT 3 AS c"}


class OneRow:
    def __init__(self, row):
        self.row = row
        self.sql = None

    def query(self, sql):
        self.sql = sql
        return self

    def result(self):
        return [self.row]


def row(**overrides):
    base = {"sales_rows": 100, "raw_order_items": 100, "sales_orders": 40, "raw_orders": 40,
            "customers_rows": 80, "raw_users": 80, "products_rows": 30, "raw_products": 30,
            "sales_revenue": [{"status": "Complete", "revenue": 500.0}, {"status": "Returned", "revenue": 50.0}],
            "raw_revenue": [{"status": "Complete", "revenue": 500.0}, {"status": "Returned", "revenue": 50.0}]}
    return {**base, **overrides}


def test_matching_tables_pass_every_check():
    checks = run_reconciliation(OneRow(row()), CURATED, "ds")
    assert all(c.ok for c in checks) and len(checks) == 6


def test_a_dropped_row_or_status_fails():
    client = OneRow(row(sales_rows=99, sales_revenue=[{"status": "Complete", "revenue": 500.0}]))
    failed = {c.name for c in run_reconciliation(client, CURATED, "ds") if not c.ok}
    assert failed == {"sales rows = order_items rows", "revenue for Returned"}  # a status missing entirely counts


def test_rounding_tolerance_and_the_query_reads_both_sides():
    assert ReconCheck("r", 100.004, 100.0).ok and not ReconCheck("r", 100.02, 100.0).ok
    sql = reconciliation_sql(CURATED, "bigquery-public-data.thelook_ecommerce")
    assert sql.startswith("WITH sales AS") and "`bigquery-public-data.thelook_ecommerce.order_items`" in sql
