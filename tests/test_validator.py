import pytest

from analyst.sql.validator import Rejection, Validated, describe, validate


def code(sql):
    out = validate(sql)
    return out.code if isinstance(out, Rejection) else "OK"


# Allowed shapes
@pytest.mark.parametrize("sql", [
    "SELECT brand, SUM(sale_price) AS revenue FROM sales GROUP BY brand",
    "select * from Sales s join PRODUCTS p using (product_id)",
    "WITH t AS (SELECT state, COUNT(DISTINCT customer_id) AS n FROM sales GROUP BY 1) SELECT * FROM t ORDER BY n DESC",
    "SELECT a FROM (SELECT state AS a FROM customers) AS sub",
    "SELECT state FROM sales UNION ALL SELECT state FROM customers",
    "-- a comment\nSELECT 1 /* inline */ FROM sales;",
    "SELECT x FROM UNNEST([1, 2, 3]) AS x",
    "SELECT country, city, age_band FROM customers",
    "SELECT COUNT(*) FROM sales WHERE brand = \"Levi's\"",
    "SELECT COUNT(*) FROM sales WHERE brand = 'Levi\\'s'",
])
def test_allowed(sql):
    assert code(sql) == "OK"


# Stop codes (D2.1)
@pytest.mark.parametrize("sql", [
    "DELETE FROM sales WHERE TRUE",
    "UPDATE sales SET sale_price = 0 WHERE TRUE",
    "INSERT INTO sales (order_id) VALUES (1)",
    "DROP TABLE sales",
    "CREATE TABLE x AS SELECT 1",
    "MERGE sales t USING products s ON t.product_id = s.product_id WHEN MATCHED THEN DELETE",
    "EXPORT DATA OPTIONS(uri='gs://bucket/*', format='CSV') AS SELECT * FROM sales",
    "CALL some_procedure()",
    "SELECT 1 FROM sales; DELETE FROM sales WHERE TRUE",
    "CREATE TEMP FUNCTION f() AS (1); SELECT f()",
    "SELECT * FROM EXTERNAL_QUERY('conn', 'SELECT 1')",
    "SELECT bqutil.fn.median([1, 2]) FROM sales",
])
def test_not_select_stops(sql):
    out = validate(sql)
    assert out.code == "NOT_SELECT" and out.stop


@pytest.mark.parametrize("col", ["email", "first_name", "last_name", "street_address", "EMAIL"])
def test_pii_columns_stop(col):
    out = validate(f"SELECT {col} FROM customers")
    assert out.code == "PII_IDENTIFIER" and out.stop


def test_pii_in_subquery_and_qualified():
    assert code("SELECT n FROM (SELECT COUNT(DISTINCT c.email) AS n FROM customers c)") == "PII_IDENTIFIER"
    assert code("SELECT SPLIT(email, '@')[OFFSET(1)] AS domain, COUNT(*) FROM customers GROUP BY 1") == "PII_IDENTIFIER"


# Repairable codes
@pytest.mark.parametrize("col", ["postal_code", "latitude", "longitude", "user_geom"])
def test_geo_too_fine_repairs(col):
    out = validate(f"SELECT {col}, COUNT(*) FROM customers GROUP BY 1")
    assert out.code == "GEO_TOO_FINE" and not out.stop and "city" in out.hint


@pytest.mark.parametrize("sql", [
    "SELECT * FROM users",
    "SELECT * FROM order_items",
    "SELECT * FROM `bigquery-public-data.thelook_ecommerce.users`",
    "SELECT * FROM thelook_ecommerce.orders",
    "SELECT * FROM region-us.INFORMATION_SCHEMA.JOBS",
    "SELECT * FROM sales WHERE customer_id IN (SELECT id FROM `bigquery-public-data.thelook_ecommerce.users`)",
    "SELECT * FROM sales UNION ALL SELECT * FROM `bigquery-public-data.thelook_ecommerce.order_items`",
    "WITH sales AS (SELECT * FROM `bigquery-public-data.thelook_ecommerce.users`) SELECT * FROM sales",
    "WITH customers AS (SELECT 1 AS x) SELECT * FROM customers",
])
def test_unknown_or_shadowed_tables(sql):
    out = validate(sql)
    assert out.code == "UNKNOWN_TABLE" and not out.stop


@pytest.mark.parametrize("sql", ["DECLARE x INT64 DEFAULT 1; SELECT x", "SELECT 1; SELECT 2", "SET x = 1"])
def test_multi_statement(sql):
    out = validate(sql)
    assert out.code == "MULTI_STATEMENT" and "CTE" in out.hint


@pytest.mark.parametrize("sql", ["SELEC brand FROM sales", "SELECT FROM WHERE", "SELECT 'Levi''s'", ""])
def test_parse_error(sql):
    assert code(sql) == "PARSE_ERROR"


def test_brand_literals_found_in_eq_in_and_functions():
    out = validate(
        "SELECT * FROM sales WHERE brand = 'Carhartt' OR brand IN ('Levi\\'s', \"Diesel\") "
        "OR LOWER(brand) = 'calvin klein' OR 'Volcom' = brand"
    )
    assert isinstance(out, Validated)
    got = {(b.value, b.case_insensitive) for b in out.brand_literals}
    assert got == {("Carhartt", False), ("Levi's", False), ("Diesel", False), ("calvin klein", True), ("Volcom", False)}


def test_category_literals_are_not_brands():
    out = validate("SELECT * FROM sales WHERE category = 'Jeans'")
    assert out.brand_literals == []


def test_describe_for_records():
    out = validate(
        "SELECT state, FORMAT_DATE('%Y-%m', DATE(order_created_at)) AS month, SUM(sale_price) AS revenue "
        "FROM sales WHERE status NOT IN ('Cancelled', 'Returned') GROUP BY state, month"
    )
    meta = describe(out.tree)
    assert meta["metric"] == "revenue"
    assert meta["grouping"] == "state, month"
    assert "Cancelled" in meta["filters"]


def test_tool_error_shape():
    err = validate("SELECT * FROM users").as_tool_error()["error"]
    assert set(err) == {"code", "message", "hint"}


@pytest.mark.parametrize("sql, expected", [
    ("SELECT `email` FROM customers", "PII_IDENTIFIER"),
    ("SELECT c.`Street_Address` FROM customers AS c", "PII_IDENTIFIER"),
    ("SELECT * FROM `sales`", "OK"),
    ("SELECT * FROM ML.PREDICT(MODEL m, (SELECT 1))", "UNKNOWN_TABLE"),  # parsed as a qualified table; blocked either way
    ("SeLeCt /* x */ brand FrOm SALES -- trailing", "OK"),
    ("SELECT * FROM sales AS users", "OK"),  # alias named like a raw table is harmless
    ("SELECT * FROM `bigquery-public-data`.thelook_ecommerce.users", "UNKNOWN_TABLE"),
    ("WITH a AS (SELECT * FROM b) SELECT * FROM a", "UNKNOWN_TABLE"),
])
def test_evasion_shapes(sql, expected):
    assert code(sql) == expected
