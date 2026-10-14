"""Live checks against BigQuery. Skipped by default; run with `pytest -m live`."""

import pytest

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def bq():
    from google.cloud import bigquery

    from analyst.config import load_env, load_settings
    from analyst.sql.gateway import load_curated

    settings = load_settings()
    client = bigquery.Client(project=load_env().google_cloud_project)
    return client, load_curated(settings.bigquery.dataset), settings


def test_reconciliation(bq):
    from analyst.sql.reconcile import run_reconciliation

    client, curated, settings = bq
    failed = [c for c in run_reconciliation(client, curated, settings.bigquery.dataset) if not c.ok]
    assert failed == []


def test_scoped_user_sees_only_their_brands(bq):
    from analyst.config import load_users
    from analyst.sql.brands import BrandMatcher
    from analyst.sql.gateway import Gateway, load_brands
    from analyst.store.results import ResultStore

    client, curated, settings = bq
    gw = Gateway(settings.bigquery, client, ResultStore(), curated,
                 BrandMatcher(load_brands(client, settings.bigquery.dataset)))
    user = load_users().get("bm_levis")
    out = gw.run("SELECT DISTINCT brand FROM sales UNION DISTINCT SELECT DISTINCT brand FROM products", user)
    assert {r["brand"] for r in out["preview"]} == {"Levi's", "Canada Goose"}
    customers = gw.run("SELECT COUNT(*) AS n FROM customers", user)["preview"][0]["n"]
    assert 0 < customers < 100000  # buyers of their brands only


def test_mixed_orders_give_own_share(bq):
    from analyst.config import load_users
    from analyst.sql.brands import BrandMatcher
    from analyst.sql.gateway import Gateway, load_brands
    from analyst.store.results import ResultStore

    client, curated, settings = bq
    gw = Gateway(settings.bigquery, client, ResultStore(), curated,
                 BrandMatcher(load_brands(client, settings.bigquery.dataset)))
    ck = load_users().get("bm_calvin")
    ceo = load_users().get("ceo")
    sql = "SELECT ROUND(SUM(sale_price), 2) AS v FROM sales WHERE order_id IN (SELECT order_id FROM sales WHERE brand = 'Calvin Klein')"
    own = gw.run(sql, ck)["preview"][0]["v"]
    whole = gw.run(sql, ceo)["preview"][0]["v"]
    assert own < whole  # the CEO sees whole mixed orders; the brand manager only their items (A1)
