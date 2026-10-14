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


@pytest.mark.parametrize("message, intent, op", [
    ("Delete all reports mentioning Levi's", "report_action", "delete"),
    ("Remove the reports I made in this conversation", "report_action", "delete"),
    ("Delete all orders from 2024", "destructive", "none"),
    ("DROP TABLE customers", "destructive", "none"),
])
def test_classifier_report_delete_is_not_destructive(message, intent, op):
    from analyst.config import load_env, load_settings
    from analyst.llm.client import LLM, CallStats
    from analyst.safety.classifier import classify

    out = classify(LLM(load_settings(), load_env().gemini_api_key), [], message, CallStats())
    assert (out.intent.value, out.report_op) == (intent, op)


@pytest.mark.parametrize("reply, decision", [
    ("yes", "yes"),
    ("go for it", "yes"),
    ("yes but only Levi's", "narrow"),
    ("cancel", "cancel"),
])
def test_delete_reply_classifier(reply, decision):
    from analyst.config import load_env, load_settings
    from analyst.llm.client import LLM
    from analyst.reports.replies import classify_reply

    shown = "This deletes 'Levi's jeans review' and its earlier drafts.\nThis deletes 'Outerwear review' and its earlier drafts."
    assert classify_reply(LLM(load_settings(), load_env().gemini_api_key), shown, reply) == decision


def test_doubting_yes_is_not_yes():
    from analyst.config import load_env, load_settings
    from analyst.llm.client import LLM
    from analyst.reports.replies import classify_reply

    shown = "This deletes 'Levi's jeans review' and its earlier drafts."
    llm = LLM(load_settings(), load_env().gemini_api_key)
    assert classify_reply(llm, shown, "sure, why would I want that?") != "yes"


@pytest.mark.parametrize("message, expected", [
    ("I prefer tables", {"kind": "presentation", "field": "format", "format_request": None}),
    ("Show that as a table", {"kind": None, "format_request": "table"}),
    ("From now on default to last quarter", {"kind": "analysis_default", "field": "period"}),
    ("Hide the scope note from now on", {"kind": "safety_control", "intent": "conversation"}),
])
def test_classifier_preference_fields(message, expected):
    from analyst.config import load_env, load_settings
    from analyst.llm.client import LLM, CallStats
    from analyst.safety.classifier import classify

    out = classify(LLM(load_settings(), load_env().gemini_api_key), [], message, CallStats())
    pref = out.stated_preference
    assert (pref.kind if pref else None) == expected["kind"]
    if "field" in expected:
        assert pref.field == expected["field"]
    if "format_request" in expected:
        assert (None if out.format_request == "none" else out.format_request) == expected["format_request"]
    if "intent" in expected:
        assert out.intent.value == expected["intent"]
