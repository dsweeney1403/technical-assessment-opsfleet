"""`doctor`: check config, Gemini, BigQuery and the curated SQL, with the exact fix for each failure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from analyst.config import ConfigError, load_env, load_settings, load_users


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""
    skipped: bool = False


def check_config() -> Check:
    try:
        settings = load_settings()
        users = load_users()
    except ConfigError as exc:
        return Check("Config", False, str(exc), "Fix the file named above.")
    return Check("Config", True, f"settings.yaml and users.yaml valid ({len(users.users)} users, agent {settings.models.agent})")


def check_env() -> Check:
    try:
        env = load_env()
    except ConfigError as exc:
        return Check("Environment", False, str(exc))  # the message is the fix
    return Check("Environment", True, f"GEMINI_API_KEY set, project {env.google_cloud_project}")


def check_gemini() -> Check:
    fix = "Check GEMINI_API_KEY in .env. Create a key at https://aistudio.google.com/apikey."
    try:
        from google import genai

        settings, env = load_settings(), load_env()
        client = genai.Client(api_key=env.gemini_api_key)  # keep a reference: the list pages lazily
        names = {m.name.removeprefix("models/") for m in client.models.list()}
    except ConfigError as exc:
        return Check("Gemini", False, str(exc), fix)
    except Exception as exc:
        return Check("Gemini", False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}", fix)
    m = settings.models
    wanted = {m.agent, m.light, m.embedding, *m.agent_fallbacks, *m.light_fallbacks}
    missing = sorted(wanted - names)
    if missing:
        return Check("Gemini", False, f"Key works, but these model IDs aren't available: {', '.join(missing)}",
                     "Update models in config/settings.yaml to IDs listed for your key.")
    return Check("Gemini", True, f"Key works; all {len(wanted)} configured models available")


def check_bigquery() -> Check:
    fix = ("Run `gcloud auth application-default login`, and set GOOGLE_CLOUD_PROJECT to a project "
           "with the BigQuery API enabled. A free BigQuery sandbox project works.")
    try:
        from google.cloud import bigquery

        settings, env = load_settings(), load_env()
        client = bigquery.Client(project=env.google_cloud_project)
        table = client.get_table(f"{settings.bigquery.dataset}.order_items")
    except ConfigError as exc:
        return Check("BigQuery", False, str(exc), fix)
    except Exception as exc:
        return Check("BigQuery", False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}", fix)
    return Check("BigQuery", True, f"Credentials work; order_items has {table.num_rows:,} rows")


def check_curated_sql() -> Check:
    fix = "Check src/analyst/sql/curated.sql against the dataset schema."
    try:
        from google.cloud import bigquery

        from analyst.sql.gateway import load_curated, wrap_sql

        settings, env = load_settings(), load_env()
        client = bigquery.Client(project=env.google_cloud_project)
        curated = load_curated(settings.bigquery.dataset)
        scanned = {}
        for name in curated:
            job = client.query(wrap_sql(f"SELECT COUNT(*) FROM {name}", curated),
                               job_config=bigquery.QueryJobConfig(dry_run=True))
            scanned[name] = job.total_bytes_processed / 1e6
    except Exception as exc:
        return Check("Curated SQL", False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}", fix)
    return Check("Curated SQL", True, "Dry run OK: " + ", ".join(f"{k} {v:.1f} MB" for k, v in scanned.items()))


def check_reconciliation() -> Check:
    fix = "The curated tables dropped rows. Check the joins in src/analyst/sql/curated.sql."
    try:
        from google.cloud import bigquery

        from analyst.sql.gateway import load_curated
        from analyst.sql.reconcile import run_reconciliation

        settings, env = load_settings(), load_env()
        client = bigquery.Client(project=env.google_cloud_project)
        checks = run_reconciliation(client, load_curated(settings.bigquery.dataset), settings.bigquery.dataset)
    except Exception as exc:
        return Check("Reconciliation", False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}", fix)
    failed = [c for c in checks if not c.ok]
    if failed:
        return Check("Reconciliation", False, "; ".join(f"{c.name}: {c.curated} vs {c.raw}" for c in failed), fix)
    return Check("Reconciliation", True, f"Curated tables match raw tables ({len(checks)} checks)")


# (name, check, checks it needs). A check whose prerequisite failed is skipped, so one
# cause shows as one FAIL with its fix instead of a FAIL on every check after it.
CHECKS: list[tuple[str, Callable[[], Check], tuple[str, ...]]] = [
    ("Config", check_config, ()),
    ("Environment", check_env, ()),
    ("Gemini", check_gemini, ("Config", "Environment")),
    ("BigQuery", check_bigquery, ("Config", "Environment")),
    ("Curated SQL", check_curated_sql, ("BigQuery",)),
    ("Reconciliation", check_reconciliation, ("BigQuery",)),
]


def run_checks(checks: list[tuple[str, Callable[[], Check], tuple[str, ...]]] | None = None) -> list[Check]:
    results: dict[str, Check] = {}
    for name, check, needs in checks or CHECKS:
        failed = [n for n in needs if not results[n].ok]
        results[name] = (Check(name, False, f"Skipped until {' and '.join(failed)} passes.", skipped=True)
                         if failed else check())
    return list(results.values())
