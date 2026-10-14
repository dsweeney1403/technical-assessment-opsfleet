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
        return Check("Environment", False, str(exc), "Copy .env.example to .env and fill it in.")
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
           "with the BigQuery API enabled and billing set up.")
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


CHECKS: list[Callable[[], Check]] = [check_config, check_env, check_gemini, check_bigquery, check_curated_sql]


def run_checks(checks: list[Callable[[], Check]] | None = None) -> list[Check]:
    return [check() for check in (checks or CHECKS)]
