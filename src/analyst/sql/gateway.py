"""SQL gateway: every query the agent runs goes through `Gateway.run`.

Steps: validate (SELECT-only, allowed tables and columns), brand backstop,
scope rewrite, dry run and bytes cap, execute, store, compact result. There is
no other path to BigQuery. Stop codes raise SafetyStop and end the turn (D2.1);
other codes return {"error": {code, message, hint}} for the agent to repair.
Phase 3 adds BQ_TRANSIENT retries, diagnosis on empty results and the stuck rule.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from google.api_core import exceptions as gexc
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery

from analyst.config import BigQuerySettings, User
from analyst.sql.brands import BrandMatcher
from analyst.sql.scope import scoped_tables
from analyst.sql.validator import Rejection, Validated, describe, validate
from analyst.store.results import ResultStore

CURATED_SQL = Path(__file__).with_name("curated.sql")
DOCTOR_HINT = "Run `python -m analyst doctor` to check your setup."


class GatewayFatalError(Exception):
    """BigQuery can't be used at all (credentials, permissions). BQ_FATAL: not for the agent to repair."""


class SafetyStop(Exception):
    """A stop code (NOT_SELECT, PII_IDENTIFIER): the turn ends with a neutral decline and a security event."""

    def __init__(self, rejection: Rejection, sql: str):
        self.rejection = rejection
        self.sql = sql
        super().__init__(rejection.code)


def load_curated(dataset: str, path: Path = CURATED_SQL) -> dict[str, str]:
    """Parse curated.sql into {table_name: select_sql}."""
    tables: dict[str, list[str]] = {}
    current = None
    for line in path.read_text().splitlines():
        m = re.match(r"--\s*name:\s*(\w+)", line)
        if m:
            current = m.group(1)
            tables[current] = []
        elif current and not line.lstrip().startswith("--"):
            tables[current].append(line)
    return {name: "\n".join(lines).strip().format(dataset=dataset) for name, lines in tables.items()}


def load_brands(client: bigquery.Client, dataset: str) -> list[str]:
    sql = f"SELECT DISTINCT brand FROM `{dataset}.products` WHERE brand IS NOT NULL"
    return [r.brand for r in client.query(sql).result()]


_LEADING_COMMENTS = re.compile(r"^\s*(--[^\n]*\n\s*|/\*.*?\*/\s*)*", re.S)
_LEADING_WITH = re.compile(r"^WITH\s+(?!RECURSIVE\b)", re.I)


def wrap_sql(sql: str, tables: dict[str, str]) -> str:
    """Prepend the tables as WITH clauses, merging with the model's own WITH.

    Safe because the validator has already proved the SQL is one SELECT over
    these names, with no CTE shadowing them.
    """
    body = _LEADING_COMMENTS.sub("", sql.strip()).rstrip(";").strip()
    ctes = ",\n".join(f"{name} AS (\n{select}\n)" for name, select in tables.items())
    if _LEADING_WITH.match(body):
        return f"WITH {ctes},\n{_LEADING_WITH.sub('', body, count=1)}"
    return f"WITH {ctes}\n{body}"


def _bq_message(exc: Exception) -> str:
    errors = getattr(exc, "errors", None) or []
    if errors and isinstance(errors[0], dict) and errors[0].get("message"):
        return errors[0]["message"]
    return str(exc).splitlines()[0]


def _error(code: str, message: str, hint: str) -> dict:
    return {"error": {"code": code, "message": message, "hint": hint}}


def compact(stored, df: pd.DataFrame, preview_rows: int) -> dict:
    preview = json.loads(df.head(preview_rows).to_json(orient="records", date_format="iso"))
    nulls = {col: int(n) for col, n in df.isna().sum().items() if n}
    return {
        "result_id": stored.result_id,
        "columns": list(df.columns),
        "row_count": len(df),
        "preview": preview,
        "truncated": len(df) > preview_rows,
        "empty": df.empty,
        "null_flags": nulls,
        "data_ts": stored.data_ts,
    }


@dataclass
class Gateway:
    settings: BigQuerySettings
    client: bigquery.Client
    store: ResultStore
    curated: dict[str, str]
    brands: BrandMatcher

    def run(self, sql: str, user: User) -> dict:
        checked = validate(sql)
        if isinstance(checked, Rejection):
            if checked.stop:
                raise SafetyStop(checked, sql)
            return checked.as_tool_error()
        assert isinstance(checked, Validated)
        brand_problem = self.brands.check_literals(checked.brand_literals, user)
        if brand_problem:
            return brand_problem.as_tool_error()

        tables, params = scoped_tables(self.curated, user)
        rewritten = wrap_sql(sql, tables)
        cap = self.settings.maximum_bytes_billed
        try:
            dry = self.client.query(
                rewritten,
                job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False, query_parameters=params),
            )
            scanned = dry.total_bytes_processed or 0
            if scanned > cap:
                return _error("COST_LIMIT", f"Query would scan {scanned / 1e6:.1f} MB; the cap is {cap / 1e6:.1f} MB.",
                              "Select fewer columns or narrow the date range.")
            job = self.client.query(
                rewritten, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=cap, query_parameters=params)
            )
            try:
                df = job.result(timeout=self.settings.job_timeout_s).to_dataframe(create_bqstorage_client=False)
            except (TimeoutError, gexc.RetryError, gexc.DeadlineExceeded):
                job.cancel()
                return _error("TIMEOUT", f"Query took longer than {self.settings.job_timeout_s:.0f} s and was cancelled.",
                              "Aggregate earlier or narrow the date range.")
        except (gexc.Unauthorized, gexc.Forbidden, DefaultCredentialsError) as exc:
            raise GatewayFatalError(f"BigQuery refused access: {_bq_message(exc)}. {DOCTOR_HINT}") from exc
        except gexc.BadRequest as exc:
            return _error("BQ_INVALID", _bq_message(exc), "Fix the SQL using BigQuery's message and try again.")
        data_ts = datetime.now(UTC).isoformat(timespec="seconds")
        stored = self.store.put(df, sql, rewritten, data_ts, meta=describe(checked.tree))
        return compact(stored, df, self.settings.preview_rows_to_model)
