"""SQL gateway v1: wrap, dry run, bytes cap, execute, store, compact result.

Every query the agent runs goes through `Gateway.run`. Phase 2 adds the
validator (SELECT-only, allowed columns, brand backstop) and the brand-scoped
WITH clauses; phase 3 adds the structured error codes and BigQuery retries.
Until scoping exists, only users with all brands may query (fail closed).
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
from analyst.store.results import ResultStore

CURATED_SQL = Path(__file__).with_name("curated.sql")
DOCTOR_HINT = "Run `python -m analyst doctor` to check your setup."


class GatewayFatalError(Exception):
    """BigQuery can't be used at all (credentials, permissions). Not for the agent to repair."""


class ScopeNotAvailableError(GatewayFatalError):
    pass


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


_LEADING_COMMENTS = re.compile(r"^\s*(--[^\n]*\n\s*|/\*.*?\*/\s*)*", re.S)
_LEADING_WITH = re.compile(r"^WITH\s+(?!RECURSIVE\b)", re.I)


def wrap_sql(sql: str, curated: dict[str, str]) -> str:
    """Prepend the curated tables as WITH clauses, merging with the model's own WITH."""
    body = _LEADING_COMMENTS.sub("", sql.strip()).rstrip(";").strip()
    ctes = ",\n".join(f"{name} AS (\n{select}\n)" for name, select in curated.items())
    if _LEADING_WITH.match(body):
        return f"WITH {ctes},\n{_LEADING_WITH.sub('', body, count=1)}"
    return f"WITH {ctes}\n{body}"


def _bq_message(exc: Exception) -> str:
    errors = getattr(exc, "errors", None) or []
    if errors and isinstance(errors[0], dict) and errors[0].get("message"):
        return errors[0]["message"]
    return str(exc).splitlines()[0]


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

    def run(self, sql: str, user: User) -> dict:
        """Return a compact result, or {"error": {...}} the agent can act on."""
        if not user.all_brands:
            raise ScopeNotAvailableError(
                "Brand scoping is not built yet (phase 2), so only the CEO can query in this build."
            )
        rewritten = wrap_sql(sql, self.curated)
        try:
            dry = self.client.query(
                rewritten, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            )
            scanned = dry.total_bytes_processed or 0
            cap = self.settings.maximum_bytes_billed
            if scanned > cap:
                return _error(
                    f"Query would scan {scanned / 1e6:.1f} MB; the cap is {cap / 1e6:.1f} MB.",
                    "Select fewer columns or narrow the date range.",
                )
            job = self.client.query(
                rewritten, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=cap)
            )
            try:
                df = job.result(timeout=self.settings.job_timeout_s).to_dataframe(create_bqstorage_client=False)
            except (TimeoutError, gexc.RetryError, gexc.DeadlineExceeded):
                job.cancel()
                return _error(
                    f"Query took longer than {self.settings.job_timeout_s:.0f} s and was cancelled.",
                    "Aggregate earlier or narrow the date range.",
                )
        except (gexc.Unauthorized, gexc.Forbidden, DefaultCredentialsError) as exc:
            raise GatewayFatalError(f"BigQuery refused access: {_bq_message(exc)}. {DOCTOR_HINT}") from exc
        except gexc.BadRequest as exc:
            return _error(_bq_message(exc), "Fix the SQL and try again.")
        data_ts = datetime.now(UTC).isoformat(timespec="seconds")
        stored = self.store.put(df, sql, rewritten, data_ts)
        return compact(stored, df, self.settings.preview_rows_to_model)


def _error(message: str, hint: str) -> dict:
    return {"error": {"message": message, "hint": hint}}
