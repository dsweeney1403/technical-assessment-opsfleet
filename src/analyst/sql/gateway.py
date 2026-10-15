"""SQL gateway: every query the agent runs goes through `Gateway.run`.

Steps: validate (SELECT-only, allowed tables and columns), brand backstop,
scope rewrite, dry run and bytes cap, execute, store, compact result. There is
no other path to BigQuery. Stop codes raise SafetyStop and end the turn (D2.1);
other codes return {"error": {code, message, hint}} for the agent to repair.

BigQuery errors are coded by their reason field, never the message or status
alone (D2.4): a cost limit arrives as HTTP 500 with reason
bytesBilledLimitExceeded, and a per-second rate limit as 403 with reason
rateLimitExceeded. BQ_TRANSIENT is retried here, invisible to the agent; when
retries run out, GatewayUnavailableError ends querying for the turn (D2.13).

Every query emits a `gateway` span for the trace (D2.16): SQL before and after
the rewrite, code, bytes, rows, diagnosis and latency. Never the rows themselves.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from google.api_core import exceptions as gexc
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from requests.exceptions import ConnectionError as RequestsConnectionError
from google.cloud import bigquery

from analyst.config import BigQuerySettings, EmptyResults, User
from analyst.observability import tracer
from analyst.sql import diagnosis
from analyst.sql.brands import BrandMatcher
from analyst.sql.scope import scoped_tables
from analyst.sql.validator import Rejection, Validated, describe, validate
from analyst.store.results import ResultStore

CURATED_SQL = Path(__file__).with_name("curated.sql")
DOCTOR_HINT = "Run `python -m analyst doctor` to check your setup."
COST_HINT = "Select fewer columns or narrow the date range."


class GatewayFatalError(Exception):
    """BigQuery can't be used at all (credentials, permissions). BQ_FATAL: not for the agent to repair."""


class GatewayUnavailableError(Exception):
    """BQ_TRANSIENT retries ran out: BigQuery is down for now (D2.13)."""


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
    for line in path.read_text(encoding="utf-8").splitlines():
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


BQ_ERRORS = (gexc.GoogleAPICallError, DefaultCredentialsError, RefreshError, RequestsConnectionError, ConnectionError)
REASON_CODES = {
    "invalidQuery": "BQ_INVALID",
    "invalid": "BQ_INVALID",
    "resourcesExceeded": "BQ_INVALID",  # query too complex: the agent can simplify it
    "bytesBilledLimitExceeded": "COST_LIMIT",
    "backendError": "BQ_TRANSIENT",
    "internalError": "BQ_TRANSIENT",
    "rateLimitExceeded": "BQ_TRANSIENT",  # per second, clears on its own
    "jobBackendError": "BQ_TRANSIENT",
    "jobInternalError": "BQ_TRANSIENT",
    "accessDenied": "BQ_FATAL",
    "quotaExceeded": "BQ_FATAL",  # usually daily: same look as rateLimitExceeded, opposite handling
    "billingNotEnabled": "BQ_FATAL",
    "billingTierLimitExceeded": "BQ_FATAL",
    "notFound": "BQ_FATAL",
}
BACKOFF_BASE_S = 1.0


def _reason(exc: Exception) -> str | None:
    errors = getattr(exc, "errors", None) or []
    if errors and isinstance(errors[0], dict):
        return errors[0].get("reason")
    return None


def classify_bq_error(exc: Exception) -> str:
    """Gateway code for a BigQuery failure. The reason field decides; status is only a fallback."""
    if isinstance(exc, (DefaultCredentialsError, RefreshError)):
        return "BQ_FATAL"
    if isinstance(exc, (RequestsConnectionError, ConnectionError)):
        return "BQ_TRANSIENT"
    code = REASON_CODES.get(_reason(exc) or "")
    if code:
        return code
    status = getattr(exc, "code", None)
    if isinstance(status, int) and status >= 500:
        return "BQ_TRANSIENT"
    if status == 400:
        return "BQ_INVALID"
    return "BQ_FATAL"


class _Timeout(Exception):
    pass


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
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    known: diagnosis.KnownValues | None = None  # loaded once per session (D2.5)
    empty: EmptyResults = field(default_factory=EmptyResults)
    transient_retries: int = 0  # cumulative for the session; each trace span gets its own count

    def _retrying(self, fn: Callable):
        """Run fn, retrying BQ_TRANSIENT with backoff and jitter (D2.4)."""
        attempts = self.settings.transient_max_attempts
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except BQ_ERRORS as exc:
                if classify_bq_error(exc) != "BQ_TRANSIENT":
                    raise
                if attempt == attempts:
                    raise GatewayUnavailableError(f"BigQuery is unavailable: {_bq_message(exc)}") from exc
                self.transient_retries += 1
                self.sleep(BACKOFF_BASE_S * 2 ** (attempt - 1) * (0.5 + random.random()))
        raise AssertionError("unreachable")

    def _execute(self, sql: str, params: list, span: dict | None = None) -> pd.DataFrame:
        job = self.client.query(
            sql, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=self.settings.maximum_bytes_billed,
                                                    query_parameters=params)
        )
        try:
            df = job.result(timeout=self.settings.job_timeout_s).to_dataframe(create_bqstorage_client=False)
        except (TimeoutError, gexc.RetryError, gexc.DeadlineExceeded) as exc:
            job.cancel()
            raise _Timeout() from exc
        if span is not None:
            span["bytes_billed"] = getattr(job, "total_bytes_billed", None)
        return df

    def run(self, sql: str, user: User) -> dict:
        """Run one query and emit its trace span, whatever the outcome."""
        span: dict = {"sql_in": sql, "sql_rewritten": None, "code": None, "bytes_processed": None,
                      "bytes_billed": None, "rows": None, "diagnosis": None}
        retries_before = self.transient_retries
        start = tracer.now_ns()
        try:
            result = self._run(sql, user, span)
            error = result.get("error")
            if error:
                span["code"] = error["code"]
            else:
                span["rows"], span["diagnosis"] = result["row_count"], result.get("diagnosis")
            return result
        except SafetyStop as stop:
            span["code"] = stop.rejection.code
            raise
        except GatewayUnavailableError:
            span["code"] = "BQ_DOWN"
            raise
        except GatewayFatalError:
            span["code"] = "BQ_FATAL"
            raise
        finally:
            tracer.emit("gateway", {**span, "error": span["code"],
                                    "transient_retries": self.transient_retries - retries_before,
                                    "start_ns": start, "end_ns": tracer.now_ns()})

    def _run(self, sql: str, user: User, span: dict) -> dict:
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
        span.update(sql_rewritten=rewritten, scoped_tables=list(tables),
                    scope_brands=None if user.all_brands else list(user.brands))
        cap = self.settings.max_bytes_processed
        try:
            dry = self._retrying(lambda: self.client.query(
                rewritten,
                job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False, query_parameters=params),
            ))
            scanned = dry.total_bytes_processed or 0
            span["bytes_processed"] = scanned
            if scanned > cap:
                return _error("COST_LIMIT", f"Query would scan {scanned / 1e6:.1f} MB; the cap is {cap / 1e6:.1f} MB.",
                              COST_HINT)
            df = self._retrying(lambda: self._execute(rewritten, params, span))
        except _Timeout:
            return _error("TIMEOUT", f"Query took longer than {self.settings.job_timeout_s:.0f} s and was cancelled.",
                          "Aggregate earlier or narrow the date range.")
        except BQ_ERRORS as exc:
            code = classify_bq_error(exc)
            if code == "COST_LIMIT":
                return _error("COST_LIMIT", f"Query is over the billing cap: {_bq_message(exc)}", COST_HINT)
            if code == "BQ_INVALID":
                return _error("BQ_INVALID", _bq_message(exc), "Fix the SQL using BigQuery's message and try again.")
            raise GatewayFatalError(f"BigQuery refused the request: {_bq_message(exc)}. {DOCTOR_HINT}") from exc
        data_ts = datetime.now(UTC).isoformat(timespec="seconds")
        stored = self.store.put(df, sql, rewritten, data_ts, meta=describe(checked.tree))
        result = compact(stored, df, self.settings.preview_rows_to_model)
        if diagnosis.no_rows_matched(df):
            found = self._diagnose(checked, tables, params)
            if found:
                result["diagnosis"] = found
        return result

    def _diagnose(self, checked: Validated, tables: dict[str, str], params: list) -> dict | None:
        """Why is it empty? Filter values first, then one count without filters (D2.5)."""
        if self.empty.diagnosis and self.known is not None:
            problems = diagnosis.diagnose(checked.tree, self.known)
            if problems:
                return {"filter_problems": problems, "hint": diagnosis.FILTER_HINT}
        if not self.empty.structural_check:
            return None
        unfiltered = wrap_sql(diagnosis.without_filters(checked.tree), tables)
        try:
            still_nothing = diagnosis.no_rows_matched(self._retrying(lambda: self._execute(unfiltered, params)))
        except (_Timeout, GatewayUnavailableError, *BQ_ERRORS):
            return None  # best effort: the result still goes back as it is
        if still_nothing:
            return {"structural": "empty_without_filters", "hint": diagnosis.JOIN_HINT}
        return {"structural": "genuine_empty", "hint": diagnosis.GENUINE_HINT}
