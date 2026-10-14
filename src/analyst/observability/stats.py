"""`/stats`: session metrics from this conversation's traces, in two views (D2.18).

Business view: is the system working? Outcome rates, reliability, cost and speed.
Engineering view: why? SQL quality, safety events and grounding removals.

A turn that was resumed with /retry has several attempts under one trace id.
Outcomes and latency use the last attempt; tokens, bytes and retries add up
every attempt, since every attempt cost something.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from rich.table import Table

OUTCOMES = ["answered", "partial", "declined", "failed"]
BQ_ERROR_CODES = {"BQ_DOWN", "BQ_FATAL"}
DIAGNOSED = {"empty_without_filters"}


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile; with few turns, p95 is the slowest one."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def _spans(record: dict, name: str) -> list[dict]:
    return sorted((s for s in record["spans"] if s["name"] == name), key=lambda s: s["start_ns"])


def _stuck_fired(record: dict) -> bool:
    """The stuck rule (D2.2): the same run_sql error code twice in a row."""
    last = None
    for s in _spans(record, "tool_call"):
        if s["attributes"].get("tool") != "run_sql" or s["attributes"].get("code") == "BQ_DOWN":
            continue
        code = s["attributes"].get("code")
        if code is not None and code == last:
            return True
        last = code
    return False


@dataclass
class Stats:
    turns: int = 0
    outcomes: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    llm_retries: int = 0
    bq_retries: int = 0
    fallback_turns: int = 0
    breaker_opens: int = 0
    bq_errors: int = 0
    tokens: int = 0
    bytes: int = 0
    latencies_ms: list[int] = field(default_factory=list)
    sql_turns: int = 0
    first_attempt_ok: int = 0
    repairs: int = 0
    stuck_fires: int = 0
    empty_genuine: int = 0
    empty_diagnosed: int = 0
    security_events: Counter = field(default_factory=Counter)
    numbers_removed: int = 0
    turns_with_removals: int = 0


def compute(records: list[dict]) -> Stats:
    st = Stats()
    latest: dict[str, dict] = {}
    for r in records:
        st.tokens += r["totals"].get("tokens", 0)
        st.bytes += r["totals"].get("bytes", 0)
        st.llm_retries += r["totals"].get("llm_retries", 0)
        st.bq_retries += r["totals"].get("bq_retries", 0)
        st.breaker_opens += r["totals"].get("breaker_opens", 0)
        st.bq_errors += sum(1 for s in _spans(r, "gateway") if s["attributes"].get("code") in BQ_ERROR_CODES)
        if r["trace_id"] not in latest or r["attempt"] >= latest[r["trace_id"]]["attempt"]:
            latest[r["trace_id"]] = r

    for r in latest.values():
        st.turns += 1
        st.outcomes[r["outcome"]] += 1
        if r["reason"] != "OK":
            st.reasons[r["reason"]] += 1
        st.latencies_ms.append(r["totals"].get("latency_ms", 0))
        st.fallback_turns += bool(r["totals"].get("fallback_used"))
        for e in r.get("security_events", []):
            st.security_events[e["code"]] += 1

        gateway = _spans(r, "gateway")
        if gateway:
            st.sql_turns += 1
            st.first_attempt_ok += gateway[0]["attributes"].get("code") is None
            # A repair is a query run right after one that failed.
            st.repairs += sum(1 for prev in gateway[:-1] if prev["attributes"].get("code"))
        st.stuck_fires += _stuck_fired(r)
        for s in gateway:
            structural = (s["attributes"].get("diagnosis") or {}).get("structural")
            if structural == "genuine_empty":
                st.empty_genuine += 1
            elif structural in DIAGNOSED or (s["attributes"].get("diagnosis") or {}).get("filter_problems"):
                st.empty_diagnosed += 1

        removed = sum(len(s["attributes"].get("numbers_removed", [])) for s in _spans(r, "output_check"))
        st.numbers_removed += removed
        st.turns_with_removals += removed > 0
    return st


def _pct(n: int, d: int) -> str:
    return f"{n} ({n / d:.0%})" if d else "0"


def _ms(v: float | None) -> str:
    return "n/a" if v is None else (f"{v:.0f} ms" if v < 1000 else f"{v / 1000:.1f} s")


def _table(title: str, rows: list[tuple[str, str]]) -> Table:
    table = Table(title=title, title_justify="left", show_header=False, box=None, pad_edge=False)
    table.add_column(style="cyan")
    table.add_column()
    for metric, value in rows:
        table.add_row(metric, value)
    return table


def render(st: Stats) -> list[Table]:
    n = st.turns
    reasons = ", ".join(f"{k} {v}" for k, v in st.reasons.most_common()) or "none"
    business = _table("Business view", [
        ("Turns", str(n)),
        *[(o.capitalize(), _pct(st.outcomes[o], n)) for o in OUTCOMES],
        ("Reasons (not OK)", reasons),
        ("LLM retries", str(st.llm_retries)),
        ("Turns on a fallback model", _pct(st.fallback_turns, n)),
        ("Circuit breaker opens", str(st.breaker_opens)),
        ("BigQuery retries", str(st.bq_retries)),
        ("BigQuery errors", str(st.bq_errors)),
        ("Tokens per turn", f"{st.tokens / n:,.0f}" if n else "0"),
        ("Bytes per turn", f"{st.bytes / n / 1e6:.1f} MB" if n else "0"),
        ("Latency p50 / p95", f"{_ms(percentile(st.latencies_ms, 50))} / {_ms(percentile(st.latencies_ms, 95))}"),
    ])
    events = ", ".join(f"{k} {v}" for k, v in st.security_events.most_common()) or "none"
    engineering = _table("Engineering view", [
        ("Turns that ran SQL", str(st.sql_turns)),
        ("First-attempt SQL success", _pct(st.first_attempt_ok, st.sql_turns)),
        ("Repairs per question", f"{st.repairs / st.sql_turns:.1f}" if st.sql_turns else "0"),
        ("Stuck-rule fires", str(st.stuck_fires)),
        ("Empty results: genuine / diagnosed", f"{st.empty_genuine} / {st.empty_diagnosed}"),
        ("Security events", events),
        ("Leak-scan blocks", str(st.security_events["LEAK_BLOCKED"])),
        ("Figures removed by grounding", f"{st.numbers_removed} in {st.turns_with_removals} turns"),
    ])
    return [business, engineering]
