"""Persona smoke test (D4.23): a handful of questions, including red-team ones, against the active persona.

It runs the real graph, so the gateway, output check and declines all apply. It
protects quality, not safety: a persona can't weaken safety, because safety is
enforced in code. Run it before pointing `active` at a new version.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ASSUMPTIONS = re.compile(r"assumptions?\b", re.I)


class Case(BaseModel):
    id: str
    user: str
    question: str
    expect: list[str]
    checks: list[str] = Field(default_factory=list)


class Result(BaseModel):
    case: Case
    outcome: str
    reason: str
    trace_id: str
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def load_cases(path: Path) -> list[Case]:
    return [Case.model_validate(c) for c in yaml.safe_load(Path(path).read_text(encoding="utf-8"))["cases"]]


def check(case: Case, record: dict) -> list[str]:
    """What went wrong for one case; empty when it passed."""
    text = record.get("answer") or ""
    failures = []
    if record["outcome"] not in case.expect:
        failures.append(f"expected {' or '.join(case.expect)}, got {record['outcome']} ({record['reason']})")
    for c in case.checks:
        if c == "assumptions_line" and not ASSUMPTIONS.search(text):
            failures.append("no assumptions line")
        elif c == "no_em_dash" and "—" in text:
            failures.append("em dash in the answer")
        elif c == "scope_note" and "Scope: covers" not in text:
            failures.append("no scope note")
        elif c.startswith("contains:") and c.removeprefix("contains:").lower() not in text.lower():
            failures.append(f"doesn't mention '{c.removeprefix('contains:')}'")
    return failures


def run(cases: list[Case], session_for: Callable[[str], object], requests_per_minute: int,
        sleep: Callable[[float], None] = time.sleep) -> list[Result]:
    """One fresh session per case, throttled for free-tier rate limits."""
    from analyst.evals.harness import Conversation
    from analyst.evals.runner import Throttle

    throttle = Throttle(requests_per_minute, sleep=sleep)
    results = []
    for i, case in enumerate(cases):
        turn = Conversation(session_for(case.user)).say(case.question)
        record = turn.record or {"outcome": "failed", "reason": "INTERNAL", "trace_id": None, "totals": {}}
        results.append(Result(case=case, outcome=record["outcome"], reason=record["reason"],
                              trace_id=record["trace_id"] or "", failures=check(case, {**record, "answer": turn.text})))
        if i < len(cases) - 1:
            throttle.after(record["totals"].get("llm_calls", 1))
    return results
