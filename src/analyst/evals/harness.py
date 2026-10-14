"""Scripted conversations on real sessions, for evals, the persona smoke test and the example run.

Each case gets its own session: its own conversation, its own SQLite file for
reports and preferences, and optionally its own persona. A turn is a message,
or the reply to a confirmation the previous turn showed. Every turn is traced
like a chat turn, so a failed case opens with `trace <id>`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from analyst.config import load_users
from analyst.store.base import NewVersion


class SeedReport(BaseModel):
    """A saved report put in place before the case runs."""

    owner: str
    title: str
    mentions: list[str] = Field(default_factory=list)
    body: str = ""


@dataclass
class TurnResult:
    said: str
    text: str  # what the user saw: the answer, or the confirmation text
    awaiting: bool  # the turn stopped to ask for a confirmation
    record: dict | None  # the turn's trace record
    state: dict = field(default_factory=dict)

    @property
    def outcome(self) -> str:
        return self.record["outcome"] if self.record else "failed"

    @property
    def reason(self) -> str:
        return self.record["reason"] if self.record else "INTERNAL"

    @property
    def trace_id(self) -> str | None:
        return self.record["trace_id"] if self.record else None


class Conversation:
    def __init__(self, session):
        self.session = session
        self.turns: list[TurnResult] = []

    def say(self, text: str) -> TurnResult:
        from analyst.cli import run_turn

        try:
            state, asking = run_turn(self.session, text)
        except Exception:
            state, asking = {}, None  # the trace records the failure; the case scores it
        draft = state.get("draft")
        shown = asking or (draft.text if draft is not None else "")
        result = TurnResult(text, shown, asking is not None, self.session.tracer.last, state)
        self.turns.append(result)
        return result

    def run(self, messages: list[str]) -> list[TurnResult]:
        return [self.say(m) for m in messages]


def write_persona(directory: Path, text: str) -> Path:
    """A one-version persona dir, for a case that tests a persona."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "v1.md").write_text(text)
    (directory / "active").write_text("v1\n")
    return directory


def new_session(user_id: str, workdir: Path, persona: str | None = None, seed: list[SeedReport] = (),
                faults=None, build=None):
    """A fresh session with its own files under `workdir`. `build` defaults to `cli.build_session`."""
    if build is None:
        from analyst.cli import build_session as build
    workdir.mkdir(parents=True, exist_ok=True)
    persona_dir = write_persona(workdir / "persona", persona) if persona else None
    session = build(user_id, faults, reports_path=workdir / "reports.db", persona_dir=persona_dir)
    users = load_users()
    for r in seed:
        session.reports.save(users.get(r.owner), NewVersion(
            conversation_id="seed", title=r.title, body=r.body or f"# {r.title}\nSeeded for an eval case.",
            data_ts="2026-10-01T00:00:00+00:00", trace_id="seed", mentions=r.mentions))
    return session
