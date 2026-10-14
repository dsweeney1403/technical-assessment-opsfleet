"""SQLite report store (D3.10, D3.11, D3.23, D3.24). One local file, zero setup.

The owner filter is in every query's WHERE clause, so no read path can forget
it (D3.13). The hidden filter (D3.15) runs on the rows that come back. A delete
and its audit row commit in one transaction, so neither exists without the other.
"""

from __future__ import annotations

import difflib
import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from analyst.config import User
from analyst.sql.brands import CUTOFF, normalise
from analyst.store.base import (
    AuditEntry,
    NewVersion,
    ReportStore,
    ReportVersion,
    SavedFinding,
    brands_scope,
    visible_to,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    owner_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    chart_files TEXT NOT NULL,
    created_at TEXT NOT NULL,
    brands_scope TEXT NOT NULL,
    data_ts TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    findings TEXT NOT NULL,
    mentions TEXT NOT NULL,
    deleted_at TEXT,
    PRIMARY KEY (report_id, version)
);
CREATE INDEX IF NOT EXISTS reports_owner ON reports (owner_id);
CREATE TABLE IF NOT EXISTS audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    at TEXT NOT NULL,
    action TEXT NOT NULL,
    request_text TEXT NOT NULL,
    report_ids TEXT NOT NULL,
    titles TEXT NOT NULL,
    trace_id TEXT NOT NULL
);
"""
JSON_FIELDS = ("chart_files", "brands_scope", "findings", "mentions")
LATEST = """
SELECT r.* FROM reports r
JOIN (SELECT report_id, MAX(version) AS v FROM reports WHERE owner_id = ? GROUP BY report_id) m
  ON r.report_id = m.report_id AND r.version = m.v
WHERE r.owner_id = ? AND r.deleted_at IS NULL
"""


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds")


class SQLiteReportStore(ReportStore):
    def __init__(self, path: Path | str, clock=_now):
        self.path = str(path)
        self._clock = clock
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)  # graph nodes may run off the main thread
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    # Rows

    @staticmethod
    def _version(row: sqlite3.Row) -> ReportVersion:
        data = dict(row)
        for f in JSON_FIELDS:
            data[f] = json.loads(data[f])
        data["findings"] = [SavedFinding(**f) for f in data["findings"]]
        return ReportVersion(**data)

    def _latest(self, user: User, extra: str = "", params: tuple = ()) -> list[ReportVersion]:
        rows = self._conn.execute(LATEST + extra + " ORDER BY r.created_at DESC, r.report_id",
                                  (user.id, user.id, *params)).fetchall()
        return [v for v in map(self._version, rows) if visible_to(user, v.brands_scope)]

    # Reads

    def get(self, user: User, report_id: str) -> ReportVersion | None:
        found = self._latest(user, " AND r.report_id = ?", (report_id,))
        return found[0] if found else None

    def list(self, user: User) -> list[ReportVersion]:
        return self._latest(user)

    def match_mentions(self, user: User, term: str) -> list[ReportVersion]:
        key = normalise(term)
        if not key:
            return []
        out = []
        for report in self._latest(user):
            keys = [normalise(m) for m in report.mentions]
            if key in keys or difflib.get_close_matches(key, keys, n=1, cutoff=CUTOFF):
                out.append(report)
        return out

    def created_in(self, user: User, conversation_id: str) -> list[ReportVersion]:
        first = {row["report_id"] for row in self._conn.execute(
            "SELECT report_id FROM reports WHERE owner_id = ? AND version = 1 AND conversation_id = ?",
            (user.id, conversation_id))}
        return [r for r in self._latest(user) if r.report_id in first]

    # Writes

    def save(self, user: User, report: NewVersion) -> ReportVersion:
        if report.report_id is None:
            report_id, version = f"rep_{uuid.uuid4().hex[:8]}", 1
        else:
            current = self.get(user, report.report_id)
            if current is None:
                raise KeyError(f"No report {report.report_id} for this user.")
            report_id, version = current.report_id, current.version + 1
        saved = ReportVersion(**{**report.model_dump(), "report_id": report_id}, version=version, owner_id=user.id,
                              created_at=_iso(self._clock()), brands_scope=brands_scope(user))
        row = saved.model_dump()
        for f in JSON_FIELDS:
            row[f] = json.dumps(row[f])
        cols = ", ".join(row)
        with self._conn:
            self._conn.execute(f"INSERT INTO reports ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))
        return saved

    def _write_audit(self, user_id: str, action: str, request_text: str, reports: list[ReportVersion],
                     trace_id: str) -> None:
        self._conn.execute(
            "INSERT INTO audit (user_id, at, action, request_text, report_ids, titles, trace_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, _iso(self._clock()), action, request_text, json.dumps([r.report_id for r in reports]),
             json.dumps([r.title for r in reports]), trace_id),
        )

    def soft_delete(self, user: User, report_ids: list[str], request_text: str, trace_id: str) -> list[ReportVersion]:
        wanted = set(report_ids)
        targets = [r for r in self._latest(user) if r.report_id in wanted]
        if not targets:
            return []
        now = _iso(self._clock())
        with self._conn:  # one transaction: every version deleted and audited, or nothing
            for r in targets:
                self._conn.execute(
                    "UPDATE reports SET deleted_at = ? WHERE report_id = ? AND owner_id = ? AND deleted_at IS NULL",
                    (now, r.report_id, user.id))
            self._write_audit(user.id, "delete", request_text, targets, trace_id)
        return targets

    def purge(self, days: int) -> int:
        cutoff = _iso(self._clock() - timedelta(days=days))
        with self._conn:
            ids = {row[0] for row in self._conn.execute(
                "SELECT report_id FROM reports WHERE deleted_at IS NOT NULL AND deleted_at < ?", (cutoff,))}
            self._conn.execute("DELETE FROM reports WHERE deleted_at IS NOT NULL AND deleted_at < ?", (cutoff,))
        return len(ids)

    def restore(self, report_id: str, admin_id: str, reason: str) -> bool:
        rows = self._conn.execute("SELECT * FROM reports WHERE report_id = ? AND deleted_at IS NOT NULL ORDER BY version",
                                  (report_id,)).fetchall()
        if not rows:
            return False
        with self._conn:
            self._conn.execute("UPDATE reports SET deleted_at = NULL WHERE report_id = ?", (report_id,))
            self._write_audit(admin_id, "restore", reason, [self._version(rows[-1])], "")
        return True

    def audit(self) -> list[AuditEntry]:
        rows = self._conn.execute("SELECT * FROM audit ORDER BY audit_id").fetchall()
        return [AuditEntry(**{**dict(r), "report_ids": json.loads(r["report_ids"]), "titles": json.loads(r["titles"])})
                for r in rows]
