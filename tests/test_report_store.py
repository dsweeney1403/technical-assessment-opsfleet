from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from analyst.config import User
from analyst.reports.mentions import MentionScanner
from analyst.sql.diagnosis import KnownValues
from analyst.store.base import NewVersion, SavedFinding
from analyst.store.sqlite_store import SQLiteReportStore

CEO = User(id="ceo", name="CEO", role="ceo", brands=["*"])
LEVIS = User(id="bm_levis", name="BM", role="brand_manager", brands=["Levi's", "Canada Goose"])
CALVIN = User(id="bm_calvin", name="BM", role="brand_manager", brands=["Calvin Klein"])


class Clock:
    def __init__(self):
        self.t = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)

    def __call__(self):
        self.t += timedelta(seconds=1)
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(clock, tmp_path):
    return SQLiteReportStore(tmp_path / "reports.db", clock=clock)


def draft(title="Q1 report", mentions=("Levi's",), report_id=None, conversation="conv1", body="Body"):
    return NewVersion(report_id=report_id, conversation_id=conversation, title=title, body=body,
                      data_ts="2026-10-14T09:00:00+00:00", trace_id="t1", mentions=list(mentions),
                      findings=[SavedFinding(finding_id="f1", text="Revenue rose", result_id="r1", sql="SELECT 1")])


def test_save_and_read_back(store):
    saved = store.save(LEVIS, draft())
    assert saved.version == 1 and saved.owner_id == "bm_levis"
    assert saved.brands_scope == ["Canada Goose", "Levi's"]  # the user's brands at save time (D3.11)
    got = store.get(LEVIS, saved.report_id)
    assert got == saved and got.findings[0].result_id == "r1"
    assert store.save(CEO, draft()).brands_scope == ["*"]


def test_redo_is_a_new_version_under_the_same_id(store):
    first = store.save(LEVIS, draft())
    second = store.save(LEVIS, draft(title="Q1 report, short", report_id=first.report_id))
    assert (second.report_id, second.version) == (first.report_id, 2)
    assert [r.title for r in store.list(LEVIS)] == ["Q1 report, short"]  # versions never listed


def test_owner_only_for_every_role(store):
    mine = store.save(LEVIS, draft())
    store.save(CALVIN, draft(title="Calvin Q1", mentions=["Calvin Klein"]))
    assert store.get(CEO, mine.report_id) is None  # the CEO can't read other users' reports (D3.13)
    assert store.list(CEO) == [] and store.match_mentions(CEO, "Levi's") == []
    assert [r.title for r in store.list(CALVIN)] == ["Calvin Q1"]
    assert store.match_mentions(CALVIN, "Levi's") == []
    assert store.soft_delete(CALVIN, [mine.report_id], "delete it", "t9") == []
    assert store.get(LEVIS, mine.report_id) is not None
    with pytest.raises(KeyError):
        store.save(CALVIN, draft(report_id=mine.report_id))  # no new version of someone else's report


def test_lost_brand_hides_reports_and_restoring_brings_them_back(store):
    saved = store.save(LEVIS, draft())
    reduced = User(id="bm_levis", name="BM", role="brand_manager", brands=["Canada Goose"])
    assert store.list(reduced) == [] and store.get(reduced, saved.report_id) is None
    assert store.match_mentions(reduced, "Levi's") == [] and store.created_in(reduced, "conv1") == []
    assert store.soft_delete(reduced, [saved.report_id], "delete", "t") == []
    assert store.get(LEVIS, saved.report_id) is not None  # access back: visible again


def test_mentions_match_fuzzily(store):
    a = store.save(LEVIS, draft(title="A", mentions=["Levi's", "Jeans"]))
    store.save(LEVIS, draft(title="B", mentions=["Canada Goose"]))
    assert [r.report_id for r in store.match_mentions(LEVIS, "levis")] == [a.report_id]
    assert [r.report_id for r in store.match_mentions(LEVIS, "Levi")] == [a.report_id]
    assert store.match_mentions(LEVIS, "loyalty programme") == []  # not a known value: matches nothing
    assert store.match_mentions(LEVIS, "") == []


def test_created_in_uses_the_first_version(store):
    old = store.save(LEVIS, draft(title="Old", conversation="conv0"))
    store.save(LEVIS, draft(title="Old redone", report_id=old.report_id, conversation="conv1"))
    new = store.save(LEVIS, draft(title="New", conversation="conv1"))
    assert [r.report_id for r in store.created_in(LEVIS, "conv1")] == [new.report_id]


def test_soft_delete_removes_every_version_and_audits_titles_only(store):
    a = store.save(LEVIS, draft(title="Q1", body="SECRET BODY"))
    store.save(LEVIS, draft(title="Q1 short", report_id=a.report_id, body="SECRET BODY"))
    b = store.save(LEVIS, draft(title="Other"))
    deleted = store.soft_delete(LEVIS, [a.report_id, "rep_unknown"], "delete the Q1 one", "trace9")
    assert [r.report_id for r in deleted] == [a.report_id]
    assert store.get(LEVIS, a.report_id) is None and store.get(LEVIS, b.report_id) is not None
    rows = store._conn.execute("SELECT deleted_at FROM reports WHERE report_id = ?", (a.report_id,)).fetchall()
    assert len(rows) == 2 and all(r[0] for r in rows)  # all versions
    (entry,) = store.audit()
    assert (entry.user_id, entry.action, entry.request_text, entry.trace_id) == ("bm_levis", "delete",
                                                                                 "delete the Q1 one", "trace9")
    assert entry.report_ids == [a.report_id] and entry.titles == ["Q1 short"]
    assert "SECRET BODY" not in store._conn.execute("SELECT * FROM audit").fetchone()[5:]


def test_delete_and_audit_commit_together(store, monkeypatch):
    saved = store.save(LEVIS, draft())

    def broken(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "_write_audit", broken)
    with pytest.raises(RuntimeError):
        store.soft_delete(LEVIS, [saved.report_id], "delete", "t")
    assert store.get(LEVIS, saved.report_id) is not None  # neither the delete nor the audit row
    assert store.audit() == []


def test_purge_after_30_days_and_admin_restore(store, clock):
    a = store.save(LEVIS, draft(title="A"))
    b = store.save(LEVIS, draft(title="B"))
    store.soft_delete(LEVIS, [a.report_id, b.report_id], "delete all", "t")
    assert store.restore(b.report_id, "admin", "deleted by mistake")
    assert store.get(LEVIS, b.report_id) is not None
    assert store.audit()[-1].action == "restore"
    assert store.purge(30) == 0
    clock.t += timedelta(days=31)
    assert store.purge(30) == 1
    assert not store.restore(a.report_id, "admin", "too late")
    assert store._conn.execute("SELECT COUNT(*) FROM reports WHERE report_id = ?", (a.report_id,)).fetchone()[0] == 0


def test_store_persists_across_sessions(tmp_path, clock):
    path = tmp_path / "reports.db"
    saved = SQLiteReportStore(path, clock=clock).save(LEVIS, draft())
    assert SQLiteReportStore(path, clock=clock).get(LEVIS, saved.report_id).title == "Q1 report"


# Mentions (D3.18)

KNOWN = KnownValues({"category": ["Jeans", "Outerwear & Coats"], "department": ["Men", "Women"],
                     "country": ["Brasil", "Spain"], "state": ["California"], "city": ["Reading", "Madrid"]},
                    date(2019, 1, 1), date(2026, 10, 14))


def test_mentions_from_body_and_results():
    scanner = MentionScanner(["Levi's", "Calvin Klein", "Calvin Klein Jeans", "Carolina"], KNOWN, stoplist={"Carolina"})
    body = "Levi's Jeans grew in Spain. Calvin Klein Jeans fell. Reading the trend, Carolina wins. Menswear."
    df = pd.DataFrame({"brand": ["Carolina", "Unknown"], "city": ["Madrid", None], "revenue": [1.0, 2.0]})
    found = scanner.scan(body, [df])
    assert found == ["Calvin Klein Jeans", "Carolina", "Jeans", "Levi's", "Madrid", "Reading", "Spain"]
    assert "Calvin Klein" not in found  # longest name first; "Men" inside "Menswear" doesn't count
