import json

import pytest
from rich.console import Console
from typer.testing import CliRunner

from analyst import cli
from analyst.config import load_settings
from analyst.observability import viewer

MS = 1_000_000


def span(name, sid, parent, start, attrs=None, status="ok", end=None):
    return {"name": name, "span_id": sid, "parent_span_id": parent, "start_ns": start * MS,
            "end_ns": (end or start + 5) * MS, "attributes": attrs or {}, "status": status}


def record(trace_id="7f3a1c2e", attempt=1, outcome="answered", reason="OK"):
    return {
        "trace_id": trace_id, "conversation_id": "conv1", "attempt": attempt,
        "user": {"id": "ceo", "role": "ceo"}, "started_at": "2026-10-14T09:00:00+00:00",
        "user_input": "revenue in Brazil", "outcome": outcome, "reason": reason, "error": None,
        "answer": "Revenue in Brasil was 100.", "security_events": [],
        "totals": {"tokens": 330, "bytes": 10_485_760, "latency_ms": 2400, "tool_calls": 1, "fallback_used": True},
        "spans": [
            span("turn", "root", None, 0, end=2400),
            span("classify", "c", "root", 1, {"verdict": "analysis", "confident": True, "out_of_scope_brands": []}),
            span("llm_call", "c1", "c", 2, {"role": "light", "model": "gemini-3.5-flash-lite",
                                            "tokens": {"total": 110}, "messages_in": [], "output": {"intent": "analysis"}}),
            span("agent_step", "a1", "root", 10, {"step": 1}),
            span("llm_call", "l1", "a1", 11, {
                "role": "agent", "model": "gemini-3.6-flash", "tokens": {"total": 220}, "retries": 2,
                "fallback_used": True,
                "messages_in": [{"role": "system", "content": "SYSTEM PROMPT TEXT"},
                                {"role": "human", "content": "revenue in Brazil"}],
                "output": {"role": "ai", "content": "",
                           "tool_calls": [{"name": "run_sql", "args": {"sql": "SELECT 1"}, "id": "x"}]}}),
            span("tool_call", "t1", "a1", 20, {"tool": "run_sql", "args": {"sql": "SELECT 1"},
                                               "result": '{"row_count": 1, "preview": [{"r": 100}]}'}),
            span("gateway", "g1", "t1", 21, {
                "sql_in": "SELECT SUM(x) FROM sales WHERE country = 'Brasil'",
                "sql_rewritten": "WITH sales AS (SELECT ...) SELECT SUM(x) FROM sales WHERE country = 'Brasil'",
                "code": None, "rows": 1, "bytes_billed": 10_485_760, "transient_retries": 0,
                "diagnosis": None}),
            span("output_check", "o", "root", 30, {"leak_hits": [], "numbers_removed": [], "grounding_retry": False}),
        ],
    }


def write(tmp_path, *records):
    d = tmp_path / "traces"
    d.mkdir(exist_ok=True)
    with (d / "conv1.jsonl").open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return d


def text_of(tree) -> str:
    console = Console(width=200, record=True, color_system=None)
    console.print(tree)
    return console.export_text()


def test_find_by_prefix_returns_every_attempt(tmp_path):
    d = write(tmp_path, record(attempt=1, outcome="failed", reason="LLM_DOWN"), record(attempt=2),
              record(trace_id="0000aaaa"))
    found = viewer.find(d, "7f3a")
    assert [r["attempt"] for r in found] == [1, 2]


@pytest.mark.parametrize("trace_id, message", [("ffff", "No trace ffff"), ("", "Give a trace id")])
def test_find_missing(tmp_path, trace_id, message):
    d = write(tmp_path, record())
    with pytest.raises(viewer.TraceNotFound, match=message):
        viewer.find(d, trace_id)


def test_ambiguous_prefix(tmp_path):
    d = write(tmp_path, record(trace_id="7f3a0001"), record(trace_id="7f3a0002"))
    with pytest.raises(viewer.TraceNotFound, match="ambiguous"):
        viewer.find(d, "7f3a")


def test_compact_shows_steps_sql_codes_and_outcome_without_messages():
    out = text_of(viewer.render(record()))
    for expected in ["trace 7f3a1c2e", "question: revenue in Brazil", "classify  analysis", "agent step 1",
                     "gemini-3.6-flash", "220 tokens", "retries 2", "fallback used", "run_sql  ok",
                     "gateway  1 rows · 10.5 MB", "sql in: SELECT SUM(x)", "sql rewritten: WITH sales AS",
                     "numbers removed: none", "outcome  answered (OK)", "330 tokens", "2.4 s", "1 tool calls"]:
        assert expected in out, expected
    assert "SYSTEM PROMPT TEXT" not in out and "result:" not in out


def test_full_adds_messages_results_and_answer():
    out = text_of(viewer.render(record(), full=True))
    assert "in system: SYSTEM PROMPT TEXT" in out
    assert 'out: [tool calls] run_sql({"sql": "SELECT 1"})' in out
    assert 'result: {"row_count": 1' in out
    assert "answer: Revenue in Brasil was 100." in out


def test_failed_turn_shows_error_and_security_events():
    r = record(outcome="declined", reason="NOT_SELECT")
    r["error"] = None
    r["security_events"] = [{"code": "NOT_SELECT", "source": "gateway", "detail": "DELETE FROM sales"}]
    r["spans"][6]["attributes"].update(code="NOT_SELECT", rows=None)
    r["spans"][6]["status"] = "error"
    out = text_of(viewer.render(r))
    assert "gateway  NOT_SELECT" in out and "outcome  declined (NOT_SELECT)" in out
    assert "security event NOT_SELECT: gateway: DELETE FROM sales" in out


def test_trace_command(tmp_path, monkeypatch):
    d = write(tmp_path, record())
    settings = load_settings()
    settings.tracing.dir = d
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["trace", "7f3a", "--full"], env={"COLUMNS": "200"})
    assert result.exit_code == 0 and "outcome  answered (OK)" in result.output and "in system:" in result.output
    missing = runner.invoke(cli.app, ["trace", "beef"])
    assert missing.exit_code == 1 and "No trace beef" in missing.output
