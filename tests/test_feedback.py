"""Thumbs feedback next to the trace ID (D4.19, D8.1)."""

from pathlib import Path

import pytest
import typer
from test_cli import BM, CEO, FakeGraph, runner, use_session

from analyst import cli
from analyst.config import load_settings
from analyst.observability import stats, viewer
from analyst.observability.tracer import JsonlExporter, Tracer, read_traces


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def chat(monkeypatch, keys, lines, outcome="**42**"):
    """Run the chat with a terminal attached: `keys` are single key presses, `lines` the typed input."""
    keys = list(keys)
    monkeypatch.setattr(cli, "interactive", lambda: True)
    monkeypatch.setattr(typer, "getchar", lambda: keys.pop(0))
    use_session(monkeypatch, outcome)
    return runner.invoke(cli.app, ["chat", "--user", "ceo"], input="".join(f"{x}\n" for x in lines))


def feedback_lines():
    return [r for r in read_traces(Path("traces/t.jsonl")) if r.get("type") == "feedback"]


def test_thumbs_up_is_stored_next_to_the_trace(monkeypatch):
    result = chat(monkeypatch, ["u"], ["What was revenue?", "/exit"])
    assert "Helpful? u = yes, d = no, Enter = skip" in result.output
    (fb,) = feedback_lines()
    turn = next(r for r in read_traces(Path("traces/t.jsonl")) if r.get("type") != "feedback")
    assert (fb["trace_id"], fb["thumbs"], fb["reason"]) == (turn["trace_id"], "up", None)


def test_thumbs_down_asks_for_an_optional_reason(monkeypatch):
    chat(monkeypatch, ["d", "d"], ["What was revenue?", "Wrong date range", "And returns?", "", "/exit"])
    first, second = feedback_lines()
    assert (first["thumbs"], first["reason"]) == ("down", "Wrong date range")
    assert (second["thumbs"], second["reason"]) == ("down", None)


def test_any_other_key_skips(monkeypatch):
    chat(monkeypatch, ["\r"], ["What was revenue?", "/exit"])
    assert [f["thumbs"] for f in feedback_lines()] == ["skip"]


def test_no_prompt_without_a_terminal_after_a_failure_or_when_switched_off(monkeypatch):
    use_session(monkeypatch)
    result = runner.invoke(cli.app, ["chat", "--user", "ceo"], input="What was revenue?\n/exit\n")
    assert "Helpful?" not in result.output and feedback_lines() == []
    result = chat(monkeypatch, [], ["What was revenue?", "/exit"], outcome=RuntimeError("boom"))
    assert "Helpful?" not in result.output
    settings = load_settings()
    settings.feedback.thumbs_prompt = False
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    assert "Helpful?" not in chat(monkeypatch, [], ["What was revenue?", "/exit"]).output


def test_stats_count_thumbs_and_the_last_rating_of_an_answer_wins(tmp_path):
    tracer = Tracer(load_settings(), BM, "c", JsonlExporter(tmp_path))
    turn = {"trace_id": "a1", "conversation_id": "c", "attempt": 1, "outcome": "answered", "reason": "OK",
            "totals": {"latency_ms": 10}, "spans": []}
    tracer.exporter.write(turn)
    tracer.exporter.write({**turn, "trace_id": "b2"})
    tracer.add_feedback("a1", "down", "too long")
    tracer.add_feedback("a1", "up")
    tracer.add_feedback("b2", "skip")
    st = stats.compute(read_traces(tracer.path))
    assert st.turns == 2 and st.thumbs == {"up": 1, "skip": 1}
    engineering = stats.render(st)[1]
    assert "Thumbs up / down / skipped" in str(engineering.columns[0]._cells)


def test_viewer_shows_the_feedback_under_the_answer(tmp_path):
    from rich.console import Console

    tracer = Tracer(load_settings(), CEO, "c", JsonlExporter(tmp_path))
    tracer.exporter.write({"trace_id": "a1b2c3d4", "conversation_id": "c", "attempt": 1, "outcome": "answered",
                           "reason": "OK", "started_at": "now", "user": {"id": "ceo", "role": "ceo"},
                           "totals": {"tokens": 1, "bytes": 0, "latency_ms": 5, "tool_calls": 0},
                           "spans": [{"name": "turn", "span_id": "r", "parent_span_id": None, "start_ns": 0,
                                      "end_ns": 1, "attributes": {}, "status": "ok"}]})
    tracer.add_feedback("a1b2c3d4", "down", "wrong period")
    (record,) = viewer.find(tmp_path, "a1b2")
    console = Console(record=True, width=150)
    console.print(viewer.render(record))
    assert "feedback: thumbs down: wrong period" in console.export_text()
