from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from analyst import cli
from analyst.config import User, load_settings
from analyst.observability import stats
from analyst.observability.tracer import JsonlExporter, Tracer

CEO = User(id="ceo", name="CEO", role="ceo", brands=["*"])


def gw(start, code=None, rows=1, diagnosis=None):
    return {"name": "gateway", "start_ns": start, "attributes": {"code": code, "rows": rows, "diagnosis": diagnosis}}


def sql_tool(start, code=None):
    return {"name": "tool_call", "start_ns": start, "attributes": {"tool": "run_sql", "code": code}}


def turn(trace_id, outcome="answered", reason="OK", attempt=1, latency=1000, tokens=100, bytes_=0, spans=(),
         events=(), fallback=False, llm_retries=0, bq_retries=0, breaker_opens=0):
    return {"trace_id": trace_id, "attempt": attempt, "outcome": outcome, "reason": reason,
            "security_events": [{"code": c, "source": "gateway"} for c in events],
            "totals": {"tokens": tokens, "bytes": bytes_, "latency_ms": latency, "fallback_used": fallback,
                       "llm_retries": llm_retries, "bq_retries": bq_retries, "breaker_opens": breaker_opens},
            "spans": list(spans)}


@pytest.mark.parametrize("values, p, expected", [
    ([], 50, None), ([5], 95, 5), ([1, 2, 3, 4], 50, 2), (list(range(1, 21)), 95, 19), (list(range(1, 21)), 50, 10),
])
def test_percentile_nearest_rank(values, p, expected):
    assert stats.percentile(values, p) == expected


def test_business_metrics():
    records = [
        turn("a", latency=1000, tokens=100, bytes_=10_000_000, fallback=True, llm_retries=2),
        turn("b", "failed", "LLM_DOWN", attempt=1, latency=500, tokens=50, llm_retries=6, breaker_opens=1),
        turn("b", "answered", "OK", attempt=2, latency=3000, tokens=150),  # /retry: latest attempt counts
        turn("c", "declined", "NOT_SELECT", events=["NOT_SELECT"], latency=200),
        turn("d", "partial", "BQ_DOWN", bq_retries=2, spans=[gw(1, "BQ_DOWN")], latency=4000),
    ]
    st = stats.compute(records)
    assert st.turns == 4
    assert st.outcomes == {"answered": 2, "declined": 1, "partial": 1}
    assert st.reasons == {"NOT_SELECT": 1, "BQ_DOWN": 1}  # the failed first attempt of b is superseded
    assert (st.llm_retries, st.bq_retries, st.breaker_opens, st.bq_errors) == (8, 2, 1, 1)
    assert st.fallback_turns == 1
    assert st.tokens == 100 + 50 + 150 + 100 + 100  # every attempt cost tokens
    assert sorted(st.latencies_ms) == [200, 1000, 3000, 4000]


def test_engineering_metrics():
    records = [
        turn("a", spans=[gw(1)]),  # first attempt ok
        turn("b", spans=[gw(1, "PARSE_ERROR"), gw(2)]),  # one repair
        turn("c", "partial", "STUCK", spans=[gw(1, "BQ_INVALID"), sql_tool(1, "BQ_INVALID"),
                                             gw(2, "BQ_INVALID"), sql_tool(2, "BQ_INVALID")]),
        turn("d", spans=[gw(1, rows=0, diagnosis={"filter_problems": [{"column": "country"}]}), gw(2)]),
        turn("e", spans=[gw(1, rows=0, diagnosis={"structural": "genuine_empty"})]),
        turn("f", "declined", "LEAK_BLOCKED", events=["LEAK_BLOCKED"]),
        turn("g", "partial", "GROUNDING_REMOVED",
             spans=[{"name": "output_check", "start_ns": 1, "attributes": {"numbers_removed": ["12", "40%"]}}]),
    ]
    st = stats.compute(records)
    assert st.sql_turns == 5
    assert st.first_attempt_ok == 3  # a, d (empty is not an error), e
    assert st.repairs == 2  # b's second query and c's second query
    assert st.stuck_fires == 1
    assert (st.empty_genuine, st.empty_diagnosed) == (1, 1)
    assert st.security_events == {"LEAK_BLOCKED": 1}
    assert (st.numbers_removed, st.turns_with_removals) == (2, 1)


def test_render_shows_both_views():
    st = stats.compute([turn("a", spans=[gw(1, "PARSE_ERROR"), gw(2)]), turn("b", "failed", "LLM_DOWN")])
    console = Console(width=120, record=True, color_system=None)
    for table in stats.render(st):
        console.print(table)
    out = console.export_text()
    for expected in ["Business view", "Answered", "1 (50%)", "Reasons (not OK)", "LLM_DOWN 1", "Latency p50 / p95",
                     "Engineering view", "First-attempt SQL success", "0 (0%)", "Repairs per question", "1.0",
                     "Empty results: genuine / diagnosed", "Figures removed by grounding"]:
        assert expected in out, expected


def test_stats_command_reads_this_conversation(tmp_path, monkeypatch):
    from test_cli import FakeGraph

    monkeypatch.chdir(tmp_path)
    tracer = Tracer(load_settings(), CEO, "conv1", JsonlExporter(Path("traces")))
    monkeypatch.setattr(cli, "build_session", lambda user_id, faults=None: cli.Session(CEO, FakeGraph("**42**"),
                                                                                      "conv1", tracer))
    runner = CliRunner()
    empty = runner.invoke(cli.app, ["chat", "--user", "ceo"], input="/stats\n/exit\n")
    assert "No turns yet." in empty.output
    result = runner.invoke(cli.app, ["chat", "--user", "ceo"], input="revenue?\n/stats\n/exit\n",
                           env={"COLUMNS": "120"})
    assert "Business view" in result.output and "Engineering view" in result.output
    assert "Answered" in result.output and "1 (100%)" in result.output
