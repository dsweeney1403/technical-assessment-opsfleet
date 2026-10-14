from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from analyst import cli
from analyst.config import ConfigError, User, load_settings
from analyst.doctor import Check
from analyst.llm.client import LLMFatalError, LLMUnavailableError
from analyst.observability.tracer import Tracer, read_traces
from analyst.state import Draft

runner = CliRunner()
CEO = User(id="ceo", name="CEO", role="ceo", brands=["*"])
BM = User(id="bm_calvin", name="Calvin Klein brand manager", role="brand_manager", brands=["Calvin Klein"])


class FakeGraph:
    """`outcomes` are used one per turn; an exception outcome fails that turn."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.inputs = []
        self.text = None

    def stream(self, payload, config, stream_mode=None):
        self.inputs.append(payload)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        yield {"classify": {}}
        if isinstance(outcome, Exception):
            raise outcome
        self.text = outcome

    def get_state(self, config):
        return SimpleNamespace(values={"draft": Draft(text=self.text or "", chart_paths=["charts/x.png"]),
                                       "trace_id": "7f3a1c2e"})


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # traces/ is written relative to the working directory


def use_session(monkeypatch, outcome="**42**", user=CEO, *more):
    graph = FakeGraph(outcome, *more)
    tracer = Tracer(load_settings(), user, "t")
    monkeypatch.setattr(cli, "build_session", lambda user_id, faults=None: cli.Session(user, graph, "t", tracer))
    return graph


def test_whoami_help_and_exit(monkeypatch):
    use_session(monkeypatch)
    result = runner.invoke(cli.app, ["chat", "--user", "ceo"], input="/whoami\n/help\n/exit\n")
    assert result.exit_code == 0
    assert "Brands: all brands" in result.output
    assert "/exit" in result.output


def test_answer_rendered_with_chart_path(monkeypatch):
    graph = use_session(monkeypatch)
    result = runner.invoke(cli.app, ["chat", "--user", "ceo"], input="revenue?\n/exit\n")
    assert "42" in result.output and "Chart saved: charts/x.png" in result.output
    assert graph.inputs[0]["user"] == CEO  # identity injected by code
    assert "trace 7f3a1c2e" in result.output
    (record,) = read_traces(Path("traces/t.jsonl"))
    assert record["trace_id"] == "7f3a1c2e" and record["outcome"] == "answered"


@pytest.mark.parametrize(
    "exc, expected",
    [
        (LLMFatalError(400, "API key not valid"), "doctor"),
        (LLMUnavailableError("down"), "Type /retry"),
        (RuntimeError("bug"), "Something went wrong"),
    ],
)
def test_errors_are_friendly_and_chat_continues(monkeypatch, exc, expected):
    use_session(monkeypatch, outcome=exc)
    result = runner.invoke(cli.app, ["chat", "--user", "ceo"], input="revenue?\n/whoami\n/exit\n")
    assert result.exit_code == 0
    assert expected in result.output
    assert "Traceback" not in result.output
    assert result.output.index(expected) < result.output.index("trace 7f3a1c2e")  # message first, then the id
    assert "Brands: all brands" in result.output  # still running after the error


def test_unknown_user_is_friendly(monkeypatch):
    def boom(user_id, faults=None):
        raise ConfigError("Unknown user 'zed'. Known users: ceo, bm_calvin, bm_levis.")

    monkeypatch.setattr(cli, "build_session", boom)
    result = runner.invoke(cli.app, ["chat", "--user", "zed"])
    assert result.exit_code == 1 and "Known users" in result.output and "Traceback" not in result.output


def test_scoped_user_can_chat(monkeypatch):
    use_session(monkeypatch, user=BM)
    result = runner.invoke(cli.app, ["chat", "--user", "bm_calvin"], input="/whoami\n/exit\n")
    assert result.exit_code == 0 and "Brands: Calvin Klein." in result.output


def test_doctor_reports_failures_with_fix(monkeypatch):
    monkeypatch.setattr(cli.doctor_checks, "run_checks", lambda: [
        Check("Config", True, "ok"), Check("Gemini", False, "API key not valid", "Check GEMINI_API_KEY")])
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "FAIL" in result.output and "Check GEMINI_API_KEY" in result.output


def test_doctor_all_ok(monkeypatch):
    monkeypatch.setattr(cli.doctor_checks, "run_checks", lambda: [Check("Config", True, "ok")])
    assert runner.invoke(cli.app, ["doctor"]).exit_code == 0


def test_doctor_gemini_check_flags_missing_models(monkeypatch):
    from analyst import doctor

    class FakeModels:
        def list(self):
            return [SimpleNamespace(name="models/gemini-3.8-flash")]

    monkeypatch.setattr("google.genai.Client", lambda api_key: SimpleNamespace(models=FakeModels()))
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    check = doctor.check_gemini()
    assert not check.ok and "gemini-3.5-flash-lite" in check.detail


def test_retry_resumes_the_stopped_turn_from_the_checkpoint(monkeypatch):
    graph = use_session(monkeypatch, LLMUnavailableError("down"), CEO, "Resumed answer")
    result = runner.invoke(cli.app, ["chat", "--user", "ceo"], input="revenue?\n/retry\n/retry\n/exit\n")
    assert "Type /retry" in result.output and "Resumed answer" in result.output
    assert graph.inputs == [{"user_input": "revenue?", "user": CEO, "conversation_id": "t"}, None]  # None resumes
    assert "Nothing to retry." in result.output  # second /retry: nothing pending


def test_simulate_flags_reach_the_session_and_are_announced(monkeypatch):
    seen = {}

    def build(user_id, faults=None):
        seen["faults"] = faults
        return cli.Session(CEO, FakeGraph("ok"), "t", Tracer(load_settings(), CEO, "t"))

    monkeypatch.setattr(cli, "build_session", build)
    result = runner.invoke(cli.app, ["chat", "--user", "ceo", "--simulate-llm-429", "--simulate-bq-error"],
                           input="/exit\n")
    assert seen["faults"] == cli.Faults(llm_429=True, bq_error=True)
    assert "Simulating: Gemini 429" in result.output and "BigQuery errors" in result.output


def test_status_line_text_for_updates():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    query = AIMessage("", tool_calls=[{"name": "run_sql", "args": {}, "id": "1"}])
    bad = ToolMessage('{"error": {"code": "UNKNOWN_TABLE"}}', tool_call_id="1", name="run_sql")
    assert cli.status_for({"agent": {"turn_messages": [query]}}) == "Running a query..."
    assert cli.status_for({"tools": {"turn_messages": [query, bad]}}) == "Checking the query..."
    assert cli.status_for({"output_check": {"turn_messages": [HumanMessage("fix")]}}) == "Checking the answer..."
    assert cli.status_for({"classify": {}}) is None


class ConfirmingGraph(FakeGraph):
    """First turn stops at a confirmation; the reply resumes it."""

    def __init__(self):
        super().__init__("Deleted 'Q1'.")
        self.asking = False

    def stream(self, payload, config, stream_mode=None):
        self.inputs.append(payload)
        self.asking = not self.asking and isinstance(payload, dict)
        self.text = "Deleted 'Q1'."
        yield {"confirm_delete": {}}

    def get_state(self, config):
        snap = super().get_state(config)
        if self.asking:
            snap.interrupts = (SimpleNamespace(value={"kind": "delete", "text": "This deletes 'Q1' and its earlier drafts."}),)
        return snap


def test_reply_to_a_confirmation_resumes_the_turn(monkeypatch):
    from langgraph.types import Command

    graph = ConfirmingGraph()
    tracer = Tracer(load_settings(), CEO, "t")
    monkeypatch.setattr(cli, "build_session", lambda user_id, faults=None: cli.Session(CEO, graph, "t", tracer))
    result = runner.invoke(cli.app, ["chat", "--user", "ceo"], input="delete my Q1 report\ngo for it\n/exit\n")
    assert "This deletes 'Q1' and its earlier drafts." in result.output and "Deleted 'Q1'." in result.output
    resume = graph.inputs[1]
    assert isinstance(resume, Command) and resume.resume == "go for it"
    assert resume.update["continues"] == "7f3a1c2e" and resume.update["trace_id"] != "7f3a1c2e"
    first, second = read_traces(Path("traces/t.jsonl"))
    assert first["reason"] == "AWAITING_CONFIRMATION" and second["user_input"] == "go for it"


def test_reports_and_download_commands(monkeypatch, tmp_path):
    from analyst.store.base import NewVersion
    from analyst.store.sqlite_store import SQLiteReportStore

    reports = SQLiteReportStore(":memory:")
    reports.save(CEO, NewVersion(conversation_id="c", title="Q1 report", body="# Q1 report\nBody", data_ts="2026-10-14",
                                 trace_id="t"))
    graph = FakeGraph("ok")
    tracer = Tracer(load_settings(), CEO, "t")
    monkeypatch.setattr(cli, "build_session",
                        lambda user_id, faults=None: cli.Session(CEO, graph, "t", tracer, reports))
    result = runner.invoke(cli.app, ["chat", "--user", "ceo"], input="/reports\n/download 1\n/download 9\n/exit\n",
                           env={"COLUMNS": "150"})
    assert "Q1 report" in result.output and "Saved a copy to reports/q1-report.md" in result.output
    assert Path("reports/q1-report.md").read_text().startswith("# Q1 report")
    assert "Give a report number from /reports" in result.output
