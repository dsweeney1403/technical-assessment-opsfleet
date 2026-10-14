from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from analyst import cli
from analyst.config import ConfigError, User
from analyst.doctor import Check
from analyst.llm.client import LLMFatalError, LLMUnavailableError
from analyst.state import Draft

runner = CliRunner()
CEO = User(id="ceo", name="CEO", role="ceo", brands=["*"])
BM = User(id="bm_calvin", name="Calvin Klein brand manager", role="brand_manager", brands=["Calvin Klein"])


class FakeGraph:
    def __init__(self, outcome):
        self.outcome = outcome
        self.inputs = []

    def invoke(self, payload, config):
        self.inputs.append(payload)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return {"draft": Draft(text=self.outcome, chart_paths=["charts/x.png"])}


def use_session(monkeypatch, outcome="**42**", user=CEO):
    graph = FakeGraph(outcome)
    monkeypatch.setattr(cli, "build_session", lambda user_id: cli.Session(user, graph, "t"))
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


@pytest.mark.parametrize(
    "exc, expected",
    [
        (LLMFatalError(400, "API key not valid"), "doctor"),
        (LLMUnavailableError("down"), "try again"),
        (RuntimeError("bug"), "Something went wrong"),
    ],
)
def test_errors_are_friendly_and_chat_continues(monkeypatch, exc, expected):
    use_session(monkeypatch, outcome=exc)
    result = runner.invoke(cli.app, ["chat", "--user", "ceo"], input="revenue?\n/whoami\n/exit\n")
    assert result.exit_code == 0
    assert expected in result.output
    assert "Traceback" not in result.output
    assert "Brands: all brands" in result.output  # still running after the error


def test_unknown_user_is_friendly(monkeypatch):
    def boom(user_id):
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
