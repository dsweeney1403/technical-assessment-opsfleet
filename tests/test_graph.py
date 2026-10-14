from datetime import date

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, SystemMessage

from analyst.config import User, load_settings
from analyst.graph import Deps, build_graph
from analyst.llm.client import LLMFatalError, LLMUnavailableError
from analyst.safety.classifier import ClassifierOutput
from analyst.state import Intent
from analyst.store.results import ResultStore

CEO = User(id="ceo", name="CEO", role="ceo", brands=["*"])


class FakeLLM:
    """Scripted stand-in for analyst.llm.client.LLM."""

    def __init__(self, intents, agent_replies=(), light_reply="You're welcome."):
        self.intents = list(intents)
        self.agent_replies = list(agent_replies)
        self.light_reply = light_reply
        self.calls = []

    def structured(self, role, schema, messages, stats=None):
        self.calls.append(("structured", role))
        intent = self.intents.pop(0)
        if isinstance(intent, Exception):
            raise intent
        return ClassifierOutput(intent=intent, reason="test")

    def invoke(self, role, messages, tools=None, stats=None):
        self.calls.append(("invoke", role, bool(tools)))
        if stats is not None:
            stats.input_tokens += 10
        if role == "light":
            if isinstance(messages[0], SystemMessage) and "running summary" in messages[0].content:
                return AIMessage("summary text")
            return AIMessage(self.light_reply)
        reply = self.agent_replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeGateway:
    def __init__(self, store):
        self.store = store
        self.sqls = []

    def run(self, sql, user):
        self.sqls.append(sql)
        df = pd.DataFrame({"month": ["2026-01", "2026-02"], "revenue": [100.0, 120.0]})
        stored = self.store.put(df, sql, "WITH ... " + sql, "2026-10-14T00:00:00+00:00")
        return {"result_id": stored.result_id, "columns": list(df.columns), "row_count": 2,
                "preview": df.to_dict("records"), "truncated": False, "empty": False,
                "null_flags": {}, "data_ts": stored.data_ts}


def sql_call(i, sql="SELECT month, SUM(sale_price) AS revenue FROM sales GROUP BY 1"):
    return AIMessage("", tool_calls=[{"name": "run_sql", "args": {"sql": sql}, "id": f"c{i}"}])


def chart_call(result_id="r1"):
    return AIMessage("", tool_calls=[{"name": "make_chart", "id": "ch1", "args": {
        "result_id": result_id, "chart_type": "bar", "x": "month", "y": ["revenue"], "title": "Revenue"}}])


@pytest.fixture
def setup(tmp_path):
    settings = load_settings()
    settings.charts.dir = tmp_path / "charts"
    store = ResultStore()
    gateway = FakeGateway(store)

    def make(llm):
        graph = build_graph(Deps(settings, llm, gateway, store, today=lambda: date(2026, 10, 14)))
        config = {"configurable": {"thread_id": "t1"}}

        def turn(text):
            return graph.invoke({"user_input": text, "user": CEO}, config)

        return turn

    return make, gateway, store


def test_thanks_takes_light_path(setup):
    make, gateway, _ = setup
    llm = FakeLLM(["conversation"])
    state = make(llm)("thanks")
    assert state["draft"].text == "You're welcome."
    assert state["budget"].tool_calls_used == 0 and gateway.sqls == []
    assert ("invoke", "agent", True) not in llm.calls


def test_analysis_runs_sql_and_records_result(setup):
    make, gateway, _ = setup
    llm = FakeLLM(["analysis"], [sql_call(1), AIMessage("| month | revenue |\n|---|---|\n| 2026-01 | 100 |")])
    state = make(llm)("What was revenue by month this year?")
    assert state["budget"].tool_calls_used == 1
    assert state["records"][0].result_id == "r1"
    assert state["records"][0].sql.startswith("SELECT month")
    assert state["draft"].text.startswith("| month")
    assert [m.text for m in state["messages"]][0] == "What was revenue by month this year?"


def test_make_it_a_chart_reuses_stored_result(setup):
    make, gateway, store = setup
    llm = FakeLLM(["analysis", "analysis"],
                  [sql_call(1), AIMessage("table"), chart_call("r1"), AIMessage("Here is the chart.")])
    turn = make(llm)
    turn("revenue by month this year")
    state = turn("make it a chart")
    assert len(gateway.sqls) == 1  # no second query
    assert len(state["draft"].chart_paths) == 1
    assert state["draft"].chart_paths[0].endswith(".png")


def test_tool_cap_stops_the_loop(setup):
    make, gateway, _ = setup
    llm = FakeLLM(["analysis"], [sql_call(i) for i in range(6)] + [AIMessage("Answer with what I have.")])
    state = make(llm)("why is Texas underspending?")
    assert state["budget"].tool_calls_used == 6 and len(gateway.sqls) == 6
    assert llm.calls[-1] == ("invoke", "agent", False)  # final agent call has no tools bound


def test_parallel_calls_past_the_cap_are_refused(setup):
    make, gateway, _ = setup
    many = AIMessage("", tool_calls=[{"name": "run_sql", "args": {"sql": "SELECT 1"}, "id": f"p{i}"} for i in range(8)])
    llm = FakeLLM(["analysis"], [many, AIMessage("done")])
    state = make(llm)("lots of queries")
    assert len(gateway.sqls) == 6 and state["budget"].tool_calls_used == 6


def test_classifier_failure_fails_open_to_agent(setup):
    make, _, _ = setup
    llm = FakeLLM([LLMUnavailableError("down")], [AIMessage("answered anyway")])
    state = make(llm)("revenue?")
    assert state["classified"].intent == Intent.unavailable
    assert state["draft"].text == "answered anyway"


def test_fatal_llm_error_propagates(setup):
    make, _, _ = setup
    llm = FakeLLM(["analysis"], [LLMFatalError(400, "API key not valid")])
    with pytest.raises(LLMFatalError):
        make(llm)("revenue?")


def test_history_window_and_summary(setup):
    make, _, _ = setup
    settings_turns = load_settings().memory.verbatim_turns
    llm = FakeLLM(["conversation"] * (settings_turns + 1))
    turn = make(llm)
    for i in range(settings_turns + 1):
        state = turn(f"hello {i}")
    assert len(state["messages"]) == settings_turns * 2
    assert state["summary"] == "summary text"
    assert state["messages"][0].text == "hello 1"
