import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from google.api_core import exceptions as gexc
from google.genai.errors import ClientError, ServerError
from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console

from analyst.config import User, load_settings
from analyst.graph import Deps, build_graph
from analyst.llm.client import LLM, LLMFatalError, LLMUnavailableError
from analyst.observability import tracer, viewer
from analyst.observability.tracer import JsonlExporter, Tracer, outcome_for, read_traces
from analyst.safety.classifier import ClassifierOutput
from analyst.sql.brands import BrandMatcher
from analyst.sql.gateway import Gateway, GatewayFatalError
from analyst.state import Budget, Classified, Intent
from analyst.store.results import ResultStore

CEO = User(id="ceo", name="CEO", role="ceo", brands=["*"])
CURATED = {"sales": "SELECT 1 AS a", "customers": "SELECT 2 AS b", "products": "SELECT 3 AS c"}
GOOD_SQL = "SELECT month, SUM(sale_price) AS revenue FROM sales GROUP BY 1"
ROW_ONLY_IN_BIGQUERY = "zz-row-55-never-seen"


def usage(msg: AIMessage) -> AIMessage:
    msg.usage_metadata = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
    return msg


def sql_call(i, sql=GOOD_SQL):
    return usage(AIMessage("", tool_calls=[{"name": "run_sql", "args": {"sql": sql}, "id": f"c{i}"}]))


class ScriptedChat:
    """A chat model per model id. `replies` are AIMessages or exceptions, used in order."""

    def __init__(self, replies):
        self.replies = replies

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, include_raw=False):
        outer = self

        class Structured:
            def invoke(self, messages):
                reply = outer._next()
                return {"raw": usage(AIMessage("")), "parsed": reply, "parsing_error": None}

        return Structured()

    def _next(self):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def invoke(self, messages):
        return self._next()


class FakeJob:
    def __init__(self, df=None, exc=None):
        self.total_bytes_processed = 2_000_000
        self.total_bytes_billed = 10_485_760
        self._df, self._exc = df, exc

    def result(self, timeout=None):
        if self._exc:
            raise self._exc
        return self

    def to_dataframe(self, create_bqstorage_client=True):
        return self._df

    def cancel(self):
        pass


class FakeBigQuery:
    def __init__(self, down=False):
        self.down = down

    def query(self, sql, job_config=None):
        if job_config.dry_run:
            return FakeJob()
        if self.down:
            return FakeJob(exc=gexc.ServiceUnavailable("down", errors=[{"reason": "backendError", "message": "x"}]))
        months = [f"2026-{i:02d}" for i in range(1, 61)]
        months[54] = ROW_ONLY_IN_BIGQUERY  # past the 50-row preview, so the model never sees it
        return FakeJob(df=pd.DataFrame({"month": months, "revenue": [100.0] * 60}))


@pytest.fixture
def run(tmp_path):
    settings = load_settings()
    settings.charts.dir = tmp_path / "charts"
    settings.tracing.dir = tmp_path / "traces"

    def make(chats: dict, bq=None):
        store = ResultStore()
        gateway = Gateway(settings.bigquery, bq or FakeBigQuery(), store, CURATED, BrandMatcher(["Levi's"]),
                          sleep=lambda s: None)
        llm = LLM(settings, "key", chat_factory=lambda model, **kw: ScriptedChat(chats[model]),
                  sleep=lambda s: None)
        graph = build_graph(Deps(settings, llm, gateway, store, today=lambda: date(2026, 10, 14)))
        t = Tracer(settings, CEO, "conv1")
        config = {"configurable": {"thread_id": "t1"}}

        def turn(text):
            payload = None if text is None else {"user_input": text, "user": CEO}
            with t.turn(graph, config, text) as cfg:
                for _ in graph.stream(payload, cfg, stream_mode="updates"):
                    pass
            return t.last

        return turn, t

    return settings, make


def intent(name):
    return ClassifierOutput(intent=name, reason="test")


def spans(record, name):
    return [s for s in record["spans"] if s["name"] == name]


def by_id(record):
    return {s["span_id"]: s for s in record["spans"]}


def test_repaired_answer_has_full_span_tree(run):
    settings, make = run
    chats = {
        settings.models.light: [intent("analysis")],
        settings.models.agent: [sql_call(1, "SELEC month FROM sales"), sql_call(2),
                                usage(AIMessage("Revenue in January 2026 was 100."))],
    }
    turn, t = make(chats)
    record = turn("revenue by month")

    assert record["outcome"] == "answered" and record["reason"] == "OK"
    assert record["user"] == {"id": "ceo", "role": "ceo"} and record["conversation_id"] == "conv1"
    assert record["models"]["agent"][0] == settings.models.agent
    ids = by_id(record)
    root = record["spans"][0]
    assert root["name"] == "turn" and root["parent_span_id"] is None

    (classify,) = spans(record, "classify")
    assert classify["attributes"]["verdict"] == "analysis"
    steps = spans(record, "agent_step")
    assert [s["attributes"]["step"] for s in steps] == [1, 2, 3]
    assert all(s["parent_span_id"] == root["span_id"] for s in [classify, *steps])

    llm_calls = spans(record, "llm_call")
    assert len(llm_calls) == 4  # classifier plus three agent calls
    assert ids[llm_calls[0]["parent_span_id"]]["name"] == "classify"
    agent_call = llm_calls[1]
    assert ids[agent_call["parent_span_id"]]["name"] == "agent_step"
    assert agent_call["attributes"]["model"] == settings.models.agent
    assert agent_call["attributes"]["tokens"] == {"input": 100, "output": 10, "total": 110}
    assert agent_call["attributes"]["messages_in"][0]["role"] == "system"
    assert agent_call["attributes"]["output"]["tool_calls"][0]["name"] == "run_sql"
    assert agent_call["attributes"]["fallback_used"] is False

    tool_calls = spans(record, "tool_call")
    assert [ids[s["parent_span_id"]]["attributes"]["step"] for s in tool_calls] == [1, 2]
    assert tool_calls[0]["attributes"]["code"] == "PARSE_ERROR"
    assert json.loads(tool_calls[1]["attributes"]["result"])["row_count"] == 60

    first, second = spans(record, "gateway")
    assert ids[first["parent_span_id"]] is ids[tool_calls[0]["span_id"]]
    assert first["attributes"]["code"] == "PARSE_ERROR" and first["status"] == "error"
    assert second["attributes"]["sql_in"] == GOOD_SQL
    assert second["attributes"]["sql_rewritten"].startswith("WITH sales AS (")
    assert second["attributes"]["rows"] == 60 and second["attributes"]["bytes_billed"] == 10_485_760

    (check,) = spans(record, "output_check")
    assert check["attributes"]["numbers_removed"] == [] and check["attributes"]["leak_hits"] == []
    assert record["totals"]["tokens"] == 4 * 110 and record["totals"]["tool_calls"] == 2
    assert record["totals"]["bytes"] == 10_485_760
    assert record["answer"].startswith("Revenue in January")
    assert read_traces(t.path) == [json.loads(json.dumps(record, default=str))]

    console = Console(width=200, record=True, color_system=None)
    console.print(viewer.render(read_traces(t.path)[0], full=True))
    shown = console.export_text()
    assert "run_sql  PARSE_ERROR" in shown and "gateway  60 rows" in shown and "outcome  answered (OK)" in shown


def test_trace_never_holds_rows_the_model_did_not_see(run):
    settings, make = run
    turn, t = make({settings.models.light: [intent("analysis")],
                    settings.models.agent: [sql_call(1), usage(AIMessage("Revenue was 100 each month."))]})
    turn("revenue by month")
    assert ROW_ONLY_IN_BIGQUERY not in t.path.read_text()


def test_fallback_model_and_retries_recorded(run):
    settings, make = run
    rate_limited = ClientError(429, {"error": {"code": 429, "message": "slow down"}})
    fallback = settings.models.agent_fallbacks[0]
    turn, _ = make({
        settings.models.light: [intent("analysis")],
        settings.models.agent: [rate_limited] * settings.llm_retry.max_attempts,
        fallback: [usage(AIMessage("Hello, no data needed."))],
    })
    record = turn("hi there")
    agent_call = spans(record, "llm_call")[1]
    assert agent_call["attributes"]["model"] == fallback
    assert agent_call["attributes"]["fallback_used"] is True
    assert agent_call["attributes"]["retries"] == settings.llm_retry.max_attempts - 1
    assert record["totals"]["fallback_used"] is True
    assert record["totals"]["llm_retries"] == settings.llm_retry.max_attempts - 1


def test_llm_down_is_traced_and_retry_is_a_second_attempt(run):
    settings, make = run
    down = ServerError(503, {"error": {"code": 503, "message": "down"}})
    agent_chain = [settings.models.agent, *settings.models.agent_fallbacks]
    chats = {m: [down] * 20 for m in {settings.models.light, *settings.models.light_fallbacks, *agent_chain}}
    chats[settings.models.light].insert(0, intent("analysis"))  # the light model is also the agent's last fallback
    turn, t = make(chats)
    with pytest.raises(LLMUnavailableError):
        turn("revenue by month")
    first = t.last
    assert (first["outcome"], first["reason"]) == ("failed", "LLM_DOWN")
    assert first["error"].startswith("LLMUnavailableError") and first["answer"] is None
    assert spans(first, "llm_call")[-1]["status"] == "error"

    chats[settings.models.agent][:] = [usage(AIMessage("Back now, nothing to query."))]
    second = turn(None)  # /retry resumes from the checkpoint
    assert second["trace_id"] == first["trace_id"] and second["attempt"] == 2
    assert second["outcome"] == "answered"
    assert [r["attempt"] for r in read_traces(t.path)] == [1, 2]


def test_bigquery_down_marks_code_made_results(run):
    settings, make = run
    two_queries = usage(AIMessage("", tool_calls=[
        {"name": "run_sql", "args": {"sql": GOOD_SQL}, "id": "a"},
        {"name": "run_sql", "args": {"sql": GOOD_SQL + " ORDER BY 1"}, "id": "b"},
    ]))
    turn, _ = make({settings.models.light: [intent("analysis")],
                    settings.models.agent: [two_queries, usage(AIMessage("BigQuery is down, so no figures."))]},
                   bq=FakeBigQuery(down=True))
    record = turn("revenue by month")
    assert (record["outcome"], record["reason"]) == ("partial", "BQ_DOWN")
    invoked, refused = spans(record, "tool_call")
    assert invoked["attributes"]["code"] == "BQ_DOWN" and "handled_in_code" not in invoked["attributes"]
    assert refused["attributes"]["handled_in_code"] is True and refused["attributes"]["code"] == "BQ_DOWN"
    assert spans(record, "gateway")[0]["attributes"]["code"] == "BQ_DOWN"
    assert record["totals"]["bq_retries"] == settings.bigquery.transient_max_attempts - 1


def test_stop_code_declines_with_security_event(run):
    settings, make = run
    turn, _ = make({settings.models.light: [intent("analysis")],
                    settings.models.agent: [sql_call(1, "DELETE FROM sales WHERE 1=1")]})
    record = turn("clear the sales table")
    assert (record["outcome"], record["reason"]) == ("declined", "NOT_SELECT")
    assert record["security_events"][0]["code"] == "NOT_SELECT"
    (tool,) = spans(record, "tool_call")
    assert tool["status"] == "error" and tool["attributes"]["code"] == "NOT_SELECT"
    assert spans(record, "gateway")[0]["attributes"]["code"] == "NOT_SELECT"


def test_emit_outside_a_run_does_nothing():
    tracer.emit("gateway", {"start_ns": 0, "end_ns": 1})


# Outcome and reason (D2.17)

def state(**kw):
    base = {"budget": Budget(), "removed_numbers": [], "classified": None, "turn_messages": [], "decline": None}
    return {**base, **kw}


def brands_out(*names):
    return Classified(intent=Intent.analysis, out_of_scope_brands=list(names))


def no_access_brand_result():
    return ToolMessage(json.dumps({"error": {"code": "NO_ACCESS_BRAND"}}), tool_call_id="x", name="run_sql")


@pytest.mark.parametrize("st, expected", [
    (state(), ("answered", "OK")),
    (state(decline="off_topic"), ("declined", "OFF_TOPIC")),
    (state(decline="no_access"), ("declined", "NO_ACCESS")),
    (state(decline="report_action"), ("declined", "REPORT_ACTION")),
    (state(decline="PII_IDENTIFIER"), ("declined", "PII_IDENTIFIER")),
    (state(decline="LEAK_BLOCKED"), ("declined", "LEAK_BLOCKED")),
    (state(budget=Budget(bq_down=True, stuck=True)), ("partial", "BQ_DOWN")),
    (state(budget=Budget(stuck=True, tool_calls_used=6)), ("partial", "STUCK")),
    (state(budget=Budget(tool_calls_used=6), removed_numbers=["12"]), ("partial", "BUDGET_SPENT")),
    (state(removed_numbers=["12"], classified=brands_out("Gucci")), ("partial", "GROUNDING_REMOVED")),
    (state(classified=brands_out("Gucci")), ("partial", "NO_ACCESS_BRAND")),
    (state(turn_messages=[no_access_brand_result()]), ("partial", "NO_ACCESS_BRAND")),
])
def test_outcome_and_reason(st, expected):
    assert outcome_for(st, None, cap=6) == expected


@pytest.mark.parametrize("error, reason", [
    (LLMUnavailableError("down"), "LLM_DOWN"),
    (GatewayFatalError("no access"), "BQ_FATAL"),
    (LLMFatalError(401, "bad key"), "LLM_FATAL"),
    (KeyError("bug"), "INTERNAL"),
])
def test_failures(error, reason):
    assert outcome_for(state(), error, cap=6) == ("failed", reason)


def test_exporter_appends_one_line_per_turn(tmp_path):
    exporter = JsonlExporter(tmp_path / "traces")
    exporter.write({"conversation_id": "c", "trace_id": "a"})
    exporter.write({"conversation_id": "c", "trace_id": "b"})
    assert [r["trace_id"] for r in read_traces(exporter.path("c"))] == ["a", "b"]


def test_failed_turn_without_state_still_gets_a_trace_id(tmp_path):
    settings = load_settings()
    settings.tracing.dir = tmp_path
    t = Tracer(settings, CEO, "c")
    graph = SimpleNamespace(get_state=lambda config: SimpleNamespace(values={}))
    with pytest.raises(RuntimeError), t.turn(graph, {}, "hi"):
        raise RuntimeError("boom")
    assert len(t.last["trace_id"]) == 8 and t.last["reason"] == "INTERNAL"
