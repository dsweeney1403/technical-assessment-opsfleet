"""Forced trio retrieval (D4.4, D4.5, D4.6, D4.8, D4.9)."""

from datetime import date

import numpy as np
from langchain_core.messages import AIMessage
from langgraph.types import Command
from test_graph import BRANDS, CEO, FakeGateway, FakeLLM, sql_call
from test_reports_flow import ScriptLLM, new_section, op, outline, report_draft, reply

from analyst.config import Golden, User, load_settings
from analyst.faults import Faults, embedding_client_factory
from analyst.golden.index import IndexedTrio, TrioIndex
from analyst.golden.retrieve import Retriever, render
from analyst.golden.vector_store import VectorStore
from analyst.graph import Deps, build_graph
from analyst.llm.client import LLMUnavailableError
from analyst.llm.embeddings import Embedder
from analyst.observability.tracer import TraceHandler
from analyst.safety.classifier import ClassifierOutput
from analyst.sql.brands import BrandMatcher
from analyst.store.results import ResultStore
from analyst.store.sqlite_store import SQLiteReportStore

CALVIN = User(id="bm_calvin", name="CK", role="brand_manager", brands=["Calvin Klein"])


def trio(trio_id, vector, brands=()):
    return IndexedTrio(trio_id=trio_id, question=f"Question {trio_id}?", description=f"Approach {trio_id}.",
                       report_stripped=f"Report {trio_id}: up [X].", brands=list(brands), embedding=vector,
                       source_ref=f"trios/{trio_id}.yaml", indexed_at="2026-10-14")


# Query vector is [1, 0, 0], so each trio's score is its first coordinate.
STORE = VectorStore(TrioIndex(model="m", dimensions=3, built_at="x", trios=[
    trio("levis_texas", [1.0, 0.0, 0.0], ["Levi's"]),
    trio("company_texas", [0.9, 0.43589, 0.0]),
    trio("ck_margin", [0.8, 0.6, 0.0], ["Calvin Klein"]),
    trio("mixed", [0.75, 0.66144, 0.0], ["Calvin Klein", "Levi's"]),
    trio("company_weak", [0.3, 0.95394, 0.0]),
]))


class QueryEmbedder:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def embed(self, texts, task, stats=None):
        self.calls.append((list(texts), task))
        if self.error:
            raise self.error
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


def retriever(threshold=0.5, k=3, embedder=None, store=STORE):
    return Retriever(store, embedder or QueryEmbedder(), Golden(threshold=threshold, k=k, overfetch=20))


def ids(result):
    return [t.trio_id for t in result.trios]


def test_a_brand_manager_never_gets_another_brands_trio():
    result = retriever().retrieve("Why is Texas underspending?", CALVIN)
    assert ids(result) == ["company_texas", "ck_margin"]  # Levi's and mixed trios dropped, though they score high
    assert result.out_of_scope == 2
    assert all(set(t.brands) <= {"Calvin Klein"} for t in result.trios)


def test_the_ceo_keeps_every_trio_up_to_k():
    assert ids(retriever(k=3).retrieve("q", CEO)) == ["levis_texas", "company_texas", "ck_margin"]


def test_below_threshold_is_dropped_and_zero_trios_is_valid():
    result = retriever(threshold=0.85).retrieve("q", CALVIN)
    assert ids(result) == ["company_texas"]
    assert [s.trio_id for s in result.below_threshold] == ["ck_margin", "company_weak"]
    empty = retriever(threshold=0.95).retrieve("q", CALVIN)
    assert (empty.status, empty.trios) == ("ok", [])


def test_embedding_outage_gives_no_trios_and_says_so():
    result = retriever(embedder=QueryEmbedder(LLMUnavailableError("503"))).retrieve("q", CEO)
    assert (result.status, result.trios) == ("trios_unavailable", [])


def test_simulated_outage_through_the_real_embedder():
    embedder = Embedder(load_settings(), "key", client_factory=embedding_client_factory(None, Faults(embedding_down=True)),
                        sleep=lambda s: None)
    assert retriever(embedder=embedder).retrieve("q", CEO).status == "trios_unavailable"


def test_unset_threshold_or_missing_index_retrieves_nothing():
    embedder = QueryEmbedder()
    assert retriever(threshold=None, embedder=embedder).retrieve("q", CEO).status == "threshold_unset"
    assert Retriever(None, embedder, Golden(threshold=0.5)).retrieve("q", CEO).status == "no_index"
    assert embedder.calls == []


def test_examples_block_holds_no_scores_or_tags():
    text = render(retriever().retrieve("q", CEO).trios)
    assert "Question levis_texas?" in text and "Approach levis_texas." in text and "Report levis_texas: up [X]." in text
    assert "0.9" not in text and "Levi's" not in text


# In the graph


def chat_graph(llm, embedder):
    settings = load_settings()
    store = ResultStore()
    deps = Deps(settings, llm, FakeGateway(store), store, today=lambda: date(2026, 10, 14),
                brands=BrandMatcher(BRANDS), retriever=retriever(embedder=embedder))
    return build_graph(deps)


def test_analysis_turn_retrieves_and_the_agent_sees_the_examples():
    embedder = QueryEmbedder()
    llm = FakeLLM(["analysis"], [AIMessage("Assumptions: year to date. Texas trails.")])
    handler = TraceHandler()
    graph = chat_graph(llm, embedder)
    graph.invoke({"user_input": "Why is Texas underspending?", "user": CALVIN},
                 {"configurable": {"thread_id": "t"}, "callbacks": [handler]})
    assert embedder.calls == [(["Why is Texas underspending?"], "query")]
    system = llm.agent_messages[0][0].content
    assert "Worked examples" in system and "Question company_texas?" in system
    assert "levis_texas" not in system
    span = next(s for s in handler.spans if s.name == "retrieval")
    assert [t["trio_id"] for t in span.attributes["trios"]] == ["company_texas", "ck_margin"]
    assert span.attributes["out_of_scope"] == 2 and span.attributes["status"] == "ok"
    assert all(isinstance(t["score"], float) for t in span.attributes["trios"])


def test_light_path_never_retrieves():
    embedder = QueryEmbedder()
    chat_graph(FakeLLM(["conversation", "data_question"]), embedder).invoke(
        {"user_input": "Thanks!", "user": CEO}, {"configurable": {"thread_id": "t"}})
    assert embedder.calls == []


def test_embedding_outage_still_answers_from_the_glossary():
    embedder = QueryEmbedder(LLMUnavailableError("503"))
    llm = FakeLLM(["analysis"], [AIMessage("Assumptions: year to date. Fine.")])
    handler = TraceHandler()
    out = chat_graph(llm, embedder).invoke({"user_input": "How are we doing?", "user": CEO},
                                           {"configurable": {"thread_id": "t"}, "callbacks": [handler]})
    assert out["draft"].text.startswith("Assumptions")
    assert "Worked examples" not in llm.agent_messages[0][0].content
    assert "Glossary (company definitions" in llm.agent_messages[0][0].content
    assert next(s for s in handler.spans if s.name == "retrieval").attributes["status"] == "trios_unavailable"


def test_report_retrieves_once_and_sections_get_no_examples(tmp_path):
    embedder = QueryEmbedder()
    llm = ScriptLLM(
        ClassifierOutput=[op("create"), ClassifierOutput(intent="report_action", report_op="redo", reason="t")],
        Outline=[outline(new_section(), new_section("Returns")), outline(new_section())],
        ReplyOutput=[reply("yes"), reply("yes")],
        agent=[sql_call(1), AIMessage("Notes (r1)."), sql_call(2), AIMessage("Notes (r2)."),
               sql_call(3), AIMessage("Notes (r3).")],
        ReportDraft=[report_draft(), report_draft(title="Shorter")],
    )
    settings = load_settings()
    store = ResultStore()
    graph = build_graph(Deps(settings, llm, FakeGateway(store), store, today=lambda: date(2026, 10, 14),
                             brands=BrandMatcher(BRANDS), reports=SQLiteReportStore(tmp_path / "r.db"),
                             retriever=retriever(embedder=embedder)))
    cfg = {"configurable": {"thread_id": "t"}}
    seen_sections = []
    real_invoke = llm.invoke

    def invoke(role, messages, tools=None, stats=None):
        if role == "agent":
            seen_sections.append(messages[0].content)
        return real_invoke(role, messages, tools, stats)

    llm.invoke = invoke
    for payload in ({"user_input": "Create a Q1 report", "user": CEO, "conversation_id": "c"}, Command(resume="yes"),
                    {"user_input": "Make it about revenue only", "user": CEO, "conversation_id": "c"},
                    Command(resume="yes")):
        graph.invoke(payload, cfg)
    assert len(embedder.calls) == 1  # once per report: not per section, not on the redo (D4.5)
    planner_inputs = [m[0].content for m in llm.messages["Outline"]]
    writer_inputs = [m[0].content for m in llm.messages["ReportDraft"]]
    assert all("Question levis_texas?" in t for t in planner_inputs + writer_inputs)  # the redo reuses them
    assert seen_sections and all("Worked examples" not in s for s in seen_sections)
    assert graph.get_state(cfg).values["report"].trios[0].trio_id == "levis_texas"
