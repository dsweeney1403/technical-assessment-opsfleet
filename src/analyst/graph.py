"""LangGraph definition v1 (D1.10).

load_context -> classify -> light_reply | agent <-> tools -> output_check -> respond -> save

Forced steps are graph edges, so the model can't skip them. The agent loop is
capped at max_tool_calls_per_turn, repairs included (D1.18, D2.2). Phase 2 adds
the decline paths and the real output check; phase 5 adds the report flow.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from analyst import memory, prompts
from analyst.config import Settings, User
from analyst.llm.client import LLM, CallStats, LLMError
from analyst.safety.classifier import classify
from analyst.sql.gateway import Gateway
from analyst.state import AgentState, Budget, Draft, Intent
from analyst.store.results import ResultStore
from analyst.tools.chart import make_chart_tool
from analyst.tools.run_sql import make_run_sql_tool

LIGHT_INTENTS = {Intent.conversation, Intent.data_question}
BUDGET_SPENT = (
    "The tool budget for this turn is used up. Answer now with the results you have, "
    "and say plainly what you could not get."
)


@dataclass
class Deps:
    settings: Settings
    llm: LLM
    gateway: Gateway
    store: ResultStore
    today: Callable[[], date] = field(default=date.today)


def _charge(budget: Budget, stats: CallStats) -> Budget:
    return budget.model_copy(update={
        "tokens_used": budget.tokens_used + stats.input_tokens + stats.output_tokens,
        "llm_retries": budget.llm_retries + stats.retries,
    })


def build_graph(deps: Deps, checkpointer=None):
    cap = deps.settings.budgets.max_tool_calls_per_turn

    def tools_for(user: User):
        return {
            "run_sql": make_run_sql_tool(deps.gateway, user),
            "make_chart": make_chart_tool(deps.store, deps.settings.charts.dir),
        }

    def load_context(state: AgentState) -> dict:
        return {
            "trace_id": uuid.uuid4().hex[:8],
            "messages": state.get("messages", []),
            "summary": state.get("summary", ""),
            "records": state.get("records", []),
            "turn_messages": [],
            "classified": None,
            "budget": Budget(),
            "draft": Draft(),
        }

    def classify_node(state: AgentState) -> dict:
        stats = CallStats()
        classified = classify(deps.llm, state["messages"], state["user_input"], stats)
        return {"classified": classified, "budget": _charge(state["budget"], stats)}

    def route_after_classify(state: AgentState) -> str:
        return "light_reply" if state["classified"].intent in LIGHT_INTENTS else "agent"

    def light_reply(state: AgentState) -> dict:
        stats = CallStats()
        system = prompts.light_system(state["user"], deps.today(), state["summary"])
        msg = deps.llm.invoke(
            "light", [SystemMessage(system), *state["messages"], HumanMessage(state["user_input"])], stats=stats
        )
        return {"draft": Draft(text=msg.text), "budget": _charge(state["budget"], stats)}

    def agent(state: AgentState) -> dict:
        stats = CallStats()
        system = prompts.agent_system(
            state["user"], deps.today(), memory.records_block(state["records"]), state["summary"]
        )
        messages = [SystemMessage(system), *state["messages"], HumanMessage(state["user_input"]),
                    *state["turn_messages"]]
        budget = state["budget"]
        if budget.tool_calls_used >= cap:
            msg = deps.llm.invoke("agent", [*messages, HumanMessage(BUDGET_SPENT)], stats=stats)
        else:
            msg = deps.llm.invoke("agent", messages, tools=list(tools_for(state["user"]).values()), stats=stats)
        update = {"turn_messages": [*state["turn_messages"], msg], "budget": _charge(budget, stats)}
        if not msg.tool_calls:
            update["draft"] = Draft(text=msg.text)
        return update

    def route_after_agent(state: AgentState) -> str:
        last = state["turn_messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "output_check"

    def tools(state: AgentState) -> dict:
        available = tools_for(state["user"])
        budget = state["budget"]
        used = budget.tool_calls_used
        results = []
        for call in state["turn_messages"][-1].tool_calls:
            if used >= cap:
                content = json.dumps({"error": {"message": "Tool budget used up for this turn.", "hint": BUDGET_SPENT}})
            elif call["name"] not in available:
                used += 1
                content = json.dumps({"error": {"message": f"Unknown tool {call['name']}.", "hint": ""}})
            else:
                used += 1
                content = available[call["name"]].invoke(call["args"])
            results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {
            "turn_messages": [*state["turn_messages"], *results],
            "budget": budget.model_copy(update={"tool_calls_used": used}),
        }

    def output_check(state: AgentState) -> dict:
        # Stub until phase 2: leak scan and number grounding (D1.16).
        return {}

    def respond(state: AgentState) -> dict:
        charts = []
        for m in state["turn_messages"]:
            if isinstance(m, ToolMessage) and m.name == "make_chart":
                path = json.loads(m.content).get("chart_path")
                if path:
                    charts.append(path)
        draft = state["draft"].model_copy(update={"chart_paths": charts})
        messages = [*state["messages"], HumanMessage(state["user_input"]), AIMessage(draft.text)]
        return {"draft": draft, "messages": messages}

    def save(state: AgentState) -> dict:
        sql_by_call = {}
        new_records = []
        for m in state["turn_messages"]:
            if isinstance(m, AIMessage):
                for call in m.tool_calls:
                    if call["name"] == "run_sql":
                        sql_by_call[call["id"]] = call["args"].get("sql", "")
            elif isinstance(m, ToolMessage) and m.name == "run_sql":
                result = json.loads(m.content)
                if "result_id" in result:
                    new_records.append(memory.new_record(state["user_input"], result, sql_by_call.get(m.tool_call_id, "")))
        stats = CallStats()

        def summarise(previous, turns):
            out = deps.llm.invoke("light", memory.summary_messages(previous, turns), stats=stats)
            return out.text

        try:
            messages, summary = memory.trim_history(
                state["messages"], state["summary"], deps.settings.memory.verbatim_turns, summarise
            )
        except LLMError:
            messages, summary = state["messages"], state["summary"]  # try again next turn
        return {
            "records": [*state["records"], *new_records],
            "messages": messages,
            "summary": summary,
            "budget": _charge(state["budget"], stats),
        }

    g = StateGraph(AgentState)
    for name, fn in [("load_context", load_context), ("classify", classify_node), ("light_reply", light_reply),
                     ("agent", agent), ("tools", tools), ("output_check", output_check),
                     ("respond", respond), ("save", save)]:
        g.add_node(name, fn)
    g.add_edge(START, "load_context")
    g.add_edge("load_context", "classify")
    g.add_conditional_edges("classify", route_after_classify, ["light_reply", "agent"])
    g.add_edge("light_reply", "output_check")
    g.add_conditional_edges("agent", route_after_agent, ["tools", "output_check"])
    g.add_edge("tools", "agent")
    g.add_edge("output_check", "respond")
    g.add_edge("respond", "save")
    g.add_edge("save", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())

