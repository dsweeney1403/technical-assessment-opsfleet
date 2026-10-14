"""LangGraph definition (D1.10, Group 1 Interfaces).

load_context -> classify -> light_reply | decline | agent <-> tools
             -> output_check (-> agent once, on ungrounded figures) -> respond -> save

Forced steps are graph edges, so the model can't skip them. The agent loop is
capped at max_tool_calls_per_turn, repairs included (D1.18, D2.2). Declines are
written by code (D1.8, D2.1); a gateway stop code ends the loop in the decline
node. Phase 5 adds the report flow.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from analyst import memory, prompts
from analyst.config import Settings, User
from analyst.llm.client import LLM, CallStats, LLMError
from analyst.safety import output_check as oc
from analyst.safety.classifier import classify
from analyst.safety.declines import DECLINES, SECURITY_REASONS, no_access, scope_note
from analyst.sql.brands import BrandMatcher
from analyst.sql.diagnosis import KnownValues
from analyst.sql.gateway import Gateway, GatewayUnavailableError, SafetyStop
from analyst.state import AgentState, Budget, Draft, Intent, SecurityEvent
from analyst.store.results import ResultStore
from analyst.tools.chart import make_chart_tool
from analyst.tools.compute import make_compute_tool
from analyst.tools.run_sql import make_run_sql_tool

LIGHT_INTENTS = {Intent.conversation, Intent.data_question}
DECLINE_INTENTS = {Intent.off_topic, Intent.pii_individuals, Intent.injection, Intent.destructive,
                   Intent.report_action}  # report_action until phase 5
UNSURE_GOES_TO_AGENT = {Intent.off_topic, Intent.pii_individuals}  # D1.8: fail open on intent
RESULT_TOOLS = {"run_sql", "compute"}
# State types the checkpointer may restore. Anything else is refused.
CHECKPOINT_TYPES = [
    ("analyst.config", "User"),
    ("analyst.state", "Record"),
    ("analyst.state", "Intent"),
    ("analyst.state", "Classified"),
    ("analyst.state", "Budget"),
    ("analyst.state", "Draft"),
    ("analyst.state", "SecurityEvent"),
]
BUDGET_SPENT = (
    "The tool budget for this turn is used up. Answer now with the results you have, "
    "and say plainly what you could not get."
)
STUCK = (
    "The same query error came back twice in a row, so stop retrying it. Answer now with the results you have, "
    "and say plainly what you could not get."
)
BQ_DOWN = json.dumps({"error": {
    "code": "BQ_DOWN",
    "message": "BigQuery is unavailable right now.",
    "hint": "Run no more queries this turn. compute and make_chart still work on stored results. "
            "Answer with the results you have and say plainly what you couldn't fetch. The user can try again later.",
}})


@dataclass
class Deps:
    settings: Settings
    llm: LLM
    gateway: Gateway
    store: ResultStore
    today: Callable[[], date] = field(default=date.today)
    brands: BrandMatcher = field(default_factory=lambda: BrandMatcher([]))
    known: KnownValues | None = None  # loaded once per session (D2.5)


def _charge(budget: Budget, stats: CallStats) -> Budget:
    return budget.model_copy(update={
        "tokens_used": budget.tokens_used + stats.input_tokens + stats.output_tokens,
        "llm_retries": budget.llm_retries + stats.retries,
        "models_used": [*budget.models_used, *stats.models_used],
        "fallback_used": budget.fallback_used or stats.fallback_used,
        "breaker_opens": budget.breaker_opens + stats.breaker_opens,
    })


def _user_texts(state: AgentState) -> list[str]:
    """What the user wrote, this turn and earlier. Brands and numbers in it are theirs to repeat."""
    return [state["user_input"], *(m.text for m in state["messages"] if isinstance(m, HumanMessage))]


def build_graph(deps: Deps, checkpointer=None):
    cap = deps.settings.budgets.max_tool_calls_per_turn
    scanner = oc.LeakScanner(deps.brands.all_brands)

    def tools_for(user: User):
        return {
            "run_sql": make_run_sql_tool(deps.gateway, user),
            "compute": make_compute_tool(deps.store),
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
            "decline": None,
            "security_events": [],
            "grounding_retried": False,
            "removed_numbers": [],
        }

    def classify_node(state: AgentState) -> dict:
        stats = CallStats()
        user = state["user"]
        classified = classify(deps.llm, state["messages"], state["user_input"], stats)
        out_of_scope = deps.brands.out_of_scope_mentions(classified.brands_mentioned, user)
        classified = classified.model_copy(update={"out_of_scope_brands": out_of_scope})

        decline = None
        intent = classified.intent
        if intent in DECLINE_INTENTS and not (intent in UNSURE_GOES_TO_AGENT and not classified.confident):
            decline = intent.value
        elif intent not in LIGHT_INTENTS and out_of_scope and len(out_of_scope) == len(classified.brands_mentioned):
            # Every brand asked about is outside the user's access (D1.8 row 6).
            decline = "no_access"
        return {"classified": classified, "decline": decline, "budget": _charge(state["budget"], stats)}

    def route_after_classify(state: AgentState) -> str:
        if state["decline"]:
            return "decline"
        return "light_reply" if state["classified"].intent in LIGHT_INTENTS else "agent"

    def decline(state: AgentState) -> dict:
        reason = state["decline"]
        if reason == "no_access":
            text = no_access(state["classified"].out_of_scope_brands, state["user"])
        else:
            text = DECLINES[reason]
        events = list(state["security_events"])
        if reason in SECURITY_REASONS and not any(e.code == reason for e in events):
            events.append(SecurityEvent(code=reason, source="classifier", detail=state["classified"].reason))
        return {"draft": Draft(text=text, source="code"), "security_events": events}

    def light_reply(state: AgentState) -> dict:
        stats = CallStats()
        system = prompts.light_system(state["user"], deps.today(), state["summary"], deps.known)
        msg = deps.llm.invoke(
            "light", [SystemMessage(system), *state["messages"], HumanMessage(state["user_input"])], stats=stats
        )
        return {"draft": Draft(text=msg.text, source="light"), "budget": _charge(state["budget"], stats)}

    def agent(state: AgentState) -> dict:
        stats = CallStats()
        classified = state["classified"]
        system = prompts.agent_system(
            state["user"], deps.today(), memory.records_block(state["records"]), state["summary"],
            classified.out_of_scope_brands if classified else [], deps.known,
        )
        messages = [SystemMessage(system), *state["messages"], HumanMessage(state["user_input"]),
                    *state["turn_messages"]]
        budget = state["budget"]
        if budget.tool_calls_used >= cap or budget.stuck:
            # Code decides when the loop ends: no tools bound, so the model must answer (D2.2).
            stop = STUCK if budget.stuck else BUDGET_SPENT
            msg = deps.llm.invoke("agent", [*messages, HumanMessage(stop)], stats=stats)
        else:
            msg = deps.llm.invoke("agent", messages, tools=list(tools_for(state["user"]).values()), stats=stats)
        update = {"turn_messages": [*state["turn_messages"], msg], "budget": _charge(budget, stats)}
        if not msg.tool_calls:
            update["draft"] = Draft(text=msg.text, source="agent")
        return update

    def route_after_agent(state: AgentState) -> str:
        last = state["turn_messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "output_check"

    def tools(state: AgentState) -> dict:
        available = tools_for(state["user"])
        budget = state["budget"]
        used = budget.tool_calls_used
        last_code, stuck, bq_down = budget.last_error_code, budget.stuck, budget.bq_down
        results = []
        update: dict = {}
        for call in state["turn_messages"][-1].tool_calls:
            if used >= cap:
                content = json.dumps({"error": {"message": "Tool budget used up for this turn.", "hint": BUDGET_SPENT}})
            elif call["name"] not in available:
                used += 1
                content = json.dumps({"error": {"message": f"Unknown tool {call['name']}.", "hint": ""}})
            elif call["name"] == "run_sql" and bq_down:
                used += 1
                content = BQ_DOWN  # refused in code: no new queries once BigQuery is down (D2.13)
            else:
                used += 1
                try:
                    content = available[call["name"]].invoke(call["args"])
                except GatewayUnavailableError:
                    bq_down = True
                    content = BQ_DOWN
                except SafetyStop as stop:
                    # A stop code ends the turn: neutral decline plus a security event (D2.1).
                    code = stop.rejection.code
                    event = SecurityEvent(code=code, source="gateway", detail=" ".join(stop.sql.split())[:1000])
                    update = {"decline": code, "security_events": [*state["security_events"], event]}
                    results.append(ToolMessage(content=json.dumps({"error": {"code": code}}),
                                               tool_call_id=call["id"], name=call["name"]))
                    break
            if call["name"] == "run_sql" and content != BQ_DOWN:
                # Stuck rule (D2.2): the same error code twice in a row on run_sql.
                code = json.loads(content).get("error", {}).get("code")
                stuck = stuck or (code is not None and code == last_code)
                last_code = code
            results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {
            **update,
            "turn_messages": [*state["turn_messages"], *results],
            "budget": budget.model_copy(update={"tool_calls_used": used, "last_error_code": last_code,
                                                "stuck": stuck, "bq_down": bq_down}),
        }

    def route_after_tools(state: AgentState) -> str:
        return "decline" if state["decline"] else "agent"

    def output_check(state: AgentState) -> dict:
        """Leak scan, then number grounding (D1.16). Code-written text is only leak-scanned."""
        draft, user = state["draft"], state["user"]
        frames = deps.store.frames()
        user_texts = _user_texts(state)
        classified = state["classified"]
        allowed = "\n".join([*user_texts, *(classified.out_of_scope_brands if classified else [])])
        leaks = scanner.scan(draft.text, user, allowed, oc.result_text(frames))
        if leaks:
            detail = "; ".join(f"{leak.kind}: {leak.match}" for leak in leaks)
            event = SecurityEvent(code="LEAK_BLOCKED", source="output_check", detail=detail[:1000])
            return {"draft": Draft(text=DECLINES["LEAK_BLOCKED"], source="code"), "decline": "LEAK_BLOCKED",
                    "security_events": [*state["security_events"], event]}
        if draft.source == "code":
            return {}
        missing = oc.ungrounded(draft.text, oc.ground_values(frames, user_texts))
        if not missing:
            return {}
        if draft.source == "agent" and not state["grounding_retried"]:
            # One retry: the agent is told which figures had no source.
            return {"grounding_retried": True,
                    "turn_messages": [*state["turn_messages"], HumanMessage(oc.retry_instruction(missing))]}
        return {"draft": draft.model_copy(update={"text": oc.strip_figures(draft.text, missing)}),
                "removed_numbers": [f.text.strip() for f in missing]}

    def route_after_output_check(state: AgentState) -> str:
        last = state["turn_messages"][-1] if state["turn_messages"] else None
        return "agent" if isinstance(last, HumanMessage) else "respond"

    def respond(state: AgentState) -> dict:
        draft = state["draft"]
        charts = []
        if draft.source != "code":
            for m in state["turn_messages"]:
                if isinstance(m, ToolMessage) and m.name == "make_chart":
                    path = json.loads(m.content).get("chart_path")
                    if path:
                        charts.append(path)
        text = draft.text
        if draft.source == "agent":
            classified = state["classified"]
            note = scope_note(state["user"], classified.out_of_scope_brands if classified else [])
            if note:
                text = f"{text}\n\n{note}"
        draft = draft.model_copy(update={"text": text, "chart_paths": charts})
        messages = [*state["messages"], HumanMessage(state["user_input"]), AIMessage(draft.text)]
        return {"draft": draft, "messages": messages}

    def save(state: AgentState) -> dict:
        args_by_call = {}
        new_records = []
        for m in state["turn_messages"]:
            if isinstance(m, AIMessage):
                for call in m.tool_calls:
                    args_by_call[call["id"]] = call["args"]
            elif isinstance(m, ToolMessage) and m.name in RESULT_TOOLS:
                result = json.loads(m.content)
                if "result_id" in result:
                    stored = deps.store.get(result["result_id"])
                    sql = args_by_call.get(m.tool_call_id, {}).get("sql", "") if m.name == "run_sql" else stored.sql
                    new_records.append(memory.new_record(state["user_input"], result, sql, stored.meta))
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
                     ("decline", decline), ("agent", agent), ("tools", tools), ("output_check", output_check),
                     ("respond", respond), ("save", save)]:
        g.add_node(name, fn)
    g.add_edge(START, "load_context")
    g.add_edge("load_context", "classify")
    g.add_conditional_edges("classify", route_after_classify, ["light_reply", "decline", "agent"])
    g.add_edge("light_reply", "output_check")
    g.add_edge("decline", "output_check")
    g.add_conditional_edges("agent", route_after_agent, ["tools", "output_check"])
    g.add_conditional_edges("tools", route_after_tools, ["agent", "decline"])
    g.add_conditional_edges("output_check", route_after_output_check, ["agent", "respond"])
    g.add_edge("respond", "save")
    g.add_edge("save", END)
    saver = checkpointer or InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))
    return g.compile(checkpointer=saver)
