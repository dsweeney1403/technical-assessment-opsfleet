"""LangGraph definition (D1.10, Group 1 and Group 3 Interfaces).

load_context -> classify -> light_reply | decline | agent <-> tools | report flow | save_report | download
             -> output_check (-> agent or write once, on ungrounded figures) -> respond -> save

Report flow (D3.1, D3.2): plan -> confirm_outline (interrupt, only if a section
needs new queries; feedback goes back to plan) -> section, one per pass, each
new-queries section in its own bounded loop -> write -> output_check.

Delete flow (D3.26): the agent's propose_delete is resolved in code by the
tools node -> confirm_delete (interrupt, reply classified, narrow re-shows the
list) -> delete_reports (code: exactly the ids shown, soft, audited).

Forced steps are graph edges, so the model can't skip them. Agent loops are
capped at max_tool_calls_per_turn, repairs included (D1.18, D2.2), and each
report section gets its own cap (D3.1). Declines are written by code (D1.8,
D2.1); a gateway stop code ends the loop in the decline node.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from analyst import memory, preferences, prompts
from analyst.persona import load_persona
from analyst.config import AnalysisDefaults, Glossary, Settings, User, load_analysis_defaults, load_glossary
from analyst.llm.client import LLM, CallStats, LLMError, LLMOutputError
from analyst.reports import delete as deletion
from analyst.reports import flow
from analyst.reports.mentions import MentionScanner
from analyst.reports.replies import classify_reply
from analyst.safety import output_check as oc
from analyst.safety.classifier import classify
from analyst.safety.declines import DECLINES, SECURITY_REASONS, no_access, scope_note
from analyst.sql.brands import BrandMatcher
from analyst.sql.diagnosis import KnownValues
from analyst.sql.gateway import Gateway, GatewayUnavailableError, SafetyStop
from analyst.state import (
    AgentState,
    Budget,
    CurrentReport,
    Draft,
    Intent,
    Outline,
    PendingDelete,
    PersonaInUse,
    ReportDraft,
    SectionResult,
    SecurityEvent,
)
from analyst.store.base import NewVersion, ProfileStore, ReportStore, SavedFinding
from analyst.store.results import ResultStore
from analyst.store.sqlite_store import SQLiteProfileStore, SQLiteReportStore
from analyst.tools.chart import make_chart_tool
from analyst.tools.compute import make_compute_tool
from analyst.tools.reports import make_report_tools
from analyst.tools.run_sql import make_run_sql_tool

LIGHT_INTENTS = {Intent.conversation, Intent.data_question}
DECLINE_INTENTS = {Intent.off_topic, Intent.pii_individuals, Intent.injection, Intent.destructive}
UNSURE_GOES_TO_AGENT = {Intent.off_topic, Intent.pii_individuals}  # D1.8: fail open on intent
RESULT_TOOLS = {"run_sql", "compute"}
# State types the checkpointer may restore. Anything else is refused.
CHECKPOINT_TYPES = [
    ("analyst.config", "User"),
    *[("analyst.state", name) for name in (
        "Record", "Intent", "Classified", "Budget", "Draft", "SecurityEvent", "Section", "Outline", "SectionResult",
        "Finding", "Target", "Action", "ReportDraft", "CurrentReport", "ShownReport", "PendingDelete", "PersonaInUse", "StatedPreference")],
    *[("analyst.store.base", name) for name in ("UserProfile", "Preference")],
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
NO_REPORT = "There's no report in this session yet. Ask me to create one, or open a saved one with /reports."
EARLIER_SESSION = ("That report was saved in an earlier session, so its results aren't loaded. "
                   "I can create a new report on the same topic.")
OUTLINE_CANCELLED = "Cancelled. Nothing was run."
PLAN_FAILED = "I couldn't plan that report this time. Try rephrasing the request."
WRITE_FAILED = "I ran the sections but couldn't write the report this time. Say \"make it again\" to retry the writing."


@dataclass
class Deps:
    settings: Settings
    llm: LLM
    gateway: Gateway
    store: ResultStore
    today: Callable[[], date] = field(default=date.today)
    brands: BrandMatcher = field(default_factory=lambda: BrandMatcher([]))
    known: KnownValues | None = None  # loaded once per session (D2.5)
    reports: ReportStore | None = None  # in-memory SQLite when not given (tests)
    mentions: MentionScanner | None = None
    glossary: Glossary | None = field(default_factory=load_glossary)  # D4.2
    defaults: AnalysisDefaults | None = field(default_factory=load_analysis_defaults)  # D4.15
    persona_dir: Path | None = None  # settings.persona.dir when not given
    profiles: ProfileStore | None = None  # in-memory SQLite when not given (tests)


def _charge(budget: Budget, stats: CallStats) -> Budget:
    return budget.model_copy(update={
        "tokens_used": budget.tokens_used + stats.input_tokens + stats.output_tokens,
        "llm_retries": budget.llm_retries + stats.retries,
        "models_used": [*budget.models_used, *stats.models_used],
        "fallback_used": budget.fallback_used or stats.fallback_used,
        "breaker_opens": budget.breaker_opens + stats.breaker_opens,
    })


def _merge_section(budget: Budget, section: Budget) -> Budget:
    """A section's own budget folds into the turn's, apart from its tool-call cap (D3.1)."""
    return budget.model_copy(update={
        "tokens_used": budget.tokens_used + section.tokens_used,
        "llm_retries": budget.llm_retries + section.llm_retries,
        "models_used": [*budget.models_used, *section.models_used],
        "fallback_used": budget.fallback_used or section.fallback_used,
        "breaker_opens": budget.breaker_opens + section.breaker_opens,
        "stuck": budget.stuck or section.stuck,
        "bq_down": budget.bq_down or section.bq_down,
        "cap_hit": budget.cap_hit or section.cap_hit,
        "section_tool_calls": budget.section_tool_calls + section.tool_calls_used,
    })


def _user_texts(state: AgentState) -> list[str]:
    """What the user wrote, this turn and earlier. Brands and numbers in it are theirs to repeat."""
    return [state["user_input"], *(m.text for m in state["messages"] if isinstance(m, HumanMessage)),
            *state.get("outline_feedback", [])]


def _exempt(figure: oc.Figure, text: str, exempt: list[str]) -> bool:
    for phrase in exempt:
        for m in re.finditer(re.escape(phrase), text):
            if m.start() <= figure.start and figure.end <= m.end():
                return True
    return False


def _tool_results(messages: list[BaseMessage]) -> tuple[list[dict], list[str]]:
    """Stored results and chart paths produced by a loop's tool calls."""
    args_by_call, results, charts = {}, [], []
    for m in messages:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                args_by_call[call["id"]] = call["args"]
        elif isinstance(m, ToolMessage):
            try:
                content = json.loads(m.content)
            except (TypeError, ValueError):
                continue
            if m.name in RESULT_TOOLS and "result_id" in content:
                results.append({"tool": m.name, "result": content, "args": args_by_call.get(m.tool_call_id, {})})
            elif m.name == "make_chart" and content.get("chart_path"):
                charts.append(content["chart_path"])
    return results, charts


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "report"


def build_graph(deps: Deps, checkpointer=None):
    cap = deps.settings.budgets.max_tool_calls_per_turn
    max_sections = deps.settings.reports.max_sections
    scanner = oc.LeakScanner(deps.brands.all_brands)
    reports = deps.reports or SQLiteReportStore(":memory:")
    mentions = deps.mentions or MentionScanner(deps.brands.all_brands, deps.known)
    persona_dir = deps.persona_dir or deps.settings.persona.dir
    profiles = deps.profiles or SQLiteProfileStore(":memory:")
    infer_after = deps.settings.profile.inferred_preference_after

    def is_report_turn(state: AgentState) -> bool:
        classified = state.get("classified")
        return classified is not None and classified.intent == Intent.report_action

    def tools_for(state: AgentState) -> dict:
        user = state["user"]
        tools = {
            "run_sql": make_run_sql_tool(deps.gateway, user),
            "compute": make_compute_tool(deps.store),
            "make_chart": make_chart_tool(deps.store, deps.settings.charts.dir),
        }
        if is_report_turn(state):
            tools.update(make_report_tools(reports, user))  # read-only tools plus propose_delete (D3.26)
        return tools

    def layers_for(state: AgentState) -> prompts.Layers:
        """The prompt layers for this turn (D4.24)."""
        persona, profile = state.get("persona"), state.get("profile")
        return prompts.Layers(glossary=deps.glossary, defaults=deps.defaults,
                              preferences=preferences.render(profile), own_defaults=preferences.own_defaults(profile),
                              persona=persona.text if persona else "", persona_version=persona.version if persona else "")

    def chat_messages(state: AgentState) -> list[BaseMessage]:
        classified = state["classified"]
        system = prompts.agent_system(
            state["user"], deps.today(), memory.records_block(state["records"]), state["summary"],
            classified.out_of_scope_brands if classified else [], deps.known, report_tools=is_report_turn(state),
            layers=layers_for(state),
        )
        return [SystemMessage(system), *state["messages"], HumanMessage(state["user_input"])]

    def section_messages(state: AgentState) -> list[BaseMessage]:
        return state["task_messages"]

    # The bounded agent loop, shared by the chat and by report sections (D1.18, D2.2, D3.1)

    def make_agent(messages_for: Callable[[AgentState], list[BaseMessage]]):
        def agent(state: AgentState) -> dict:
            stats = CallStats()
            messages = [*messages_for(state), *state["turn_messages"]]
            budget = state["budget"]
            if budget.tool_calls_used >= cap or budget.stuck:
                # Code decides when the loop ends: no tools bound, so the model must answer (D2.2).
                stop = STUCK if budget.stuck else BUDGET_SPENT
                msg = deps.llm.invoke("agent", [*messages, HumanMessage(stop)], stats=stats)
                budget = budget.model_copy(update={"cap_hit": budget.cap_hit or not budget.stuck})
            else:
                msg = deps.llm.invoke("agent", messages, tools=list(tools_for(state).values()), stats=stats)
            update = {"turn_messages": [*state["turn_messages"], msg], "budget": _charge(budget, stats)}
            if not msg.tool_calls:
                update["draft"] = Draft(text=msg.text, source="agent")
            return update

        return agent

    def route_after_agent(state: AgentState) -> str:
        last = state["turn_messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "output_check"

    def tools(state: AgentState) -> dict:
        available = tools_for(state)
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
            elif call["name"] == "propose_delete":
                used += 1
                content, proposal = propose_delete(state, call["args"])
                update.update(proposal)
                if proposal.get("pending_delete") is not None:
                    results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
                    break  # the confirmation step takes over (D3.26)
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
                if call["name"] == "get_report" and "error" not in json.loads(content):
                    update["opened_report"] = json.loads(content)["report_id"]
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
        if state.get("decline"):
            return "decline"
        if state.get("pending_delete") is not None:
            return "confirm_delete"
        return "agent"

    # Report sections run the same loop in a subgraph with a fresh context (D3.1)

    sg = StateGraph(AgentState)
    sg.add_node("agent", make_agent(section_messages))
    sg.add_node("tools", tools)
    sg.add_edge(START, "agent")
    sg.add_conditional_edges("agent", lambda s: route_after_agent(s) if route_after_agent(s) == "tools" else END,
                             ["tools", END])
    sg.add_conditional_edges("tools", lambda s: END if s.get("decline") else "agent", ["agent", END])
    section_loop = sg.compile(checkpointer=False)

    # Chat nodes

    def load_context(state: AgentState) -> dict:
        p = load_persona(persona_dir)  # every turn, so an edit applies to the next answer (D4.22)
        return {
            "persona": PersonaInUse(version=p.version, text=p.text, fallback_from=p.fallback_from, error=p.error),
            "profile": profiles.get(state["user"]),
            "preference_note": None,
            "definition_request": False,
            "trace_id": uuid.uuid4().hex[:8],
            "continues": None,
            "messages": state.get("messages", []),
            "summary": state.get("summary", ""),
            "records": state.get("records", []),
            "report": state.get("report"),
            "turn_messages": [],
            "classified": None,
            "budget": Budget(),
            "draft": Draft(),
            "decline": None,
            "security_events": [],
            "grounding_retried": False,
            "removed_numbers": [],
            "outline": None,
            "outline_feedback": [],
            "sections_done": [],
            "write_feedback": None,
            "redo_of": None,
            "opened_report": None,
            "pending_delete": None,
            "decision": None,
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
        elif (intent not in LIGHT_INTENTS and intent != Intent.report_action and out_of_scope
              and len(out_of_scope) == len(classified.brands_mentioned)):
            # Every brand asked about is outside the user's access (D1.8 row 6).
            decline = "no_access"
        return {"classified": classified, "decline": decline, "budget": _charge(state["budget"], stats)}

    def route_after_classify(state: AgentState) -> str:
        if state["decline"]:
            return "decline"
        classified = state["classified"]
        if classified.stated_preference is not None or classified.format_request != "none":
            return "preferences"
        return route_intent(state)

    def preferences_node(state: AgentState) -> dict:
        """Code decides what is saved (D4.15, D4.16). The classifier only reported what was asked."""
        classified, user = state["classified"], state["user"]
        pref = preferences.valid(classified.stated_preference) if classified.stated_preference else None
        update: dict = {}
        note = None
        if pref is not None and pref.kind == "safety_control":
            return {"decline": "safety_control"}  # never a preference, never saved
        if pref is not None and pref.kind == "definition":
            update["definition_request"] = True  # answered from the glossary, not saved (D4.2)
        elif pref is not None and pref.kind == "presentation":
            profiles.set_presentation(user, pref.field, pref.value, "stated")
            note = preferences.acknowledgement(pref)
        elif pref is not None:
            profiles.set_default(user, pref.field, pref.value)
            note = preferences.acknowledgement(pref)
        if classified.format_request in preferences.FORMAT_REQUESTS:
            field_, value = preferences.FORMAT_REQUESTS[classified.format_request]
            current = profiles.get(user).presentation.get(field_)
            count = profiles.count_format(user, classified.format_request)
            if count >= infer_after and (current is None or (current.source == "inferred" and current.value != value)):
                profiles.set_presentation(user, field_, value, "inferred")  # silent, visible in /prefs (D4.16)
        update["profile"] = profiles.get(user)
        if note and classified.intent in LIGHT_INTENTS | {Intent.unavailable}:
            update["draft"] = Draft(text=note, source="code")  # nothing else to answer
        elif note:
            update["preference_note"] = note
        return update

    def route_after_preferences(state: AgentState) -> str:
        if state.get("decline"):
            return "decline"
        if state["draft"].text:
            return "output_check"
        if state.get("definition_request"):
            return "light_reply"
        return route_intent(state)

    def route_intent(state: AgentState) -> str:
        classified = state["classified"]
        if classified.intent in LIGHT_INTENTS:
            return "light_reply"
        if classified.intent == Intent.report_action:
            op = classified.report_op
            if op in ("create", "redo"):
                return "plan"
            if op == "save":
                return "save_report"
            if op == "download":
                return "download_report"
        return "agent"

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
        layers = layers_for(state)
        if state["classified"].intent != Intent.data_question and not state.get("definition_request"):
            layers = replace(layers, glossary=None)  # the glossary is for analysis and data questions (D4.2)
        system = prompts.light_system(state["user"], deps.today(), state["summary"], deps.known, layers)
        msg = deps.llm.invoke(
            "light", [SystemMessage(system), *state["messages"], HumanMessage(state["user_input"])], stats=stats
        )
        return {"draft": Draft(text=msg.text, source="light"), "budget": _charge(state["budget"], stats)}

    # Report flow (D3.1 to D3.9)

    def plan(state: AgentState) -> dict:
        redo = state["classified"].report_op == "redo"
        previous = state.get("report") if redo else None
        if redo and previous is None:
            return {"draft": Draft(text=NO_REPORT, source="code")}
        if previous is not None and not previous.from_this_session:
            return {"draft": Draft(text=EARLIER_SESSION, source="code")}
        stats = CallStats()
        messages = flow.plan_messages(
            state["user"], deps.today(), state["user_input"], state["messages"], state["summary"],
            memory.records_block(state["records"]), max_sections, deps.known, previous, state["outline_feedback"],
            layers_for(state))
        try:
            outline = deps.llm.structured("report", Outline, messages, stats)
        except LLMOutputError:
            return {"draft": Draft(text=PLAN_FAILED, source="code"), "budget": _charge(state["budget"], stats)}
        outline = flow.checked_outline(outline, deps.store, _user_texts(state), max_sections)
        return {"outline": outline, "redo_of": previous, "sections_done": [], "decision": None,
                "budget": _charge(state["budget"], stats)}

    def route_after_plan(state: AgentState) -> str:
        if state["draft"].text:
            return "output_check"
        return "confirm_outline" if flow.needs_confirmation(state["outline"]) else "section"

    def confirm_outline(state: AgentState) -> dict:
        """D3.2: the full outline every round; yes runs it, cancel stops, anything else is feedback."""
        shown = flow.outline_text(state["outline"], confirm=True)
        reply = interrupt({"kind": "outline", "text": shown})
        stats = CallStats()
        decision = classify_reply(deps.llm, shown, reply, stats)
        update = {"budget": _charge(state["budget"], stats), "decision": decision}
        if decision == "cancel":
            update["draft"] = Draft(text=OUTLINE_CANCELLED, source="code")
        elif decision != "yes":
            update["outline_feedback"] = [*state["outline_feedback"], reply]
        return update

    def route_after_confirm_outline(state: AgentState) -> str:
        decision = state["decision"]
        if decision == "cancel":
            return "output_check"
        return "section" if decision == "yes" else "plan"

    def section(state: AgentState) -> dict:
        """One section per pass, in order (D3.5), so each shows in the status line and the checkpoint."""
        outline, done = state["outline"], state["sections_done"]
        s = outline.sections[len(done)]
        if s.source == "from_records":
            result = SectionResult(title=s.title, notes=s.brief, result_ids=s.result_ids)
            return {"sections_done": [*done, result]}
        start = {
            "user": state["user"], "classified": None, "turn_messages": [], "decline": None,
            "security_events": state["security_events"], "draft": Draft(),
            "budget": Budget(bq_down=state["budget"].bq_down),
            "task_messages": flow.section_messages(state["user"], deps.today(), memory.records_block(state["records"]),
                                                   deps.known, s, layers_for(state)),
        }
        out = section_loop.invoke(start)
        update: dict = {"budget": _merge_section(state["budget"], out["budget"]),
                        "security_events": out["security_events"]}
        if out.get("decline"):
            return {**update, "decline": out["decline"]}
        produced, charts = _tool_results(out["turn_messages"])
        records = [memory.new_record(f"{outline.title}: {s.title}", p["result"],
                                     p["args"].get("sql", "") or deps.store.get(p["result"]["result_id"]).sql,
                                     deps.store.get(p["result"]["result_id"]).meta)
                   for p in produced]
        result = SectionResult(title=s.title, notes=out["draft"].text, chart_paths=charts,
                               result_ids=[*s.result_ids, *(p["result"]["result_id"] for p in produced)])
        return {**update, "sections_done": [*done, result], "records": [*state["records"], *records]}

    def route_after_section(state: AgentState) -> str:
        if state.get("decline"):
            return "decline"
        return "section" if len(state["sections_done"]) < len(state["outline"].sections) else "write"

    def write(state: AgentState) -> dict:
        outline, sections, previous = state["outline"], state["sections_done"], state.get("redo_of")
        stats = CallStats()
        messages = flow.write_messages(state["user"], deps.today(), state["user_input"], outline, sections,
                                       deps.store, previous, state.get("write_feedback"), layers_for(state))
        try:
            draft = deps.llm.structured("report", ReportDraft, messages, stats)
        except LLMOutputError:
            return {"draft": Draft(text=WRITE_FAILED, source="code"), "write_feedback": None,
                    "budget": _charge(state["budget"], stats)}
        draft = flow.anchor_check(draft, deps.store, outline.user_context)
        result_ids = list(dict.fromkeys([*(r for s in sections for r in s.result_ids),
                                         *(r for f in draft.findings for r in f.result_ids if r in deps.store)]))
        data_ts = flow.data_as_of(deps.store, result_ids)
        body, exempt = flow.render(draft, outline.user_context, data_ts)
        charts = [c for s in sections for c in s.chart_paths]
        if previous is not None and not charts:
            charts = previous.chart_paths
        report = CurrentReport(
            report_id=previous.report_id if previous is not None else None,
            version=previous.version if previous is not None else 0,
            title=draft.title, body=body, request=state["user_input"], outline=outline, draft=draft,
            result_ids=result_ids, chart_paths=charts, data_ts=data_ts,
        )
        return {"draft": Draft(text=body, source="report", chart_paths=charts, exempt=exempt), "report": report,
                "write_feedback": None, "budget": _charge(state["budget"], stats)}

    def save_report(state: AgentState) -> dict:
        """D3.8, D3.11: code saves the latest report with code-built mentions (D3.18)."""
        report = state.get("report")
        if report is None or not report.body:
            return {"draft": Draft(text=NO_REPORT, source="code")}
        if report.saved:
            return {"draft": Draft(text=f"'{report.title}' is already saved.", source="code")}
        frames = [deps.store.get(r).df for r in report.result_ids if r in deps.store]
        findings = [SavedFinding(finding_id=f.id, text=f.text, result_id=r, sql=deps.store.get(r).sql)
                    for f in (report.draft.findings if report.draft else []) for r in f.result_ids if r in deps.store]
        saved = reports.save(state["user"], NewVersion(
            report_id=report.report_id, conversation_id=state.get("conversation_id", ""), title=report.title,
            body=report.body, chart_files=report.chart_paths, data_ts=report.data_ts, trace_id=state["trace_id"],
            findings=findings, mentions=mentions.scan(report.body, frames)))
        report = report.model_copy(update={"report_id": saved.report_id, "version": saved.version, "saved": True})
        return {"draft": Draft(text=f"Saved '{report.title}'.", source="code"), "report": report}

    def download_report(state: AgentState) -> dict:
        report = state.get("report")
        if report is None or not report.body:
            return {"draft": Draft(text=NO_REPORT, source="code")}
        path = download(report.title, report.body, report.chart_paths, deps.settings.reports.download_dir)
        return {"draft": Draft(text=f"Saved a copy to {path}.", source="code")}

    # Delete flow (D3.17 to D3.27)

    def propose_delete(state: AgentState, args: dict) -> tuple[str, dict]:
        """Resolve the model's filter in code. Matches go to the confirmation step; the model never picks ids."""
        filter_type = args.get("type", "mentions")
        shown, unsaved = deletion.resolve(reports, state["user"], filter_type, args.get("value", ""),
                                          state.get("conversation_id", ""), state.get("report"))
        if shown:
            pending = PendingDelete(request=state["user_input"], shown=shown, discard_unsaved=unsaved is not None)
            return json.dumps({"proposed": len(shown), "message": "Shown to the user for confirmation."}), \
                {"pending_delete": pending}
        if unsaved:
            # Unsaved session reports are dropped with a mention, no confirmation (D3.19).
            return json.dumps({"matches": 0, "discarded_unsaved": unsaved,
                               "message": f"No saved reports matched. The unsaved report '{unsaved}' was discarded."}), \
                {"report": None}
        what = "made in this conversation" if filter_type == "conversation" else f"mentioning '{args.get('value', '')}'"
        return json.dumps({"matches": 0, "message": f"No saved reports {what}. Nothing to delete."}), {}

    def confirm_delete(state: AgentState) -> dict:
        pending = state["pending_delete"]
        shown = deletion.confirm_text(pending, again=state.get("decision") == "unclear")
        reply = interrupt({"kind": "delete", "text": shown})
        stats = CallStats()
        decision = classify_reply(deps.llm, shown, reply, stats)
        pending = pending.model_copy(update={"replies": [*pending.replies, reply]})
        update = {"decision": decision, "pending_delete": pending}
        if decision == "cancel":
            update.update(pending_delete=None, draft=Draft(text=deletion.CANCELLED, source="code"))
        elif decision == "narrow":
            kept = deletion.narrow(deps.llm, pending, reply, stats)
            if kept:
                update["pending_delete"] = pending.model_copy(update={"shown": kept})
            else:
                update.update(pending_delete=None, draft=Draft(text=deletion.NONE_LEFT, source="code"))
        update["budget"] = _charge(state["budget"], stats)
        return update

    def route_after_confirm_delete(state: AgentState) -> str:
        if state.get("pending_delete") is None:
            return "output_check"
        return "delete_reports" if state["decision"] == "yes" else "confirm_delete"

    def delete_reports(state: AgentState) -> dict:
        """D3.22 to D3.24: exactly the ids shown, every version, soft, audited in the same transaction."""
        pending = state["pending_delete"]
        request = " | ".join([pending.request, *pending.replies])
        deleted = reports.soft_delete(state["user"], [r.report_id for r in pending.shown], request, state["trace_id"])
        report = state.get("report")
        discarded = None
        if pending.discard_unsaved and report is not None and not report.saved:
            discarded, report = report.title, None
        if report is not None and report.report_id in {r.report_id for r in deleted}:
            report = None
        return {"draft": Draft(text=deletion.deleted_text([r.title for r in deleted], discarded), source="code"),
                "report": report, "pending_delete": None}

    # Output check and the end of the turn

    def output_check(state: AgentState) -> dict:
        """Leak scan, then number grounding (D1.16). Code-written text is only leak-scanned; report targets
        are labelled and exempt from grounding (D3.7)."""
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
        missing = [f for f in oc.ungrounded(draft.text, oc.ground_values(frames, user_texts))
                   if not _exempt(f, draft.text, draft.exempt)]
        if not missing:
            return {}
        if draft.source in ("agent", "report") and not state["grounding_retried"]:
            # One retry: the agent or writer is told which figures had no source.
            if draft.source == "report":
                return {"grounding_retried": True, "write_feedback": flow.write_retry([f.text.strip() for f in missing])}
            return {"grounding_retried": True,
                    "turn_messages": [*state["turn_messages"], HumanMessage(oc.retry_instruction(missing))]}
        return {"draft": draft.model_copy(update={"text": oc.strip_figures(draft.text, missing)}),
                "removed_numbers": [f.text.strip() for f in missing]}

    def route_after_output_check(state: AgentState) -> str:
        if state.get("write_feedback"):
            return "write"
        last = state["turn_messages"][-1] if state["turn_messages"] else None
        return "agent" if isinstance(last, HumanMessage) else "respond"

    def respond(state: AgentState) -> dict:
        draft = state["draft"]
        update: dict = {}
        charts = list(draft.chart_paths) if draft.source == "report" else []
        if draft.source in ("agent", "light"):
            charts = _tool_results(state["turn_messages"])[1]
        text = draft.text
        if state.get("preference_note"):
            text = f"{state['preference_note']}\n\n{text}"
        if draft.source == "agent":
            classified = state["classified"]
            note = scope_note(state["user"], classified.out_of_scope_brands if classified else [])
            if note:
                text = f"{text}\n\n{note}"
        if draft.source == "report":
            # The body as checked is what gets saved; the offer is not part of it (D3.8).
            update["report"] = state["report"].model_copy(update={"body": draft.text})
            text = f"{text}\n\n{flow.SAVE_OFFER}"
        opened = state.get("opened_report")
        if opened and state.get("decline") is None:
            saved = reports.get(state["user"], opened)
            if saved is not None:
                # Shown as saved, with its as-of time (D3.16). Checked when it was made; owner and scope checked now.
                text = f"{text}\n\n---\n\n*Saved {saved.created_at}. Data as of {saved.data_ts}.*\n\n{saved.body}"
                charts += [c for c in saved.chart_files if Path(c).exists()]
                update["report"] = CurrentReport(report_id=saved.report_id, version=saved.version, saved=True,
                                                 title=saved.title, body=saved.body, chart_paths=saved.chart_files,
                                                 data_ts=saved.data_ts, from_this_session=False)
        draft = draft.model_copy(update={"text": text, "chart_paths": charts})
        messages = [*state["messages"], HumanMessage(state["user_input"]), AIMessage(draft.text)]
        return {**update, "draft": draft, "messages": messages}

    def save(state: AgentState) -> dict:
        new_records = [memory.new_record(state["user_input"], p["result"],
                                         p["args"].get("sql", "") if p["tool"] == "run_sql"
                                         else deps.store.get(p["result"]["result_id"]).sql,
                                         deps.store.get(p["result"]["result_id"]).meta)
                       for p in _tool_results(state["turn_messages"])[0]]
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
                     ("decline", decline), ("preferences", preferences_node), ("agent", make_agent(chat_messages)),
                     ("tools", tools),
                     ("plan", plan), ("confirm_outline", confirm_outline), ("section", section), ("write", write),
                     ("save_report", save_report), ("download_report", download_report),
                     ("confirm_delete", confirm_delete), ("delete_reports", delete_reports),
                     ("output_check", output_check), ("respond", respond), ("save", save)]:
        g.add_node(name, fn)
    g.add_edge(START, "load_context")
    g.add_edge("load_context", "classify")
    g.add_conditional_edges("classify", route_after_classify,
                            ["preferences", "light_reply", "decline", "agent", "plan", "save_report", "download_report"])
    g.add_conditional_edges("preferences", route_after_preferences,
                            ["output_check", "light_reply", "decline", "agent", "plan", "save_report", "download_report"])
    g.add_edge("light_reply", "output_check")
    g.add_edge("decline", "output_check")
    g.add_conditional_edges("agent", route_after_agent, ["tools", "output_check"])
    g.add_conditional_edges("tools", route_after_tools, ["agent", "decline", "confirm_delete"])
    g.add_conditional_edges("plan", route_after_plan, ["confirm_outline", "section", "output_check"])
    g.add_conditional_edges("confirm_outline", route_after_confirm_outline, ["section", "plan", "output_check"])
    g.add_conditional_edges("section", route_after_section, ["section", "write", "decline"])
    g.add_edge("write", "output_check")
    g.add_edge("save_report", "output_check")
    g.add_edge("download_report", "output_check")
    g.add_conditional_edges("confirm_delete", route_after_confirm_delete,
                            ["confirm_delete", "delete_reports", "output_check"])
    g.add_edge("delete_reports", "output_check")
    g.add_conditional_edges("output_check", route_after_output_check, ["agent", "write", "respond"])
    g.add_edge("respond", "save")
    g.add_edge("save", END)
    saver = checkpointer or InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))
    return g.compile(checkpointer=saver)


def download(title: str, body: str, charts: list[str], directory: Path) -> Path:
    """D3.14: markdown plus charts alongside, in the gitignored reports/ folder."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slug(title)}.md"
    lines = [body]
    for chart in charts:
        src = Path(chart)
        if src.exists():
            dest = directory / src.name
            shutil.copy(src, dest)
            lines.append(f"\n![{src.stem}]({dest.name})")
    path.write_text("\n".join(lines))
    return path
