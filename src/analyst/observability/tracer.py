"""One trace per turn, OpenTelemetry-shaped, written as JSONL (D2.16, D2.17, D2.23).

Spans come from three places:
- LangGraph callbacks: node runs (classify, agent steps, output check) and tool runs.
- A custom event from the LLM wrapper per call: model, messages in and out, tokens,
  latency, retries and fallback use. Only the wrapper knows the retries and the chain.
- A custom event from the SQL gateway per query: SQL before and after the rewrite,
  code, bytes, rows, diagnosis and latency.
- A custom event per trio retrieval: the trio ids and scores injected, how many
  fell below the threshold or outside the user's brands, and `trios_unavailable`
  on an embedding outage (D4.8, D4.9).

The trace stores what the model saw, which is PII-free by construction (D1.4).
Raw BigQuery rows from before the gateway are never passed in. Spans keep
OpenTelemetry's fields (trace and span ids, parent, start and end, attributes,
status), so production swaps the JSONL exporter for an OpenTelemetry one (D2.23).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler, dispatch_custom_event
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel

from analyst.config import Settings, User

# Graph nodes that become spans. load_context and respond only move state around.
TRACED_NODES = {"classify", "preferences", "light_reply", "decline", "agent", "output_check", "save", "plan", "confirm_outline",
                "section", "write", "save_report", "download_report", "confirm_delete", "delete_reports"}
SPAN_NAMES = {"agent": "agent_step"}

# Outcome and reason (D2.17). Precedence for partial answers, first match wins.
STOP_REASONS = {"NOT_SELECT", "PII_IDENTIFIER", "LEAK_BLOCKED"}
PARTIAL_ORDER = ["BQ_DOWN", "STUCK", "BUDGET_SPENT", "GROUNDING_REMOVED", "NO_ACCESS_BRAND"]


def emit(name: str, data: dict) -> None:
    """Send a span to the trace handler, if one is listening. Outside a graph run it does nothing."""
    try:
        dispatch_custom_event(name, data)
    except RuntimeError:
        pass  # no parent run: called directly, e.g. from a test or the doctor


def now_ns() -> int:
    return time.time_ns()


def jsonable(value: Any) -> Any:
    """Plain JSON types for the trace file."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, BaseMessage):
        return message_dict(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def message_dict(m: BaseMessage) -> dict:
    out: dict[str, Any] = {"role": m.type, "content": m.text if not isinstance(m.content, str) else m.content}
    if isinstance(m, AIMessage) and m.tool_calls:
        out["tool_calls"] = [{"name": c["name"], "args": jsonable(c["args"]), "id": c.get("id")} for c in m.tool_calls]
    if isinstance(m, ToolMessage):
        out["name"], out["tool_call_id"] = m.name, m.tool_call_id
    return out


def messages_dicts(messages: Sequence[BaseMessage]) -> list[dict]:
    return [message_dict(m) for m in messages]


@dataclass
class Span:
    name: str
    span_id: str
    parent_span_id: str | None
    start_ns: int
    end_ns: int | None = None
    attributes: dict = field(default_factory=dict)
    status: str = "ok"  # ok | error

    @property
    def latency_ms(self) -> int:
        return round(((self.end_ns or self.start_ns) - self.start_ns) / 1e6)


def _span_id(run_id: uuid.UUID | str | None = None) -> str:
    return (uuid.UUID(str(run_id)) if run_id else uuid.uuid4()).hex[:16]


class TraceHandler(BaseCallbackHandler):
    """Collects one turn's spans from LangGraph callbacks and custom events."""

    def __init__(self) -> None:
        self.root_id = _span_id()
        self.spans: list[Span] = []
        self._by_run: dict[str, Span] = {}
        self._parent_run: dict[str, str | None] = {}
        self._steps = 0
        self._last_step: Span | None = None
        self._tools_runs: set[str] = set()  # runs of the tools node
        self._tools_node_spans: list[Span] = []  # tool runs inside the current tools node

    # Parents

    def _traced_parent(self, run_id) -> str:
        """Nearest traced ancestor of a run, or the root span."""
        key = str(run_id) if run_id else None
        while key is not None:
            if key in self._by_run:
                return self._by_run[key].span_id
            key = self._parent_run.get(key)
        return self.root_id

    def _open(self, name: str, run_id, parent_id: str, attributes: dict | None = None) -> Span:
        span = Span(name, _span_id(run_id), parent_id, now_ns(), attributes=attributes or {})
        self.spans.append(span)
        if run_id is not None:
            self._by_run[str(run_id)] = span
        return span

    # Graph nodes

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, name=None, **kw):
        self._parent_run[str(run_id)] = str(parent_run_id) if parent_run_id else None
        node = (metadata or {}).get("langgraph_node")
        if name != node:
            return  # routers, channel writes and inner chains are not spans
        if node == "tools":
            self._tools_runs.add(str(run_id))
            self._tools_node_spans = []
        elif node in TRACED_NODES:
            attrs = {}
            if node == "agent":
                self._steps += 1
                attrs["step"] = self._steps
            span = self._open(SPAN_NAMES.get(node, node), run_id, self._traced_parent(parent_run_id), attrs)
            if node == "agent":
                self._last_step = span

    def on_chain_end(self, outputs, *, run_id, **kw):
        span = self._by_run.get(str(run_id))
        if span is not None:
            span.end_ns = now_ns()
            span.attributes.update(_node_attributes(span.name, outputs))
        elif str(run_id) in self._tools_runs and isinstance(outputs, dict):
            self._match_tool_results(outputs)

    def on_chain_error(self, error, *, run_id, **kw):
        span = self._by_run.get(str(run_id))
        if span is not None:
            span.end_ns = now_ns()
            span.status = "error"
            span.attributes["error"] = type(error).__name__

    # Tools

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, inputs=None, **kw):
        self._parent_run[str(run_id)] = str(parent_run_id) if parent_run_id else None
        parent = self._last_step.span_id if self._last_step else self._traced_parent(parent_run_id)
        span = self._open("tool_call", run_id, parent, {"tool": serialized.get("name"), "args": jsonable(inputs)})
        self._tools_node_spans.append(span)

    def on_tool_end(self, output, *, run_id, **kw):
        span = self._by_run.get(str(run_id))
        if span is not None:
            span.end_ns = now_ns()

    def on_tool_error(self, error, *, run_id, **kw):
        span = self._by_run.get(str(run_id))
        if span is not None:
            span.end_ns = now_ns()
            span.status = "error"
            span.attributes["error"] = type(error).__name__

    def _match_tool_results(self, outputs: dict) -> None:
        """Attach each tool result exactly as the model got it. Results written by code
        (budget used up, BigQuery down, unknown tool) get a span of their own."""
        messages = outputs["turn_messages"]
        calls = {}
        for m in messages:
            if isinstance(m, AIMessage):
                for c in m.tool_calls:
                    calls[c["id"]] = c
        new = []
        for m in reversed(messages):
            if not isinstance(m, ToolMessage):
                break
            new.append(m)
        unmatched = list(self._tools_node_spans)
        for m in reversed(new):
            call = calls.get(m.tool_call_id, {})
            args = jsonable(call.get("args"))
            span = next((s for s in unmatched if s.attributes["tool"] == m.name and s.attributes["args"] == args), None)
            if span is None:
                parent = self._last_step.span_id if self._last_step else self.root_id
                span = self._open("tool_call", None, parent, {"tool": m.name, "args": args, "handled_in_code": True})
                span.end_ns = span.start_ns
            else:
                unmatched.remove(span)
            span.attributes["result"] = m.content
            code = _error_code(m.content)
            if code:
                span.attributes["code"] = code
        self._tools_node_spans = []

    # Custom events: llm_call from the LLM wrapper, gateway from the SQL gateway

    def on_custom_event(self, name, data, *, run_id, **kw):
        if name not in ("llm_call", "gateway", "retrieval"):
            return
        data = dict(data)
        start, end = data.pop("start_ns"), data.pop("end_ns")
        status = "error" if data.get("error") else "ok"
        span = Span(name, _span_id(), self._traced_parent(run_id), start, end, jsonable(data), status)
        self.spans.append(span)


def _error_code(content) -> str | None:
    try:
        return json.loads(content).get("error", {}).get("code")
    except (TypeError, ValueError, AttributeError):
        return None


def _node_attributes(span_name: str, outputs) -> dict:
    """What each node decided, from its state update."""
    if not isinstance(outputs, dict):
        return {}
    if span_name == "classify" and outputs.get("classified") is not None:
        c = outputs["classified"]
        return {"verdict": c.intent.value, "confident": c.confident, "brands_mentioned": c.brands_mentioned,
                "out_of_scope_brands": c.out_of_scope_brands, "decline": outputs.get("decline"),
                "format_request": c.format_request,
                "stated_preference": c.stated_preference.model_dump() if c.stated_preference else None}
    if span_name == "preferences":
        profile = outputs.get("profile")
        return {"decline": outputs.get("decline"), "definition_request": outputs.get("definition_request", False),
                "note": outputs.get("preference_note") or (outputs["draft"].text if outputs.get("draft") else None),
                "profile": profile.model_dump(mode="json") if profile is not None else None}
    if span_name == "decline":
        return {"text": outputs["draft"].text} if outputs.get("draft") else {}
    if span_name == "plan" and outputs.get("outline") is not None:
        o = outputs["outline"]
        return {"title": o.title, "sections": [f"{x.title} ({x.source})" for x in o.sections],
                "user_context": o.user_context}
    if span_name in ("confirm_outline", "confirm_delete") and outputs.get("decision"):
        return {"decision": outputs["decision"]}
    if span_name == "section" and outputs.get("sections_done"):
        done = outputs["sections_done"][-1]
        return {"section": done.title, "result_ids": done.result_ids}
    if span_name == "write" and outputs.get("report") is not None:
        r = outputs["report"]
        return {"title": r.title, "findings": len(r.draft.findings) if r.draft else 0,
                "actions": len(r.draft.actions) if r.draft else 0}
    if span_name == "delete_reports" and outputs.get("draft") is not None:
        return {"text": outputs["draft"].text}
    if span_name == "output_check":
        leaks = [e.detail for e in outputs.get("security_events", []) if e.code == "LEAK_BLOCKED"]
        return {"leak_hits": leaks, "numbers_removed": outputs.get("removed_numbers", []),
                "grounding_retry": bool(outputs.get("grounding_retried"))}
    return {}


def outcome_for(state: dict, error: BaseException | None, cap: int, awaiting: bool = False) -> tuple[str, str]:
    """One outcome and one reason code per turn (D2.17)."""
    if awaiting and error is None:
        return "answered", "AWAITING_CONFIRMATION"  # the reply is its own trace, linked by `continues`
    if error is not None:
        from analyst.llm.client import LLMFatalError, LLMUnavailableError
        from analyst.sql.gateway import GatewayFatalError

        if isinstance(error, LLMUnavailableError):
            return "failed", "LLM_DOWN"
        if isinstance(error, GatewayFatalError):
            return "failed", "BQ_FATAL"
        if isinstance(error, LLMFatalError):
            return "failed", "LLM_FATAL"
        return "failed", "INTERNAL"
    decline = state.get("decline")
    if decline:
        return "declined", decline if decline in STOP_REASONS else decline.upper()
    found = set()
    budget = state.get("budget")
    if budget is not None:
        if budget.bq_down:
            found.add("BQ_DOWN")
        if budget.stuck:
            found.add("STUCK")
        if budget.cap_hit or budget.tool_calls_used >= cap:
            found.add("BUDGET_SPENT")
    if state.get("removed_numbers"):
        found.add("GROUNDING_REMOVED")
    classified = state.get("classified")
    tool_codes = {_error_code(m.content) for m in state.get("turn_messages", []) if isinstance(m, ToolMessage)}
    if (classified is not None and classified.out_of_scope_brands) or "NO_ACCESS_BRAND" in tool_codes:
        found.add("NO_ACCESS_BRAND")
    for reason in PARTIAL_ORDER:
        if reason in found:
            return "partial", reason
    return "answered", "OK"


class JsonlExporter:
    """One file per conversation in traces/, one line per turn (gitignored, D2.16)."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def path(self, conversation_id: str) -> Path:
        return self.dir / f"{conversation_id}.jsonl"

    def write(self, record: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.path(record["conversation_id"]).open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def read_traces(path: Path) -> list[dict]:
    """Every line of a conversation's file: turn records, plus feedback lines (`"type": "feedback"`)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_feedback(record: dict) -> bool:
    return record.get("type") == "feedback"


class Tracer:
    """Builds and writes the trace for each turn of one conversation."""

    def __init__(self, settings: Settings, user: User, conversation_id: str, exporter: JsonlExporter | None = None):
        self.settings = settings
        self.user = user
        self.conversation_id = conversation_id
        self.exporter = exporter or JsonlExporter(settings.tracing.dir)
        self._attempts: dict[str, int] = {}
        self.last: dict | None = None  # the most recent turn's trace record

    @contextmanager
    def turn(self, graph, config: dict, user_input: str | None = None) -> Iterator[dict]:
        """Wrap one graph run. Yields the run config with the trace handler attached, then
        writes the trace from the final checkpoint, failed turns included."""
        handler, started_ns, started_at = self.start()
        try:
            yield {**config, "callbacks": [handler]}
        except Exception as exc:
            try:
                self.last = self.finish(handler, started_ns, started_at, graph.get_state(config).values, exc,
                                        user_input)
            except Exception:
                pass  # a tracing problem must never hide the error the user needs to see
            raise
        snapshot = graph.get_state(config)
        interrupts = getattr(snapshot, "interrupts", ()) or ()
        self.last = self.finish(handler, started_ns, started_at, snapshot.values, None, user_input,
                                awaiting=interrupts[0].value.get("text") if interrupts else None)

    @property
    def path(self) -> Path:
        return self.exporter.path(self.conversation_id)

    def add_feedback(self, trace_id: str, thumbs: str, reason: str | None = None) -> dict:
        """Thumbs up, down or skip on an answer, stored next to its trace (D4.19, D8.1)."""
        record = {"type": "feedback", "trace_id": trace_id, "conversation_id": self.conversation_id,
                  "user": {"id": self.user.id, "role": self.user.role},
                  "at": datetime.now(UTC).isoformat(timespec="seconds"), "thumbs": thumbs, "reason": reason}
        self.exporter.write(record)
        return record

    def start(self) -> tuple[TraceHandler, int, str]:
        return TraceHandler(), now_ns(), datetime.now(UTC).isoformat(timespec="seconds")

    def finish(self, handler: TraceHandler, started_ns: int, started_at: str, state: dict,
               error: BaseException | None = None, user_input: str | None = None,
               awaiting: str | None = None) -> dict:
        """`awaiting` is the confirmation text shown, when the turn stopped to ask the user."""
        trace_id = state.get("trace_id") or uuid.uuid4().hex[:8]
        attempt = self._attempts.get(trace_id, 0) + 1
        self._attempts[trace_id] = attempt
        cap = self.settings.budgets.max_tool_calls_per_turn
        outcome, reason = outcome_for(state, error, cap, awaiting is not None)
        end = now_ns()
        spans = handler.spans
        gateway = [s for s in spans if s.name == "gateway"]
        llm = [s for s in spans if s.name == "llm_call"]
        budget = state.get("budget")
        totals = {
            "tokens": sum(s.attributes.get("tokens", {}).get("total", 0) for s in llm),
            "bytes": sum(s.attributes.get("bytes_billed") or 0 for s in gateway),
            "latency_ms": round((end - started_ns) / 1e6),
            "tool_calls": budget.tool_calls_used + budget.section_tool_calls if budget else 0,
            "llm_calls": len(llm),
            "llm_retries": sum(s.attributes.get("retries", 0) for s in llm),
            "bq_retries": sum(s.attributes.get("transient_retries", 0) for s in gateway),
            "fallback_used": any(s.attributes.get("fallback_used") for s in llm),
            "breaker_opens": sum(s.attributes.get("breaker_opens", 0) for s in llm),
        }
        draft = state.get("draft")
        persona = state.get("persona")
        root = Span("turn", handler.root_id, None, started_ns, end,
                    {"outcome": outcome, "reason": reason, "totals": totals},
                    "error" if outcome == "failed" else "ok")
        models = self.settings.models
        record = {
            "trace_id": trace_id,
            "conversation_id": self.conversation_id,
            "attempt": attempt,
            "continues": state.get("continues"),
            "user": {"id": self.user.id, "role": self.user.role},
            "started_at": started_at,
            "models": {"agent": [models.agent, *models.agent_fallbacks],
                       "light": [models.light, *models.light_fallbacks]},
            "user_input": user_input if user_input is not None else state.get("user_input", ""),
            "persona": ({"version": persona.version, "fallback_from": persona.fallback_from, "error": persona.error}
                        if persona is not None else None),
            "outcome": outcome,
            "reason": reason,
            "error": f"{type(error).__name__}: {error}" if error is not None else None,
            "answer": awaiting or (draft.text if draft is not None and error is None else None),
            "totals": totals,
            "security_events": jsonable(state.get("security_events", [])),
            "spans": [asdict(s) for s in [root, *spans]],
        }
        self.exporter.write(record)
        return record
