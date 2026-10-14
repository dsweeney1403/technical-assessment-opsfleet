"""Graph state (Group 1 and 2 interfaces)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from analyst.config import User


class Intent(StrEnum):
    """D1.8 intents, plus `unavailable` when the classifier fails open (D2.11)."""

    analysis = "analysis"
    data_question = "data_question"
    conversation = "conversation"
    report_action = "report_action"
    off_topic = "off_topic"
    pii_individuals = "pii_individuals"
    injection = "injection"
    destructive = "destructive"
    unavailable = "unavailable"


ReportOp = Literal["create", "redo", "save", "library", "download", "delete", "none"]


class Classified(BaseModel):
    intent: Intent
    confident: bool = True  # an unsure decline intent goes to the agent (D1.8)
    brands_mentioned: list[str] = []
    out_of_scope_brands: list[str] = []  # filled by code against entitlements (D1.8 row 6)
    is_followup: bool = False
    report_op: ReportOp = "none"  # which report action, for report_action only (routing, phase 5)
    reason: str = ""


class Record(BaseModel):
    """One per successful query in an answered question (D1.13). Never summarised (D1.14)."""

    record_id: str
    question: str
    sql: str
    result_id: str
    metric: str | None = None
    filters: str | None = None
    grouping: str | None = None
    data_ts: str
    columns: list[str] = []
    row_count: int = 0


class Budget(BaseModel):
    """Per-turn counters. LLM retries are kept apart from tool calls (D2.8)."""

    tool_calls_used: int = 0
    tokens_used: int = 0
    llm_retries: int = 0
    last_error_code: str | None = None  # stuck rule (D2.2): the last run_sql error code in a row
    stuck: bool = False  # same code twice in a row: answer now, no more tools
    bq_down: bool = False  # BigQuery retries ran out this turn (D2.13)
    models_used: list[str] = Field(default_factory=list)  # for the trace (D2.10)
    fallback_used: bool = False
    breaker_opens: int = 0
    cap_hit: bool = False  # a loop was told to answer because its tool budget ran out
    section_tool_calls: int = 0  # tool calls made by report sections, each with its own cap (D3.1)


class Draft(BaseModel):
    text: str = ""
    source: Literal["agent", "light", "code", "report"] = "agent"  # code-written text skips number grounding
    chart_paths: list[str] = Field(default_factory=list)
    exempt: list[str] = Field(default_factory=list)  # labelled targets, not grounded (D3.7)


class SecurityEvent(BaseModel):
    """Written to the trace in phase 4 (D2.1, D1.8, D1.16)."""

    code: str  # injection, destructive, NOT_SELECT, PII_IDENTIFIER, LEAK_BLOCKED
    source: Literal["classifier", "gateway", "output_check"]
    detail: str = ""


# Reports (Group 3 Interfaces)


class Section(BaseModel):
    title: str
    brief: str = Field(description="What the section shows and the metrics it needs.")
    source: Literal["from_records", "new_queries"] = Field(
        description="from_records: built only from stored results listed. new_queries: needs new queries.")
    result_ids: list[str] = Field(default_factory=list, description="Stored results the section uses, if any.")


class Outline(BaseModel):
    title: str
    sections: list[Section]
    user_context: list[str] = Field(
        default_factory=list,
        description="Facts about the business the user stated in this conversation, quoted word for word.")


class SectionResult(BaseModel):
    title: str
    notes: str = ""
    result_ids: list[str] = Field(default_factory=list)
    chart_paths: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    id: str = Field(description="f1, f2, ...")
    text: str
    result_ids: list[str] = Field(description="Stored results the finding comes from.")


class Target(BaseModel):
    value: str = Field(description="The suggested target, e.g. '12% repeat rate'.")
    basis: str = Field(description="What it is in line with, e.g. 'the 11% store average'. Empty if none.")


class Action(BaseModel):
    why: str = Field(description="Why this matters here, from the findings or the user's context.")
    how: str = Field(description="A suggested way to act on it. Worded as a suggestion.")
    based_on: list[str] = Field(description="Finding ids (f1) or user context ids (c1) behind the why.")
    target: Target | None = None
    offer: str = Field(description="The same idea as an offer to investigate, used if it has no anchor. "
                                   "E.g. 'Retention isn't covered here. Want me to look at repeat purchase rates?'")


class ReportDraft(BaseModel):
    title: str
    assumptions: str = Field(description="One line: date range, metric definitions and scope used.")
    findings: list[Finding]
    actions: list[Action] = Field(default_factory=list)
    next_questions: list[str] = Field(default_factory=list)


class CurrentReport(BaseModel):
    """The latest report in this session, saved or not (D3.8)."""

    report_id: str | None = None  # set once saved
    version: int = 0  # saved version; 0 while unsaved
    saved: bool = False
    title: str
    body: str = ""
    request: str = ""
    outline: Outline | None = None
    draft: ReportDraft | None = None
    result_ids: list[str] = Field(default_factory=list)
    chart_paths: list[str] = Field(default_factory=list)
    data_ts: str = ""
    from_this_session: bool = True  # False when opened from the library: its results are gone (D3.6)


class ShownReport(BaseModel):
    report_id: str
    title: str
    mentions: list[str] = Field(default_factory=list)


class PendingDelete(BaseModel):
    """What the confirmation shows. Only these ids can be deleted (D3.22)."""

    request: str
    shown: list[ShownReport]
    discard_unsaved: bool = False  # "made in this conversation" also drops the unsaved report (D3.19)
    replies: list[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    user: User  # injected by code from --user, never chosen by the model
    user_input: str
    messages: list[BaseMessage]  # verbatim turns: user messages and final answers
    turn_messages: list[BaseMessage]  # this turn's tool calls and results, cleared each turn
    summary: str  # rolling summary of turns older than the verbatim window
    records: list[Record]
    classified: Classified | None
    budget: Budget
    draft: Draft
    trace_id: str
    decline: str | None  # why the turn was declined: an intent, a stop code or no_access
    security_events: list[SecurityEvent]
    grounding_retried: bool  # one grounding retry per turn (D1.16)
    removed_numbers: list[str]  # ungrounded figures removed from the answer (D1.16)
    conversation_id: str  # the session, for "made in this conversation" (D3.19)
    continues: str | None  # the trace this reply continues, after a confirmation (D2.19)
    report: CurrentReport | None
    outline: Outline | None
    outline_feedback: list[str]
    redo_of: CurrentReport | None  # the report being changed (D3.9)
    task_messages: list[BaseMessage]  # a report section's fresh context (D3.1)
    decision: str | None  # the last confirmation reply: yes, narrow, cancel or unclear
    sections_done: list[SectionResult]
    write_feedback: str | None  # figures the writer must drop or source, one retry (D1.16)
    opened_report: str | None  # report_id opened this turn, shown as saved (D3.16)
    pending_delete: PendingDelete | None
