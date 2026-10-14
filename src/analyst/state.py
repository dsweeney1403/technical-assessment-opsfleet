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


class Classified(BaseModel):
    intent: Intent
    confident: bool = True  # an unsure decline intent goes to the agent (D1.8)
    brands_mentioned: list[str] = []
    out_of_scope_brands: list[str] = []  # filled by code against entitlements (D1.8 row 6)
    is_followup: bool = False
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


class Draft(BaseModel):
    text: str = ""
    source: Literal["agent", "light", "code"] = "agent"  # code-written text skips number grounding
    chart_paths: list[str] = Field(default_factory=list)


class SecurityEvent(BaseModel):
    """Written to the trace in phase 4 (D2.1, D1.8, D1.16)."""

    code: str  # injection, destructive, NOT_SELECT, PII_IDENTIFIER, LEAK_BLOCKED
    source: Literal["classifier", "gateway", "output_check"]
    detail: str = ""


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
