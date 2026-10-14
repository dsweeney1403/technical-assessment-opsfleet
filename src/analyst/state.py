"""Graph state (Group 1 and 2 interfaces)."""

from __future__ import annotations

from enum import StrEnum
from typing import TypedDict

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
    brands_mentioned: list[str] = []
    out_of_scope_brands: list[str] = []  # filled by code against entitlements (phase 2)
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
    last_error_code: str | None = None  # stuck rule (D2.2), used from phase 3


class Draft(BaseModel):
    text: str = ""
    chart_paths: list[str] = Field(default_factory=list)


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
