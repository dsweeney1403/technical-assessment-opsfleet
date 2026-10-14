"""Intent classifier (D1.8, D1.9). Flash-Lite, structured output, sees recent turns.

Not a security control: it routes, and code enforces safety behind it. If it
fails, the message goes to the agent as unclassified (D2.11).
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from analyst.llm.client import LLM, CallStats, LLMError
from analyst.memory import split_turns
from analyst.prompts import CLASSIFIER
from analyst.state import Classified, Intent

RECENT_TURNS = 3


class ClassifierOutput(BaseModel):
    intent: Literal[
        "analysis", "data_question", "conversation", "report_action",
        "off_topic", "pii_individuals", "injection", "destructive",
    ]
    brands_mentioned: list[str] = Field(default_factory=list)
    is_followup: bool = False
    reason: str = Field(description="One short sentence.")


def _context(history: list[BaseMessage], user_input: str) -> str:
    lines = []
    for turn in split_turns(history)[-RECENT_TURNS:]:
        for m in turn:
            speaker = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{speaker}: {m.text[:500]}")
    recent = "\n".join(lines) or "(no earlier turns)"
    return f"Recent turns:\n{recent}\n\nLatest message:\n{user_input}"


def classify(llm: LLM, history: list[BaseMessage], user_input: str, stats: CallStats) -> Classified:
    messages = [SystemMessage(CLASSIFIER), HumanMessage(_context(history, user_input))]
    try:
        out = llm.structured("light", ClassifierOutput, messages, stats)
    except LLMError as exc:
        return Classified(intent=Intent.unavailable, reason=f"classifier failed: {type(exc).__name__}")
    return Classified(
        intent=Intent(out.intent),
        brands_mentioned=out.brands_mentioned,
        is_followup=out.is_followup,
        reason=out.reason,
    )
