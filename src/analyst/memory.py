"""Conversation memory (D1.13, D1.14).

The last K turns stay verbatim. Older turns fold into a rolling summary that
carries the narrative only. Records are never summarised, and numbers always
come from records, never from the summary.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from analyst.state import Record

SUMMARY_PROMPT = """You keep a running summary of a conversation between a retail executive and a data analyst assistant.
Update the summary with the turns below. Keep it under 150 words.
Record the narrative only: what the user asked about, which brands, periods and breakdowns, and any stated preferences or corrections.
Do not include any figures, amounts or percentages. Exact numbers are kept elsewhere."""


def new_record(question: str, compact_result: dict, sql: str) -> Record:
    return Record(
        record_id=f"rec_{uuid.uuid4().hex[:8]}",
        question=question,
        sql=sql,
        result_id=compact_result["result_id"],
        data_ts=compact_result["data_ts"],
        columns=compact_result["columns"],
        row_count=compact_result["row_count"],
    )


def split_turns(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    turns: list[list[BaseMessage]] = []
    for message in messages:
        if isinstance(message, HumanMessage) or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def trim_history(
    messages: list[BaseMessage],
    summary: str,
    verbatim_turns: int,
    summarise: Callable[[str, list[BaseMessage]], str],
) -> tuple[list[BaseMessage], str]:
    """Keep the last `verbatim_turns` turns; fold anything older into the summary."""
    turns = split_turns(messages)
    if len(turns) <= verbatim_turns:
        return messages, summary
    old, keep = turns[:-verbatim_turns], turns[-verbatim_turns:]
    folded = [m for turn in old for m in turn]
    return [m for turn in keep for m in turn], summarise(summary, folded)


def summary_messages(previous: str, turns: list[BaseMessage]) -> list[BaseMessage]:
    lines = [f"Current summary: {previous or '(none yet)'}", "", "Turns to add:"]
    for m in turns:
        speaker = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{speaker}: {m.text}")
    return [SystemMessage(SUMMARY_PROMPT), HumanMessage("\n".join(lines))]


def records_block(records: list[Record]) -> str:
    """Compact listing of stored results the agent can reuse by result_id (D1.15)."""
    if not records:
        return "No stored results yet."
    lines = []
    for r in records:
        lines.append(
            f"- {r.result_id}: question \"{r.question}\"; columns {', '.join(r.columns)}; "
            f"{r.row_count} rows; data as of {r.data_ts}; SQL: {' '.join(r.sql.split())}"
        )
    return "\n".join(lines)
