"""Report library tools (D3.13, D3.26). The signed-in user is bound by code; the model never passes an owner.

`list_reports`, `search_reports` and `get_report` are read-only. `propose_delete`
is the model's only delete-related tool: it deletes nothing. The graph's tools
node resolves its filter in code and hands the matches to the confirmation step.
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from analyst.config import User
from analyst.store.base import ReportStore, ReportVersion

OPENED_NOTE = ("The app shows this saved report to the user below your reply, exactly as saved. "
               "Introduce it in one line. Don't restate its figures.")


class DeleteFilter(BaseModel):
    type: Literal["mentions", "conversation"] = Field(
        description="mentions: reports that mention a brand, category, department or place. "
                    "conversation: reports made in this conversation.")
    value: str = Field("", description="For mentions: the brand, category, department or place, as the user said it.")


def summary(r: ReportVersion) -> dict:
    return {"report_id": r.report_id, "title": r.title, "saved_at": r.created_at, "data_as_of": r.data_ts,
            "mentions": r.mentions[:10]}


def make_report_tools(store: ReportStore, user: User) -> dict[str, StructuredTool]:
    def list_reports() -> str:
        return json.dumps({"reports": [summary(r) for r in store.list(user)]})

    def search_reports(term: str) -> str:
        t = term.lower().strip()
        found = {r.report_id: r for r in store.match_mentions(user, term)}
        for r in store.list(user):
            if t and t in r.title.lower():
                found.setdefault(r.report_id, r)
        return json.dumps({"reports": [summary(r) for r in found.values()]})

    def get_report(report_id: str) -> str:
        r = store.get(user, report_id)
        if r is None:
            return json.dumps({"error": {"message": f"No saved report {report_id}.", "hint": "Use list_reports."}})
        return json.dumps({**summary(r), "note": OPENED_NOTE})

    def propose_delete(type: Literal["mentions", "conversation"], value: str = "") -> str:
        return json.dumps({"message": "Handled by the app."})  # the graph's tools node resolves it in code

    return {
        "list_reports": StructuredTool.from_function(
            list_reports, name="list_reports", description="List the user's saved reports, newest first."),
        "search_reports": StructuredTool.from_function(
            search_reports, name="search_reports",
            description="Find the user's saved reports by a brand, category, place or title word."),
        "get_report": StructuredTool.from_function(
            get_report, name="get_report", description="Open one saved report by report_id. The app shows it."),
        "propose_delete": StructuredTool.from_function(
            propose_delete, name="propose_delete", args_schema=DeleteFilter,
            description="Propose deleting the user's reports that match a filter. Deletes nothing: the app "
                        "shows the matching reports and asks the user to confirm."),
    }
