"""`make_chart` tool: renders a stored result to a PNG. Takes a result ID, never raw data (D1.12)."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402

from analyst.store.results import ResultNotFound, ResultStore  # noqa: E402

DESCRIPTION = (
    "Draw a bar or line chart from a stored result. Give the result_id, the x column and one or "
    "more y columns. Returns the chart file path, or an error naming the valid columns."
)


def render_chart(
    store: ResultStore,
    out_dir: Path,
    result_id: str,
    chart_type: Literal["bar", "line"],
    x: str,
    y: list[str],
    title: str,
) -> dict:
    try:
        stored = store.get(result_id)
    except ResultNotFound as exc:
        return {"error": {"message": str(exc), "hint": "Use a result_id from the stored results."}}
    df = stored.df
    missing = [c for c in [x, *y] if c not in df.columns]
    if missing:
        return {"error": {"message": f"Unknown columns: {', '.join(missing)}.",
                          "hint": f"Columns in {result_id}: {', '.join(df.columns)}."}}
    if df.empty:
        return {"error": {"message": f"{result_id} has no rows to chart.", "hint": ""}}
    fig, ax = plt.subplots(figsize=(9, 5))
    df.plot(x=x, y=y, kind=chart_type, ax=ax, legend=len(y) > 1)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y[0] if len(y) == 1 else "")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "chart"
    path = out_dir / f"{slug}-{uuid.uuid4().hex[:6]}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"chart_path": str(path), "result_id": result_id}


def make_chart_tool(store: ResultStore, out_dir: Path) -> StructuredTool:
    def make_chart(result_id: str, chart_type: Literal["bar", "line"], x: str, y: list[str], title: str) -> str:
        return json.dumps(render_chart(store, out_dir, result_id, chart_type, x, y, title))

    return StructuredTool.from_function(make_chart, name="make_chart", description=DESCRIPTION)
