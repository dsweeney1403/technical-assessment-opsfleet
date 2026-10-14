"""`python -m analyst trace <id>`: one turn as a tree (D2.20).

Compact by default: steps, codes, SQL before and after the rewrite, rows,
latency and tokens, which shows where a turn went wrong. `--full` adds the
complete messages in and out of each LLM call and each tool result, which
shows why the model did it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rich.console import Group
from rich.text import Text
from rich.tree import Tree

from analyst.observability.tracer import read_traces


class TraceNotFound(Exception):
    pass


def find(directory: Path, trace_id: str) -> list[dict]:
    """Every attempt of one trace. A unique prefix of the id is enough."""
    prefix = trace_id.strip().lower()
    if not prefix:
        raise TraceNotFound("Give a trace id, e.g. `python -m analyst trace 7f3a`.")
    matches = [r for path in sorted(Path(directory).glob("*.jsonl")) for r in read_traces(path)
               if r["trace_id"].startswith(prefix)]
    ids = sorted({r["trace_id"] for r in matches})
    if not ids:
        raise TraceNotFound(f"No trace {trace_id} in {directory}/.")
    if len(ids) > 1:
        raise TraceNotFound(f"Trace id {trace_id} is ambiguous: {', '.join(ids)}. Give more characters.")
    return sorted(matches, key=lambda r: r["attempt"])


def _ms(span: dict) -> int:
    return round(((span.get("end_ns") or span["start_ns"]) - span["start_ns"]) / 1e6)


def _duration(ms: int) -> str:
    return f"{ms} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def _mb(n: int | None) -> str | None:
    return None if n is None else f"{n / 1e6:.1f} MB"


def _line(name: str, *parts, style: str = "") -> Text:
    text = Text(name, style="bold " + style if style else "bold")
    detail = " · ".join(str(p) for p in parts if p not in (None, "", []))
    if detail:
        text.append("  " + detail)
    return text


def _block(label: str, body: str, style: str = "") -> Text:
    text = Text(f"{label}: ", style="cyan")
    text.append(body, style=style)
    return text


def _message_text(m: dict) -> str:
    body = m.get("content") or ""
    if m.get("tool_calls"):
        calls = "; ".join(f"{c['name']}({json.dumps(c['args'], default=str)})" for c in m["tool_calls"])
        body = f"{body}\n[tool calls] {calls}".strip()
    return body


def _label(span: dict) -> Text:
    a, ms = span["attributes"], _duration(_ms(span))
    error = "red" if span["status"] == "error" else ""
    name = span["name"]
    if name == "classify":
        scope = f"out of scope: {', '.join(a['out_of_scope_brands'])}" if a.get("out_of_scope_brands") else None
        unsure = "unsure" if a.get("confident") is False else None
        return _line("classify", a.get("verdict"), unsure, scope, f"decline {a['decline']}" if a.get("decline") else None, ms)
    if name == "agent_step":
        return _line(f"agent step {a.get('step')}", ms, style=error)
    if name == "llm_call":
        tokens = a.get("tokens", {}).get("total")
        return _line(f"llm {a.get('role')}", a.get("model") or "no model answered", f"{tokens} tokens", ms,
                     f"retries {a['retries']}" if a.get("retries") else None,
                     "fallback used" if a.get("fallback_used") else None,
                     f"breaker opened {a['breaker_opens']}x" if a.get("breaker_opens") else None,
                     a.get("error"), style=error)
    if name == "tool_call":
        return _line(a.get("tool", "tool"), a.get("code") or "ok", "handled in code" if a.get("handled_in_code") else None,
                     ms, style="red" if a.get("code") else "")
    if name == "gateway":
        diag = a.get("diagnosis") or {}
        diag_text = diag.get("structural") or ("filter problems" if diag.get("filter_problems") else None)
        return _line("gateway", a.get("code") or f"{a.get('rows')} rows", _mb(a.get("bytes_billed")),
                     f"BigQuery retries {a['transient_retries']}" if a.get("transient_retries") else None,
                     f"diagnosis: {diag_text}" if diag_text else None, ms, style=error)
    if name == "output_check":
        return _line("output check", f"leaks blocked: {len(a.get('leak_hits', []))}",
                     f"numbers removed: {', '.join(a['numbers_removed']) if a.get('numbers_removed') else 'none'}",
                     "grounding retry" if a.get("grounding_retry") else None, ms)
    if name == "decline":
        return _line("decline", ms)
    return _line(name.replace("_", " "), ms, style=error)


def _details(span: dict, full: bool) -> list[Text]:
    """Lines under a span: SQL always, messages and results with --full."""
    a, name = span["attributes"], span["name"]
    out = []
    if name == "gateway":
        out.append(_block("sql in", a.get("sql_in") or ""))
        if a.get("sql_rewritten"):
            out.append(_block("sql rewritten", a["sql_rewritten"], "dim"))
        if a.get("diagnosis"):
            out.append(_block("diagnosis", json.dumps(a["diagnosis"], default=str)))
    elif name == "tool_call" and a.get("tool") != "run_sql":
        out.append(_block("args", json.dumps(a.get("args"), default=str)))
    elif name == "decline" and a.get("text"):
        out.append(_block("text", a["text"]))
    elif name == "output_check" and a.get("leak_hits"):
        out.append(_block("leaks", "; ".join(a["leak_hits"]), "red"))
    if full:
        if name == "llm_call":
            for m in a.get("messages_in", []):
                out.append(_block(f"in {m['role']}", _message_text(m)))
            output = a.get("output")
            if output is not None:
                out.append(_block("out", _message_text(output) if "role" in output else json.dumps(output), "green"))
        elif name == "tool_call" and "result" in a:
            out.append(_block("result", a["result"], "green"))
    return out


def render(record: dict, full: bool = False) -> Tree:
    spans = record["spans"]
    root = spans[0]
    children = defaultdict(list)
    for s in spans[1:]:
        children[s["parent_span_id"]].append(s)
    user = record.get("user", {})
    header = Text(f"trace {record['trace_id']}", style="bold")
    header.append(f"  attempt {record['attempt']} · {user.get('id')} ({user.get('role')}) · {record['started_at']}"
                  f" · conversation {record['conversation_id']}", style="dim")
    tree = Tree(Group(header, _block("question", record.get("user_input") or "")))

    def add(parent: Tree, span: dict) -> None:
        node = parent.add(_label(span))
        for line in _details(span, full):
            node.add(line)
        for child in sorted(children[span["span_id"]], key=lambda s: s["start_ns"]):
            add(node, child)

    for span in sorted(children[root["span_id"]], key=lambda s: s["start_ns"]):
        add(tree, span)

    totals = record["totals"]
    style = {"answered": "green", "partial": "yellow", "declined": "magenta", "failed": "red"}[record["outcome"]]
    outcome = tree.add(_line("outcome", f"{record['outcome']} ({record['reason']})", f"{totals['tokens']} tokens",
                             _mb(totals["bytes"]), _duration(totals["latency_ms"]), f"{totals['tool_calls']} tool calls",
                             "fallback used" if totals.get("fallback_used") else None, style=style))
    if record.get("error"):
        outcome.add(_block("error", record["error"], "red"))
    for event in record.get("security_events", []):
        outcome.add(_block(f"security event {event['code']}", f"{event['source']}: {event.get('detail', '')}", "red"))
    if full and record.get("answer"):
        outcome.add(_block("answer", record["answer"], "green"))
    return tree
