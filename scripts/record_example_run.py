"""Record the example run (deliverable 2d) into docs/example-run.md.

Drives real sessions (Gemini and BigQuery) through the same CLI functions a user
types into, with the CLI's console swapped for a recording one, so the
transcript is exactly what the terminal shows. About 150 Gemini requests.

    python scripts/record_example_run.py

Each section uses its own temporary SQLite file and persona folder, so the
repo's reports.db and config/persona/ are never touched. Traces go to traces/
as usual, so every trace id in the transcript opens with `python -m analyst trace <id>`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyst import cli  # noqa: E402
from analyst.config import load_settings  # noqa: E402
from analyst.evals.harness import SeedReport, new_session  # noqa: E402
from analyst.faults import Faults  # noqa: E402
from analyst.observability import viewer  # noqa: E402

WIDTH = 110
WORK = Path(tempfile.mkdtemp(prefix="analyst-example-"))
sections: list[tuple[str, str, str]] = []  # (title, what it shows, transcript)


def recording() -> Console:
    cli.console = Console(record=True, width=WIDTH, force_terminal=False, color_system=None)
    return cli.console


def section(title: str, note: str):
    """Use as `with section(...)`: everything the CLI prints inside is recorded under this title."""
    class _Section:
        def __enter__(self):
            self.console = recording()
            return self.console

        def __exit__(self, *exc):
            sections.append((title, note, self.console.export_text().rstrip()))
            return False

    return _Section()


def session(user_id: str, key: str, faults: Faults | None = None, persona_dir: Path | None = None, seed=()):
    return new_session(user_id, WORK / key, faults=faults, seed=[SeedReport(**r) for r in seed],
                       persona_dir=persona_dir)


def say(s, text: str) -> None:
    cli.console.print(f"you> {text}", markup=False)
    with cli.friendly_errors():
        cli.answer(s, text)
    cli.trace_line(s)


def command(text: str) -> None:
    cli.console.print(f"you> {text}", markup=False)


def show_trace(s) -> None:
    trace_id = s.tracer.last["trace_id"]
    cli.console.print(f"\n$ python -m analyst trace {trace_id}", markup=False)
    for record in viewer.find(load_settings().tracing.dir, trace_id):
        cli.console.print(viewer.render(record))


def main() -> None:
    started = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    with section("Schema question, analysis and follow-up", "The light path answers a question about the data "
                 "with no query. An analysis answer opens with its assumptions line. A follow-up reuses the "
                 "conversation, and a chart is drawn from a stored result without querying again."):
        ceo = session("ceo", "analysis")
        say(ceo, "What data do you have, and what kind of questions can I ask?")
        say(ceo, "What was revenue by month in 2025?")
        say(ceo, "Just the last three months of 2025, and make it a chart.")

    with section("A multi-step why question, with retrieved worked examples", "A brand manager asks why. "
                 "The trace shows the retrieval step (the trio used and its score, and trios outside their brands "
                 "dropped), each agent step, and every query with its brand scope."):
        ck = session("bm_calvin", "why")
        say(ck, "Why are Texas customers spending less than Californians?")
        show_trace(ck)

    with section("Report with outline confirmation, redo and save", "The outline needs new queries, so it "
                 "waits for a yes. The report has an assumptions line, an as-of time, findings, anchored action "
                 "items and next questions. \"Make it shorter\" creates a new version of the same report."):
        ceo = session("ceo", "report")
        say(ceo, "Create a Q1 2026 report with action items for Q2.")
        say(ceo, "yes")
        say(ceo, "Make it shorter.")
        if ceo.awaiting:
            say(ceo, "yes")
        say(ceo, "Save it.")
        command("/reports")
        cli.show_reports(ceo)

    with section("Scoped user, out-of-scope brand and a PII request", "A brand manager asking about another "
                 "brand gets their own data with the gap flagged, never \"no sales\". A request for customers' "
                 "personal details is declined in neutral words."):
        ck = session("bm_calvin", "scope")
        say(ck, "How did Calvin Klein do in 2025 compared with Levi's?")
        say(ck, "List the email addresses of our ten best customers.")

    with section("Deleting reports, with narrowing", "Code resolves the request to the user's own reports and "
                 "shows them. A reply with a condition narrows the list for a fresh yes. Only the reports shown "
                 "are deleted, softly, with an audit row."):
        lv = session("bm_levis", "delete", seed=[
            {"owner": "bm_levis", "title": "Levi's jeans review", "mentions": ["Levi's", "Jeans"]},
            {"owner": "bm_levis", "title": "Outerwear review",
             "mentions": ["Canada Goose", "Levi's", "Outerwear & Coats"]}])
        say(lv, "Delete all reports mentioning Levi's.")
        say(lv, "only the jeans one")
        say(lv, "go for it")
        command("/reports")
        cli.show_reports(lv)
        audit = lv.reports.audit()
        cli.console.print(f"\naudit table: {[a.model_dump(include={'user_id', 'action', 'titles', 'trace_id'}) for a in audit]}",
                          markup=False)

    for flag, faults, question, note in [
        ("--simulate-llm-429", Faults(llm_429=True), "What was revenue in 2025?",
         "The agent's main model returns 429. Retries run out, the backup model answers, and the trace records "
         "the fallback."),
        ("--simulate-llm-down", Faults(llm_down=True), "What was revenue in 2025?",
         "Every Gemini model returns 503. The classifier fails open, the agent's chain fails, and the chat says "
         "so plainly. /retry resumes from the checkpoint once Gemini is back."),
        ("--simulate-bq-error", Faults(bq_error=True), "What was revenue in 2025?",
         "Every BigQuery query fails. Retries run out, further queries are refused for the turn, and the agent "
         "says what it couldn't fetch."),
        ("--simulate-embedding-down", Faults(embedding_down=True), "Why are Texas customers spending less "
         "than Californians?", "The embedding API fails. The turn gets no worked examples, the agent works from "
         "the glossary, and the trace shows trios_unavailable."),
    ]:
        with section(f"Fault injection: {flag}", note):
            cli.console.print(f"$ python -m analyst chat --user ceo {flag}", markup=False)
            s = session("ceo", f"fault{flag}", faults=faults)
            say(s, question)
            if flag == "--simulate-embedding-down":
                show_trace(s)

    with section("A learned preference", "Asking for tables three times makes tables the default, marked as "
                 "inferred. A stated preference would be saved at once and would win over an inferred one."):
        ck = session("bm_calvin", "prefs")
        for q in ["Show 2025 revenue by category as a table.", "Now 2025 orders by country, in a table.",
                  "And 2025 returns by department as a table please."]:
            say(ck, q)
        command("/prefs")
        cli.show_prefs(ck, "")

    with section("Persona edit mid-session, and a broken persona", "The persona file is re-read every turn, so "
                 "pointing `active` at v2 changes the next answer's tone with no restart. An empty persona file "
                 "falls back to the previous version, and the trace says so."):
        pdir = WORK / "persona"
        shutil.copytree(ROOT / "config" / "persona", pdir)
        (pdir / "active").write_text("v1\n")
        ceo = session("ceo", "persona", persona_dir=pdir)
        say(ceo, "What was revenue in September 2026?")
        cli.console.print("\n# config/persona/active changed from v1 to v2\n", markup=False)
        (pdir / "active").write_text("v2\n")
        say(ceo, "And in August 2026?")
        cli.console.print("\n# config/persona/v2.md emptied by mistake\n", markup=False)
        (pdir / "v2.md").write_text("")
        say(ceo, "How many orders did we have in August 2026?")
        cli.console.print(f"trace persona: {ceo.tracer.last.get('persona')}", markup=False)

    with section("Thumbs down with a reason, and /stats", "In a terminal, a one-key prompt follows each answer. "
                 "Here the rating is recorded through the same call the prompt uses. It is stored next to the "
                 "trace and counted in the engineering view of /stats."):
        ck = session("bm_calvin", "thumbs")
        say(ck, "What was our revenue in 2025?")
        cli.console.print("Helpful? u = yes, d = no, Enter = skip d", markup=False)
        cli.console.print("What was wrong? (optional, Enter to skip) I wanted it split by quarter", markup=False)
        ck.tracer.add_feedback(ck.tracer.last["trace_id"], "down", "I wanted it split by quarter")
        command("/stats")
        cli.show_stats(ck)
        show_trace(ck)

    with section("python -m analyst eval", "One eval set, as a reviewer on the free tier would run it. The full "
                 "baseline is in evals/results/baseline.json and docs/evaluation.md."):
        cli.console.print("$ python -m analyst eval --only delete_reply --rpm 60", markup=False)
        out = subprocess.run([sys.executable, "-m", "analyst", "eval", "--only", "delete_reply", "--rpm", "60"],
                             cwd=ROOT, capture_output=True, text=True, env={**os.environ, "COLUMNS": str(WIDTH)})
        cli.console.print(out.stdout.rstrip(), markup=False)

    lines = [
        "# Example run",
        "",
        f"Recorded {started} with `python scripts/record_example_run.py`, on the real models and data. Each block "
        "is exactly what the terminal showed. Trace ids open with `python -m analyst trace <id>` on the machine "
        "that recorded them.",
        "",
        "Contents:",
        "",
        *[f"{i}. [{title}](#{i}-{_anchor(title)})" for i, (title, _, _) in enumerate(sections, 1)],
    ]
    for i, (title, note, transcript) in enumerate(sections, 1):
        lines += ["", f"## {i}. {title}", "", note, "", "```text", transcript, "```"]
    (ROOT / "docs" / "example-run.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote docs/example-run.md ({len(sections)} sections). Work files in {WORK}.")


def _anchor(title: str) -> str:
    """GitHub's heading anchor: lowercase, punctuation dropped, spaces to hyphens."""
    return re.sub(r"[^a-z0-9 -]", "", title.lower()).replace(" ", "-")


if __name__ == "__main__":
    main()
