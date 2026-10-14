"""Command line: `chat --user <id>`, `trace <id>` and `doctor`."""

from __future__ import annotations

import functools
import json
import sys
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import typer
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from analyst import doctor as doctor_checks
from analyst.config import ConfigError, User, load_env, load_settings, load_users
from analyst.faults import Faults
from analyst.llm.client import LLMFatalError, LLMUnavailableError
from analyst.observability.tracer import Tracer
from analyst.sql.gateway import GatewayFatalError

app = typer.Typer(add_completion=False, no_args_is_help=True, help="OpsFleet Analyst")
console = Console()

HELP = """Ask a question about sales, customers or products, for example "What was revenue by month this year?".

Commands:
  /help     show this help
  /whoami   show who you are signed in as and which brands you can see
  /retry    pick up a question that stopped because Gemini was unavailable
  /stats    session metrics: business view and engineering view
  /prefs    show your saved preferences and defaults; /prefs reset clears them
  /persona  show which persona version is active
  /reports  list your saved reports
  /download <number>  save a copy of a report from /reports to the reports/ folder
  /exit     leave the chat"""
LLM_DOWN = "Gemini is busy or unreachable right now. Type /retry to pick up where it stopped."
REPAIRABLE = {"PARSE_ERROR", "MULTI_STATEMENT", "UNKNOWN_TABLE", "GEO_TOO_FINE", "NO_ACCESS_BRAND",
              "BQ_INVALID", "COST_LIMIT", "TIMEOUT"}

DOCTOR_HINT = "Run `python -m analyst doctor` to check your setup."


@dataclass
class Session:
    user: User
    graph: object
    thread_id: str
    tracer: Tracer
    reports: object = None  # ReportStore
    profiles: object = None  # ProfileStore
    results: object = None  # ResultStore: the session's post-gateway results (evals compare them)
    gateway: object = None  # the session's SQL gateway (evals run reference SQL through it)
    pending: bool = False  # the last turn stopped on an LLM outage and can resume (D2.13)
    awaiting: str | None = None  # trace id of a turn waiting for a confirmation reply (D3.2, D3.20)

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 50}


@functools.lru_cache(maxsize=4)
def _catalogue(project: str, dataset: str):
    """Brand list and known values, loaded once per process (D2.5); evals build many sessions."""
    from google.cloud import bigquery

    from analyst.sql.diagnosis import load_known_values
    from analyst.sql.gateway import BQ_ERRORS, load_brands, load_curated

    client = bigquery.Client(project=project)
    brands = load_brands(client, dataset)
    try:
        known = load_known_values(client, load_curated(dataset))
    except BQ_ERRORS:
        known = None  # prompts fall back to fixed lists; queries still report their own errors
    return brands, known


def build_session(user_id: str, faults: Faults | None = None, reports_path=None, persona_dir=None) -> Session:
    """`reports_path` and `persona_dir` override the settings, so each eval case gets its own files."""
    from google.cloud import bigquery

    from analyst.config import resolve
    from analyst.golden.retrieve import Retriever
    from analyst.golden.vector_store import VectorStore
    from analyst.graph import Deps, build_graph
    from analyst.llm.client import LLM
    from analyst.llm.embeddings import Embedder, genai_client
    from analyst.sql.brands import BrandMatcher
    from analyst.sql.gateway import Gateway, load_curated
    from analyst.reports.mentions import MentionScanner
    from analyst.store.results import ResultStore
    from analyst.store.sqlite_store import SQLiteProfileStore, SQLiteReportStore

    from langchain_google_genai import ChatGoogleGenerativeAI

    from analyst import faults as fault_injection

    settings = load_settings()
    user = load_users().get(user_id)
    env = load_env()
    faults = faults or Faults()
    store = ResultStore()
    client = bigquery.Client(project=env.google_cloud_project)
    curated = load_curated(settings.bigquery.dataset)
    brand_list, known = _catalogue(env.google_cloud_project, settings.bigquery.dataset)
    brands = BrandMatcher(brand_list)
    if faults.bq_error:
        client = fault_injection.FailingBigQuery(client)  # after the session's own loads, so startup works
    gateway = Gateway(settings.bigquery, client, store, curated, brands, known=known, empty=settings.empty_results)
    factory = fault_injection.chat_factory(ChatGoogleGenerativeAI, faults, settings.models.agent)
    llm = LLM(settings, env.gemini_api_key, chat_factory=factory)
    index_path = resolve(settings.golden.index)
    store_ = VectorStore.load(index_path) if index_path.exists() else None
    embedder = Embedder(settings, env.gemini_api_key,
                        client_factory=fault_injection.embedding_client_factory(genai_client, faults))
    retriever = Retriever(store_, embedder, settings.golden)
    db = reports_path or settings.reports.sqlite_path
    reports = SQLiteReportStore(db)
    reports.purge(settings.reports.purge_after_days)  # D3.23; production runs this as a scheduled job
    profiles = SQLiteProfileStore(db)  # same file as the reports (D4.25)
    graph = build_graph(Deps(settings, llm, gateway, store, brands=brands, known=known, reports=reports,
                             mentions=MentionScanner(brands.all_brands, known), profiles=profiles,
                             retriever=retriever, persona_dir=persona_dir))
    thread_id = uuid.uuid4().hex
    return Session(user, graph, thread_id, Tracer(settings, user, thread_id), reports, profiles, store, gateway)


@contextmanager
def friendly_errors(debug: bool = False) -> Iterator[None]:
    """Turn known failures into plain messages. Never show a traceback unless --debug."""
    try:
        yield
    except (ConfigError, LLMFatalError, GatewayFatalError) as exc:
        console.print(f"[red]{exc}[/red]")
    except LLMUnavailableError:
        console.print(f"[yellow]{LLM_DOWN}[/yellow]")
    except Exception:
        console.print(f"[red]Something went wrong on our side. {DOCTOR_HINT}[/red]")
        if debug:
            console.print(traceback.format_exc(), markup=False)


def whoami(user: User) -> str:
    brands = "all brands" if user.all_brands else ", ".join(user.brands)
    return f"{user.name} (user id {user.id}). Brands: {brands}."


def status_for(update: dict) -> str | None:
    """Status line text for a graph update, so a longer turn never looks frozen (D2.5)."""
    for node, delta in update.items():
        if not isinstance(delta, dict):
            continue  # e.g. "__interrupt__" carries a tuple
        messages = (delta or {}).get("turn_messages") or []
        last = messages[-1] if messages else None
        if node == "agent" and isinstance(last, AIMessage) and last.tool_calls:
            return "Running a query..." if any(c["name"] == "run_sql" for c in last.tool_calls) else "Working..."
        if node == "tools":
            for m in messages:
                if isinstance(m, ToolMessage) and _error_code(m.content) in REPAIRABLE:
                    return "Checking the query..."
            return "Thinking..."
        if node == "output_check" and isinstance(last, HumanMessage):
            return "Checking the answer..."
        if node == "plan" and (delta or {}).get("outline") is not None:
            outline = delta["outline"]
            if not any(sec.source == "new_queries" for sec in outline.sections):
                return f"Writing '{outline.title}' from this chat's results..."  # outline as a status line (D3.2)
            return "Planning the report..."
        if node == "confirm_outline" and (delta or {}).get("decision") == "yes":
            return "Running the report sections..."
        if node == "section" and (delta or {}).get("sections_done"):
            done = delta["sections_done"]
            return f"Finished section {len(done)}: {done[-1].title}. Working..."
        if node == "write":
            return "Checking the report..."
    return None


def _error_code(content) -> str | None:
    try:
        return json.loads(content).get("error", {}).get("code")
    except (TypeError, ValueError, AttributeError):
        return None


def run_turn(session: Session, text: str | None, on_update=None) -> tuple[dict, str | None]:
    """Run one turn quietly, or resume the stopped one when text is None (/retry). Returns the final state and,
    if the turn stopped to ask for a confirmation, the text shown. Used by the chat and by the evals."""
    if text is None:
        payload = None
    elif session.awaiting:
        # A reply to a confirmation resumes the stopped turn, as a new trace that continues it (D2.19).
        payload = Command(resume=text, update={"trace_id": uuid.uuid4().hex[:8], "continues": session.awaiting})
    else:
        payload = {"user_input": text, "user": session.user, "conversation_id": session.thread_id}
    session.pending = False
    session.awaiting = None
    session.tracer.last = None
    try:
        with session.tracer.turn(session.graph, session.config, text) as config:
            for update in session.graph.stream(payload, config, stream_mode="updates"):
                if on_update is not None:
                    on_update(update)
    except LLMUnavailableError:
        session.pending = True  # the checkpoint holds the turn up to the failed step
        raise
    snapshot = session.graph.get_state(session.config)
    interrupts = getattr(snapshot, "interrupts", ()) or ()
    if interrupts:
        session.awaiting = snapshot.values.get("trace_id") or "unknown"
        return snapshot.values, interrupts[0].value["text"]
    return snapshot.values, None


def answer(session: Session, text: str | None) -> None:
    """Run one turn in the chat, with a live status line, and print the answer."""
    with console.status("Thinking...") as status:
        def update(u):
            message = status_for(u)
            if message:
                status.update(message)

        state, asking = run_turn(session, text, update)
    if asking:
        console.print(asking, markup=False)
        return
    console.print(Markdown(state["draft"].text or "(no answer)"))
    for path in state["draft"].chart_paths:
        console.print(f"[green]Chart saved:[/green] {path}")


def show_reports(session: Session) -> list:
    saved = session.reports.list(session.user) if session.reports is not None else []
    if not saved:
        console.print("No saved reports.")
        return []
    table = Table(show_header=True, header_style="bold")
    for col in ("#", "Title", "Saved", "Data as of"):
        table.add_column(col)
    for i, r in enumerate(saved, 1):
        table.add_row(str(i), r.title, r.created_at, r.data_ts)
    console.print(table)
    return saved


def download_numbered(session: Session, arg: str) -> None:
    from analyst.graph import download

    saved = session.reports.list(session.user) if session.reports is not None else []
    if not arg.isdigit() or not 1 <= int(arg) <= len(saved):
        console.print("Give a report number from /reports, for example /download 1.")
        return
    r = saved[int(arg) - 1]
    path = download(r.title, r.body, r.chart_files, load_settings().reports.download_dir)
    console.print(f"Saved a copy to {path}. A downloaded copy is outside the app's access controls.", markup=False)


def show_prefs(session: Session, arg: str) -> None:
    """D4.15: preferences are visible, changeable and resettable."""
    from analyst.config import load_analysis_defaults
    from analyst.preferences import LABELS

    if session.profiles is None:
        console.print("Preferences aren't available in this session.")
        return
    if arg == "reset":
        session.profiles.reset(session.user)
        console.print("Cleared your preferences and defaults. Answers use the system defaults again.")
        return
    if arg:
        console.print("Use /prefs to see your preferences, or /prefs reset to clear them.")
        return
    profile = session.profiles.get(session.user)
    table = Table(show_header=True, header_style="bold")
    for col in ("Preference", "Value", "Source", "Set"):
        table.add_column(col)
    for p in [*profile.presentation.values(), *profile.analysis_defaults.values()]:
        table.add_row(LABELS.get(p.field, p.field), p.value, p.source, p.set_at)
    defaults = load_analysis_defaults(load_settings().profile.analysis_defaults)
    for field, value in defaults.model_dump().items():
        if field not in profile.analysis_defaults:
            table.add_row(LABELS.get(field, field), value, "system default", "")
    console.print(table)


def persona_line() -> str:
    from analyst.persona import load_persona

    p = load_persona(load_settings().persona.dir)
    if p.fallback_from is None:
        return f"Persona {p.version} is active."
    using = f"using {p.version}" if p.version != "none" else "using no persona"
    return f"Persona {p.fallback_from} can't be loaded ({p.error}), so the chat is {using}."


THUMBS = {"u": "up", "d": "down"}


def interactive() -> bool:
    return sys.stdin.isatty()


def ask_thumbs(session: Session, getchar=None) -> None:
    """One key under each answer: u, d or Enter to skip. A thumbs down asks for an optional reason (D4.19, D8.1).
    Only in a terminal, so piped and scripted runs are never blocked."""
    feedback = load_settings().feedback
    last = session.tracer.last
    if (not feedback.thumbs_prompt or not interactive() or last is None or session.awaiting
            or last["outcome"] == "failed"):
        return
    console.print("[grey50]Helpful? u = yes, d = no, Enter = skip[/grey50] ", end="")
    try:
        key = (getchar or typer.getchar)()
    except (EOFError, KeyboardInterrupt):
        key = ""
    thumbs = THUMBS.get(key.lower(), "skip")
    console.print(f"[grey50]{thumbs}[/grey50]")
    reason = None
    if thumbs == "down" and feedback.thumbs_down_reason == "optional":
        try:
            reason = console.input("[grey50]What was wrong? (optional, Enter to skip)[/grey50] ").strip() or None
        except (EOFError, KeyboardInterrupt):
            reason = None
    session.tracer.add_feedback(last["trace_id"], thumbs, reason)


def trace_line(session: Session) -> None:
    """The turn's trace id in grey, under every answer and failure, so it can be quoted (D2.19)."""
    if session.tracer.last:
        console.print(f"[grey50]trace {session.tracer.last['trace_id']}[/grey50]")


def show_stats(session: Session) -> None:
    from analyst.observability import stats
    from analyst.observability.tracer import read_traces

    records = read_traces(session.tracer.path)
    if not records:
        console.print("No turns yet.")
        return
    for table in stats.render(stats.compute(records)):
        console.print(table)
        console.print()


@app.command()
def chat(
    user: str = typer.Option(..., "--user", help="User id from config/users.yaml"),
    debug: bool = typer.Option(False, "--debug", help="Show tracebacks for unexpected errors"),
    simulate_llm_429: bool = typer.Option(False, "--simulate-llm-429", help="Agent's primary model returns 429"),
    simulate_llm_down: bool = typer.Option(False, "--simulate-llm-down", help="Every Gemini model returns 503"),
    simulate_bq_error: bool = typer.Option(False, "--simulate-bq-error", help="Every BigQuery query fails"),
    simulate_embedding_down: bool = typer.Option(False, "--simulate-embedding-down",
                                                 help="Every embedding call fails, so no trios are retrieved"),
) -> None:
    """Chat with the analyst as a configured user."""
    session = None
    with friendly_errors(debug):
        configured = load_settings().faults
        faults = Faults(
            llm_429=simulate_llm_429 or configured.simulate_llm_429,
            llm_down=simulate_llm_down or configured.simulate_llm_down,
            bq_error=simulate_bq_error or configured.simulate_bq_error,
            embedding_down=simulate_embedding_down or configured.simulate_embedding_down,
        )
        session = build_session(user, faults)
    if session is None:
        raise typer.Exit(1)
    console.print(f"Signed in as {whoami(session.user)} Type /help for help, /exit to leave.")
    if faults.active:
        console.print(f"[yellow]Simulating: {', '.join(faults.active)}.[/yellow]")
    while True:
        try:
            text = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not text:
            continue
        if text in ("/exit", "/quit"):
            break
        if text == "/help":
            console.print(HELP, markup=False)
        elif text == "/whoami":
            console.print(whoami(session.user), markup=False)
        elif text == "/stats":
            show_stats(session)
        elif text.startswith("/prefs"):
            show_prefs(session, text.removeprefix("/prefs").strip())
        elif text == "/persona":
            console.print(persona_line(), markup=False)
        elif text == "/reports":
            show_reports(session)
        elif text.startswith("/download"):
            download_numbered(session, text.removeprefix("/download").strip())
        elif text == "/retry":
            if not session.pending:
                console.print("Nothing to retry.")
            else:
                with friendly_errors(debug):
                    answer(session, None)
                trace_line(session)
                ask_thumbs(session)
        elif text.startswith("/"):
            console.print(f"Unknown command {text}. Type /help.", markup=False)
        else:
            with friendly_errors(debug):
                answer(session, text)
            trace_line(session)
            ask_thumbs(session)


golden_app = typer.Typer(no_args_is_help=True, help="Golden Bucket: index the seeded trios, calibrate the threshold.")
app.add_typer(golden_app, name="golden")


@golden_app.command("index")
def golden_index(debug: bool = typer.Option(False, "--debug", help="Show tracebacks for unexpected errors")) -> None:
    """Strip, tag, describe and embed every trio in data/golden/trios/ (D4.3, D4.4, D4.7, D4.11).
    The index is committed, so reviewers don't need to run this (D8.3)."""
    from google.cloud import bigquery

    from analyst.config import ROOT, resolve
    from analyst.golden.index import build_index, load_trios, write_index
    from analyst.llm.client import LLM
    from analyst.llm.embeddings import Embedder
    from analyst.sql.brands import BrandMatcher
    from analyst.sql.gateway import load_brands

    done = False
    with friendly_errors(debug):
        settings, env = load_settings(), load_env()
        brands = BrandMatcher(load_brands(bigquery.Client(project=env.google_cloud_project), settings.bigquery.dataset))
        trios = load_trios(resolve(settings.golden.trios_dir))
        with console.status(f"Indexing {len(trios)} trios..."):
            index, dropped = build_index(trios, ROOT, LLM(settings, env.gemini_api_key),
                                         Embedder(settings, env.gemini_api_key), brands)
        write_index(index, resolve(settings.golden.index))
        table = Table(show_header=True, header_style="bold")
        for col in ("Trio", "Brands", "Description", "Dropped"):
            table.add_column(col)
        for t in index.trios:
            table.add_row(t.trio_id, ", ".join(t.brands) or "(company-wide)", t.description,
                          ", ".join(dropped.get(t.trio_id, [])))
        console.print(table)
        console.print(f"Wrote {len(index.trios)} trios to {settings.golden.index} "
                      f"({index.model}, {index.dimensions} dimensions).", markup=False)
        done = True
    if not done:
        raise typer.Exit(1)


@golden_app.command("calibrate")
def golden_calibrate(
    out: str = typer.Option("", "--out", help="Also write the result as markdown, e.g. docs/golden-calibration.md"),
    debug: bool = typer.Option(False, "--debug", help="Show tracebacks for unexpected errors"),
) -> None:
    """Score the labelled set against every trio and suggest a threshold (D4.9). Set it in settings.yaml by hand."""
    from pathlib import Path

    from analyst.config import resolve
    from analyst.golden.calibrate import calibrate, load_labelled_set, markdown, summary, sweep
    from analyst.golden.vector_store import VectorStore
    from analyst.llm.embeddings import Embedder

    done = False
    with friendly_errors(debug):
        settings, env = load_settings(), load_env()
        store = VectorStore.load(resolve(settings.golden.index))
        labelled = load_labelled_set(resolve(settings.golden.trios_dir).parent / "labelled_set.yaml")
        embedder = Embedder(settings, env.gemini_api_key)
        with console.status("Embedding the labelled questions..."):
            cal = calibrate(labelled, store, lambda texts: embedder.embed(texts, "query"))
        table = Table(show_header=True, header_style="bold")
        for col in ("Group", "Pairs", "Min", "P5", "Median", "P95", "Max"):
            table.add_column(col)
        for name, pairs in (("Correct pairs", cal.correct), ("Everything else", cal.other),
                            ("Related (excluded)", cal.related)):
            if pairs:
                s = summary([p.score for p in pairs])
                table.add_row(name, str(s["pairs"]), *(f"{s[k]:.3f}" for k in ("min", "p5", "median", "p95", "max")))
        console.print(table)
        swept = Table(show_header=True, header_style="bold")
        for col in ("Threshold", "Correct pairs passing", "Wrong pairs passing"):
            swept.add_column(col)
        for t, r, w in sweep(cal):
            swept.add_row(f"{t:.3f}", f"{r:.0%}", str(w))
        console.print(swept)
        where = "just above the best wrong pair (groups overlap)" if cal.overlap else "in the middle of the gap"
        console.print(f"Suggested threshold: {cal.suggested:.3f}, {where}. At that value "
                      f"{cal.recall(cal.suggested):.0%} of correct pairs pass and "
                      f"{cal.wrong_above(cal.suggested)} wrong pairs do.", markup=False)
        console.print(f"Current setting: golden.threshold = {settings.golden.threshold}. "
                      "Write the chosen value to config/settings.yaml.", markup=False)
        if out:
            path = Path(out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(markdown(cal, store.model, store.dimensions))
            console.print(f"Wrote {path}.", markup=False)
        done = True
    if not done:
        raise typer.Exit(1)


persona_app = typer.Typer(no_args_is_help=True, help="Persona: smoke-test the active version.")
app.add_typer(persona_app, name="persona")


@persona_app.command("test")
def persona_test(debug: bool = typer.Option(False, "--debug", help="Show tracebacks for unexpected errors")) -> None:
    """Run evals/persona_smoke.yaml against the active persona and print pass or fail (D4.23)."""
    from analyst.config import ROOT
    from analyst.persona import load_persona
    from analyst.persona_smoke import load_cases, run

    results = None
    with friendly_errors(debug):
        settings = load_settings()
        persona = load_persona(settings.persona.dir)
        console.print(persona_line(), markup=False)
        cases = load_cases(ROOT / "evals" / "persona_smoke.yaml")
        rpm = settings.evals.requests_per_minute
        with console.status(f"Running {len(cases)} cases against persona {persona.version}..."):
            results = run(cases, build_session, rpm)
    if results is None:
        raise typer.Exit(1)
    table = Table(show_header=True, header_style="bold")
    for col in ("Case", "User", "Outcome", "Result", "Trace"):
        table.add_column(col)
    for r in results:
        verdict = "[green]pass[/green]" if r.passed else "[red]FAIL[/red]: " + "; ".join(r.failures)
        table.add_row(r.case.id, r.case.user, f"{r.outcome} ({r.reason})", verdict, r.trace_id)
    console.print(table)
    failed = sum(not r.passed for r in results)
    console.print(f"{len(results) - failed} of {len(results)} passed.", markup=False)
    if failed:
        raise typer.Exit(1)


@app.command("eval")
def eval_cmd(
    only: list[str] = typer.Option([], "--only", help="Run just these components (repeatable): classifier, "
                                                      "delete_reply, retrieval, sql, e2e, red_team"),
    repeats: int = typer.Option(0, "--repeats", help="Runs per red-team attack (default from settings)"),
    rpm: int = typer.Option(0, "--rpm", help="Gemini requests per minute (default from settings; 8 suits the free tier)"),
    resume: str = typer.Option("", "--resume", help="Carry on an interrupted run by its run id"),
    baseline: bool = typer.Option(False, "--baseline", help="Also write evals/results/baseline.json (scores only)"),
    trigger: str = typer.Option("manual", "--trigger", help="Recorded with the run: manual, commit, pr, nightly"),
    debug: bool = typer.Option(False, "--debug", help="Show tracebacks for unexpected errors"),
) -> None:
    """Run the eval sets (D5.1, D5.2) and write eval_runs and eval_cases (D6.18). Exits 1 if the red-team
    safety outcome is under 100% (D5.12)."""
    from analyst.evals.runner import COMPONENTS, run

    unknown = [c for c in only if c not in COMPONENTS]
    if unknown:
        console.print(f"Unknown component {', '.join(unknown)}. Choose from {', '.join(COMPONENTS)}.", markup=False)
        raise typer.Exit(2)
    components = only or COMPONENTS
    row = None

    def progress(comp, result):
        mark = "[green]pass[/green]" if result.passed else "[red]FAIL[/red]"
        trace = f" [grey50]trace {result.trace_id}[/grey50]" if result.trace_id else ""
        note = f" [grey50]{result.notes[:150]}[/grey50]" if result.notes and not result.passed else ""
        console.print(f"{comp:<12} {result.case_id:<38} {mark}{trace}{note}")

    with friendly_errors(debug):
        settings = load_settings()
        row = run(settings, components, rpm=rpm or None, repeats=repeats or None, resume=resume or None,
                  baseline=baseline, trigger=trigger, on_case=progress)
    if row is None:
        raise typer.Exit(1)
    table = Table(title=f"Eval run {row['run_id']} ({row['git_sha']})", title_justify="left", header_style="bold")
    for col in ("Component", "Score"):
        table.add_column(col)
    for comp in components:
        if comp in row["scores"]:
            table.add_row(comp, f"{row['scores'][comp]:.0%}")
    if row["red_team_outcome"] is not None:
        table.add_row("red_team safety (blocking)", f"{row['red_team_outcome']:.0%}")
        table.add_row("red_team behaviour", f"{row['red_team_behaviour']:.0%}")
    console.print(table)
    console.print(f"{row['cases']} cases, {row['failed']} failed. Results in {settings.evals.results_dir}/. "
                  "Open a failed case with `python -m analyst trace <id>`.", markup=False)
    if baseline:
        console.print(f"Baseline written to {settings.evals.results_dir}/baseline.json.", markup=False)
    if row["red_team_outcome"] is not None and row["red_team_outcome"] < 1.0:
        console.print("[red]Red-team safety outcome is under 100%. This blocks a release (D5.12).[/red]")
        raise typer.Exit(1)


@app.command("trace")
def trace_cmd(
    trace_id: str = typer.Argument(..., help="Trace id shown under an answer; a unique prefix is enough"),
    full: bool = typer.Option(False, "--full", help="Add the complete messages and tool results"),
) -> None:
    """Show one turn's trace as a tree."""
    from analyst.observability import viewer

    try:
        records = viewer.find(load_settings().tracing.dir, trace_id)
    except (viewer.TraceNotFound, ConfigError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    for record in records:
        console.print(viewer.render(record, full))


@app.command()
def doctor() -> None:
    """Check credentials, BigQuery, Gemini and config."""
    results = doctor_checks.run_checks()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Detail")
    for r in results:
        table.add_row(r.name, "[green]OK[/green]" if r.ok else "[red]FAIL[/red]",
                      r.detail if r.ok else f"{r.detail}\n[bold]Fix:[/bold] {r.fix}")
    console.print(table)
    if not all(r.ok for r in results):
        raise typer.Exit(1)
