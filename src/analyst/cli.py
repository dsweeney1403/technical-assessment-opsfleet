"""Command line: `chat --user <id>`, `trace <id>` and `doctor`."""

from __future__ import annotations

import json
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
    pending: bool = False  # the last turn stopped on an LLM outage and can resume (D2.13)
    awaiting: str | None = None  # trace id of a turn waiting for a confirmation reply (D3.2, D3.20)

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 50}


def build_session(user_id: str, faults: Faults | None = None) -> Session:
    from google.cloud import bigquery

    from analyst.graph import Deps, build_graph
    from analyst.llm.client import LLM
    from analyst.sql.brands import BrandMatcher
    from analyst.sql.diagnosis import load_known_values
    from analyst.sql.gateway import BQ_ERRORS, Gateway, load_brands, load_curated
    from analyst.reports.mentions import MentionScanner
    from analyst.store.results import ResultStore
    from analyst.store.sqlite_store import SQLiteReportStore

    from langchain_google_genai import ChatGoogleGenerativeAI

    from analyst import faults as fault_injection

    settings = load_settings()
    user = load_users().get(user_id)
    env = load_env()
    faults = faults or Faults()
    store = ResultStore()
    client = bigquery.Client(project=env.google_cloud_project)
    curated = load_curated(settings.bigquery.dataset)
    brands = BrandMatcher(load_brands(client, settings.bigquery.dataset))
    try:
        known = load_known_values(client, curated)  # once per session (D2.5)
    except BQ_ERRORS:
        known = None  # prompts fall back to fixed lists; queries still report their own errors
    if faults.bq_error:
        client = fault_injection.FailingBigQuery(client)  # after the session's own loads, so startup works
    gateway = Gateway(settings.bigquery, client, store, curated, brands, known=known, empty=settings.empty_results)
    factory = fault_injection.chat_factory(ChatGoogleGenerativeAI, faults, settings.models.agent)
    llm = LLM(settings, env.gemini_api_key, chat_factory=factory)
    reports = SQLiteReportStore(settings.reports.sqlite_path)
    reports.purge(settings.reports.purge_after_days)  # D3.23; production runs this as a scheduled job
    graph = build_graph(Deps(settings, llm, gateway, store, brands=brands, known=known, reports=reports,
                             mentions=MentionScanner(brands.all_brands, known)))
    thread_id = uuid.uuid4().hex
    return Session(user, graph, thread_id, Tracer(settings, user, thread_id), reports)


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


def answer(session: Session, text: str | None) -> None:
    """Run one turn, or resume the stopped one when text is None (/retry)."""
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
        with (session.tracer.turn(session.graph, session.config, text) as config,
              console.status("Thinking...") as status):
            for update in session.graph.stream(payload, config, stream_mode="updates"):
                message = status_for(update)
                if message:
                    status.update(message)
    except LLMUnavailableError:
        session.pending = True  # the checkpoint holds the turn up to the failed step
        raise
    snapshot = session.graph.get_state(session.config)
    interrupts = getattr(snapshot, "interrupts", ()) or ()
    if interrupts:
        session.awaiting = snapshot.values.get("trace_id") or "unknown"
        console.print(interrupts[0].value["text"], markup=False)
        return
    state = snapshot.values
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


def persona_line() -> str:
    from analyst.persona import load_persona

    p = load_persona(load_settings().persona.dir)
    if p.fallback_from is None:
        return f"Persona {p.version} is active."
    using = f"using {p.version}" if p.version != "none" else "using no persona"
    return f"Persona {p.fallback_from} can't be loaded ({p.error}), so the chat is {using}."


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
) -> None:
    """Chat with the analyst as a configured user."""
    session = None
    with friendly_errors(debug):
        configured = load_settings().faults
        faults = Faults(
            llm_429=simulate_llm_429 or configured.simulate_llm_429,
            llm_down=simulate_llm_down or configured.simulate_llm_down,
            bq_error=simulate_bq_error or configured.simulate_bq_error,
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
        elif text.startswith("/"):
            console.print(f"Unknown command {text}. Type /help.", markup=False)
        else:
            with friendly_errors(debug):
                answer(session, text)
            trace_line(session)


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
