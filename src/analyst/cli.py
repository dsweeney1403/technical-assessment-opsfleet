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
    pending: bool = False  # the last turn stopped on an LLM outage and can resume (D2.13)

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
    from analyst.store.results import ResultStore

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
    graph = build_graph(Deps(settings, llm, gateway, store, brands=brands, known=known))
    thread_id = uuid.uuid4().hex
    return Session(user, graph, thread_id, Tracer(settings, user, thread_id))


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
    return None


def _error_code(content) -> str | None:
    try:
        return json.loads(content).get("error", {}).get("code")
    except (TypeError, ValueError, AttributeError):
        return None


def answer(session: Session, text: str | None) -> None:
    """Run one turn, or resume the stopped one when text is None (/retry)."""
    payload = None if text is None else {"user_input": text, "user": session.user}
    session.pending = False
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
    state = session.graph.get_state(session.config).values
    console.print(Markdown(state["draft"].text or "(no answer)"))
    for path in state["draft"].chart_paths:
        console.print(f"[green]Chart saved:[/green] {path}")


def trace_line(session: Session) -> None:
    """The turn's trace id in grey, under every answer and failure, so it can be quoted (D2.19)."""
    if session.tracer.last:
        console.print(f"[grey50]trace {session.tracer.last['trace_id']}[/grey50]")


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
