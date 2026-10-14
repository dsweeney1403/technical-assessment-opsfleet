"""Command line: `chat --user <id>` and `doctor`."""

from __future__ import annotations

import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from analyst import doctor as doctor_checks
from analyst.config import ConfigError, User, load_env, load_settings, load_users
from analyst.llm.client import LLMFatalError, LLMUnavailableError
from analyst.sql.gateway import GatewayFatalError

app = typer.Typer(add_completion=False, no_args_is_help=True, help="OpsFleet Analyst")
console = Console()

HELP = """Ask a question about sales, customers or products, for example "What was revenue by month this year?".

Commands:
  /help     show this help
  /whoami   show who you are signed in as and which brands you can see
  /exit     leave the chat"""

DOCTOR_HINT = "Run `python -m analyst doctor` to check your setup."


@dataclass
class Session:
    user: User
    graph: object
    thread_id: str


def build_session(user_id: str) -> Session:
    from google.cloud import bigquery

    from analyst.graph import Deps, build_graph
    from analyst.llm.client import LLM
    from analyst.sql.gateway import Gateway, load_curated
    from analyst.store.results import ResultStore

    settings = load_settings()
    user = load_users().get(user_id)
    env = load_env()
    store = ResultStore()
    gateway = Gateway(
        settings.bigquery,
        bigquery.Client(project=env.google_cloud_project),
        store,
        load_curated(settings.bigquery.dataset),
    )
    graph = build_graph(Deps(settings, LLM(settings, env.gemini_api_key), gateway, store))
    return Session(user, graph, uuid.uuid4().hex)


@contextmanager
def friendly_errors(debug: bool = False) -> Iterator[None]:
    """Turn known failures into plain messages. Never show a traceback unless --debug."""
    try:
        yield
    except (ConfigError, LLMFatalError, GatewayFatalError) as exc:
        console.print(f"[red]{exc}[/red]")
    except LLMUnavailableError:
        console.print("[yellow]Gemini is busy or unreachable right now. Please try again in a moment.[/yellow]")
    except Exception:
        console.print(f"[red]Something went wrong on our side. {DOCTOR_HINT}[/red]")
        if debug:
            console.print(traceback.format_exc(), markup=False)


def whoami(user: User) -> str:
    brands = "all brands" if user.all_brands else ", ".join(user.brands)
    return f"{user.name} ({user.id}, {user.role.replace('_', ' ')}). Brands: {brands}."


def answer(session: Session, text: str) -> None:
    with console.status("Thinking..."):
        state = session.graph.invoke(
            {"user_input": text, "user": session.user},
            {"configurable": {"thread_id": session.thread_id}, "recursion_limit": 50},
        )
    console.print(Markdown(state["draft"].text or "(no answer)"))
    for path in state["draft"].chart_paths:
        console.print(f"[green]Chart saved:[/green] {path}")


@app.command()
def chat(
    user: str = typer.Option(..., "--user", help="User id from config/users.yaml"),
    debug: bool = typer.Option(False, "--debug", help="Show tracebacks for unexpected errors"),
) -> None:
    """Chat with the analyst as a configured user."""
    session = None
    with friendly_errors(debug):
        session = build_session(user)
    if session is None:
        raise typer.Exit(1)
    if not session.user.all_brands:
        console.print("[yellow]Brand scoping arrives in phase 2. In this build only the CEO can chat.[/yellow]")
        raise typer.Exit(1)
    console.print(f"Signed in as {whoami(session.user)} Type /help for help, /exit to leave.")
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
        elif text.startswith("/"):
            console.print(f"Unknown command {text}. Type /help.", markup=False)
        else:
            with friendly_errors(debug):
                answer(session, text)


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
