"""`python -m analyst eval`: run the eval sets and write results in the D6.18 shape.

Components (D5.1): classifier, delete_reply, retrieval and sql are component
sets with exact checks; e2e is scripted conversations; red_team is scored twice
(D5.2): the safety outcome is binary and must be 100% (the run exits 1
otherwise, D5.12), and behaviour is a rate.

Results go to two append-only files, `eval_runs.jsonl` and `eval_cases.jsonl`,
the local stand-ins for the BigQuery tables (D6.18). Each case is written as it
finishes, so `--resume <run_id>` carries on after an interruption. Failed
conversation cases carry a trace id: open them with `trace <id>`.
"""

from __future__ import annotations

import json
import subprocess
import time
import traceback
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from analyst.config import ROOT, Settings, User, load_users, resolve
from analyst.evals import scoring
from analyst.evals.harness import Conversation, SeedReport, new_session

COMPONENTS = ["classifier", "delete_reply", "retrieval", "sql", "e2e", "red_team"]
EVALS_DIR = ROOT / "evals"


@dataclass
class CaseResult:
    case_id: str
    score: float
    passed: bool
    trace_id: str | None = None
    notes: str = ""
    llm_calls: int = 1  # for the throttle
    extra: dict = field(default_factory=dict)


class Throttle:
    """Keeps the request rate under `rpm`, counting each case's model calls (free-tier limits, D8.1)."""

    def __init__(self, rpm: int, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.rpm, self.sleep, self.clock = rpm, sleep, clock
        self.started = clock()

    def after(self, calls: int) -> None:
        budget = 60.0 / self.rpm * max(1, calls)
        self.sleep(max(0.0, budget - (self.clock() - self.started)))
        self.started = self.clock()


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load(name: str) -> dict:
    return yaml.safe_load((EVALS_DIR / name).read_text(encoding="utf-8"))


def _calls(turns) -> int:
    return sum((t.record or {}).get("totals", {}).get("llm_calls", 1) for t in turns) or 1


# Components. Each yields one CaseResult per case.


class Components:
    def __init__(self, settings: Settings, workdir: Path, repeats: int, build=None, llm=None, embedder=None,
                 brands: list[str] | None = None):
        self.settings = settings
        self.workdir = workdir
        self.repeats = repeats
        self.build = build  # session builder; cli.build_session when None
        self._llm, self._embedder, self._brands = llm, embedder, brands
        self.users = load_users()

    # Shared clients, made on first use so `--only` never builds what it doesn't need.

    @property
    def llm(self):
        if self._llm is None:
            from analyst.config import load_env
            from analyst.llm.client import LLM

            self._llm = LLM(self.settings, load_env().gemini_api_key)
        return self._llm

    @property
    def embedder(self):
        if self._embedder is None:
            from analyst.config import load_env
            from analyst.llm.embeddings import Embedder

            self._embedder = Embedder(self.settings, load_env().gemini_api_key)
        return self._embedder

    @property
    def brands(self) -> list[str]:
        if self._brands is None:
            from analyst.cli import _catalogue
            from analyst.config import load_env

            self._brands = _catalogue(load_env().google_cloud_project, self.settings.bigquery.dataset)[0]
        return self._brands

    def session(self, user_id: str, key: str, persona: str | None = None, seed=()):
        return new_session(user_id, self.workdir / key, persona=persona,
                           seed=[SeedReport(**s) for s in seed], build=self.build)

    # Each component yields (case_id, run_case) pairs, so the runner can skip finished cases without running them.

    def classifier(self) -> Iterator[tuple[str, Callable[[], CaseResult]]]:
        from analyst.llm.client import CallStats
        from analyst.safety.classifier import classify

        def one(c):
            out = classify(self.llm, [], c["message"], CallStats())
            ok = out.intent.value == c["intent"] and ("report_op" not in c or out.report_op == c["report_op"])
            return CaseResult(c["id"], float(ok), ok, notes="" if ok else f"got {out.intent.value}/{out.report_op}")

        for c in _load("classifier.yaml")["cases"]:
            yield c["id"], lambda c=c: one(c)

    def delete_reply(self) -> Iterator[tuple[str, Callable[[], CaseResult]]]:
        from analyst.reports.replies import classify_reply

        data = _load("delete_replies.yaml")

        def one(c):
            got = classify_reply(self.llm, data["shown"], c["reply"])
            ok = got == c["decision"]
            return CaseResult(c["id"], float(ok), ok, notes="" if ok else f"got {got}")

        for c in data["cases"]:
            yield c["id"], lambda c=c: one(c)

    def retrieval(self) -> Iterator[tuple[str, Callable[[], CaseResult]]]:
        """The D4.9 labelled set at the configured threshold, as the CEO (no brand filter)."""
        from analyst.golden.calibrate import load_labelled_set
        from analyst.golden.retrieve import Retriever
        from analyst.golden.vector_store import VectorStore

        labelled = load_labelled_set(resolve(self.settings.golden.trios_dir).parent / "labelled_set.yaml")
        retriever = None

        def one(case_id, question, trio):
            nonlocal retriever
            if retriever is None:
                retriever = Retriever(VectorStore.load(resolve(self.settings.golden.index)), self.embedder,
                                      self.settings.golden)
            found = [t.trio_id for t in retriever.retrieve(question, self.users.get("ceo")).trios]
            ok = (trio in found) if trio else not found
            return CaseResult(case_id, float(ok), ok, notes="" if ok else f"retrieved {found or 'nothing'}")

        cases = [(f"{m.trio}_{i}", q, m.trio) for m in labelled.matches for i, q in enumerate(m.questions, 1)]
        cases += [(f"none_{i}", q, None) for i, q in enumerate(labelled.none, 1)]
        for case_id, question, trio in cases:
            yield case_id, lambda a=(case_id, question, trio): one(*a)

    def sql(self) -> Iterator[tuple[str, Callable[[], CaseResult]]]:
        from analyst.graph import _tool_results

        tol = self.settings.evals.sql_numeric_tolerance

        def one(c):
            user = self.users.get(c["user"])
            session = self.session(c["user"], f"sql/{c['id']}")
            turn = Conversation(session).say(c["question"])
            produced = _tool_results(turn.state.get("turn_messages", []))[0]
            frames = [session.results.get(p["result"]["result_id"]).df for p in produced]
            reference = session.gateway.run(c["reference"], user)
            if "error" in reference:
                return CaseResult(c["id"], 0.0, False, turn.trace_id,
                                  f"reference failed: {reference['error']['code']}", _calls([turn]))
            expected = session.results.get(reference["result_id"]).df
            ok = any(scoring.result_matches(expected, df, tol) for df in frames)
            note = "" if ok else (f"{len(frames)} result(s), none matched the reference" if frames
                                  else f"no query result ({turn.outcome}, {turn.reason})")
            return CaseResult(c["id"], float(ok), ok, turn.trace_id, note, _calls([turn]))

        for c in _load("sql.yaml")["cases"]:
            yield c["id"], lambda c=c: one(c)

    # Conversations

    def e2e(self) -> Iterator[tuple[str, Callable[[], CaseResult]]]:
        def one(c):
            user = self.users.get(c["user"])
            session = self.session(c["user"], f"e2e/{c['id']}", c.get("persona"), c.get("seed", []))
            conv = Conversation(session)
            failures, checks, failed = [], 0, 0
            for i, t in enumerate(c["turns"], 1):
                turn = conv.say(t["say"])
                if turn.awaiting and t.get("reply_if_asked"):
                    turn = conv.say(t["reply_if_asked"])
                checks += 1
                found = scoring.turn_failures(turn, t.get("expect", {}))
                failed += bool(found)
                failures += [f"turn {i}: {f}" for f in found]
            if c.get("expect_case"):
                checks += 1
                found = scoring.case_failures(session, user, c["expect_case"])
                failed += bool(found)
                failures += found
            last = next((t.trace_id for t in reversed(conv.turns) if t.trace_id), None)
            return CaseResult(c["id"], (checks - failed) / checks, not failures, last, "; ".join(failures),
                              _calls(conv.turns))

        for path in sorted((EVALS_DIR / "e2e").glob("*.yaml")):
            c = yaml.safe_load(path.read_text(encoding="utf-8"))
            yield c["id"], lambda c=c: one(c)

    def red_team(self) -> Iterator[tuple[str, Callable[[], CaseResult]]]:
        from analyst.safety.output_check import LeakScanner

        scanner = None

        def one(c, key):
            nonlocal scanner
            scanner = scanner or LeakScanner(self.brands)
            user: User = self.users.get(c["user"])
            session = self.session(c["user"], f"red_team/{key}", c.get("persona"), c.get("seed", []))
            conv = Conversation(session)
            conv.run(c["turns"])
            violations = scoring.safety_violations(conv.turns, user, scanner, len(session.reports.audit()),
                                                   c.get("allow_delete", False))
            behaviour = scoring.turn_failures(conv.turns[-1], c.get("expect", {}))
            notes = "; ".join([*(f"SAFETY: {v}" for v in violations), *behaviour])
            return CaseResult(key, float(not violations), not violations, conv.turns[-1].trace_id, notes,
                              _calls(conv.turns), {"safety_ok": not violations, "behaviour_ok": not behaviour})

        for c in _load("red_team.yaml")["cases"]:
            for rep in range(1, self.repeats + 1):
                key = f"{c['id']}#{rep}"
                yield key, lambda c=c, key=key: one(c, key)


# The run


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def scores_for(rows: list[dict]) -> tuple[dict[str, float], float | None, float | None]:
    """Mean score per component, the red-team safety outcome and the behaviour rate."""
    by_comp: dict[str, list[dict]] = {}
    for r in rows:
        by_comp.setdefault(r["component"], []).append(r)
    scores = {comp: round(sum(r["score"] for r in rs) / len(rs), 4) for comp, rs in by_comp.items()}
    red = by_comp.get("red_team", [])
    safety = round(sum(r["safety_ok"] for r in red) / len(red), 4) if red else None
    behaviour = round(sum(r["behaviour_ok"] for r in red) / len(red), 4) if red else None
    return scores, safety, behaviour


def run(settings: Settings, components: list[str], components_impl: Components | None = None, rpm: int | None = None,
        repeats: int | None = None, resume: str | None = None, baseline: bool = False, trigger: str = "manual",
        on_case: Callable[[str, CaseResult], None] | None = None, sleep: Callable[[float], None] = time.sleep) -> dict:
    """Run the chosen components and return the run row. Cases already recorded under `resume` are skipped."""
    results_dir = resolve(settings.evals.results_dir)
    runs_path, cases_path = results_dir / "eval_runs.jsonl", results_dir / "eval_cases.jsonl"
    run_id = resume or f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    done = {(r["component"], r["case_id"]) for r in _read_jsonl(cases_path) if r["run_id"] == run_id}
    impl = components_impl or Components(settings, results_dir / "work" / run_id,
                                         repeats or settings.evals.red_team_repeats)
    throttle = Throttle(rpm or settings.evals.requests_per_minute, sleep=sleep)
    for comp in components:
        for case_id, run_case in getattr(impl, comp)():
            if (comp, case_id) in done:
                continue  # finished in an earlier attempt of this run
            try:
                result = run_case()
            except Exception as exc:  # a broken case is a failed case, never a crashed run
                result = CaseResult(case_id, 0.0, False, notes=f"{type(exc).__name__}: {exc}"[:500],
                                    extra={"safety_ok": False, "behaviour_ok": False} if comp == "red_team" else {})
                result.extra["traceback"] = traceback.format_exc()[-2000:]
            _append(cases_path, {"run_id": run_id, "component": comp, "case_id": result.case_id,
                                 "score": result.score, "passed": result.passed, "trace_id": result.trace_id,
                                 "notes": result.notes, **result.extra})
            if on_case:
                on_case(comp, result)
            throttle.after(result.llm_calls)
    rows = [r for r in _read_jsonl(cases_path) if r["run_id"] == run_id]
    scores, safety, behaviour = scores_for(rows)
    models = settings.models
    row = {"run_id": run_id, "started_at": started_at, "git_sha": git_sha(), "trigger": trigger,
           "model_versions": {"agent": [models.agent, *models.agent_fallbacks],
                              "light": [models.light, *models.light_fallbacks],
                              "report": models.report, "embedding": models.embedding,
                              "embedding_dimensions": models.embedding_dimensions},
           "scores": scores, "red_team_outcome": safety, "red_team_behaviour": behaviour,
           "cases": len(rows), "failed": sum(not r["passed"] for r in rows), "is_baseline": baseline,
           "approved_by": None, "approved_at": None, "override": None}
    _append(runs_path, row)
    if baseline:
        summary = {k: row[k] for k in ("run_id", "started_at", "git_sha", "model_versions", "scores",
                                       "red_team_outcome", "red_team_behaviour", "cases", "failed")}
        (results_dir / "baseline.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return row
