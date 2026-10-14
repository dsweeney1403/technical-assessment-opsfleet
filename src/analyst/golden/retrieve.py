"""Forced trio retrieval (D4.5, D4.6, D4.8, D4.9).

A graph step, not a tool the agent can skip: on every analysis turn, and once
at the start of a report. Steps: embed the question, take the top `overfetch`
by similarity, drop any trio tagged with a brand outside the user's access
(the CEO keeps all; untagged trios reach everyone), drop trios below the
threshold, keep up to k. Zero trios is a valid outcome: the agent works from
the glossary. An embedding outage gives no trios and `trios_unavailable` in
the trace (D4.8). Trios improve answers but never protect anything.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from analyst.config import Golden, User
from analyst.golden.vector_store import VectorStore
from analyst.llm.client import CallStats, LLMError
from analyst.observability import tracer
from analyst.state import RetrievedTrio

Status = Literal["ok", "trios_unavailable", "threshold_unset", "no_index"]


class Scored(BaseModel):
    trio_id: str
    score: float


class Retrieval(BaseModel):
    """What the trace records for one retrieval (D4.9)."""

    status: Status = "ok"
    trios: list[RetrievedTrio] = Field(default_factory=list)
    below_threshold: list[Scored] = Field(default_factory=list)
    out_of_scope: int = 0  # dropped by the brand filter; counted, not named
    error: str | None = None


def allowed(brands: list[str], user: User) -> bool:
    """D4.4: every tagged brand must be within the user's brands. Untagged means company-wide."""
    return user.all_brands or set(brands) <= set(user.brands)


class Retriever:
    def __init__(self, store: VectorStore | None, embedder, settings: Golden):
        self.store = store
        self.embedder = embedder
        self.settings = settings

    def retrieve(self, question: str, user: User, stats: CallStats | None = None) -> Retrieval:
        stats = stats if stats is not None else CallStats()
        start = tracer.now_ns()
        result = self._retrieve(question, user, stats)
        tracer.emit("retrieval", {
            "status": result.status,
            "trios": [{"trio_id": t.trio_id, "score": round(t.score, 4), "brands": t.brands} for t in result.trios],
            "below_threshold": [s.model_dump() for s in result.below_threshold],
            "out_of_scope": result.out_of_scope,
            "threshold": self.settings.threshold,
            "k": self.settings.k,
            "retries": stats.retries,
            "error": result.error,
            "start_ns": start,
            "end_ns": tracer.now_ns(),
        })
        return result

    def _retrieve(self, question: str, user: User, stats: CallStats) -> Retrieval:
        if self.store is None:
            return Retrieval(status="no_index")
        threshold = self.settings.threshold
        if threshold is None:
            return Retrieval(status="threshold_unset")  # until `golden calibrate` sets it (D4.9)
        try:
            vector = self.embedder.embed([question], "query", stats)[0]
        except LLMError as exc:
            return Retrieval(status="trios_unavailable", error=f"{type(exc).__name__}: {exc}")
        result = Retrieval()
        for trio, score in self.store.search(vector, self.settings.overfetch):
            if not allowed(trio.brands, user):
                result.out_of_scope += 1
            elif score < threshold:
                result.below_threshold.append(Scored(trio_id=trio.trio_id, score=round(score, 4)))
            elif len(result.trios) < self.settings.k:
                result.trios.append(RetrievedTrio(trio_id=trio.trio_id, question=trio.question,
                                                  description=trio.description, report=trio.report_stripped,
                                                  brands=trio.brands, score=score))
        return result


def render(trios: list[RetrievedTrio]) -> str:
    """The examples prompt block. Scores and brand tags stay out of the prompt."""
    blocks = []
    for i, t in enumerate(trios, 1):
        blocks.append(f"Example {i}\nQuestion: {t.question}\nApproach: {t.description}\n"
                      f"Analyst report (numbers hidden):\n{t.report.strip()}")
    return "\n\n".join(blocks)
