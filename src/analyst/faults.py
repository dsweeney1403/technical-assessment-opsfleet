"""Fault injection for reviewers (D2.14).

Each flag makes a real dependency fail at its boundary, so the real retry,
fallback, breaker and gateway code handles it:
- --simulate-llm-429: the agent's primary model answers 429 on every call.
  Retries run out, the fallback model answers, and the breaker opens after 5.
- --simulate-llm-down: every Gemini model answers 503. The classifier fails
  open to the agent, the agent's chain fails, and the chat says so (/retry).
- --simulate-bq-error: every BigQuery query fails with a transient 503 at
  execution (dry runs pass, so it fails mid-turn). Retries run out and the
  agent answers with what it has.
"""

from __future__ import annotations

from dataclasses import dataclass

from google.api_core import exceptions as gexc
from google.genai.errors import ClientError, ServerError


@dataclass(frozen=True)
class Faults:
    llm_429: bool = False
    llm_down: bool = False
    bq_error: bool = False

    @property
    def active(self) -> list[str]:
        return [name for name, on in (("Gemini 429 on the agent's primary model", self.llm_429),
                                      ("every Gemini model down", self.llm_down),
                                      ("BigQuery errors", self.bq_error)) if on]


def _rate_limited() -> Exception:
    return ClientError(429, {"error": {"code": 429, "message": "Simulated rate limit (--simulate-llm-429).",
                                       "status": "RESOURCE_EXHAUSTED"}})


def _unavailable() -> Exception:
    return ServerError(503, {"error": {"code": 503, "message": "Simulated outage (--simulate-llm-down).",
                                       "status": "UNAVAILABLE"}})


class FailingChat:
    """Stands in for a chat model that fails on every call."""

    def __init__(self, error):
        self._error = error

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, include_raw=False):
        return self

    def invoke(self, messages):
        raise self._error()


def chat_factory(factory, faults: Faults, agent_primary: str):
    """Wrap the chat factory so the chosen models fail."""

    def make(model: str, **kwargs):
        if faults.llm_down:
            return FailingChat(_unavailable)
        if faults.llm_429 and model == agent_primary:
            return FailingChat(_rate_limited)
        return factory(model=model, **kwargs)

    return make


class FailingBigQuery:
    """Wraps a BigQuery client: dry runs pass, executions fail with a transient error."""

    def __init__(self, client):
        self._client = client

    def query(self, sql, job_config=None):
        if job_config is not None and job_config.dry_run:
            return self._client.query(sql, job_config=job_config)
        raise gexc.ServiceUnavailable("Simulated BigQuery outage (--simulate-bq-error).",
                                      errors=[{"reason": "backendError", "message": "Simulated outage."}])
