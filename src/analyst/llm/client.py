"""Gemini client with transient-only retries (D2.8).

Retries 429, 5xx, timeouts and connection resets with exponential backoff,
full jitter and the server's retry delay, up to max_attempts and a total wait
cap. Any other 4xx fails the same way every time, so it stops at once with a
message pointing to `doctor`. Retries are counted in CallStats, never against
the agent's tool-call budget. Fallback chain and breaker arrive in phase 3.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

import httpx
from langchain_core.messages import AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from analyst.config import LLMRetry, Settings

Role = Literal["agent", "light"]
T = TypeVar("T")
S = TypeVar("S", bound=BaseModel)

BACKOFF_BASE_S = 2.0
DOCTOR_HINT = "Run `python -m analyst doctor` to check your setup."


class LLMError(Exception):
    """Base for LLM failures the graph handles."""


class LLMFatalError(LLMError):
    """A non-transient error (bad key, bad request, no permission). Not retried."""

    def __init__(self, status: int | None, detail: str):
        self.status = status
        super().__init__(f"Gemini rejected the request ({status or 'error'}): {detail}. {DOCTOR_HINT}")


class LLMUnavailableError(LLMError):
    """Transient errors continued after every retry."""


class LLMOutputError(LLMError):
    """The model answered, but not in the requested structure."""


@dataclass
class CallStats:
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add_usage(self, message: AIMessage) -> None:
        usage = getattr(message, "usage_metadata", None) or {}
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def status_code(exc: BaseException) -> int | None:
    for e in _chain(exc):
        code = getattr(e, "code", None)
        if isinstance(code, int) and 100 <= code < 600:
            return code
    return None


def is_transient(exc: BaseException) -> bool:
    for e in _chain(exc):
        if isinstance(e, (httpx.TimeoutException, httpx.TransportError, TimeoutError, ConnectionError)):
            return True
    code = status_code(exc)
    return code is not None and (code == 429 or code >= 500)


def retry_after_s(exc: BaseException) -> float | None:
    """Server-suggested wait: Gemini's RetryInfo detail, or a Retry-After header."""
    for e in _chain(exc):
        body = getattr(e, "details", None) or getattr(e, "response_json", None)
        if isinstance(body, dict):
            for item in body.get("error", {}).get("details", []) or []:
                delay = item.get("retryDelay") if isinstance(item, dict) else None
                m = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(delay or ""))
                if m:
                    return float(m.group(1))
        response = getattr(e, "response", None)
        header = getattr(response, "headers", {}).get("retry-after") if response is not None else None
        if header and str(header).replace(".", "", 1).isdigit():
            return float(header)
    return None


def _detail(exc: BaseException) -> str:
    for e in _chain(exc):
        body = getattr(e, "details", None)
        if isinstance(body, dict) and body.get("error", {}).get("message"):
            return body["error"]["message"]
    return str(exc).splitlines()[0][:200]


def with_retries(
    fn: Callable[[], T],
    policy: LLMRetry,
    stats: CallStats | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    waited = 0.0
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if not is_transient(exc):
                if status_code(exc) is not None:
                    raise LLMFatalError(status_code(exc), _detail(exc)) from exc
                raise
            if attempt == policy.max_attempts:
                raise LLMUnavailableError(_detail(exc)) from exc
            suggested = retry_after_s(exc) if policy.honour_retry_after else None
            delay = suggested if suggested is not None else rand() * BACKOFF_BASE_S * 2 ** (attempt - 1)
            if waited + delay > policy.max_total_wait_s:
                raise LLMUnavailableError(_detail(exc)) from exc
            sleep(delay)
            waited += delay
            if stats is not None:
                stats.retries += 1
    raise AssertionError("unreachable")


class LLM:
    """Role-based access to Gemini models. Model IDs come from settings."""

    def __init__(self, settings: Settings, api_key: str, chat_factory=ChatGoogleGenerativeAI):
        self._settings = settings
        self._api_key = api_key
        self._factory = chat_factory
        self._models: dict[str, object] = {}

    def model_id(self, role: Role) -> str:
        return getattr(self._settings.models, role)

    def _chat(self, role: Role):
        model_id = self.model_id(role)
        if model_id not in self._models:
            self._models[model_id] = self._factory(
                model=model_id,
                api_key=self._api_key,
                max_retries=0,  # retries are ours (D2.8)
                timeout=self._settings.llm_retry.request_timeout_s,
            )
        return self._models[model_id]

    def invoke(
        self,
        role: Role,
        messages: Sequence[BaseMessage],
        tools: Sequence | None = None,
        stats: CallStats | None = None,
    ) -> AIMessage:
        chat = self._chat(role)
        runnable = chat.bind_tools(list(tools)) if tools else chat
        message = with_retries(lambda: runnable.invoke(list(messages)), self._settings.llm_retry, stats)
        if stats is not None:
            stats.add_usage(message)
        return message

    def structured(
        self,
        role: Role,
        schema: type[S],
        messages: Sequence[BaseMessage],
        stats: CallStats | None = None,
    ) -> S:
        runnable = self._chat(role).with_structured_output(schema, include_raw=True)
        out = with_retries(lambda: runnable.invoke(list(messages)), self._settings.llm_retry, stats)
        if stats is not None and out.get("raw") is not None:
            stats.add_usage(out["raw"])
        if out.get("parsing_error") or out.get("parsed") is None:
            raise LLMOutputError(f"Structured output failed: {out.get('parsing_error')}")
        return out["parsed"]
