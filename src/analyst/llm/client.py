"""Gemini client: transient-only retries, fallback chain, circuit breaker.

Retries 429, 5xx, timeouts and connection resets with exponential backoff,
full jitter and the server's retry delay, up to max_attempts and a total wait
cap (D2.8). Any other 4xx fails the same way every time, so it stops at once
with a message pointing to `doctor`. Retries are counted in CallStats, never
against the agent's tool-call budget.

When a model's retries run out, the next model in the role's chain is tried
(D2.10, D2.11). A breaker on each chain's primary model opens after N
consecutive transient errors and skips that model for a cooldown (D2.12).
Fallback use is recorded in CallStats for the trace only.

Each call emits an `llm_call` span for the trace (D2.16): model, messages in and
out, tokens, latency, retries and fallback use.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeVar

import httpx
from langchain_core.messages import AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from analyst.config import CircuitBreakerSettings, LLMRetry, Settings
from analyst.observability import tracer

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
        super().__init__(f"Gemini rejected the request ({status or 'error'}): {detail.rstrip('.')}. {DOCTOR_HINT}")


class LLMUnavailableError(LLMError):
    """Transient errors continued after every retry, on every model in the chain."""


class LLMOutputError(LLMError):
    """The model answered, but not in the requested structure."""


@dataclass
class CallStats:
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    models_used: list[str] = field(default_factory=list)
    fallback_used: bool = False
    breaker_opens: int = 0

    def snapshot(self) -> tuple[int, int, int, int, int]:
        return (self.retries, self.input_tokens, self.output_tokens, len(self.models_used), self.breaker_opens)

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
    on_transient: Callable[[], bool] | None = None,
) -> T:
    """`on_transient` is told about each transient error; returning True stops retrying (breaker open)."""
    waited = 0.0
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if not is_transient(exc):
                if status_code(exc) is not None:
                    raise LLMFatalError(status_code(exc), _detail(exc)) from exc
                raise
            if on_transient is not None and on_transient():
                raise LLMUnavailableError(_detail(exc)) from exc
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


class CircuitBreaker:
    """Opens after `failures` consecutive transient errors; stays open for `cooldown_s` (D2.12)."""

    def __init__(self, policy: CircuitBreakerSettings, clock: Callable[[], float] = time.monotonic):
        self._policy = policy
        self._clock = clock
        self._failures = 0
        self._open_until = 0.0

    def is_open(self) -> bool:
        return self._clock() < self._open_until

    def record_failure(self) -> bool:
        """Count a transient error. True when this one opens the breaker."""
        self._failures += 1
        if self._failures >= self._policy.failures:
            self._failures = 0
            self._open_until = self._clock() + self._policy.cooldown_s
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0


class LLM:
    """Role-based access to Gemini models. Model IDs and chains come from settings."""

    def __init__(self, settings: Settings, api_key: str, chat_factory=ChatGoogleGenerativeAI,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self._settings = settings
        self._api_key = api_key
        self._factory = chat_factory
        self._sleep = sleep
        self._models: dict[str, object] = {}
        primaries = {settings.models.agent, settings.models.light}
        self.breakers = {m: CircuitBreaker(settings.circuit_breaker, clock) for m in primaries}

    def model_id(self, role: Role) -> str:
        return getattr(self._settings.models, role)

    def chain(self, role: Role) -> list[str]:
        models = self._settings.models
        fallbacks = models.agent_fallbacks if role == "agent" else models.light_fallbacks
        return list(dict.fromkeys([self.model_id(role), *fallbacks]))

    def _chat_for(self, model_id: str):
        if model_id not in self._models:
            self._models[model_id] = self._factory(
                model=model_id,
                api_key=self._api_key,
                max_retries=0,  # retries are ours (D2.8)
                timeout=self._settings.llm_retry.request_timeout_s,
            )
        return self._models[model_id]

    def _chat(self, role: Role):
        return self._chat_for(self.model_id(role))

    def _run(self, role: Role, call: Callable[[object], T], stats: CallStats | None) -> T:
        """Try each model in the role's chain until one answers."""
        stats = stats if stats is not None else CallStats()
        chain = self.chain(role)
        last: LLMUnavailableError | None = None
        for model_id in chain:
            breaker = self.breakers.get(model_id)
            if breaker is not None and breaker.is_open():
                continue

            def on_transient(breaker=breaker) -> bool:
                if breaker is None:
                    return False
                if breaker.record_failure():
                    stats.breaker_opens += 1
                    return True
                return False

            chat = self._chat_for(model_id)
            try:
                out = with_retries(lambda: call(chat), self._settings.llm_retry, stats,
                                   sleep=self._sleep, on_transient=on_transient)
            except LLMUnavailableError as exc:
                last = exc
                continue
            if breaker is not None:
                breaker.record_success()
            stats.models_used.append(model_id)
            stats.fallback_used = stats.fallback_used or model_id != chain[0]
            return out
        raise last or LLMUnavailableError("Every model is skipped while the breaker is open.")

    def _traced(self, role: Role, messages: Sequence[BaseMessage], stats: CallStats | None,
                call: Callable[[CallStats], T], tools: Sequence | None = None) -> T:
        """Run one logical call (retries and fallbacks included) and emit its trace span."""
        stats = stats if stats is not None else CallStats()
        retries, tokens_in, tokens_out, n_models, opens = stats.snapshot()
        start = tracer.now_ns()
        out, error = None, None
        try:
            out = call(stats)
            return out
        except Exception as exc:
            error = exc
            raise
        finally:
            model = stats.models_used[-1] if len(stats.models_used) > n_models else None
            if isinstance(out, AIMessage):
                output = tracer.message_dict(out)
            elif isinstance(out, BaseModel):
                output = out.model_dump(mode="json")
            else:
                output = None
            used_in, used_out = stats.input_tokens - tokens_in, stats.output_tokens - tokens_out
            tracer.emit("llm_call", {
                "role": role,
                "model": model,
                "chain": self.chain(role),
                "fallback_used": model is not None and model != self.chain(role)[0],
                "retries": stats.retries - retries,
                "breaker_opens": stats.breaker_opens - opens,
                "tokens": {"input": used_in, "output": used_out, "total": used_in + used_out},
                "tools": [getattr(t, "name", str(t)) for t in tools or []],
                "messages_in": tracer.messages_dicts(messages),
                "output": output,
                "error": f"{type(error).__name__}: {error}" if error is not None else None,
                "start_ns": start,
                "end_ns": tracer.now_ns(),
            })

    def invoke(
        self,
        role: Role,
        messages: Sequence[BaseMessage],
        tools: Sequence | None = None,
        stats: CallStats | None = None,
    ) -> AIMessage:
        def call(chat):
            runnable = chat.bind_tools(list(tools)) if tools else chat
            return runnable.invoke(list(messages))

        def run(stats: CallStats) -> AIMessage:
            message = self._run(role, call, stats)
            stats.add_usage(message)
            return message

        return self._traced(role, messages, stats, run, tools)

    def structured(
        self,
        role: Role,
        schema: type[S],
        messages: Sequence[BaseMessage],
        stats: CallStats | None = None,
    ) -> S:
        def run(stats: CallStats) -> S:
            out = self._run(
                role, lambda chat: chat.with_structured_output(schema, include_raw=True).invoke(list(messages)), stats
            )
            if out.get("raw") is not None:
                stats.add_usage(out["raw"])
            if out.get("parsing_error") or out.get("parsed") is None:
                raise LLMOutputError(f"Structured output failed: {out.get('parsing_error')}")
            return out["parsed"]

        return self._traced(role, messages, stats, run)
