import httpx
import pytest
from google.genai.errors import ClientError, ServerError
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from analyst.config import LLMRetry, load_settings
from analyst.llm.client import (
    LLM,
    CallStats,
    LLMFatalError,
    LLMOutputError,
    LLMUnavailableError,
    is_transient,
    retry_after_s,
    with_retries,
)

POLICY = LLMRetry(max_attempts=3, max_total_wait_s=30, honour_retry_after=True, request_timeout_s=60)


def client_error(code, retry_delay=None, message="boom"):
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}] if retry_delay else []
    return ClientError(code, {"error": {"code": code, "message": message, "details": details}})


def wrapped(inner):
    """Mimic langchain-google-genai, which re-raises with the SDK error as the cause."""
    try:
        raise inner
    except Exception as e:
        try:
            raise RuntimeError("Error calling model") from e
        except RuntimeError as outer:
            return outer


def flaky(errors, result="ok"):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if errors:
            raise errors.pop(0)
        return result

    return fn, calls


@pytest.mark.parametrize(
    "exc",
    [client_error(429), ServerError(500, {"error": {}}), ServerError(503, {"error": {}}),
     httpx.ReadTimeout("t"), httpx.ConnectError("reset"), wrapped(client_error(429))],
)
def test_transient(exc):
    assert is_transient(exc)


@pytest.mark.parametrize("exc", [client_error(400), client_error(401), client_error(403), wrapped(client_error(404))])
def test_not_transient(exc):
    assert not is_transient(exc)


def test_retry_after_from_retry_info():
    assert retry_after_s(wrapped(client_error(429, retry_delay="7s"))) == 7.0
    assert retry_after_s(client_error(429)) is None


def test_succeeds_after_transient_errors_and_counts_retries():
    sleeps, stats = [], CallStats()
    fn, calls = flaky([client_error(429), ServerError(503, {"error": {}})])
    assert with_retries(fn, POLICY, stats, sleep=sleeps.append, rand=lambda: 0.5) == "ok"
    assert calls["n"] == 3 and stats.retries == 2
    assert sleeps == [1.0, 2.0]  # full jitter at rand=0.5 on a 2 s base


def test_gives_up_after_max_attempts():
    fn, calls = flaky([client_error(429)] * 5)
    with pytest.raises(LLMUnavailableError):
        with_retries(fn, POLICY, sleep=lambda s: None, rand=lambda: 0.1)
    assert calls["n"] == 3


def test_honours_retry_after():
    sleeps = []
    fn, _ = flaky([client_error(429, retry_delay="4s")])
    with_retries(fn, POLICY, sleep=sleeps.append)
    assert sleeps == [4.0]


def test_total_wait_cap_stops_early():
    sleeps = []
    fn, calls = flaky([client_error(429, retry_delay="45s")])
    with pytest.raises(LLMUnavailableError):
        with_retries(fn, POLICY, sleep=sleeps.append)
    assert sleeps == [] and calls["n"] == 1


def test_bad_key_not_retried_and_points_to_doctor():
    fn, calls = flaky([wrapped(client_error(400, message="API key not valid."))])
    with pytest.raises(LLMFatalError, match="doctor") as info:
        with_retries(fn, POLICY, sleep=lambda s: None)
    assert calls["n"] == 1 and info.value.status == 400
    assert "API key not valid. Run" in str(info.value)


def test_unrelated_exception_propagates():
    fn, _ = flaky([ValueError("bug")])
    with pytest.raises(ValueError):
        with_retries(fn, POLICY)


class FakeChat:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tools = None
        FakeChat.instances.append(self)

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        return AIMessage(content="hi", usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7})

    def with_structured_output(self, schema, include_raw):
        parent = self

        class R:
            def invoke(self, messages):
                return {"raw": parent.invoke(messages), "parsed": parent.parsed, "parsing_error": None}

        return R()


class Answer(BaseModel):
    value: str


def test_llm_uses_config_model_ids_and_disables_library_retries():
    FakeChat.instances.clear()
    settings = load_settings()
    llm = LLM(settings, "key", chat_factory=FakeChat)
    stats = CallStats()
    llm.invoke("agent", [HumanMessage("x")], tools=["t"], stats=stats)
    chat = FakeChat.instances[0]
    assert chat.kwargs["model"] == settings.models.agent
    assert chat.kwargs["max_retries"] == 0
    assert chat.tools == ["t"]
    assert stats.input_tokens == 5 and stats.output_tokens == 2


def test_structured_parse_failure_raises():
    FakeChat.instances.clear()
    llm = LLM(load_settings(), "key", chat_factory=FakeChat)
    llm._chat("light").parsed = None
    with pytest.raises(LLMOutputError):
        llm.structured("light", Answer, [HumanMessage("x")])


# Fallback chain and circuit breaker (D2.10 to D2.12)

class ScriptedChat:
    """Per-model behaviour: `failing[model]` is an exception raised on every call to that model."""

    failing: dict = {}
    calls: list = []

    def __init__(self, model, **kwargs):
        self.model = model

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        ScriptedChat.calls.append(self.model)
        if self.model in ScriptedChat.failing:
            raise ScriptedChat.failing[self.model]
        return AIMessage(content=f"from {self.model}")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def scripted_llm(failing, clock=None):
    ScriptedChat.failing = failing
    ScriptedChat.calls = []
    return LLM(load_settings(), "key", chat_factory=ScriptedChat, clock=clock or Clock(), sleep=lambda s: None)


def test_chains_come_from_config():
    llm = scripted_llm({})
    m = load_settings().models
    assert llm.chain("agent") == [m.agent, *m.agent_fallbacks]
    assert llm.chain("light") == [m.light, *m.light_fallbacks]


def test_fallback_model_used_after_429s_and_recorded():
    m = load_settings().models
    llm = scripted_llm({m.agent: client_error(429)})
    stats = CallStats()
    out = llm.invoke("agent", [HumanMessage("x")], stats=stats)
    assert out.content == f"from {m.agent_fallbacks[0]}"
    assert stats.fallback_used and stats.models_used == [m.agent_fallbacks[0]]
    assert ScriptedChat.calls.count(m.agent) == POLICY.max_attempts and stats.retries == POLICY.max_attempts - 1


def test_fatal_error_does_not_fall_back():
    m = load_settings().models
    llm = scripted_llm({m.agent: client_error(401)})
    with pytest.raises(LLMFatalError):
        llm.invoke("agent", [HumanMessage("x")])
    assert ScriptedChat.calls == [m.agent]


def test_every_model_down_is_unavailable():
    m = load_settings().models
    llm = scripted_llm({mid: ServerError(503, {"error": {}}) for mid in [m.agent, *m.agent_fallbacks]})
    with pytest.raises(LLMUnavailableError):
        llm.invoke("agent", [HumanMessage("x")])
    assert set(ScriptedChat.calls) == {m.agent, *m.agent_fallbacks}


def test_breaker_opens_after_five_errors_then_skips_and_closes_after_cooldown():
    m = load_settings().models
    cfg = load_settings().circuit_breaker
    clock = Clock()
    llm = scripted_llm({m.agent: client_error(429)}, clock)
    first, second = CallStats(), CallStats()
    llm.invoke("agent", [HumanMessage("x")], stats=first)  # 3 errors
    llm.invoke("agent", [HumanMessage("x")], stats=second)  # 2 more: opens on the fifth
    assert ScriptedChat.calls.count(m.agent) == cfg.failures and second.breaker_opens == 1
    assert llm.breakers[m.agent].is_open()

    ScriptedChat.calls = []
    llm.invoke("agent", [HumanMessage("x")])
    assert m.agent not in ScriptedChat.calls  # skipped, no waiting

    clock.now += cfg.cooldown_s + 1
    ScriptedChat.failing = {}
    assert llm.invoke("agent", [HumanMessage("x")]).content == f"from {m.agent}"
    assert not llm.breakers[m.agent].is_open()


def test_success_resets_the_breaker_count():
    m = load_settings().models
    llm = scripted_llm({m.agent: client_error(429)})
    llm.invoke("agent", [HumanMessage("x")])  # 3 errors
    ScriptedChat.failing = {}
    llm.invoke("agent", [HumanMessage("x")])  # success resets
    ScriptedChat.failing = {m.agent: client_error(429)}
    llm.invoke("agent", [HumanMessage("x")])  # 3 more: still closed
    assert not llm.breakers[m.agent].is_open()


def test_light_chain_falls_back_for_the_classifier():
    m = load_settings().models
    llm = scripted_llm({m.light: ServerError(500, {"error": {}})})
    assert llm.invoke("light", [HumanMessage("x")]).content == f"from {m.light_fallbacks[0]}"
