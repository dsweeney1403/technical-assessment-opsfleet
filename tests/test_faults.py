import pytest
from langchain_core.messages import AIMessage, HumanMessage

from analyst import faults
from analyst.config import User, load_settings
from analyst.llm.client import LLM, CallStats, LLMUnavailableError, is_transient
from analyst.sql.brands import BrandMatcher
from analyst.sql.gateway import Gateway, GatewayUnavailableError, classify_bq_error
from analyst.store.results import ResultStore
from test_gateway import CURATED, FakeClient

CEO = User(id="ceo", name="CEO", role="ceo", brands=["*"])


class OkChat:
    def __init__(self, model, **kwargs):
        self.model = model

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content=f"from {self.model}")


def llm_with(flags):
    settings = load_settings()
    factory = faults.chat_factory(OkChat, flags, settings.models.agent)
    return LLM(settings, "key", chat_factory=factory, sleep=lambda s: None), settings.models


def test_simulated_errors_are_transient():
    with pytest.raises(Exception) as rate:
        faults.FailingChat(faults._rate_limited).invoke([])
    with pytest.raises(Exception) as down:
        faults.FailingChat(faults._unavailable).invoke([])
    assert is_transient(rate.value) and is_transient(down.value)


def test_simulate_429_falls_back_and_opens_the_breaker():
    llm, models = llm_with(faults.Faults(llm_429=True))
    stats = CallStats()
    for _ in range(2):
        out = llm.invoke("agent", [HumanMessage("x")], stats=stats)
    assert out.content == f"from {models.agent_fallbacks[0]}" and stats.fallback_used
    assert stats.breaker_opens == 1 and llm.breakers[models.agent].is_open()
    assert llm.invoke("light", [HumanMessage("x")]).content == f"from {models.light}"  # classifier unaffected


def test_simulate_llm_down_fails_every_model():
    llm, _ = llm_with(faults.Faults(llm_down=True))
    for role in ("agent", "light"):
        with pytest.raises(LLMUnavailableError):
            llm.invoke(role, [HumanMessage("x")])


def test_no_flags_leaves_models_alone():
    llm, models = llm_with(faults.Faults())
    assert llm.invoke("agent", [HumanMessage("x")]).content == f"from {models.agent}"


def test_simulate_bq_error_fails_execution_after_retries():
    client = faults.FailingBigQuery(FakeClient())
    gateway = Gateway(load_settings().bigquery, client, ResultStore(), CURATED, BrandMatcher([]), sleep=lambda s: None)
    with pytest.raises(GatewayUnavailableError):
        gateway.run("SELECT a FROM sales", CEO)
    assert gateway.transient_retries == load_settings().bigquery.transient_max_attempts - 1


def test_simulated_bq_error_is_coded_transient_by_reason():
    with pytest.raises(Exception) as info:
        faults.FailingBigQuery(FakeClient()).query("SELECT 1", job_config=None)
    assert classify_bq_error(info.value) == "BQ_TRANSIENT"
