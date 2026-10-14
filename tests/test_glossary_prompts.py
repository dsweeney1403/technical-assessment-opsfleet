"""Glossary, system defaults and prompt layers (D4.2, D4.15, D4.24)."""

from datetime import date

from langchain_core.messages import AIMessage
from test_graph import CEO, FakeLLM, setup  # noqa: F401  (setup is a fixture)

from analyst import prompts
from analyst.config import load_analysis_defaults, load_glossary
from analyst.reports import flow
from analyst.state import Section

TODAY = date(2026, 10, 14)


def full_layers(**kw):
    return prompts.Layers(glossary=load_glossary(), defaults=load_analysis_defaults(), **kw)


def test_glossary_covers_the_agreed_definitions():
    g = load_glossary()
    terms = " ".join(e.term for section in (g.metrics, g.time_rules, g.business_terms, g.scope_rules) for e in section)
    for term in ("revenue", "margin", "average order value", "basket size", "repeat rate", "churn",
                 "year to date", "Q1", "top or best customers", "mixed-brand orders"):
        assert term in terms
    churn = next(e for e in g.metrics if e.term == "churn")
    assert "90 days" in churn.definition  # C4


def test_whole_glossary_is_in_the_agent_prompt():
    text = prompts.agent_system(CEO, TODAY, "(none)", "", layers=full_layers())
    for e in load_glossary().metrics:
        assert e.definition in text
    assert "worked examples never override it" in text


def test_layers_follow_the_precedence_order():
    layers = full_layers(preferences="- format: table (stated)", persona="Be warm.", persona_version="v1",
                         examples="Q: revenue by month")
    text = prompts.agent_system(CEO, TODAY, "(none)", "", layers=layers)
    order = [text.index(marker) for marker in (
        "Safety rules", "Glossary (company definitions", "The user's preferences", "House style (persona v1)",
        "Defaults (use only", "Worked examples")]
    assert order == sorted(order)


def test_system_defaults_and_the_users_own():
    text = prompts.defaults_block(load_analysis_defaults())
    assert "Period: year to date (system default)" in text
    own = prompts.defaults_block(load_analysis_defaults(), {"period": "last quarter"})
    assert "last quarter, your default" in own
    assert "Breakdown: none (system default)" in own


def test_report_prompts_get_the_glossary_and_sections_get_no_examples():
    layers = full_layers(examples="Q: revenue by month")
    plan = flow.plan_messages(CEO, TODAY, "Q1 report", [], "", "(none)", 4, None, None, [], layers)
    assert "Glossary (company definitions" in plan[0].content and "Worked examples" in plan[0].content
    section = flow.section_messages(CEO, TODAY, "(none)", None, Section(title="t", brief="b", source="new_queries"),
                                    layers)
    assert "Glossary (company definitions" in section[0].content
    assert "Worked examples" not in section[0].content


class RecordingLLM(FakeLLM):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.light_messages = []

    def invoke(self, role, messages, tools=None, stats=None):
        if role == "light":
            self.light_messages.append(messages)
        return super().invoke(role, messages, tools, stats)


def test_glossary_on_data_questions_not_small_talk(setup):
    make, _, _ = setup
    llm = RecordingLLM(["data_question", "conversation"], light_reply="Revenue excludes cancelled items.")
    turn = make(llm)
    turn("What does revenue mean?")
    turn("Thanks!")
    assert "Glossary (company definitions" in llm.light_messages[0][0].content
    assert "Glossary (company definitions" not in llm.light_messages[1][0].content


def test_analysis_turn_gets_glossary_and_defaults(setup):
    make, _, _ = setup
    llm = FakeLLM(["analysis"], [AIMessage("Assumptions: year to date. Nothing to report.")])
    make(llm)("How are we doing?")
    system = llm.agent_messages[0][0].content
    assert "Glossary (company definitions" in system and "Period: year to date (system default)" in system
