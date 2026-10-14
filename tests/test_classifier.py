from langchain_core.messages import AIMessage, HumanMessage

from analyst.config import User
from analyst.llm.client import CallStats, LLMUnavailableError
from analyst.safety.classifier import ClassifierOutput, classify
from analyst.safety.declines import DECLINES, no_access, scope_note
from analyst.state import Intent

CALVIN = User(id="bm_calvin", name="CK", role="brand_manager", brands=["Calvin Klein"])
LEVIS = User(id="bm_levis", name="LV", role="brand_manager", brands=["Levi's", "Canada Goose"])
CEO = User(id="ceo", name="CEO", role="ceo", brands=["*"])


class OneShot:
    def __init__(self, out):
        self.out = out
        self.messages = None

    def structured(self, role, schema, messages, stats=None):
        self.messages = messages
        if isinstance(self.out, Exception):
            raise self.out
        return self.out


def test_classifier_sees_recent_turns_and_maps_fields():
    llm = OneShot(ClassifierOutput(intent="analysis", confident=False, brands_mentioned=["Levis"],
                                   is_followup=True, reason="follow-up"))
    history = [HumanMessage("revenue by state"), AIMessage("| state | revenue |")]
    out = classify(llm, history, "now by month", CallStats())
    assert (out.intent, out.confident, out.brands_mentioned, out.is_followup) == (Intent.analysis, False, ["Levis"], True)
    assert "revenue by state" in llm.messages[1].content and "now by month" in llm.messages[1].content


def test_classifier_failure_is_unavailable():
    assert classify(OneShot(LLMUnavailableError("down")), [], "hi", CallStats()).intent == Intent.unavailable


def test_no_access_offers_own_brands():
    assert no_access(["Carhartt"], CALVIN) == (
        "Carhartt is not in your access, so I can't report on it. I can answer this for Calvin Klein instead.")
    assert no_access(["Carhartt", "Wrangler"], LEVIS).startswith("Carhartt and Wrangler are not in your access")
    assert no_access(["Carhartt"], LEVIS).endswith("for Levi's and Canada Goose instead.")


def test_scope_note():
    assert scope_note(CEO, []) == ""
    assert scope_note(LEVIS, []) == "Scope: covers Levi's and Canada Goose only."
    assert scope_note(CALVIN, ["Carhartt", "Wrangler"]).endswith("Carhartt and Wrangler are outside your access, so not included.")


def test_user_facing_strings_have_no_em_dashes():
    assert not any("—" in text for text in DECLINES.values())
