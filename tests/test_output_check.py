from decimal import Decimal

import pandas as pd
import pytest

from analyst.config import User
from analyst.safety.output_check import (
    GROUNDING_NOTE,
    LeakScanner,
    figures,
    ground_values,
    load_stoplist,
    result_text,
    strip_figures,
    ungrounded,
)

CEO = User(id="ceo", name="CEO", role="ceo", brands=["*"])
CALVIN = User(id="bm_calvin", name="CK", role="brand_manager", brands=["Calvin Klein"])
LEVIS = User(id="bm_levis", name="LV", role="brand_manager", brands=["Levi's", "Canada Goose"])
BRANDS = ["Calvin Klein", "Calvin Klein Jeans", "Calvin Klein Performance", "Levi's", "Canada Goose",
          "Carhartt", "Basic", "Carolina", "adidas", "The North Face", "GUESS"]


@pytest.fixture
def scanner():
    return LeakScanner(BRANDS, stoplist={"Basic", "Carolina"})


def kinds(leaks):
    return sorted({leak.kind for leak in leaks})


# Leak scan: personal data shapes

@pytest.mark.parametrize("text", [
    "The top buyer was jane.doe@example.com.",
    "Most orders ship to CEP 01310-100 in São Paulo.",
    "Customers in zip code 90210 spent the most.",
    "The postcode SW1A 1AA leads the list.",
    "Customer 48213 placed 14 orders.",
    "customer_id: 48213 is the biggest spender.",
    "See order #918273 for details.",
    "| customer_id | revenue |\n|---|---|\n| 1 | 2 |",
    "| Email | orders |\n| --- | --- |\n| a | 2 |",
])
def test_personal_data_shapes_are_leaks(scanner, text):
    assert scanner.scan(text, CEO)


@pytest.mark.parametrize("text", [
    "Customers: 1,204 in Texas, 980 in California.",
    "We had 48,213 customers this year.",
    "Revenue by postcode isn't available; the finest level is city.",
    "Orders rose 12% in Q3 2026.",
    "Age band 25-34 leads with 31.2% of revenue.",
    "| state | customers |\n|---|---|\n| Texas | 1204 |",
    "Data from 2026-01-01 to 2026-10-14.",
])
def test_ordinary_answers_are_not_leaks(scanner, text):
    assert scanner.scan(text, CEO) == []


# Leak scan: brands outside scope

def test_other_brand_in_scoped_answer_is_a_leak(scanner):
    leaks = scanner.scan("Calvin Klein grew 5%, ahead of Carhartt.", CALVIN)
    assert [(leak.kind, leak.match) for leak in leaks] == [("brand", "Carhartt")]


def test_ceo_sees_every_brand(scanner):
    assert scanner.scan("Calvin Klein grew 5%, ahead of Carhartt.", CEO) == []


def test_brand_the_user_named_is_allowed(scanner):
    assert scanner.scan("Carhartt is outside your access.", CALVIN, allowed_text="how is carhartt doing?") == []


def test_longer_brand_is_read_as_that_brand(scanner):
    assert kinds(scanner.scan("Calvin Klein Jeans did well.", CALVIN)) == ["brand"]
    # Lowercase "jeans" is the category, so this is the user's own brand.
    assert scanner.scan("Calvin Klein jeans sold well.", CALVIN) == []


def test_own_brand_in_any_case_is_fine(scanner):
    assert scanner.scan("calvin klein and CALVIN KLEIN.", CALVIN) == []
    assert scanner.scan("Levi’s and Canada Goose grew.", LEVIS) == []  # curly apostrophe


def test_stoplisted_and_lowercase_words_dont_trigger(scanner):
    assert scanner.scan("Basic metrics for North Carolina: revenue rose.", CALVIN) == []
    assert scanner.scan("Some guess this is noise.", CALVIN) == []  # GUESS is case-sensitive


def test_lowercase_catalogue_brand_matches_capitalised(scanner):
    assert kinds(scanner.scan("Adidas leads the category.", CALVIN)) == ["brand"]


def test_names_in_the_users_own_results_are_allowed(scanner):
    frames = [pd.DataFrame({"product_name": ["Calvin Klein Performance Tee"], "revenue": [10.0]})]
    text = "The best seller was Calvin Klein Performance Tee."
    assert kinds(scanner.scan(text, CALVIN)) == ["brand"]
    assert scanner.scan(text, CALVIN, result_text=result_text(frames)) == []


def test_real_stoplist_loads_and_keeps_demo_brands():
    stop = load_stoplist()
    assert len(stop) > 100
    assert {"Basic", "Carolina"} <= stop
    assert not {"Calvin Klein", "Levi's", "Canada Goose", "Carhartt", "The North Face"} & stop


# Number grounding

FRAMES = [
    pd.DataFrame({"month": ["2026-01", "2026-02"], "revenue": [2859097.43, 1204.5]}),
    pd.DataFrame({"age_band": ["25-34", "65+"], "share": [0.179, 0.021], "customers": [1204, 980]}),
    pd.DataFrame({"total_revenue": [Decimal("48213.27")]}),
]


@pytest.fixture
def values():
    return ground_values(FRAMES, ["Show revenue for orders over $10,000"])


def texts(figs):
    return [f.text.strip() for f in figs]


@pytest.mark.parametrize("answer", [
    "Revenue was $2,859,097.43 in January.",
    "Revenue was $2.86M in January.",
    "Revenue was about $3M in January.",
    "Revenue was 2.9 million in January.",
    "The 25-34 band holds 17.9% of revenue.",
    "The 25-34 band holds 18% of revenue.",
    "The 65+ band holds 2.1% of revenue.",
    "We had 1,204 customers aged 25-34 and 980 aged 65+.",
    "February revenue was $1,204.50.",
    "Total revenue was $48,213.",
    "Orders over $10,000 are rare.",
    "Two months returned rows.",
])
def test_grounded_figures_pass(values, answer):
    assert ungrounded(answer, values) == []


@pytest.mark.parametrize("answer, bad", [
    ("Revenue was $2.87M in January.", ["$2.87M"]),
    ("| month | revenue |\n|---|---|\n| 2026-01 | 2,859,097.43 |\n| Total | 2,860,301.93 |", ["2,860,301.93"]),
    ("Revenue grew 23.4% month on month.", ["23.4%"]),
    ("We had 5,000 customers.", ["5,000"]),
])
def test_invented_figures_fail(values, answer, bad):
    assert texts(ungrounded(answer, values)) == bad


@pytest.mark.parametrize("answer", [
    "Assumptions: 1 Jan to 14 Oct 2026, revenue excluding cancelled and returned items, Calvin Klein only.",
    "Assumptions: 1 to 30 Sept 2026.",
    "In Q3 2026 the top 10 states drove growth.",
    "Over the last 3 months, from 2026-07-01 to 2026-09-30 at 10:30.",
    "1. Texas\n2. California\n13. Ohio",
    "The 3rd and 4th largest states.",
    "Since March 15th sales rose.",
])
def test_non_data_numbers_are_ignored(values, answer):
    assert figures(answer) == [] or ungrounded(answer, values) == []


def test_strip_removes_only_ungrounded_and_adds_note(values):
    answer = "Revenue was $2.86M, up 23.4% on last year."
    out = strip_figures(answer, ungrounded(answer, values))
    assert "$2.86M" in out and "23.4%" not in out and "[removed]" in out
    assert out.endswith(GROUNDING_NOTE)


def test_compute_output_grounds_a_total():
    frames = [pd.DataFrame({"revenue": [100.0, 120.0]}), pd.DataFrame({"total_revenue": [220.0]})]
    assert ungrounded("Total revenue was $220.", ground_values(frames)) == []
    assert texts(ungrounded("Total revenue was $220.", ground_values(frames[:1]))) == ["$220"]
