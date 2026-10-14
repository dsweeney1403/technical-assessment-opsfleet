from analyst.config import User
from analyst.sql.brands import BrandMatcher, normalise
from analyst.sql.validator import BrandLiteral

ALL = ["Levi's", "Calvin Klein", "Calvin Klein Jeans", "Canada Goose", "Carhartt", "Diesel",
       "BLOOMINGDALES", "Bloomingdale's"]
CEO = User(id="ceo", name="CEO", role="ceo", brands=["*"])
BM = User(id="bm_levis", name="BM", role="brand_manager", brands=["Levi's", "Canada Goose"])
CK = User(id="bm_calvin", name="CK", role="brand_manager", brands=["Calvin Klein"])
m = BrandMatcher(ALL)


def test_normalise():
    assert normalise("Levi’s") == normalise("levis") == "levis"


def test_own_brand_ok():
    assert m.check("Levi's", BM).status == "ok"


def test_typo_of_own_brand_suggests_it():
    for typo in ["levis", "Levis", "LEVI'S", "Levvi's"]:
        r = m.check(typo, BM)
        assert r.status == "typo" and r.did_you_mean == ("Levi's",), typo


def test_case_insensitive_comparison_is_fine():
    assert m.check("levi's", BM, case_insensitive=True).status == "ok"


def test_other_brand_is_no_access_without_revealing_existence():
    assert m.check("Carhartt", BM).status == "no_access"
    assert m.check("Made Up Brand", BM).status == "no_access"  # same answer for a brand that doesn't exist


def test_scoped_user_only_matched_against_own_brands():
    r = m.check("Calvin Klein Jeans", CK)
    assert r.status == "typo" and r.did_you_mean == ("Calvin Klein",)  # never confirms the sibling brand exists
    assert m.check("calvin", BM).status == "no_access"


def test_ceo_matched_against_all_brands():
    assert m.check("Carhartt", CEO).status == "ok"
    assert m.check("carhart", CEO).did_you_mean == ("Carhartt",)
    assert set(m.check("bloomingdales", CEO).did_you_mean) == {"BLOOMINGDALES", "Bloomingdale's"}
    assert m.check("Unknown Brand", CEO).status == "ok"  # just returns empty


def test_backstop_rejection_lists_each_brand():
    rej = m.check_literals([BrandLiteral("Carhartt", False), BrandLiteral("levis", False), BrandLiteral("Levi's", False)], BM)
    assert rej.code == "NO_ACCESS_BRAND" and not rej.stop
    assert rej.brands == [{"value": "Carhartt", "status": "not in your access"},
                          {"value": "levis", "did_you_mean": ["Levi's"]}]
    assert "no sales" in rej.hint


def test_backstop_passes_own_brands():
    assert m.check_literals([BrandLiteral("Canada Goose", False)], BM) is None


def test_out_of_scope_mentions():
    assert m.out_of_scope_mentions(["Carhartt", "levis", "Levi's"], BM) == ["Carhartt"]
    assert m.out_of_scope_mentions(["Carhartt"], CEO) == []
