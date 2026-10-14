"""Brand matching against entitlements (D2.6, D1.8 row 6).

A scoped user is matched against their own brands only, so they learn nothing
about the wider catalogue: close to one of theirs gets "did you mean", anything
else is "not in your access" whether or not it exists. The CEO is matched
against every brand.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from analyst.config import User
from analyst.sql.validator import BrandLiteral, Rejection

CUTOFF = 0.8


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass(frozen=True)
class BrandCheck:
    value: str
    status: str  # ok | typo | no_access
    did_you_mean: tuple[str, ...] = ()


class BrandMatcher:
    def __init__(self, all_brands: list[str]):
        self.all_brands = sorted(set(all_brands))

    def candidates(self, user: User) -> list[str]:
        return self.all_brands if user.all_brands else list(user.brands)

    def check(self, value: str, user: User, case_insensitive: bool = False) -> BrandCheck:
        pool = self.candidates(user)
        if value in pool or (case_insensitive and value.lower() in {b.lower() for b in pool}):
            return BrandCheck(value, "ok")
        by_norm: dict[str, list[str]] = {}
        for b in pool:
            by_norm.setdefault(normalise(b), []).append(b)
        key = normalise(value)
        if key in by_norm:
            return BrandCheck(value, "typo", tuple(by_norm[key]))
        close = difflib.get_close_matches(key, list(by_norm), n=3, cutoff=CUTOFF)
        if close:
            return BrandCheck(value, "typo", tuple(b for k in close for b in by_norm[k]))
        # The CEO sees every brand, so an unknown name is just an empty result.
        return BrandCheck(value, "ok" if user.all_brands else "no_access")

    def check_literals(self, literals: list[BrandLiteral], user: User) -> Rejection | None:
        """Gateway backstop: a query naming brands outside the user's access never runs."""
        problems = []
        for lit in literals:
            result = self.check(lit.value, user, lit.case_insensitive)
            if result.status == "typo":
                problems.append({"value": lit.value, "did_you_mean": list(result.did_you_mean)})
            elif result.status == "no_access":
                problems.append({"value": lit.value, "status": "not in your access"})
        if not problems:
            return None
        return Rejection(
            "NO_ACCESS_BRAND",
            "Some brand names in the query don't match the user's brands.",
            "For a typo, use the suggested spelling. Drop any brand marked not in your access, "
            "rerun, and tell the user it is outside their access. Never report it as having no sales.",
            brands=problems,
        )

    def out_of_scope_mentions(self, mentioned: list[str], user: User) -> list[str]:
        """Brands the user mentioned that aren't theirs, for the agent's scope instructions."""
        if user.all_brands:
            return []
        return [m for m in mentioned if self.check(m, user, case_insensitive=True).status == "no_access"]
