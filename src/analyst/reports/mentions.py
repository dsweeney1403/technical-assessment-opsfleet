"""Code-built `mentions` for a saved report (D3.18).

At save time, code scans the report body and its results for known values:
brands, categories, departments, countries, states and cities. Delete requests
("reports mentioning Levi's") match on this field, so matching is deterministic
and needs no search index. A term that isn't a known value matches nothing.

The body is matched case-sensitively as written in the data, the same rule as
the brand leak scan (D1.16), and stoplisted brand names (common words such as
"Carolina") are skipped in prose. Result columns are matched exactly, so a
stoplisted brand still counts when a result row names it.
"""

from __future__ import annotations

import re

import pandas as pd

from analyst.safety.output_check import load_stoplist
from analyst.sql.diagnosis import KnownValues

MENTION_COLUMNS = ("category", "department", "country", "state", "city")


class MentionScanner:
    def __init__(self, brands: list[str], known: KnownValues | None, stoplist: set[str] | None = None):
        values = {b for b in brands if b}
        if known is not None:
            for col in MENTION_COLUMNS:
                values.update(v for v in known.values.get(col, []) if v)
        self.values = values
        skip = load_stoplist() if stoplist is None else stoplist
        in_prose = sorted((v for v in values if v not in skip and len(v) > 2), key=len, reverse=True)
        self._pattern = (re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, in_prose)) + r")(?!\w)")
                         if in_prose else None)

    def scan(self, body: str, frames: list[pd.DataFrame] = ()) -> list[str]:
        found = set(self._pattern.findall(body)) if self._pattern else set()
        for df in frames:
            for col in df.columns:
                if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                    found.update(v for v in df[col].dropna().unique() if isinstance(v, str) and v in self.values)
        return sorted(found)
