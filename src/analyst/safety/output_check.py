"""Output check (D1.16): leak scan and number grounding, both in code.

Runs on every answer before the user sees it.

Leak scan: email shapes, postcode shapes, ID shapes ("customer 48213"), tables
with identifier columns, and brands outside the user's scope. A hit blocks the
answer and logs a security event. Brands the user named themselves, and names
that come from the user's own (already scoped) results, are allowed.

Number grounding: every figure must match a value in a stored result, within
the rounding the answer shows (17.9% supports "18%", 2,859,097 supports
"$2.86M"). Years, dates, "top 10", "Q3", list markers and small bare counts are
not data figures and are ignored. Numbers the user typed are allowed.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analyst.config import CONFIG_DIR, User

STOPLIST = CONFIG_DIR / "brand_stoplist.txt"
STRING_ROWS_SCANNED = 5000  # per text column, when collecting numbers from strings
REMOVED = "[removed]"
GROUNDING_NOTE = "Note: some figures were removed because they could not be traced to a query result."

# --- Leak scan -------------------------------------------------------------

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Hyphenated postcode shapes in the data: Brazil 99999-999, Japan 999-9999,
# South Korea 999-999, Poland 99-999. Plain digit postcodes look like counts,
# so they are only caught next to a postcode word.
POSTCODE_SHAPE = re.compile(r"(?<![\w.,$€£-])(?:\d{5}-\d{3}|\d{3}-\d{4}|\d{3}-\d{3}|\d{2}-\d{3})(?![\w-]|[.,]\d)")
POSTCODE_CONTEXT = re.compile(
    r"\b(?:post(?:al)?[\s_]?codes?|zip(?:[\s_]?codes?)?|CEP)\b[^\n\d]{0,20}?"
    r"(?:[A-Z]{1,2}\d[A-Z\d]?(?:\s*\d[A-Z]{2})?\b|\d{3,6}(?:-\d{3,4})?\b)",
    re.I,
)
ID_SHAPE = re.compile(
    r"\b(?:customer|user|buyer|shopper|client|order)(?:[ _-]?id)?\s*(?:#|no\.?|number|id)?\s*[:#=]?\s*#?\d{3,}(?![\d,.]?\d)",
    re.I,
)
IDENTIFIER_HEADERS = {
    "customer_id", "user_id", "order_id", "email", "first_name", "last_name", "full_name",
    "street_address", "address", "postal_code", "postcode", "zip", "zip_code", "latitude", "longitude",
}
TABLE_HEADER = re.compile(r"^[ \t]*\|(.+)\|[ \t]*\n[ \t]*\|[ \t:|-]+\|[ \t]*$", re.M)


def load_stoplist(path: Path = STOPLIST) -> set[str]:
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text().splitlines():
        name = line.split("\t#")[0].strip()
        if name and not name.startswith("#"):
            names.add(name)
    return names


def _norm_quotes(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u2018", "'")


@dataclass(frozen=True)
class Leak:
    kind: str  # email, postcode, id, identifier_column, brand
    match: str


class LeakScanner:
    def __init__(self, all_brands: list[str], stoplist: set[str] | None = None):
        self.all_brands = sorted(set(all_brands))
        self.stoplist = load_stoplist() if stoplist is None else stoplist
        self._patterns: dict[tuple[str, ...], re.Pattern | None] = {}

    def _brand_pattern(self, own: tuple[str, ...]) -> re.Pattern | None:
        if own not in self._patterns:
            self._patterns[own] = self._build_pattern(own)
        return self._patterns[own]

    def _build_pattern(self, own: tuple[str, ...]) -> re.Pattern | None:
        """Own brands (any case) and other brands (as written) in one pattern, longest first.

        Longest first, so "Calvin Klein Jeans" is read as that brand, not as
        "Calvin Klein" followed by a word.
        """
        own_set = set(own)
        others = [b for b in self.all_brands if b not in own_set and b not in self.stoplist]
        if not others:
            return None
        alts: list[tuple[str, str]] = [(b, f"(?i:{re.escape(_norm_quotes(b))})") for b in own]
        for b in others:
            alts.append((b, re.escape(_norm_quotes(b))))
            if b.islower():
                cap = b[0].upper() + b[1:]
                alts.append((cap, re.escape(_norm_quotes(cap))))
        alts.sort(key=lambda a: -len(a[0]))
        return re.compile("(?<!\\w)(?:" + "|".join(p for _, p in alts) + ")(?!\\w)")

    def out_of_scope_brands(self, text: str, user: User, allowed_text: str, result_text: str) -> list[str]:
        if user.all_brands:
            return []
        pattern = self._brand_pattern(tuple(user.brands))
        if pattern is None:
            return []
        own = {b.lower() for b in user.brands}
        allowed = allowed_text.lower()
        found = []
        for m in pattern.finditer(_norm_quotes(text)):
            name = m.group(0)
            low = name.lower()
            if low in own or low in allowed or low in result_text:
                continue
            found.append(name)
        return sorted(set(found))

    def scan(self, text: str, user: User, allowed_text: str = "", result_text: str = "") -> list[Leak]:
        """`allowed_text`: what the user wrote. `result_text`: string cells of the user's scoped results, lowercased."""
        leaks = [Leak("email", m.group(0)) for m in EMAIL.finditer(text)]
        leaks += [Leak("postcode", m.group(0)) for m in POSTCODE_SHAPE.finditer(text)]
        leaks += [Leak("postcode", m.group(0)) for m in POSTCODE_CONTEXT.finditer(text)]
        leaks += [Leak("id", m.group(0)) for m in ID_SHAPE.finditer(text)]
        for header in TABLE_HEADER.finditer(text):
            for cell in header.group(1).split("|"):
                name = re.sub(r"[\s`*]+", "_", cell.strip().strip("`* ").lower())
                if name in IDENTIFIER_HEADERS:
                    leaks.append(Leak("identifier_column", name))
        leaks += [Leak("brand", b) for b in self.out_of_scope_brands(text, user, allowed_text, result_text)]
        return leaks


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def result_text(frames: list[pd.DataFrame]) -> str:
    """Lowercased string cells of stored results, for the brand allowance."""
    parts = []
    for df in frames:
        for col in df.columns:
            if _is_text(df[col]):
                parts.extend(str(v).lower() for v in df[col].dropna().head(STRING_ROWS_SCANNED))
    return "\n".join(parts)


# --- Number grounding ------------------------------------------------------

MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
         r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")
DAY = r"\d{1,2}(?:st|nd|rd|th)?"
NON_DATA = re.compile("|".join([
    r"\b\d{4}-\d{2}(?:-\d{2})?(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b",  # ISO dates
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{1,2}:\d{2}(?::\d{2})?\b",  # times
    rf"\b{DAY}(?:\s*(?:to|-|\u2013|and)\s*{DAY})?\s+(?:of\s+)?{MONTH}(?!\w)",  # 1 to 30 Sept
    rf"\b{MONTH}\s+{DAY}(?:\s*(?:to|-|\u2013)\s*{DAY})?(?![\d,.]?\d)(?!\w)",  # Sept 30
    r"(?i:\b(?:top|bottom|first|last|past|previous|next|best|worst)\s+\d{1,3}\b)",  # top 10, last 3 months
    r"(?m:^[ \t]*(?:[-*][ \t]+)?\d{1,2}[.)][ \t])",  # numbered list markers
]))
NUMBER = re.compile(
    r"(?<![\w.,])(?P<cur>[$\u20ac\u00a3])?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<suf>%|percent\b|per\s?cent\b|pp\b|bn\b|million\b|billion\b|thousand\b|[kKmMbB](?!\w)))?"
    r"(?(suf)|(?!\w))"
)
SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
SMALL_COUNT = 12  # bare integers up to this are list positions and counts of things named, not figures


@dataclass(frozen=True)
class Figure:
    text: str
    start: int
    end: int
    value: float
    tolerance: float
    percent: bool


def _mask(text: str) -> str:
    return NON_DATA.sub(lambda m: " " * len(m.group(0)), text)


def figures(text: str) -> list[Figure]:
    """Data figures in the text, with the tolerance their rounding implies."""
    out = []
    for m in NUMBER.finditer(_mask(text)):
        raw, suffix = m.group("num"), (m.group("suf") or "").lower().replace(" ", "")
        value = float(raw.replace(",", ""))
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        percent = suffix in ("%", "percent", "pp")
        bare = not (m.group("cur") or suffix or "," in raw or "." in raw)
        if bare and (value <= SMALL_COUNT or 1900 <= value <= 2100):
            continue
        scale = SCALE.get(suffix, 1.0)
        tolerance = 0.5 * 10 ** -decimals * scale * (1 + 1e-6) + 1e-9
        out.append(Figure(m.group(0), m.start(), m.end(), value * scale, tolerance, percent))
    return out


def _numbers_in(texts: list[str]) -> list[float]:
    vals = []
    for t in texts:
        for m in NUMBER.finditer(t):
            vals.append(float(m.group("num").replace(",", "")))
    return vals


def ground_values(frames: list[pd.DataFrame], user_texts: list[str] = ()) -> np.ndarray:
    """Every number a figure may be grounded in: result cells (numeric and inside text), row counts, user input."""
    vals: list[float] = []
    for df in frames:
        vals.append(float(len(df)))
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                continue
            if pd.api.types.is_numeric_dtype(s):
                vals.extend(pd.to_numeric(s, errors="coerce").dropna().astype(float).tolist())
            elif _is_text(s):
                head = s.dropna().head(STRING_ROWS_SCANNED)
                vals.extend(pd.to_numeric(head, errors="coerce").dropna().astype(float).tolist())  # NUMERIC as Decimal
                vals.extend(_numbers_in([str(v) for v in head]))
    vals.extend(_numbers_in(list(user_texts)))
    arr = np.abs(np.asarray(vals, dtype=float))
    return np.unique(arr[np.isfinite(arr)])


def _near(values: np.ndarray, target: float, tolerance: float) -> bool:
    i = bisect_left(values, target - tolerance)
    return i < len(values) and values[i] <= target + tolerance


def ungrounded(text: str, values: np.ndarray) -> list[Figure]:
    missing = []
    for f in figures(text):
        target = abs(f.value)
        if _near(values, target, f.tolerance):
            continue
        # A share stored as a fraction (0.179) supports "17.9%".
        if f.percent and _near(values, target / 100, f.tolerance / 100):
            continue
        missing.append(f)
    return missing


def strip_figures(text: str, figs: list[Figure]) -> str:
    for f in sorted(figs, key=lambda f: f.start, reverse=True):
        text = text[: f.start] + REMOVED + text[f.end:]
    return f"{text}\n\n{GROUNDING_NOTE}"


def retry_instruction(figs: list[Figure]) -> str:
    listed = ", ".join(dict.fromkeys(f.text.strip() for f in figs))
    return (
        f"These figures in your answer do not come from any tool result: {listed}. "
        "Every figure must come from a tool result. For a total, average, share, difference, percent change, "
        "ratio or rank of results you already have, call compute. Otherwise remove the figure. "
        "Then give the full answer again."
    )
