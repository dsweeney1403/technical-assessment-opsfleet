"""User preferences: what counts, how each kind is learned, and how they reach the model (D4.15, D4.16).

The classifier only reports what the user asked for; code decides what is saved.
- Presentation (format, length, charts): stated ones are saved at once; one-off
  format requests are counted across sessions and set the preference after N,
  as `inferred`. A stated preference beats an inferred one.
- Analysis defaults (period, breakdown, brand order): saved only when stated,
  and shown in the assumptions line as "your default".
- Definitions belong to the glossary, never to a user (D4.2).
- Safety controls are never preferences: declined, never saved.
Values come from fixed lists, or are short plain text for analysis defaults,
so a preference can't smuggle instructions into the prompt.
"""

from __future__ import annotations

import re

from analyst.state import StatedPreference
from analyst.store.base import UserProfile

# format_request from the classifier -> (presentation field, value)
FORMAT_REQUESTS = {
    "table": ("format", "table"),
    "bullets": ("format", "bullets"),
    "just_the_number": ("format", "just the number"),
    "chart": ("charts", "include a chart"),
    "shorter": ("length", "short"),
    "longer": ("length", "long"),
}
PRESENTATION = {
    "format": {"table": "table", "tables": "table", "bullets": "bullets", "bullet points": "bullets",
               "just the number": "just the number", "number only": "just the number", "prose": "prose",
               "paragraphs": "prose"},
    "length": {"short": "short", "shorter": "short", "brief": "short", "long": "long", "longer": "long",
               "detailed": "long"},
    "charts": {"yes": "include a chart", "include a chart": "include a chart", "charts": "include a chart",
               "no": "no charts", "no charts": "no charts"},
}
DEFAULT_FIELDS = {"period": "period", "breakdown": "breakdown", "brand_order": "brand order"}
PLAIN_VALUE = re.compile(r"[\w][\w '&/-]{0,39}")
LABELS = {"format": "Format", "length": "Length", "charts": "Charts", **{k: v.capitalize() for k, v in
                                                                         DEFAULT_FIELDS.items()}}


def presentation_value(field: str, value: str) -> str | None:
    """The canonical value for a presentation field, or None if it isn't one we support."""
    return PRESENTATION.get(field, {}).get(" ".join(value.lower().replace("_", " ").split()))


def default_value(field: str, value: str) -> str | None:
    """A stated analysis default, as short plain text, or None."""
    value = " ".join(value.split())
    if field not in DEFAULT_FIELDS or not PLAIN_VALUE.fullmatch(value):
        return None
    return value


def valid(pref: StatedPreference) -> StatedPreference | None:
    """Code's check on what the classifier extracted. Unsupported values are ignored, never saved."""
    if pref.kind == "presentation":
        value = presentation_value(pref.field, pref.value)
        return pref.model_copy(update={"value": value}) if value else None
    if pref.kind == "analysis_default":
        value = default_value(pref.field, pref.value)
        return pref.model_copy(update={"value": value}) if value else None
    return pref  # definitions and safety controls are handled, never saved


def acknowledgement(pref: StatedPreference) -> str:
    if pref.kind == "analysis_default":
        return (f"Noted: {pref.value} is your default {DEFAULT_FIELDS[pref.field]}. "
                "Answers that use it say so in the assumptions line. See /prefs to review or reset.")
    return f"Noted: {LABELS[pref.field].lower()} set to {pref.value} for your answers. See /prefs to review or reset."


def render(profile: UserProfile | None) -> str:
    """The preferences prompt layer (D4.24)."""
    if profile is None or not profile.presentation:
        return ""
    lines = [f"- {LABELS.get(p.field, p.field)}: {p.value}" for p in profile.presentation.values()]
    lines.append("A request in the current message wins over these for that answer.")
    return "\n".join(lines)


def own_defaults(profile: UserProfile | None) -> tuple[tuple[str, str], ...]:
    if profile is None:
        return ()
    return tuple((f, p.value) for f, p in profile.analysis_defaults.items())
