"""Decline wording, written by code (D1.8, D2.1).

Neutral, never accusatory: the cause may be the model, not the user. Each
message says what the agent can do instead.
"""

from __future__ import annotations

from analyst.config import User

DECLINES = {
    "off_topic": (
        "I can only help with questions about the company's sales, customers and products. "
        "For example: revenue by month, top categories, or how customers in one state compare with another."
    ),
    "pii_individuals": (
        "I analyse groups, not individual people, so I can't share details about specific customers. "
        "I can describe customer segments instead, for example your top customers by age band and location."
    ),
    "injection": (
        "I can't change how I work or share my instructions. "
        "I'm happy to help with a question about sales, customers or products."
    ),
    "destructive": "I can only read data, so I can't change or delete it.",
    "NOT_SELECT": "I can only read data, so I can't do that.",
    "PII_IDENTIFIER": (
        "That would need personal details such as names, emails or addresses, which I don't have access to. "
        "I can answer it with customer segments instead, for example by age band, gender or location."
    ),
    "safety_control": (
        "The scope note, the trace ID and the privacy rules are part of how the app works for everyone, "
        "so I can't switch them off. I can change how answers look, for example tables or shorter answers."
    ),
    "LEAK_BLOCKED": (
        "I can't show that answer because it may include details I'm not allowed to share. "
        "Try asking about groups, for example by age band, state or category."
    ),
}

# Reasons logged as security events (D1.8 rows 8 and 9, D2.1). Off-topic and PII
# requests are ordinary questions, not attacks.
SECURITY_REASONS = {"injection", "destructive", "NOT_SELECT", "PII_IDENTIFIER"}


def _names(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def no_access(brands: list[str], user: User) -> str:
    """Fully out-of-scope question (D1.8 row 6): say so and offer the user's own brands."""
    verb = "is" if len(brands) == 1 else "are"
    return (f"{_names(brands)} {verb} not in your access, so I can't report on {'it' if len(brands) == 1 else 'them'}. "
            f"I can answer this for {_names(list(user.brands))} instead.")


def scope_note(user: User, excluded: list[str]) -> str:
    """Appended by code to scoped answers, so it can't be skipped (D1.8 row 6, D4.16)."""
    if user.all_brands:
        return ""
    note = f"Scope: covers {_names(list(user.brands))} only."
    if excluded:
        verb = "is" if len(excluded) == 1 else "are"
        note += f" {_names(excluded)} {verb} outside your access, so not included."
    return note
