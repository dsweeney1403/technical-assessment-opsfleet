"""Report store interface (D3.10). The prototype uses SQLite; production adds a Firestore adapter.

Every method takes the signed-in `User`, injected by code, never an owner id
chosen by the model (D3.13). Reads exclude deleted reports and reports the user
is no longer entitled to (D3.15): a report is visible only while the user's
current brands cover its `brands_scope`. Hidden reports reappear if access
returns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

from analyst.config import User


class SavedFinding(BaseModel):
    finding_id: str
    text: str
    result_id: str | None = None
    sql: str | None = None


class NewVersion(BaseModel):
    """What the report flow hands to the store. Owner, scope, version and time are set by the store."""

    report_id: str | None = None  # None for a new report; set for a redo (D3.9)
    conversation_id: str
    title: str
    body: str
    chart_files: list[str] = Field(default_factory=list)
    data_ts: str
    trace_id: str
    findings: list[SavedFinding] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)


class ReportVersion(NewVersion):
    report_id: str
    version: int
    owner_id: str
    created_at: str
    brands_scope: list[str]
    deleted_at: str | None = None


class AuditEntry(BaseModel):
    audit_id: int
    user_id: str
    at: str
    action: str  # delete | restore
    request_text: str
    report_ids: list[str]
    titles: list[str]  # never bodies (D3.24)
    trace_id: str


def brands_scope(user: User) -> list[str]:
    """The user's brands at save time (D3.11). The CEO's reports are scoped to all brands."""
    return ["*"] if user.all_brands else sorted(user.brands)


def visible_to(user: User, scope: list[str]) -> bool:
    """D3.15: the user's current brands must cover the report's scope."""
    return set(scope) <= set(user.brands)


class ReportStore(ABC):
    @abstractmethod
    def save(self, user: User, report: NewVersion) -> ReportVersion:
        """Save a new report, or a new version of one the user can see."""

    @abstractmethod
    def get(self, user: User, report_id: str) -> ReportVersion | None:
        """Latest version, or None if missing, deleted, hidden or not the user's."""

    @abstractmethod
    def list(self, user: User) -> list[ReportVersion]:
        """Latest version of each visible report, newest first."""

    @abstractmethod
    def match_mentions(self, user: User, term: str) -> list[ReportVersion]:
        """Visible reports whose `mentions` match the term, fuzzily (D3.18)."""

    @abstractmethod
    def created_in(self, user: User, conversation_id: str) -> list[ReportVersion]:
        """Visible reports first created in this conversation (D3.19)."""

    @abstractmethod
    def soft_delete(self, user: User, report_ids: list[str], request_text: str, trace_id: str) -> list[ReportVersion]:
        """Set `deleted_at` on every version and write the audit row, in one transaction (D3.23, D3.24).
        Only the user's visible reports are touched. Returns what was deleted."""

    @abstractmethod
    def purge(self, days: int) -> int:
        """Hard-delete reports soft-deleted more than `days` ago. Returns how many reports went."""

    @abstractmethod
    def restore(self, report_id: str, admin_id: str, reason: str) -> bool:
        """Admin only, outside the chat (D3.23): undo a soft delete, audited."""

    @abstractmethod
    def audit(self) -> list[AuditEntry]:
        """Admin only: the audit trail, oldest first."""


# User profiles (D4.15, D4.18, D4.25)


class Preference(BaseModel):
    field: str  # presentation: format | length | charts; analysis defaults: period | breakdown | brand_order
    value: str
    source: Literal["stated", "inferred"]  # analysis defaults are always stated (D4.15)
    set_at: str


class UserProfile(BaseModel):
    """One per user. Remembered context items (D4.17) are design only, so not stored here."""

    user_id: str
    presentation: dict[str, Preference] = Field(default_factory=dict)
    analysis_defaults: dict[str, Preference] = Field(default_factory=dict)


class ProfileStore(ABC):
    """Every method takes the signed-in `User`, injected by code. The prototype uses SQLite in the
    reports file; production needs only a Firestore adapter (D4.25)."""

    @abstractmethod
    def get(self, user: User) -> UserProfile:
        """The user's profile; empty if they have none."""

    @abstractmethod
    def set_presentation(self, user: User, field: str, value: str, source: str) -> Preference | None:
        """Save a presentation preference. An inferred value never replaces a stated one: returns None then."""

    @abstractmethod
    def set_default(self, user: User, field: str, value: str) -> Preference:
        """Save an analysis default. Only ever stated (D4.15)."""

    @abstractmethod
    def count_format(self, user: User, request: str) -> int:
        """Count one format request across sessions (D4.16). Returns the new count."""

    @abstractmethod
    def reset(self, user: User) -> None:
        """Clear the user's preferences, defaults and request counts (/prefs reset)."""
