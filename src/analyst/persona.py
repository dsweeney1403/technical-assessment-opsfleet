"""Persona: versioned files plus an active pointer, re-read every turn (D4.22, D4.23).

`config/persona/` holds `v1.md`, `v2.md`, ... and a file `active` naming one of
them. Editing a file or the pointer changes the next answer with no restart;
rollback is pointing `active` at an earlier version. If the active version
can't be loaded, the highest earlier version that loads is used instead (last
known good) and the fallback is recorded for the trace. The persona is style
guidance only: safety lives in code, so a bad persona can't weaken it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from analyst.config import resolve

MAX_CHARS = 4000  # a persona is a short style guide; anything longer is a mistake
VERSION = re.compile(r"v(\d+)")


class PersonaError(Exception):
    """A persona version that can't be used. The message says why."""


@dataclass(frozen=True)
class Persona:
    version: str  # the version in use, or "none" when nothing loads
    text: str
    fallback_from: str | None = None  # the active version that failed, if any
    error: str | None = None  # why it failed


def _number(version: str) -> int | None:
    m = VERSION.fullmatch(version)
    return int(m.group(1)) if m else None


def versions(directory: Path) -> list[str]:
    """Persona versions on disk, oldest first."""
    found = [p.stem for p in Path(directory).glob("v*.md") if _number(p.stem) is not None]
    return sorted(found, key=_number)


def read_version(directory: Path, version: str) -> str:
    path = Path(directory) / f"{version}.md"
    if _number(version) is None:
        raise PersonaError(f"'{version}' is not a persona version (expected v1, v2, ...)")
    if not path.exists():
        raise PersonaError(f"{path.name} is missing")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PersonaError(f"{path.name} can't be read: {type(exc).__name__}") from exc
    if not text.strip():
        raise PersonaError(f"{path.name} is empty")
    if len(text) > MAX_CHARS:
        raise PersonaError(f"{path.name} is longer than {MAX_CHARS} characters")
    return text


def active_version(directory: Path) -> str:
    path = Path(directory) / "active"
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def load_persona(directory: Path) -> Persona:
    directory = resolve(Path(directory))
    active = active_version(directory)
    try:
        return Persona(active, read_version(directory, active))
    except PersonaError as exc:
        error = str(exc) if active else "the active pointer is missing or empty"
    # Last known good: the highest version below the active one, or the highest of all if the pointer is unusable.
    limit = _number(active)
    for version in reversed(versions(directory)):
        if limit is not None and _number(version) >= limit:
            continue
        try:
            return Persona(version, read_version(directory, version), fallback_from=active or "(none)", error=error)
        except PersonaError:
            continue
    return Persona("none", "", fallback_from=active or "(none)", error=error)
