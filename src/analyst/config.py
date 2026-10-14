"""Load and validate settings.yaml, users.yaml and environment variables.

Sections used by the current build are typed strictly, so a typo fails at
startup. Later sections pass through untyped until their phase lands.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"


class ConfigError(Exception):
    """Configuration is missing or invalid. The message says how to fix it."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Models(_Strict):
    agent: str
    agent_fallbacks: list[str] = []
    light: str
    light_fallbacks: list[str] = []
    report: str
    indexing: str
    embedding: str
    embedding_dimensions: int = Field(le=2048)


class Budgets(_Strict):
    max_tool_calls_per_turn: int = Field(ge=1)
    stuck_same_code_limit: int = Field(ge=1)


class LLMRetry(_Strict):
    max_attempts: int = Field(ge=1)
    max_total_wait_s: float = Field(gt=0)
    honour_retry_after: bool = True
    request_timeout_s: float = Field(gt=0)


class BigQuerySettings(_Strict):
    dataset: str
    maximum_bytes_billed: int = Field(gt=0)
    job_timeout_s: float = Field(gt=0)
    transient_max_attempts: int = Field(ge=1)
    preview_rows_to_model: int = Field(ge=1)


class Memory(_Strict):
    verbatim_turns: int = Field(ge=1)
    result_ttl: Literal["session"] = "session"


class Charts(_Strict):
    dir: Path


class Settings(BaseModel):
    model_config = ConfigDict(extra="allow")

    models: Models
    budgets: Budgets
    llm_retry: LLMRetry
    bigquery: BigQuerySettings
    memory: Memory
    charts: Charts


class User(_Strict):
    id: str
    name: str
    role: Literal["ceo", "brand_manager"]
    brands: list[str] = Field(min_length=1)

    @property
    def all_brands(self) -> bool:
        return self.brands == ["*"]


class Users(_Strict):
    users: list[User]

    def get(self, user_id: str) -> User:
        for user in self.users:
            if user.id == user_id:
                return user
        known = ", ".join(u.id for u in self.users)
        raise ConfigError(f"Unknown user '{user_id}'. Known users: {known}.")


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} should contain a mapping at the top level.")
    return data


def _validate(model: type[BaseModel], data: dict, name: str):
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"{name} is invalid: {problems}") from exc


def load_settings(path: Path | None = None) -> Settings:
    path = path or CONFIG_DIR / "settings.yaml"
    return _validate(Settings, _read_yaml(path), path.name)


def load_users(path: Path | None = None) -> Users:
    path = path or CONFIG_DIR / "users.yaml"
    users = _validate(Users, _read_yaml(path), path.name)
    for user in users.users:
        if "*" in user.brands and (user.brands != ["*"] or user.role != "ceo"):
            raise ConfigError(f"users.yaml: only a CEO entry may use '*', alone ({user.id}).")
    ids = [u.id for u in users.users]
    if len(ids) != len(set(ids)):
        raise ConfigError("users.yaml: user ids must be unique.")
    return users


class Env(BaseModel):
    gemini_api_key: str
    google_cloud_project: str


def _adc_project() -> str | None:
    """The project attached to Application Default Credentials, if any."""
    try:
        import google.auth

        return google.auth.default()[1]
    except Exception:
        return None


def load_env(env_file: Path | None = None, adc_project=_adc_project) -> Env:
    """Read credentials settings (D8.3). GOOGLE_API_KEY is accepted as an alias for
    GEMINI_API_KEY; GOOGLE_CLOUD_PROJECT falls back to the ADC project."""
    load_dotenv(env_file or ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ConfigError(
            "Missing GEMINI_API_KEY. Get a free key at https://aistudio.google.com/apikey "
            "and put it in .env (copy .env.example)."
        )
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or adc_project()
    if not project:
        raise ConfigError(
            "No Google Cloud project found. Set GOOGLE_CLOUD_PROJECT in .env, or run "
            "`gcloud auth application-default set-quota-project <project-id>`. "
            "A free BigQuery sandbox project works."
        )
    return Env(gemini_api_key=key, google_cloud_project=project)
