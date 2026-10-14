import pytest
import yaml

from analyst.config import ConfigError, load_env, load_settings, load_users


def test_repo_settings_load():
    s = load_settings()
    assert s.models.agent
    assert s.budgets.max_tool_calls_per_turn == 6
    assert s.memory.verbatim_turns == 6


def test_repo_users_load():
    users = load_users()
    assert users.get("ceo").all_brands
    assert users.get("bm_levis").brands == ["Levi's", "Canada Goose"]


def test_unknown_user_lists_known_ids():
    with pytest.raises(ConfigError, match="bm_calvin"):
        load_users().get("nobody")


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data))
    return p


def test_typo_in_typed_section_fails(tmp_path):
    data = yaml.safe_load(open("config/settings.yaml"))
    data["budgets"]["max_tool_calls_per_trun"] = 6
    with pytest.raises(ConfigError, match="max_tool_calls_per_trun"):
        load_settings(_write(tmp_path, "settings.yaml", data))


def test_embedding_dimension_cap(tmp_path):
    data = yaml.safe_load(open("config/settings.yaml"))
    data["models"]["embedding_dimensions"] = 3072
    with pytest.raises(ConfigError, match="embedding_dimensions"):
        load_settings(_write(tmp_path, "settings.yaml", data))


def test_star_only_for_ceo(tmp_path):
    data = {"users": [{"id": "x", "name": "X", "role": "brand_manager", "brands": ["*"]}]}
    with pytest.raises(ConfigError, match="only a CEO"):
        load_users(_write(tmp_path, "users.yaml", data))


def test_duplicate_user_ids(tmp_path):
    u = {"id": "x", "name": "X", "role": "brand_manager", "brands": ["Diesel"]}
    with pytest.raises(ConfigError, match="unique"):
        load_users(_write(tmp_path, "users.yaml", {"users": [u, u]}))


def test_missing_env(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        load_env(tmp_path / "missing.env")
