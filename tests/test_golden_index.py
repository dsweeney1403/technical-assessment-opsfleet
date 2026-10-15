"""Trio indexing (D4.1, D4.3, D4.4, D4.7, D4.11) and the embedder (D4.6, D4.8)."""

from types import SimpleNamespace

import numpy as np
import pytest
from google.genai.errors import ClientError, ServerError

from analyst.config import load_settings, resolve
from analyst.faults import Faults, embedding_client_factory
from analyst.golden.index import (
    IndexingOutput,
    TrioSource,
    build_index,
    index_trio,
    load_trios,
    read_index,
    redact_names,
    strip,
    write_index,
)
from analyst.llm.client import LLMFatalError, LLMUnavailableError
from analyst.llm.embeddings import Embedder
from analyst.sql.brands import BrandMatcher

BRANDS = BrandMatcher(["Levi's", "Calvin Klein", "Carhartt", "Canada Goose"])
SQL = "SELECT SUM(sale_price) FROM `bigquery-public-data.thelook_ecommerce.order_items` WHERE brand = 'Levi''s'"


@pytest.mark.parametrize("text, expected", [
    ("Revenue was $2.61M, up 38% on Q1 2024, from 31,400 orders.",
     "Revenue was [X], up [X] on Q1 2024, from [X] orders."),
    ("Spend fell 11% (from $108.30 to $96.40) over 1.5 years", "Spend fell [X] (from [X] to [X]) over [X] years"),
    ("Contact jane.smith@example.com or customer 48213", "Contact [EMAIL] or [ID]"),
    ("ships to 01310-100 in Brasil", "ships to [POSTCODE] in Brasil"),
    ("Q4 2025 and B2B stay; 3x does not", "Q4 2025 and B2B stay; [X]x does not"),
])
def test_numbers_and_pattern_pii_are_stripped(text, expected):
    assert strip(text) == expected


def test_names_are_redacted_but_brand_names_are_not():
    text = "Our top customer, Jane Smith in Austin, loves Calvin Klein. Jane bought again."
    out = redact_names(text, ["Jane Smith", "Calvin Klein"], BRANDS)
    assert out == "Our top customer, [NAME] in Austin, loves Calvin Klein. [NAME] bought again."


class IndexLLM:
    def __init__(self, out):
        self.out = out
        self.messages = None

    def structured(self, role, schema, messages, stats=None):
        assert role == "indexing" and schema is IndexingOutput
        self.messages = messages
        return self.out


def trio(**kw):
    base = dict(id="t99", question="Why are Levis sales in Texas behind?", sql=SQL,
                report="Levi's revenue in Texas is 42% of California's. Our best buyer Maria Gonzalez spent $4,210.")
    return TrioSource(**{**base, **kw})


def test_index_trio_validates_brands_redacts_names_and_keeps_no_sql():
    llm = IndexLLM(IndexingOutput(brands=["Levis", "Acme Denim Co"], person_names=["Maria Gonzalez"],
                                  description="Explains a state gap for Maria Gonzalez by comparing 42% shares"))
    item, dropped = index_trio(trio(), "data/golden/trios/t99.yaml", llm, BRANDS)
    assert item.brands == ["Levi's"]  # "Levis" snaps to the catalogue spelling (D4.4)
    assert dropped == ["Acme Denim Co"]  # an invented brand never becomes a tag
    assert "Maria" not in item.report_stripped and "[NAME]" in item.report_stripped
    assert item.description == "Explains a state gap for [NAME] by comparing [X] shares"
    assert "[X]" in item.report_stripped and "4,210" not in item.report_stripped
    assert "4,210" not in llm.messages[1].content  # the LLM only sees stripped text
    assert "sql" not in item.model_dump()


class FakeEmbedder:
    model, dimensions = "fake-embedding", 4

    def __init__(self):
        self.calls = []

    def embed(self, texts, task, stats=None):
        self.calls.append((list(texts), task))
        rows = np.array([[len(t), 1.0, 0.0, 0.0] for t in texts], dtype=np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_build_index_embeds_question_plus_description_and_writes_no_sql(tmp_path):
    llm = IndexLLM(IndexingOutput(brands=[], person_names=[], description="Compares two states."))
    root = tmp_path
    src = root / "trios" / "t99.yaml"
    src.parent.mkdir()
    embedder = FakeEmbedder()
    index, _ = build_index([(trio(), src)], root, llm, embedder, BRANDS)
    assert embedder.calls == [(["Why are Levis sales in Texas behind?\nCompares two states."], "document")]
    path = tmp_path / "index.json"
    write_index(index, path)
    raw = path.read_text(encoding="utf-8")
    assert "SELECT" not in raw and "bigquery-public-data" not in raw
    back = read_index(path)
    assert back.trios[0].source_ref == "trios/t99.yaml" and len(back.trios[0].embedding) == 4


def test_seeded_trios_load_with_unique_ids():
    trios = load_trios(resolve(load_settings().golden.trios_dir))
    ids = [t.id for t, _ in trios]
    assert len(ids) >= 15 and len(set(ids)) == len(ids)
    assert any("@" in t.report for t, _ in trios)  # one trio names a customer, to test redaction


# Embedder


class FakeModels:
    def __init__(self, errors=()):
        self.errors = list(errors)
        self.calls = []

    def embed_content(self, model, contents, config):
        self.calls.append((model, list(contents), config))
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0]) for _ in contents])


def embedder(errors=()):
    models = FakeModels(errors)
    e = Embedder(load_settings(), "key", client_factory=lambda key: SimpleNamespace(models=models),
                 sleep=lambda s: None)
    return e, models


def unavailable():
    return ServerError(503, {"error": {"code": 503, "message": "down", "status": "UNAVAILABLE"}})


def test_embedder_normalises_and_uses_the_configured_size():
    e, models = embedder()
    out = e.embed(["a", "b"], "query")
    assert np.allclose(out, [[0.6, 0.8], [0.6, 0.8]])
    model, _, config = models.calls[0]
    assert model == "gemini-embedding-001" and config.output_dimensionality == load_settings().models.embedding_dimensions
    assert config.task_type == "RETRIEVAL_QUERY"


def test_embedder_retries_transient_errors_then_gives_up():
    e, models = embedder([unavailable()])
    assert e.embed(["a"], "document").shape == (1, 2) and len(models.calls) == 2
    e, _ = embedder([unavailable()] * 5)
    with pytest.raises(LLMUnavailableError):
        e.embed(["a"], "document")
    e, models = embedder([ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID"}})])
    with pytest.raises(LLMFatalError):
        e.embed(["a"], "document")
    assert len(models.calls) == 1


def test_simulated_embedding_outage():
    factory = embedding_client_factory(lambda key: None, Faults(embedding_down=True))
    e = Embedder(load_settings(), "key", client_factory=factory, sleep=lambda s: None)
    with pytest.raises(LLMUnavailableError):
        e.embed(["a"], "query")


def test_committed_index_matches_the_trios_and_settings():
    """D8.3: reviewers use the prebuilt index, so it must stay in step with the trios and config."""
    settings = load_settings()
    index = read_index(resolve(settings.golden.index))
    trios = load_trios(resolve(settings.golden.trios_dir))
    assert {t.trio_id for t in index.trios} == {t.id for t, _ in trios}
    assert index.model == settings.models.embedding and index.dimensions == settings.models.embedding_dimensions
    raw = resolve(settings.golden.index).read_text(encoding="utf-8")
    for t, _ in trios:
        assert t.sql.strip().splitlines()[0] not in raw  # SQL never reaches the index (D4.1)
    assert "@" not in raw and "Gonzalez" not in raw  # the named customer is redacted (D4.11)
    assert all(len(t.embedding) == index.dimensions for t in index.trios)
