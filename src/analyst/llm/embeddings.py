"""Embeddings for trio retrieval (D4.6, D4.7, D4.8).

`gemini-embedding-001` at the configured output size (2048 or below, to fit
Firestore in production). Calls use the same transient-only retries as the chat
models (D2.8). There is no fallback model: another model's vectors don't match
the index, so an outage means no trios for that turn (D4.8).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

import numpy as np

from analyst.config import Settings
from analyst.llm.client import CallStats, with_retries

Task = Literal["document", "query"]
TASK_TYPES = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}
BATCH = 100  # texts per request


def _genai_client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


class Embedder:
    def __init__(self, settings: Settings, api_key: str, client_factory: Callable = _genai_client,
                 sleep: Callable[[float], None] = time.sleep):
        self.model = settings.models.embedding
        self.dimensions = settings.models.embedding_dimensions
        self._retry = settings.llm_retry
        self._client = client_factory(api_key)
        self._sleep = sleep

    def embed(self, texts: list[str], task: Task, stats: CallStats | None = None) -> np.ndarray:
        """One unit-length row per text. Raises LLMUnavailableError once retries run out."""
        from google.genai import types

        config = types.EmbedContentConfig(task_type=TASK_TYPES[task], output_dimensionality=self.dimensions)
        rows = []
        for start in range(0, len(texts), BATCH):
            batch = texts[start:start + BATCH]
            response = with_retries(
                lambda: self._client.models.embed_content(model=self.model, contents=batch, config=config),
                self._retry, stats, sleep=self._sleep)
            rows += [e.values for e in response.embeddings]
        vectors = np.asarray(rows, dtype=np.float32)
        # Below the full 3072 dimensions the vectors are not unit length, so normalise for cosine similarity.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)
