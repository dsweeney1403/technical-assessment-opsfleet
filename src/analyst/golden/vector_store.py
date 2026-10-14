"""Local vector search over the prebuilt trio index (D4.6).

The prototype holds the committed index in a numpy matrix and ranks by cosine
similarity. Production uses Firestore vector search at 2048 dimensions or
below, moving to Vector Search on the Gemini Enterprise Agent Platform at scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analyst.golden.index import IndexedTrio, TrioIndex, read_index


class VectorStore:
    def __init__(self, index: TrioIndex):
        self.model = index.model
        self.dimensions = index.dimensions
        self.trios = index.trios
        matrix = np.asarray([t.embedding for t in index.trios], dtype=np.float32).reshape(len(index.trios), -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._matrix = matrix / np.where(norms == 0, 1, norms)

    @classmethod
    def load(cls, path: Path) -> VectorStore:
        return cls(read_index(path))

    def search(self, vector: np.ndarray, n: int) -> list[tuple[IndexedTrio, float]]:
        """The n most similar trios, best first, with cosine similarity."""
        if not self.trios:
            return []
        scores = self._matrix @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores)[:n]
        return [(self.trios[i], float(scores[i])) for i in order]
