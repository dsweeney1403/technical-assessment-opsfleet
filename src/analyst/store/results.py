"""Session result store (D1.15). Holds post-gateway results under a result ID.

`compute` and `make_chart` take an ID, so full results never enter the model's
context (D1.12). Lives for the CLI session (D1.18).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StoredResult:
    result_id: str
    df: pd.DataFrame
    sql: str  # as the model wrote it
    sql_rewritten: str  # as executed
    data_ts: str


class ResultNotFound(KeyError):
    pass


class ResultStore:
    def __init__(self) -> None:
        self._results: dict[str, StoredResult] = {}

    def put(self, df: pd.DataFrame, sql: str, sql_rewritten: str, data_ts: str) -> StoredResult:
        result_id = f"r{len(self._results) + 1}"
        stored = StoredResult(result_id, df, sql, sql_rewritten, data_ts)
        self._results[result_id] = stored
        return stored

    def get(self, result_id: str) -> StoredResult:
        try:
            return self._results[result_id]
        except KeyError:
            known = ", ".join(self._results) or "none yet"
            raise ResultNotFound(f"No result '{result_id}'. Known results: {known}.") from None

    def __contains__(self, result_id: str) -> bool:
        return result_id in self._results
