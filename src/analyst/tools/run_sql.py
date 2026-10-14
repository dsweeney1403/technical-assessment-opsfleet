"""`run_sql` tool: the agent's only way to read data. Everything goes through the gateway."""

from __future__ import annotations

import json

from langchain_core.tools import StructuredTool

from analyst.config import User
from analyst.sql.gateway import Gateway

DESCRIPTION = (
    "Run one BigQuery Standard SQL SELECT against the tables sales, customers and products. "
    "Returns a result_id, the columns, row_count, a preview of up to 50 rows and quality flags, "
    "or an error with a hint. Aggregate in SQL; don't fetch raw rows to count them yourself."
)


def make_run_sql_tool(gateway: Gateway, user: User) -> StructuredTool:
    def run_sql(sql: str) -> str:
        return json.dumps(gateway.run(sql, user), default=str)

    return StructuredTool.from_function(run_sql, name="run_sql", description=DESCRIPTION)
