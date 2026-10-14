"""Scope rewrite (D1.1, D1.20, A1).

The curated tables stay unscoped (they are the Dataform definitions). Here
each one is wrapped with the user's brand filter, passed as a query parameter,
the way row-level security would filter them in production. The CEO gets the
tables unfiltered.

- sales, products: rows of the user's brands only. Mixed orders keep only the
  user's items, so order and customer metrics are their own share (A1).
  Unbranded products (NULL brand) never match, so only the CEO sees them.
- customers: buyers of the user's brands only. Customers who never ordered are
  visible to the CEO only.
"""

from __future__ import annotations

from google.cloud import bigquery

from analyst.config import User

BRAND_PARAM = "scope_brands"


def scoped_tables(curated: dict[str, str], user: User) -> tuple[dict[str, str], list]:
    """Return {table: select} for this user, and the query parameters they need."""
    if user.all_brands:
        return dict(curated), []
    tables = {
        "sales": f"SELECT * FROM (\n{curated['sales']}\n) WHERE brand IN UNNEST(@{BRAND_PARAM})",
        "customers": f"SELECT * FROM (\n{curated['customers']}\n) WHERE customer_id IN (SELECT customer_id FROM sales)",
        "products": f"SELECT * FROM (\n{curated['products']}\n) WHERE brand IN UNNEST(@{BRAND_PARAM})",
    }
    params = [bigquery.ArrayQueryParameter(BRAND_PARAM, "STRING", list(user.brands))]
    return tables, params
