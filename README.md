# OpsFleet Analyst

A data analysis chat agent for a retail company's executives. Built with LangGraph, Gemini and BigQuery (`bigquery-public-data.thelook_ecommerce`).

Work in progress. Full setup, usage and design docs come later.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in your project and Gemini key
gcloud auth application-default login
python -m analyst
```

Run the tests with `pytest`.
