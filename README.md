# OpsFleet Analyst

A data analysis chat agent for a retail company's executives. Built with LangGraph, Gemini and BigQuery (`bigquery-public-data.thelook_ecommerce`).

Work in progress. Full setup, usage and design docs come later.

## Setup

You need Python 3.12 or newer, a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey), and Google Cloud credentials for BigQuery. A free [BigQuery sandbox](https://cloud.google.com/bigquery/docs/sandbox) project works, no credit card needed. Nothing is deployed: the agent runs locally and queries the public dataset.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # add your Gemini key
gcloud auth application-default login     # BigQuery credentials
python -m analyst doctor                  # checks everything and names the fix for any failure
python -m analyst chat --user ceo
```

Run the tests with `pytest`. They are mocked and need no credentials.

## Trying the failure handling

Each flag makes one dependency fail on purpose, so you can see the agent recover without a real outage.

| Flag | What fails | What you should see |
|---|---|---|
| `--simulate-llm-429` | The agent's main Gemini model returns 429 on every call | Answers still arrive from the backup model. After 5 errors the main model is skipped for 60 s, so later answers come faster. |
| `--simulate-llm-down` | Every Gemini model returns 503 | A plain message that Gemini is unavailable. `/retry` picks the question up where it stopped. |
| `--simulate-bq-error` | Every BigQuery query fails | The agent stops querying, answers with what it already has and says what it couldn't fetch. Charts and calculations on earlier results still work. |

```bash
python -m analyst chat --user ceo --simulate-llm-429
```

To see the cost limit fire for real, set `bigquery.max_bytes_processed: 12000000` in `config/settings.yaml` and ask "Break down revenue and margin by brand, category, department, state, age band and traffic source". That query scans about 14 MB and is refused; the agent drops dimensions until the query fits and says what it couldn't include. Leave `maximum_bytes_billed` as it is: BigQuery bills every sales query at least 42 MB.
