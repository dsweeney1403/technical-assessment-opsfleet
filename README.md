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
| `--simulate-embedding-down` | Every embedding call returns 503 | Answers still arrive, without worked examples. The trace shows `trios_unavailable` on the retrieval step. |

```bash
python -m analyst chat --user ceo --simulate-llm-429
```

To see the cost limit fire for real, set `bigquery.max_bytes_processed: 12000000` in `config/settings.yaml` and ask "Break down revenue and margin by brand, category, department, state, age band and traffic source". That query scans about 14 MB and is refused; the agent drops dimensions until the query fits and says what it couldn't include. Leave `maximum_bytes_billed` as it is: BigQuery bills every sales query at least 42 MB.

## Seeing what happened in a turn

Every answer ends with a short grey line such as `trace 7f3a1c2e`. Quote it when something looks wrong. To open that turn:

```bash
python -m analyst trace 7f3a           # a unique prefix is enough
python -m analyst trace 7f3a --full    # adds the full rewritten SQL, every model message and each tool result
```

The tree shows each step: the intent classifier, every agent step with its model call (model, tokens, latency, retries, fallback), every query with its SQL and the brand scope the access rewrite applied, the error code or row count, and the output check. It ends with one outcome (answered, partial, declined or failed) and one reason code, such as `OK`, `STUCK`, `BQ_DOWN` or `NOT_SELECT`.

In the chat, `/stats` shows metrics for the session in two views. The business view covers outcome rates, retries, fallback use, tokens, bytes and p50 and p95 latency. The engineering view covers first-attempt SQL success, repairs, stuck-rule fires, empty results, security events and figures removed by the grounding check.

Traces are JSONL files in `traces/`, one per conversation. They hold what the model saw, never raw BigQuery rows, and they are gitignored.

## Reports

Ask for a report in plain words, for example "Create a Q1 report with action items for Q2".

1. The agent plans an outline. If any section needs new queries, it shows the outline and waits. Reply yes to run it, cancel to stop, or say what to change and it plans again. A report built only from results already in the chat runs straight away.
2. Each section runs its own short analysis, then the report is written: an assumptions line, the data's as-of time, findings, action items and next questions.
3. Every action item has to rest on a finding or on something you told the agent. Code checks this. An action with nothing behind it moves to Next questions as an offer to look into it. Suggested targets are labelled as suggestions.
4. Say "save it" to keep the report. "Make it shorter" or "add a section on returns" creates a new version of the same report.

`/reports` lists your saved reports and `/download <number>` writes one to `reports/` as markdown, with its charts alongside. You only ever see your own reports. If your brand access changes, reports covering brands you no longer have are hidden until access returns.

Deleting is always confirmed. Say "delete all reports mentioning Levi's" and the agent lists what that matches. Reply yes to delete them, cancel to keep them, or narrow the list ("only the jeans one"), which shows the shorter list for a fresh yes. Only the reports shown are deleted. A delete is soft: an admin can restore it for 30 days, and every delete is recorded in an audit table with the titles, the request and the trace id.

Reports and the audit table live in `reports.db`, a local SQLite file that is gitignored.

## Definitions, worked examples and tone

**Glossary.** `config/glossary.json` defines every metric and term: revenue, margin, average order value, churn (no purchase in 90 days), what "last quarter" or "year to date" means, and how mixed-brand orders are counted. The whole glossary goes to the agent on every analysis and data question, and it wins over everything else. Editing a definition is one change in one file.

**Worked examples.** `data/golden/trios/` holds 15 example analyses in the client's format: a question, the analyst's SQL and the analyst's report. Before each analysis answer, and once at the start of a report, the app finds up to 3 examples similar to the question and shows them to the agent as an approach to follow. The agent never sees their SQL or their figures, only the question, a one-line description and the report with numbers replaced by `[X]`. Names and emails are redacted. An example tagged with a brand reaches only users who can see that brand. The retrieval step in `trace <id>` shows which examples were used and their scores.

The index in `data/golden/index.json` is prebuilt and committed, so you don't need to run anything. To rebuild it after changing the examples, run `python -m analyst golden index`, then `python -m analyst golden calibrate` to check the similarity threshold. The current threshold and how it was chosen are in [docs/golden-calibration.md](docs/golden-calibration.md).

**House style.** The persona in `config/persona/` sets the tone. `active` names the version in use (`v1` or `v2`). The file is read on every turn, so editing it, or pointing `active` at another version, changes the next answer with no restart. If the active file is missing, empty or broken, the previous version is used and the trace says so. `/persona` shows which version is active. Before switching versions, run `python -m analyst persona test`. It asks a handful of questions, including attempts to get around the rules, and prints pass or fail for each. A persona only changes style. It can't change access, privacy rules or definitions, because those are enforced in code.

## Preferences and feedback

Tell the agent how you like answers, for example "I prefer tables" or "keep it short", and it remembers across sessions. If you ask for the same format three times, it becomes your default without you saying so. Something you said yourself always beats something it inferred. You can also set analysis defaults, such as "default to last quarter", and answers that use one say so in their assumptions line. Without one, questions with no period cover the year to date.

Definitions and safety are not preferences. Asking to count returns as revenue gets the company definition. Asking to hide the scope note, the trace id or customer privacy is declined.

`/prefs` shows your preferences with where each came from and when it was set. `/prefs reset` clears them. Preferences live in `reports.db`.

After each answer in a terminal, press `u` (helpful), `d` (not helpful) or Enter to skip. A thumbs down asks for an optional one-line reason. The rating is stored next to the answer's trace and counted in `/stats`.
