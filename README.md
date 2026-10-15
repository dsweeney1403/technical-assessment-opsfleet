# OpsFleet Analyst

A data analysis chat agent for a retail company's executives. Ask about sales, customers and products in plain words, discuss the answers, and turn them into reports with action items. Built with LangGraph, Gemini and BigQuery (`bigquery-public-data.thelook_ecommerce`), run from the command line.

The prototype implements safety and PII masking, report deletion with confirmation, resilience and observability in full, plus seeded knowledge retrieval, learned preferences and a swappable persona. The production design covers everything else.

| Document | What's in it |
|---|---|
| [Architecture](docs/architecture.md) | The production diagram, every service and why, how a turn flows, prototype against production, why LangGraph and Gemini |
| [Requirements](docs/requirements.md) | How each of the brief's requirements is met, extensibility, traceability |
| [Errors and fallbacks](docs/errors-and-fallbacks.md) | Every error code, who handles it, retries, fallbacks, outages |
| [Evaluation](docs/evaluation.md) | Testing before deployment, report intent, UX, and the prototype's eval baseline |
| [Example run](docs/example-run.md) | A recorded session covering every feature |
| [Trio calibration](docs/golden-calibration.md) | How the retrieval threshold was set |
| [Design decisions](docs/decisions.md) | All 144 design decisions: what was decided, why, and what was rejected. The IDs in the other documents point here |

## Setup

You need:
- Python 3.12, 3.13 or 3.14. Check with `python3 --version`. On a Mac the built-in `python3` is often older. Install a newer one, for example with `brew install python@3.13`, and create the venv with `python3.13 -m venv .venv`.
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
- Google Cloud credentials for BigQuery. A free [BigQuery sandbox](https://cloud.google.com/bigquery/docs/sandbox) project works, with no credit card needed.
- The [gcloud CLI](https://cloud.google.com/sdk/docs/install), to sign in for BigQuery.

Nothing is deployed. The agent runs locally, queries the public dataset (about 6 MB per query, inside the free tier) and keeps everything else in local files created on first run.

```bash
git clone https://github.com/dsweeney1403/technical-assessment-opsfleet.git
cd technical-assessment-opsfleet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # add your Gemini key
gcloud auth application-default login     # BigQuery credentials
python -m analyst doctor                  # checks everything and names the fix for any failure
python -m analyst chat --user ceo
```

On Windows, activate with `.venv\Scripts\activate` and copy the file with `copy .env.example .env`. The other commands are the same.

Run `doctor` first. It checks, in a few seconds:
- The config files load.
- `.env` has a Gemini key, and the key can use every model the agent needs.
- Your Google credentials can read the public dataset.
- The three tables the agent queries build correctly and match the raw data's totals.

Each line says OK or FAIL, and every FAIL names the exact fix, so a setup problem never shows up mid-chat. It's free: the Gemini check only lists models, which doesn't count against your daily requests, and the BigQuery checks stay inside the free tier. If `GOOGLE_CLOUD_PROJECT` isn't set in `.env`, the project attached to your Google Cloud credentials is used.

Three demo users are in `config/users.yaml`:

| User | Sees |
|---|---|
| `ceo` | Every brand |
| `bm_calvin` | Calvin Klein only |
| `bm_levis` | Levi's and Canada Goose |

In the chat, `/help` lists the commands and `/exit` leaves.

### First things to try

| As | Type | What you should see |
|---|---|---|
| `ceo` | `What data do you have?` | A short answer with no query |
| `ceo` | `What was revenue by month in 2025?` | A table that opens with its assumptions line |
| `ceo` | `Just the last three months of 2025, and make it a chart.` | A chart drawn from the stored result, no new query |
| `ceo` | `List the email addresses of our ten best customers.` | A polite decline: personal details are never returned |
| `bm_calvin` | `How did Calvin Klein do in 2025 compared with Levi's?` | Calvin Klein's figures, with a note that Levi's is outside this user's access |
| `ceo` | `Create a Q1 2026 report with action items for Q2.` | An outline that waits for your yes, then the report. Say `save it`, then `/reports` to list it and `/download 1` to save it as a markdown file. The full path is printed |
| `ceo` | `Delete the report I made in this conversation.` (in the same chat) | The report listed, deleted only after you reply yes. Reports can also be picked by what they mention, such as `delete all reports mentioning Jeans` |

Every answer ends with a trace ID. In a second terminal, `python -m analyst trace <id>` shows each step, query and model. To switch user, leave with `/exit` and start `python -m analyst chat --user bm_calvin`. A full recorded session is in [the example run](docs/example-run.md).

**On the free tier:** the two Flash models the agent uses first each allow about 20 requests a day. An analysis question uses 2 to 8 of them and a full report about 30. So a day's free quota covers about 6 to 8 analysis questions, or one report and a few questions, on the Flash models. Declines, the delete flow and the fault flags use almost none. After that, the agent falls back to Flash-Lite: answers keep coming, but may be less thorough. If every model is limited, the agent says so plainly, and `/retry` picks the question up later. Your exact limits are at [aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit).

Run the tests with `pytest`. They are mocked and need no credentials, and they also run on every push in GitHub Actions, on Linux and Windows.

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

`/reports` lists your saved reports and `/download <number>` writes one to the `reports/` folder in the repo as markdown, with its charts alongside, and prints the file's full path. In production, reports go to Google Docs instead of a local file (see [requirements](docs/requirements.md#r3-high-stakes-oversight-deleting-reports)). You only ever see your own reports. If your brand access changes, reports covering brands you no longer have are hidden until access returns.

Deleting is always confirmed. Say "delete all reports mentioning Levi's" and the agent lists what that matches. Reply yes to delete them, cancel to keep them, or narrow the list ("only the jeans one"), which shows the shorter list for a fresh yes. Only the reports shown are deleted. A delete is soft: an admin can restore it for 30 days, and every delete is recorded in an audit table with the titles, the request and the trace id.

Reports and the audit table live in `reports.db`, a local SQLite file that is gitignored.

## Evaluating the agent

`python -m analyst eval` runs the eval sets against the real models and data:
- The intent classifier, the delete-reply classifier, trio retrieval and SQL accuracy.
- Five end-to-end conversations.
- A red team of 16 attacks, each run three times. A red team is a set of deliberate attacks that try to make the agent break a safety rule: leak PII, show another brand's figures, run a non-SELECT query or delete a report without a yes.

It prints a score per set and the red-team safety outcome, which must be 100%: the command exits with code 1 otherwise. Results are written as `eval_runs` and `eval_cases` files in `evals/results/`. Open any failed case with `python -m analyst trace <id>`. The first baseline is committed in `evals/results/baseline.json`. A full run is about 400 Gemini requests, more than a free key allows in a day. On the free tier, try `--only classifier` or `--only delete_reply` (Flash-Lite only, a few minutes each at the default throttle) and read the baseline for the rest. See [evaluation](docs/evaluation.md).

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
