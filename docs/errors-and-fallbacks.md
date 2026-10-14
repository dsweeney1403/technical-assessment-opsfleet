# Errors and fallbacks

Every failure has one owner and one handling rule (D2.1, D2.4). The agent repairs what it caused. Code retries what an outside service caused. Anything that signals an attack stops the turn. The user only hears about a failure when nothing could recover it, and then in plain words.

## SQL gateway codes

Every query the agent writes goes through the gateway: parse, single-SELECT check, allowed tables and columns, brand fuzzy match and backstop, scope rewrite, dry run with a bytes cap, execute, store the result.

| Code | Where | Owner | Handling |
|---|---|---|---|
| `PARSE_ERROR` | Parse | Agent | Repair from the error message |
| `NOT_SELECT` | Statement check | Stop | Neutral decline, security event |
| `MULTI_STATEMENT` | Statement check | Agent | Repair, hint "single SELECT, constants in a CTE" |
| `UNKNOWN_TABLE` | Tables and columns | Agent | Repair: only `sales`, `customers`, `products` exist |
| `PII_IDENTIFIER` | Tables and columns | Stop | Neutral decline, security event |
| `GEO_TOO_FINE` | Tables and columns | Agent | Repair, hint names the finest level allowed (city) |
| `NO_ACCESS_BRAND` | Brand backstop | Agent | A typo gets "did you mean"; a brand outside the user's access is dropped, the query rerun and the gap flagged |
| `BQ_INVALID` | Dry run | Agent | Repair, BigQuery's message is the hint |
| `COST_LIMIT` | Dry run | Agent | Repair, hint "fewer columns or a narrower date range" |
| `TIMEOUT` | Execute (30 s) | Agent | The job is cancelled, hint "aggregate earlier or narrow the date range" |
| `BQ_TRANSIENT` | Execute | Code | Retried with backoff, invisible to the agent; if retries run out, BigQuery is down (below) |
| `BQ_FATAL` | Execute | User | Stop with a plain message pointing to `python -m analyst doctor` |
| `UNKNOWN_RESULT` | `compute`, `make_chart` | Agent | A result ID not in this session: rerun the query or carry on, no security event. Production code (D6.6); the prototype returns the same repairable error naming the known IDs. |

BigQuery's error reason field decides the code, never the message text. `rateLimitExceeded` is per second and transient. `quotaExceeded` is daily and fatal. They look alike but need opposite handling (D2.4, D8.5).

## Repair budget and the stuck rule (D2.2)

One budget per turn: 6 tool calls, repairs included. When the same error code comes back twice in a row, code stops that line of retrying. The next model call has no tools bound and is told to answer with what it has and say what it couldn't get. The cap is the hard guarantee; the stuck rule saves calls earlier. Each report section gets its own cap of 6 (D3.1).

## Empty results (D2.5)

An empty result isn't trusted until code has checked it:
1. **Prevention.** The schema in the prompt lists every value of the short columns (department, gender, status, traffic source, age band, country, category), so `'Brasil'` is spelled right first time.
2. **Diagnosis.** Each filter value is checked against the real values, and dates against the data's range. A mismatch returns a hint ("no match for 'Brazil' in country, closest: 'Brasil'").
3. **Structural check.** If diagnosis finds nothing, one count runs with the filters removed. Still empty points to the joins. Rows back means the empty is genuine, and the answer states the filters and dates used.

## Gemini failures

| Situation | Handling | Decision |
|---|---|---|
| 429, 5xx, timeout, connection reset | Retried in code: 3 attempts, exponential backoff with jitter, Retry-After honoured, 30 s total cap. Not counted against tool calls. | D2.8 |
| Other 4xx (bad key, bad request) | Not retried. Stops with a message pointing to `doctor`. | D2.8 |
| A model's retries run out | The next model in the chain: `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`. Fallback use is recorded in the trace, not shown to the user. | D2.10 |
| A model keeps failing | Circuit breaker: 5 consecutive transient errors skip that model for 60 s, so users don't each wait out the retries. | D2.12 |
| The classifier's models all fail | Fail open: the message goes to the agent as unclassified. Safe, because the classifier isn't a security control. | D2.11 |
| The agent's models all fail | The turn stops with "Type /retry to pick up where it stopped". `/retry` resumes the graph from the failed step through the checkpoint, keeping completed queries. | D2.13 |
| The planner or writer returns malformed output | A plain message: rephrase the request (planner), or say "make it again" to retry the writing without rerunning the sections (writer). | D3.1 |

## BigQuery down mid-turn (D2.13)

After `BQ_TRANSIENT` retries run out, code refuses further `run_sql` calls for that turn. `compute` and `make_chart` still work on stored results. The agent answers with what it has and says plainly what it couldn't fetch. Stored-result follow-ups and the light path keep working.

## Knowledge and persona failures

| Situation | Handling | Decision |
|---|---|---|
| The embedding API is down | No trios for that turn; the agent works from the glossary, and the trace logs `trios_unavailable`. Another embedding model can't stand in, because its vectors don't match the index. Production falls back to keyword search over the same trios. | D4.8 |
| The active persona can't be loaded | The highest earlier version that loads is used, and the trace records the fallback. | D4.23 |
| Grounding finds a figure with no source | One retry telling the agent which figures, then removal with a note. Glossary numbers ("90 days") count as sourced. | D1.16 |
| Leak scan hits | The answer is blocked, a safe message is shown and a security event is logged. | D1.16 |

## What the user sees

Every turn ends with one outcome and one reason code in the trace (D2.17):
- **answered** (`OK`).
- **partial**, with these reasons in this order of precedence: `BQ_DOWN`, `STUCK`, `BUDGET_SPENT`, `GROUNDING_REMOVED`, `NO_ACCESS_BRAND`.
- **declined**: a stop code, or the intent.
- **failed**: `LLM_DOWN`, `BQ_FATAL`, `LLM_FATAL`, `INTERNAL`.

The chat never shows a traceback. The trace ID under every answer opens the full story with `python -m analyst trace <id>`.

## Trying it

| Flag | What fails | What you should see |
|---|---|---|
| `--simulate-llm-429` | The agent's main model returns 429 on every call | Answers arrive from the backup model; the breaker opens after 5 errors |
| `--simulate-llm-down` | Every Gemini model returns 503 | A plain message; `/retry` picks the question up |
| `--simulate-bq-error` | Every BigQuery query fails at execution | The agent answers with what it has and says what it couldn't fetch |
| `--simulate-embedding-down` | Every embedding call returns 503 | Answers without worked examples; `trios_unavailable` in the trace |

To see the cost limit fire for real, set `bigquery.max_bytes_processed: 12000000` in `config/settings.yaml` and ask for a six-way breakdown. That query scans about 14 MB; the agent drops dimensions until it fits (D2.3).
