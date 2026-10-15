# How each requirement is met

For each of the brief's requirements: what the design does, what the prototype builds, and the decisions behind it. The prototype must implement R2, R3, R5 and R7. The rest are designed in full and built where a small change demonstrates them (D2.15).

The guiding rule throughout: **the model proposes, code decides.** Access, PII, read-only, the output check and the delete gate are enforced in code and BigQuery, never by asking the model to behave.

## R1 Hybrid intelligence: the Golden Bucket

Two kinds of knowledge reach the agent, for two different jobs (D4.1, D4.2).

- **The glossary** says what words mean: revenue, margin, average order value, churn (no purchase in 90 days), what "last quarter" means, how mixed-brand orders count. It's one file (`config/glossary.json`; in production owned by the data team next to the curated SQL). The whole glossary is in context on every analysis and data question, and it wins over everything else. A changed definition is one edit, not a fix in every trio.
- **Trios** (question, SQL, analyst report) show how an analyst approached a similar question. The agent sees the question, a one-line description and the report. It never sees the SQL, because that targets tables the agent can't query, may contain blocked columns, and may encode rules that conflict with the glossary. Converting it would be an LLM step that can hallucinate.

**Providing trios at query time (D4.5 to D4.9).** Retrieval is a forced graph step, not a tool, because the agent is least likely to ask for help when it is confidently wrong. On every analysis turn, and once at the start of a report, code:
1. Embeds the question with `gemini-embedding-001`.
2. Takes the 20 most similar trios.
3. Drops any trio tagged with a brand outside the user's access.
4. Drops anything below a similarity threshold, and keeps up to 3.

Zero trios is a valid outcome: the agent works from the glossary. The threshold (0.71) was set from a labelled set of rephrasings, leaning towards precision ([calibration](golden-calibration.md)). The trace logs which trios were used and their scores.

**Indexing (D4.3, D4.4, D4.7, D4.11).**
- Numbers are stripped to `[X]` before anything is indexed, so stale or out-of-scope figures can't leak. The agent learns approach, not figures.
- An LLM extracts brand names, a one-line description and person names. Code keeps only brands that match the catalogue and replaces names with `[NAME]`. Emails, IDs and postcodes are pattern-matched out.
- The question plus description is embedded.

**Keeping the bucket current (D4.10 to D4.14, design).**
- Analysts write trios, and saved agent answers can become candidates. A code filter admits a candidate only if it was saved, nothing was removed by grounding, the stuck rule didn't fire, and no fallback model was used. An analyst approves every addition, and each decision is logged with a reason.
- A glossary or schema change flags trios that mention the changed term for review. Near-duplicates are flagged at indexing.

**Prototype:** 15 seeded trios, the indexing command, a committed prebuilt index, forced retrieval with the brand filter and threshold, an embedding-outage fallback, and calibration.

## R2 Safety and PII masking

**Only analysis questions (D1.8, D1.9).** A classifier routes every message: analysis, data question, conversation, report action, off-topic, PII about individuals, injection, destructive. It is for routing, not security. When unsure about intent it fails open to the agent. Injection or destructive parts decline the whole message and log a security event. Declines are written by code and are neutral, never accusatory (D2.1).

**PII never reaches the model (D1.3, D1.4, D1.6, D1.7).**
- The agent queries three curated tables (`sales`, `customers`, `products`) that hold no names, emails, street addresses, postcodes or coordinates. Age is a band and sign-up time is a month.
- In the prototype, a query naming a blocked column is stopped before it runs (`PII_IDENTIFIER`, a security event).
- Answers describe groups, never individuals: "top customers" is a segment with its size, revenue share and profile.
- Nothing the model sees is PII, so traces can store what the model saw.

**Each user sees only their brands (D1.1, D1.2, D1.20, A1).**
- In production, BigQuery row access policies are granted to brand groups, and queries run with the user's own OAuth token.
- In the prototype, the gateway rewrites every query so the three tables are already filtered to the user's brands, with brands passed as parameters.
- Mixed-brand orders count only the user's own items. The CEO sees everything.
- A brand literal outside the user's access is caught before the query runs (`NO_ACCESS_BRAND`). It is fuzzy-matched against the user's own brands only, so a typo gets "did you mean" and a scoped user learns nothing about the wider catalogue (D2.6).
- Scoped answers end with a scope note added by code.

**Every query is one SELECT (D2.1).** The validator (sqlglot) accepts a single SELECT over the three tables. DML, multiple statements, unknown tables and `INFORMATION_SCHEMA` never reach BigQuery.

**Output check (D1.16).** Before an answer is shown, code scans it:
- **Leak scan.** Email, postcode and ID shapes, identifier column headers, and any brand outside the user's scope. A hit blocks the answer and logs a security event.
- **Number grounding.** Every figure must appear in a query result, the user's own words or the glossary. Otherwise the agent gets one retry, then ungrounded figures are removed with a note.

**Prototype:** all of the above, in the CLI, with three demo users (`ceo`, `bm_calvin`, `bm_levis`).

## R3 High-stakes oversight: deleting reports

**The model can only propose (D3.26).** Its one delete-related tool is `propose_delete(filter)`. The filter is `{type: mentions, value}` or `{type: conversation}`.

Code then:
1. Resolves the filter to report IDs. Only the user's own, visible reports are candidates. Matching is fuzzy on a `mentions` list that code built at save time (D3.17, D3.18).
2. Stops for confirmation with a LangGraph `interrupt()`. It shows the titles: "This deletes 'Levi's jeans review' and its earlier drafts".
3. Classifies the reply with an LLM, seeing exactly what was shown (D3.20, D3.21):
   - **yes** deletes.
   - **narrow** ("only the jeans one") shows the shorter list for a fresh yes.
   - **cancel** stops.
   - **unclear** asks again. A yes with a condition is narrow, and a doubting yes is never a yes.
4. Deletes exactly the IDs shown, never re-running the filter. The delete is soft, covers every version, and is audited (user, time, request, IDs, titles, trace ID) in the same transaction (D3.22 to D3.24).

**Owner only, for every role (D3.13).** Code injects the owner on every store method, so not even the CEO can open, search or delete someone else's reports. A report whose brands the user no longer fully covers is hidden until access returns (D3.15).

**Without breaking UX.** One confirmation, plain titles, natural replies ("go for it" works), and unsaved session reports are discarded with a one-line mention instead of a prompt (D3.19). A report title holding an injected instruction can at worst produce a proposal the user sees and rejects (D3.27).

**Reports themselves (D3.1 to D3.9).**
- A planner writes an outline, tagging each section "from this chat's results" or "new queries". Any new-queries section means the outline is shown and confirmed first.
- Sections run in order, each in its own bounded loop.
- A writer returns structure: findings with result IDs, action items with an anchored why and a suggested how, and next questions. Code checks every action's anchor; an action with nothing behind it becomes an offer to investigate. Suggested targets are labelled.
- Reports are offered for saving; "make it shorter" creates a new version.

**Saved reports go to Google Docs (D3.14).** The company is on Google Workspace, so a saved report is also written as a Google Doc. Executives read and share it where they already work, and nothing needs downloading to a laptop. The Doc stays under the app's control:
- It lives in a shared drive owned by the agent's service account and is shared with its owner only, as a commenter (owner-only reading, D3.13).
- Viewers can't download, print or copy it, and the drive only allows sharing inside the company domain.
- A new version updates the same Doc, so Docs' version history matches the report's versions.
- A delete moves the Doc to the drive's trash, which keeps it restorable for 30 days, the same window as the soft delete (D3.23). The audit row records it.
- If the owner loses access to a brand the report covers, their access to the Doc is removed until it returns (D3.15).
- Firestore stays the source of truth for versions, mentions, deletes and the audit table. The Doc is how the report is read.

**Prototype:** all of the above except the Google Doc, with a SQLite store behind the `ReportStore` interface. `/reports` lists saved reports, and `/download <number>` writes one to the `reports/` folder as markdown with its charts, printing the full path. That file stands in for the Doc. A local file is outside the app's access controls, which is the reason production uses Docs instead.

## R4 Continuous improvement

**User level (D4.15 to D4.18).**
- **Presentation** (format, length, charts) is learned two ways. A stated preference ("I prefer tables") is saved at once. Repeated requests (three times, from the classifier's `format_request` field) set an inferred one. Stated beats inferred.
- **Analysis defaults** ("default to last quarter") are saved only when stated, and every answer that uses one says so in its assumptions line. Silently learned defaults would change numbers invisibly.
- **Definitions** are never preferences; the glossary owns them. **Safety controls** (the scope note, privacy rules, the trace ID) are never preferences and are declined.
- `/prefs` shows and resets the profile.
- Remembered business context ("Q2 campaign in Spain") is announced, expires, and is design only.

**System level (D4.19 to D4.21).** Three signals feed the loop:
- System signals from traces: repairs, stuck rule, grounding removals, fallback use, declines.
- Thumbs up or down under every answer, with an optional reason.
- Implicit signals in production: rephrases and abandoned sessions, found by a nightly post-session review job.

Numbers tune themselves within limits, logged and reversible, such as the trio threshold refitted from trace scores. Anything with words in it (glossary, prompts, trios) stays human-approved and passes the eval gate. The system can draft such changes; a human approves each release (D4.20).

**Prototype:** stated and inferred presentation preferences, explicit analysis defaults, safety controls declined, `/prefs`, thumbs feedback in the trace and `/stats`.

## R5 Resilience and graceful error handling

See [errors and fallbacks](errors-and-fallbacks.md) for every code and owner. In short:
- Bad SQL comes back to the agent as a structured error with a hint, and the agent repairs it. Repairs count toward the 6-call cap. The same error twice in a row stops retrying, enforced in code (D2.2).
- An empty result is checked before it is trusted. Code compares filter values with the real values and dates with the data's range, then runs one count with the filters removed (D2.5).
- Transient Gemini and BigQuery errors are retried in code. Then the agent falls back to other models, and a circuit breaker skips a failing model for 60 s. If the classifier fails, the message goes to the agent. If BigQuery goes down, the agent answers with what it has. If every model is down, `/retry` resumes from the checkpoint (D2.8 to D2.13).
- Cost is bounded by construction: the tool-call cap, compact 50-row previews, a light path for simple turns, a bytes cap on every query, and Flash-Lite for routing (D1.12, D2.3, D2.9).
- Fault-injection flags let a reviewer trigger each failure (D2.14).

**Prototype:** all of the above.

## R6 Quality assurance

See [evaluation](evaluation.md). Deterministic code gets unit tests, which must pass at 100%. Each LLM step gets a component set with an exact check. A small end-to-end set catches interactions. The red team is scored on a safety outcome that must be 100%, and separately on behaviour (D5.1, D5.2). Report intent is checked against analyst-written required elements, for both outlines and full reports (D5.8, D5.9). UX is measured three ways: implicit signals, explicit thumbs, and a pilot (D5.14).

**Prototype:** unit tests in CI, the component sets, end-to-end conversations, the red team and `python -m analyst eval` with a committed baseline.

## R7 Observability

**Every turn is one trace (D2.16, D2.17, D2.23).** It is OpenTelemetry-shaped and holds:
- The classifier verdict.
- Each agent step's model call: model, messages, tokens, latency, retries, fallback.
- Each tool call as returned to the model, and each query's SQL before and after the scope rewrite, with error code, bytes and rows.
- Retrieval: trios and scores.
- The output check.
- One outcome (answered, partial, declined, failed) and one reason code.

"Why is it failing?" becomes one group-by.

**Metrics by audience (D2.18).** The business view covers outcome rates, retries, fallback use, breaker opens, tokens, bytes and p50 and p95 latency. The engineering view covers first-attempt SQL success, repairs, stuck-rule fires, genuine versus diagnosed empties, security events, grounding removals and thumbs. Production alerts on leak-scan blocks, breaker opens and job failures.

**Deep dive (D2.19, D2.20).** A trace ID appears under every answer, not only failures, because the worst answers are confident wrong ones. `python -m analyst trace <id>` renders the turn as a tree, and `--full` adds every message. In production that is Cloud Trace plus BigQuery, with a pre-filled ticket from a trace (D2.21).

**Prototype:** JSONL traces, the viewer and `/stats`.

## R8 Persona management

**Design (D4.22 to D4.24).**
- The persona lives in prompt management on the Gemini Enterprise Agent Platform. It is versioned, edited by the CEO's office in Agent Studio, and fetched at runtime with a short cache, so no redeploy is needed. Rollback is switching back to an earlier version.
- The persona sits in its own prompt block, framed as style guidance, below safety and the user's preferences.
- It can't weaken safety, because safety is in code: the worst a bad persona can do is make the model try something the gateway blocks.
- A smoke test of a handful of questions, including red-team ones, runs before a version goes live. If a version can't be loaded, the previous one is used.

**Prototype:** `config/persona/v1.md` and `v2.md` with an `active` pointer, re-read every turn. There is last-known-good fallback, the version in the trace, `/persona`, and `python -m analyst persona test`.

## Extensibility (D7.1 to D7.4)

A new capability is a tool with a declared risk class. The graph enforces each class once, rather than each tool remembering its own checks:

| Class | Graph behaviour | Examples |
|---|---|---|
| `read` | Runs freely, counts toward the tool cap | `run_sql`, `compute`, `make_chart`, report reads |
| `write` | Runs, audited | save report, scheduled reports (D7.4) |
| `external` | Confirmation first, showing recipients and a preview | email a report, post to Chat or Slack (D7.2) |
| `destructive` | Model proposes only; confirmation, soft, audited | `propose_delete` |

Adding one means writing the tool, declaring its class, adding unit, eval and red-team cases, and registering it.
- **Charts** already work this way.
- **Web search for trends** goes through Gemini's grounding with Google Search on the Agent Platform. Results are labelled external and never blended into internal figures (D7.3).
- **Scheduled reports** run as their owner, using the stored refresh token, so BigQuery applies the owner's row policy (D7.4).

## Traceability

| Item | Decisions | In the prototype |
|---|---|---|
| R1 Golden Bucket | D4.1 to D4.14 | Seeded trios, indexing, retrieval, calibration (D4.25) |
| R2 Safety and PII | D1.1 to D1.9, D1.16, D1.20, D2.1, D2.6, D8.2, D6.2, D6.6, D6.13, D7.5 | Yes |
| R3 Delete oversight | D3.12 to D3.27 | Yes |
| R4 Learning | D4.15 to D4.21 | Preferences and feedback (D4.25) |
| R5 Resilience | D2.1 to D2.15, D6.3 | Yes |
| R6 QA | D5.1 to D5.16, D6.16, D6.18, D6.20 | Unit tests, eval sets, baseline (D8.1) |
| R7 Observability | D2.16 to D2.23, D6.18 | Yes |
| R8 Persona | D4.22 to D4.24 | Local versioned files (D4.25) |
| Extensibility | D7.1 to D7.4 | Design only |
| Deliverable 1, diagram | D6.19 | [architecture](architecture.md) |
| Deliverable 2a, reasoning | D1.19, D6.1 to D6.20 | [architecture](architecture.md) |
| Deliverable 2b, data flow | Group 1 Interfaces, D6.19 | [architecture](architecture.md) |
| Deliverable 2c, errors and fallbacks | Group 2 | [errors and fallbacks](errors-and-fallbacks.md) |
| Deliverable 2d, setup and example run | D8.3 | [README](../README.md), [example run](example-run.md) |
| Deliverable 6, framework and experience | D1.19 | [architecture](architecture.md#why-langgraph-and-gemini) |
