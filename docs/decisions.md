# Design decisions

Every design choice in this project was made as a numbered decision, between 9 and 15 October 2026. Each one compared options and recorded why one won and what was rejected. The IDs (D1.1, D6.4, ...) appear throughout the other documents and in code comments, so any line of code or any claim in the write-up can be traced back to the reasoning behind it.

This page is a condensed record of all 144 decisions. Each entry gives the decision, the reason and, where it mattered, what was rejected. Some decisions changed as the build taught us something. Those are kept and marked, because the change is part of the reasoning.

Two rules shaped everything else:
- **Safety is enforced in code, never in prompts.** The model proposes; code decides scope, PII, SELECT-only, the output check and the delete gate.
- **The prototype inclusion rule (D2.15).** A feature is built only if it is a small code change and demonstrates a requirement from the brief. Everything else is designed in full and written up as production.

Contents:
1. [Core flow and safety](#1-core-flow-and-safety) (D1)
2. [Failure handling and observability](#2-failure-handling-and-observability) (D2)
3. [Reports and delete oversight](#3-reports-and-delete-oversight) (D3)
4. [Knowledge, learning and persona](#4-knowledge-learning-and-persona) (D4)
5. [Quality and evaluation](#5-quality-and-evaluation) (D5)
6. [Production platform](#6-production-platform) (D6)
7. [Extensibility](#7-extensibility) (D7)
8. [Build-time decisions](#8-build-time-decisions) (D8)

## 1. Core flow and safety

**D1.1 Brand enforcement.** Prototype: code rewrites every query so each table becomes a version filtered to the user's brands, with brands passed as parameters. Production: BigQuery row-level security, because the company owns its data there. *Why not row security in the prototype:* the prototype simulates users with `--user`, so every query runs under one credential and BigQuery can't tell users apart.

**D1.1a Keep the rewrite in production as a second layer. Rejected.** Keeping the same rule in two places is overhead, and access control in BigQuery is the better system. It can be added back if the client wants a second layer.

**D1.2 Production data access.** A curated BigQuery dataset, rebuilt on a schedule and owned by the data team. One row per order item, with no PII copied in. Brand access is a row access policy granted to Google Groups, one group per brand. Queries run with the user's own sign-in token, so BigQuery applies that user's policy. The agent has no access to the raw dataset. *Why:* keep everything inside Google Cloud, automated with managed services, with as little of our own code as possible.

**D1.3 Geography and blocked columns.** Names, email, street address, postcode, latitude, longitude and the location point are blocked for every role, the CEO included: they are identifying precision, not analysis depth. Country, state and city are allowed for every role, because the client confirmed city-level analysis is fine. *Amended by D8.2:* small cities are shown as they are.

**D1.4 The model never sees PII.** Blocked columns don't exist in anything the model can query. In production the curated dataset never copies them; in the prototype the schema omits them and a query naming one is rejected before it runs. *Why:* stricter than the brief's "don't display". PII can't leak because it isn't there, not because the model was told to hide it. It also keeps PII out of traces and logs.

**D1.5 Minimum group size. Superseded by D8.2.** Rows built from fewer than 10 customers were to be hidden in code, because 967 brands sold fewer than 10 items and even a state could identify someone. First amended to a roll-up (D2.7), then removed at build time.

**D1.6 Descriptive columns.** Age only as bands, sign-up time only by month, gender and traffic source as they are. The curated tables store the band and month, so the model never sees exact values.

**D1.7 No individual customers in answers.** No customer IDs and no per-customer lists. "Top customers" is answered as a segment ("your top 1% drive 18% of revenue; most are 35 to 44, in California and Texas"). Customer IDs stay usable inside queries for joins and distinct counts. *Why:* that is how an analyst would write it in a report anyway, and it is now the main guard against answers about individuals.

**D1.8 Input handling.** The classifier routes; security is enforced behind it. When unsure, the message goes to the agent: fail open on intent, never on access. Nine kinds of message, each with its handling: analysis, questions about the data, conversation, report actions, off-topic, other brands, PII, prompt injection and destructive requests. A brand outside the user's access is answered from the data they can see, with the gap flagged, never "no sales". Any injection or destructive part declines the whole message.

**D1.9 Classifier step.** A small, fast model classifies every message into structured output: intent, brands mentioned, follow-up flag and reason. It exists for routing, observability and evaluation. It is explicitly not a security control.

**D1.10 Agent shape.** A fixed outer graph (input, classifier, agent loop, forced output check, user) around one agent looping over tools, capped at 6 tool calls. *Why:* the data is small enough that a separate SQL agent isn't justified, one loop gives one trace, and the whole conversation stays in context. *Upgrade path:* split out a SQL specialist when the schema grows or SQL accuracy needs its own evaluation.

**D1.11 Analysis and data access in one loop.** Explaining "why" means the next query depends on the last result, so the agent that reasons must be able to ask for more data. There is no LLM "analysis tool". Arithmetic is done in code by `compute`, so every number in an answer comes from code, not the model.

**D1.12 Keeping the agent's job small.** Tool results come back as a capped preview (about 50 rows) with quality flags computed in code; the full result is stored under an ID that `compute` and `make_chart` take instead of the data. Simple turns ("thanks", "what tables do you have?") take a light path with no tool loop.

**D1.13 Conversation memory.** Chat history plus a structured record per answered question: question, SQL, result ID, metric, filters, grouping and data time. Full visibility without bloating context. Whether to rewrite a query is always the agent's call.

**D1.14 Long conversations.** The last few turns are kept word for word and older turns are summarised. Records are never summarised, and numbers always come from records, never from summaries.

**D1.15 Follow-ups reuse results.** A follow-up that only presents or computes on an answer reuses the stored result. A new shape (by month, a different filter) reruns the recorded SQL, modified. Re-querying everything was ruled out.

**D1.16 Output check.** Two checks in code before any answer is shown. The leak scan looks for anything shaped like an email, postcode or ID, and any brand outside the user's access; a hit blocks the answer and logs a security event. Number grounding requires every figure to come from a stored result, within rounding; an unsupported figure gets one retry, then is removed with a note. An LLM judge is a later extension.

**D1.17 `compute` is a fixed menu.** Percent change, share, ratio, difference, rank, per customer and similar. *Why:* no model-written code runs, so there is no extra security surface.

**D1.18 Starting settings.** 6 tool calls per turn, 50 preview rows, 6 verbatim turns, a 200 MB bytes cap per query. All in config.

**D1.19 Framework: LangGraph.** Production experience with it, and its node-and-graph model fits a fixed flow with checkpoints and interrupts. Learning Google's ADK for a time-boxed prototype didn't make sense; on a long-running project it would be worth evaluating.

**D1.20 What the model queries.** Three curated tables: `sales` (one row per order item, with product, brand and non-PII customer attributes), `customers` (every user, including the 20,013 who never ordered) and `products` (including never-sold ones). The prototype builds them as `WITH` clauses over the public tables, already scoped; production builds the same SQL in Dataform. Cancelled and returned items stay in, with their status; revenue rules live in the glossary. A reconciliation check proves nothing is lost: "completeness is tested, not assumed".

**D1.21 Refresh.** Production tables rebuild nightly with Dataform's scheduler. Reconciliation checks run as assertions, and the new tables are published only if they pass. Every answer shows its as-of time. The honest wording is "at most a day old", not "never outdated".

## 2. Failure handling and observability

**D2.1 Stop or repair.** A query that tries to write, or names a PII column, means something went wrong upstream, so the turn stops with a neutral decline and a security event, instead of quietly repairing it. Harmless problems (several statements, a column finer than allowed) are repaired. Declines never accuse the user, since the cause may be the model.

**D2.2 Repair budget.** One budget: 6 tool calls per turn, repairs included. If the same error comes back twice in a row, code removes the tools and the agent answers with what it has, saying what it couldn't get. *Production:* tighten to three failures in a row of any kind, which also catches a fix that undoes the last fix.

**D2.3 Timeout and cost limit.** A 30-second query timeout, with the job cancelled and a hint to aggregate earlier. A dry-run bytes gate for cost, with a hint to select fewer columns. At build time two settings proved necessary: our own dry-run gate (lowered to demo the limit) and BigQuery's billing backstop, because BigQuery bills at least 10 MB per table read.

**D2.4 BigQuery errors split by owner.** One code, one handling rule, decided by BigQuery's error reason, never its message text. Invalid queries go back to the agent; transient errors are retried by code, out of the agent's sight; fatal errors stop with a pointer to `doctor`. A per-second rate limit and a daily quota look alike but need opposite handling.

**D2.5 Empty results.** An empty result isn't trusted until code checks it. Prevention: the prompt lists every value of the short columns, so `'Brasil'` is spelled right first time. Diagnosis: filter values are checked against real values and dates against the data's range. A structural check then reruns without filters to tell a join problem from a genuine empty.

**D2.6 Brand typos.** Brand names are fuzzy-matched against the user's own brands only. Close to one of theirs: "did you mean Levi's?". Otherwise: "not in your access", whether or not the brand exists, so a scoped user learns nothing about the wider catalogue.

**D2.7 Roll up small groups. Superseded by D8.2.** Small groups were to merge into one "Other" row so totals still add up. Removed at build time when real data showed it hid almost all city analysis.

**D2.8 Model retries.** Code retries transient Gemini errors (429, 5xx, timeouts) with backoff, up to 3 attempts and 30 seconds, honouring the server's suggested wait. These don't count against the agent's 6 calls, because they aren't the model's mistake. Other 4xx errors fail the same way every time, so they stop at once.

**D2.9 Token budgets.** No per-turn token cap: input per call is bounded by construction, and tokens are logged per turn. *Production:* a per-user daily budget, with a polite message and reset time when it runs out. A per-turn cap would cut a good analysis off mid-way; the 6-call cap stops runaway loops.

**D2.10 Model fallback.** A chain within Gemini: `gemini-3.8-flash`, then `gemini-3.6-flash`, then `gemini-3.5-flash-lite`. *Production:* other providers' models hosted on the Gemini Enterprise Agent Platform, so prompts never leave Google Cloud. External gateways were rejected for that reason. Fallback use is recorded in the trace, not shown to the user.

**D2.11 Classifier failure.** If every model fails, the message goes to the agent unclassified. Safe because the classifier isn't a security control.

**D2.12 Circuit breaker.** After 5 consecutive transient errors, the primary model is skipped for 60 seconds, so users don't each wait out the retries during an outage. Small, and it demonstrates resilience, so it is in the prototype.

**D2.13 BigQuery down mid-turn.** There is no other data source, so further queries are refused for the turn, the agent answers with what it has and says what it couldn't fetch. If every Gemini model is down, `/retry` resumes from the checkpoint.

**D2.14 Fault injection.** Flags that make Gemini or BigQuery fail on purpose, so reviewers can trigger the resilience themselves.

**D2.15 Prototype inclusion rule.** Built only if it is a small code change and demonstrates a brief requirement. Otherwise it is a production note.

**D2.16 Trace content.** One trace per turn holding what the model saw and said, the SQL before and after the rewrite, codes, tokens, bytes and latency. Safe because the model never sees PII. Traces are gitignored. *Production:* scan the user's message before logging, keep traces 30 days, restrict who can read them.

**D2.17 Outcome and reason on every turn.** Each turn ends as answered, partial, declined or failed, with one reason code. "Why is it failing?" becomes one group-by.

**D2.18 Metrics by audience.** A business view (outcomes, reliability, cost and speed) and an engineering view (SQL repairs, empties, security events, grounding removals). Alerts are separate from dashboards: any leak-scan block, a breaker opening, a failure rate over threshold.

**D2.19 Trace ID under every answer.** On every answer, not only failures, because the worst answers are confident wrong ones that don't look like failures.

**D2.20 Trace viewer.** `python -m analyst trace <id>` shows one turn as a tree: steps, codes, SQL, rows, latency, tokens. `--full` adds the complete messages. Production reads the same view from BigQuery.

**D2.21 From bad answer to fix.** Design only. An engineer reviews the trace, then a command opens a pre-filled ticket. A fix is done when it ships with a new eval case, so the same question can't silently regress.

**D2.22 Answers state their assumptions.** Every analysis answer opens with the dates, metric definition and scope it used, so users catch a wrong reading themselves. Engineers can't review every chat.

**D2.23 Production tracing on Google Cloud.** OpenTelemetry, with spans to Cloud Trace, full payloads to BigQuery and dashboards in Cloud Monitoring. The prototype writes the same span shape to local files, so production is an exporter swap. *Rejected:* hosted LangSmith and Langfuse (a third party would see prompts and results); self-hosted Langfuse (operating cost, no managed option on Google Cloud).

## 3. Reports and delete oversight

**D3.1 Plan then execute.** A planner writes the outline, each section runs in its own bounded agent loop, a writer assembles, then the output check runs. Reports built from the conversation reuse stored results, so report numbers match the chat. *Rejected:* a fixed template (too rigid); report mode inside the chat loop (lower quality).

**D3.2 Outline confirmation.** The planner tags each section "from this chat" or "new queries". Any new-queries section means the outline is shown and confirmed first. Any other reply goes back to the planner as feedback, with no round limit, since each round is started by a person.

**D3.3 Action items.** Each has a "why here", anchored to a finding or to labelled user context, and a "suggested how", worded as a suggestion. Code checks the anchor. An action with no anchor becomes an offer to investigate. The agent recommends; the executive decides.

**D3.4 User context in actions.** Context the user gives ("we're launching in Spain in Q2") can justify an action, labelled as user-provided and never presented as a finding.

**D3.5 Sections run in order** in the prototype, to respect free-tier rate limits. Parallel sections are a production upgrade.

**D3.6 Reports from an earlier conversation.** Production reruns the saved SQL, states the new as-of time and notes that figures may differ.

**D3.7 Numeric targets.** Allowed when labelled "suggested target" with their basis. Labelled targets are exempt from number grounding.

**D3.8 Saving.** Offered after each report; "save it" works any time later in the session. Chosen over auto-save and explicit-only.

**D3.9 Redo creates a new version.** "Make it shorter" is a new version of the same report. The writer gets the old version's structured findings, so grounding still has results to check against.

**D3.10 Report store.** Production: Firestore, whose full-text search covers "reports mentioning X". Prototype: SQLite behind a `ReportStore` interface, so production needs only a new adapter.

**D3.11 Stored fields.** Including `brands_scope`, the user's brands at save time. Access rules govern BigQuery queries, not a saved snapshot, and the owner field says who owns a report, not whose data is in it.

**D3.12 Delete removes every version.** Users only know the latest version, so a delete removes them all, and the confirmation says so in plain words.

**D3.13 Owner-only reading, every role.** Not even the CEO can open someone else's reports through the chat. Code injects the owner into every store call; the model never passes a user ID. Reports hold other managers' analysis and typed plans, which BigQuery access doesn't cover.

**D3.14 Download, and Google Docs in production.** Prototype: `/download` writes a report as markdown, with its charts. Production: every saved report is written to a Google Doc, so nothing is downloaded locally. The Doc lives in a shared drive owned by the agent's service account and is shared with its owner only. Download, print and copy are off, and sharing stays inside the domain. Deletes and lost access apply to the Doc too. *Why:* the company is on Google Workspace, and a local file escapes every access control.

**D3.15 Lost brand access hides reports.** If a user no longer covers a report's brands, it is hidden from their list, searches and deletes until access returns.

**D3.16 Opening a saved report shows the snapshot**, with its as-of time. Refreshing on request is an extension.

**D3.17 Resolving delete targets.** The model turns the request into a filter ("mentioning Levi's" or "made in this conversation"); code resolves it to the user's own reports. The model never picks IDs.

**D3.18 Matching on mentions.** Code builds a `mentions` list at save time from known values (brands, categories, places). Fuzzy matching handles typos. *Production upgrade:* full-text search for free-text terms.

**D3.19 Edge cases, simplest defaults.** "Made in this conversation" means first created in this session. Unsaved reports are discarded with a one-line mention.

**D3.20 What counts as a yes.** An LLM classifies the reply as yes, narrow, cancel or unclear, seeing exactly the list shown. A yes with a condition is a narrow; a doubtful yes is never a yes. *Rejected:* a fixed word list, because "go for it" being asked again is clunky. The remaining risk is accepted because deletes are soft and audited.

**D3.21 Narrowing.** "Only the jeans one" narrows the list, which is shown again for a fresh yes. Nothing is deleted that the user hasn't seen.

**D3.22 Bind to the IDs shown.** Only the reports shown are deleted; the filter is never rerun, so a report saved in the meantime can't be swept in.

**D3.23 Soft delete, 30 days.** Deleted reports vanish everywhere and are purged after 30 days. An admin can restore within that window.

**D3.24 Audit record.** Every delete writes who, when, the request, the IDs, the titles and the trace ID, in the same transaction. Titles only, never bodies, so the audit log can't become a copy of deleted reports.

**D3.25 Hidden reports go to an admin**, who can reassign them, for example to a successor on the brand.

**D3.26 The model can only propose a delete.** Its only delete tool is `propose_delete(filter)`. The delete itself is a graph node that runs only after the user's yes, on the IDs shown, softly, with an audit row.

**D3.27 Injection through report content.** A report titled with an instruction can at worst produce a proposal the user sees and rejects.

## 4. Knowledge, learning and persona

**D4.1 Worked examples without SQL.** The agent sees an example's question and report, never its SQL. *Rejected:* raw SQL (copied raw table names and rules that conflict with ours, failing silently); converting SQL with an LLM (hallucination presented as expert guidance); SQL templates (similar questions can need different SQL); glossary only (definitions, not approach).

**D4.2 Metric glossary.** Definitions, time rules and business terms in one file, loaded whole on every analysis turn. The glossary beats any example when they disagree. It is small and fixed, so it is cheap. *Rejected:* selecting entries in code (a question may need a rule it never names); a lookup tool (the model only looks up what it knows it doesn't know).

**D4.3 Numbers stripped from examples.** Done in code at indexing time, for every user. The agent learns the approach; its numbers always come from its own queries. This removes stale and out-of-scope figures in one step.

**D4.4 Brand tags from example content.** An LLM extracts brand names at indexing, and code keeps only names that match the real brand list. An example is shown only if every tagged brand is within the user's access. *Rejected:* a brand enum in the model's output schema (2,756 values is too large); tagging by author.

**D4.5 When retrieval runs.** Forced by code on every analysis turn, and once at the start of a report. *Rejected:* a search tool the agent could skip, since it is least likely to ask for help when it is confidently wrong.

**D4.6 How matches are found.** Embeddings with `gemini-embedding-001` at 1536 dimensions, under Firestore's 2048 limit. The brand filter runs in code on an over-fetched result. *Rejected:* keyword search alone (misses rephrasings).

**D4.7 What gets embedded.** The question plus a one-line description of intent and approach, written at indexing. *Rejected:* question only (analysts write tersely); the whole example (a long report dilutes the match).

**D4.8 Embedding outage.** The prototype answers without examples and logs `trios_unavailable`. Production falls back to keyword search. Another embedding model can't stand in, because its vectors don't match the index.

**D4.9 How many examples.** Up to 3, above an absolute similarity threshold, set from a labelled set of rephrasings. Where scores overlap it leans high: a wrong example misleads, a missing one only falls back to the glossary. Zero examples is a valid outcome.

**D4.10 Who approves examples.** Analysts write them, saved answers can become candidates, and an analyst approves every addition. No LLM judge yet: a Gemini judge shares Gemini's blind spots. Every review decision is logged with a reason, so a judge can later be measured against analysts.

**D4.11 PII in examples is redacted at indexing.** Production uses Sensitive Data Protection. The prototype's indexing call returns person names, and code replaces them.

**D4.12 Candidate filter.** Only clean answers reach the analyst queue: saved, nothing removed by grounding, no stuck rule, no fallback model.

**D4.13 Staleness.** When the glossary or schema changes, examples mentioning the changed term go to the analyst queue. *Rejected:* manual only; expiry by age (expires good examples, keeps bad ones).

**D4.14 Duplicates.** A near-duplicate is flagged in review, and the analyst decides. *Rejected:* automatic merging.

**D4.15 What counts as a preference.** Presentation (format, length, charts): learned from behaviour or stated, visible and resettable. Analysis defaults (period, breakdown): stated only, and shown in the assumptions line. Definitions: never a preference. Safety controls: never a preference. *Why:* a wrong format guess is visible and easily fixed; a silently learned "exclude returns" would change numbers invisibly and give two users different figures.

**D4.16 Learning presentation preferences.** Prototype: the classifier flags format requests, and code sets the preference after 3. Production: a review job after each session catches subtler signals, through a schema that can only output presentation fields.

**D4.17 Context remembered across sessions.** Production only. Saved facts are announced in one line, can be corrected at once, and expire. *Rejected:* silent saving, because a hypothetical saved as a plan would later appear in a report attributed to the user.

**D4.18 User profile.** One profile per user: preferences with their source and date, analysis defaults and context items.

**D4.19 Feedback signals.** System signals from traces and thumbs up or down (prototype); rephrases and corrections (production, from the review job).

**D4.20 What changes, and who changes it.** Numbers can tune themselves within limits, logged and reversible. Anything with words in it (glossary, prompts, examples) stays human-approved and passes the eval gate. Automation is earned by measurement, not assumed.

**D4.21 Usage-triggered review.** Production: an example that keeps appearing in bad answers goes back to an analyst.

**D4.22 Persona in prompt management.** Versioned on the Gemini Enterprise Agent Platform, edited by the CEO's office with no redeploy, with rollback. The prototype uses versioned files re-read every turn. *Rejected:* our own admin editor; a Google Doc (no approval step).

**D4.23 A persona can't weaken safety.** Safety lives in code, so the worst a persona can do is change tone. A smoke test runs before a version goes live, the persona sits below the safety layer, and a broken persona falls back to the previous version.

**D4.24 Prompt precedence.** Safety, then user preference, then persona, then system defaults.

**D4.25 What the prototype builds.** Glossary, 15 seeded examples with indexing and calibration, forced retrieval with brand filter and threshold, learned presentation preferences, thumbs feedback, the persona with fallback and its smoke test, and a `ProfileStore` interface beside `ReportStore`.

## 5. Quality and evaluation

**D5.1 Two kinds of testing.** Deterministic code gets unit tests at 100%. LLM steps get eval sets scored as rates, one set per step with its own exact check, plus end-to-end conversations. A score drop then points at the component that broke.

**D5.2 Red team scored twice.** Safety outcome: no PII, no out-of-scope figures, no non-SELECT, no delete without a yes. Must be 100% and blocks every release. Behaviour (stopped at the right step, neutral wording) is a tracked rate.

**D5.3 Ownership.** Analysts own the expected answers; engineers own the harness and the safety suite.

**D5.4 SQL scored by execution.** The agent's result is compared with a reference result: any row order, column names ignored, small numeric tolerance. *Rejected:* exact SQL match (correct alternatives fail); a judge reading SQL.

**D5.5 Example SQL seeds test cases**, but an analyst confirms each answer key first.

**D5.6 Cross-family judge.** Answer quality is judged by a model from a different family, hosted in Model Garden, so it doesn't share Gemini's blind spots.

**D5.7 One trust rule for every judge.** A judge is trusted only when its agreement with analysts, corrected for chance, matches how often two analysts agree. Unsure verdicts go to a person.

**D5.8 Required elements per test case.** An analyst writes, once, what a correct report must contain. The set then reruns on every change with no analyst involved.

**D5.9 Outlines and reports both scored.** Outlines are cheap and catch most intent misses; full reports run on a smaller set.

**D5.10 Production intent signals.** Outline revisions, redo requests, reports never saved and thumbs down, tracked as rates.

**D5.11 Eval tiers.** Every commit: unit tests. A change to anything the model reads or is: component sets, red team and end to end. Persona edits: the smoke test. Nightly: the full suite on live data.

**D5.12 What blocks a release.** Unit tests and red-team safety at 100%, everywhere. In production, scores can't fall more than a tolerance below the approved baseline. Latency and cost become blocks only once the business sets thresholds.

**D5.13 Overrides.** Safety blocks can never be overridden. A quality regression can be accepted by a named owner with a written reason.

**D5.14 UX measured three ways.** Signals from traces, thumbs feedback and a pilot before launch. Each catches what the others miss.

**D5.15 Key implicit signals.** Rephrasing soon after an answer, and whether managers come back the next week.

**D5.16 UX ownership.** A named product owner reviews UX monthly; engineers supply the numbers.

## 6. Production platform

**D6.1 Clients.** A web app. The CLI stays an engineer tool. Google Chat or Slack is an extension.

**D6.2 Identity.** Identity-Aware Proxy on the Cloud Run service gates every request, granted to one group of agent users. The app runs its own Google OAuth for the BigQuery scope, so queries run as the user. *Rejected:* our own sign-in only (our code would be the only gate).

**D6.3 Stored refresh token, encrypted.** In Firestore, envelope-encrypted with Cloud KMS. Only the agent's service account can decrypt, and every decrypt is audited. Tokens never reach the model, traces or browser. *Rejected:* session-only tokens (re-sign-in every hour); one Secret Manager secret per user.

**D6.4 One Cloud Run service.** FastAPI and the LangGraph graph in one process, one warm instance. *Rejected:* the graph on Agent Runtime (the user's token would cross a service boundary on every call); GKE (overkill).

**D6.5 Checkpoints in Cloud SQL.** LangGraph's official Postgres saver, smallest instance, IAM authentication, 30-day retention. *Rejected:* Firestore with a community saver (we'd own a critical component).

**D6.6 Stored results in Cloud Storage.** One file per result, deleted after a day. The path is built by code from user and session, so another session's result is never found. *Rejected:* Firestore (1 MB document limit).

**D6.7 Background jobs on Cloud Run jobs.** Same image as the service. Cloud Scheduler runs timed jobs; Eventarc runs indexing when an example changes.

**D6.8 Purges use Firestore TTL**, so deleted reports and expired context need no job.

**D6.9 Post-session review as a nightly batch**, selected by a "reviewed" marker so a missed night catches up without double-counting.

**D6.10 Streaming with Server-Sent Events.** One-way streaming is all a chat needs, and it passes through IAP unchanged.

**D6.11 No Pub/Sub yet.** Nothing needs one event fanned out to several consumers.

**D6.12 One service account per workload.** Least privilege. The chat service has no BigQuery data access of its own, since every query runs as the user.

**D6.13 VPC Service Controls perimeter.** Around BigQuery, Firestore, Cloud Storage, Cloud SQL and the Agent Platform. IAM decides who can read data; the perimeter decides where data can go.

**D6.14 Region follows the data**, because BigQuery can't join across locations.

**D6.15 Build and deploy with Cloud Build.** One image, staging and production projects from the same Terraform, gradual traffic rollout. *Rejected:* GitHub Actions for production (a second place for credentials and logs).

**D6.16 The eval gate in the pipeline.** Wires D5.11 and D5.12 into CI.

**D6.17 Admin pages in the same web app**, behind an admin role, every action audited: restore reports, reassign hidden ones, the example review queue, budgets, overrides.

**D6.18 Eval results in BigQuery, next to the traces.** Two tables, `eval_runs` and `eval_cases`, joined to traces by trace ID. App data lives in Firestore and Cloud SQL; analytical data lives in BigQuery.

**D6.19 The production diagram is high level.** Six blocks, with detail in a component table, one row per service.

**D6.20 Nightly evals read live data in staging**, so they see today's data and never touch production.

## 7. Extensibility

**D7.1 Tools carry a risk level.** Every tool declares read, write, external or destructive, and the graph applies the matching safety steps once per class. Adding a capability means writing the tool, declaring its class and adding tests. *Rejected:* each tool handling its own confirmation (one forgotten check is a hole).

**D7.2 External sends ask first.** Email or chat posts use the same confirmation as delete, limited to the company domain.

**D7.3 Web search is untrusted and stays inside Google**, through Gemini's grounding with Google Search, labelled as external and never blended into internal figures.

**D7.4 Scheduled reports run as their owner**, with the owner's stored token and the same output check.

**D7.5 Entitlements come from Google Groups.** The app reads group memberships at sign-in, so the app and BigQuery use one source of truth. *Rejected:* a separate entitlements table (two places to keep in sync).

**D7.6 Consistency fixes.** A batch of corrections from a full review of the design: current product names, model IDs re-checked at build time, and IAP granted to one group instead of individuals.

## 8. Build-time decisions

**D8.1 What the prototype builds for evaluation.** Unit tests for all deterministic code, a red team of 16 attacks each run three times, small component sets with exact checks, 5 end-to-end conversations, one `eval` command and a committed baseline. GitHub Actions runs the unit tests on every push. The cross-family judge is design only.

**D8.2 No small-group roll-up.** Supersedes D1.5 and D2.7. Real data showed about 90% of a brand manager's customers live in cities with fewer than 10 customers, so roll-up turned almost every city answer into one "Other" row. The client had confirmed city analysis is wanted. Blocked columns, no individual customers, the scope rewrite and the leak scan still protect people. *Accepted gap:* a row can describe a single customer in a small city, acceptable for internal executives.

**D8.3 Runs on a reviewer's machine.** No cloud services are created. Two credentials: a Gemini key and Google Cloud sign-in. The example index is prebuilt and committed, and `doctor` names the fix for any setup problem. *Rejected:* a hosted demo or shared credentials.

**D8.4 Phase 2 build notes.** How the classifier handles uncertainty, the code-written scope note, number grounding tolerance, and the exact shapes the leak scan matches. No decision changed.

**D8.5 Phase 3 build notes.** The mapping from BigQuery error reasons to codes, how empty aggregates are detected, the fault flags, and keeping the cost-limit hint as written.
