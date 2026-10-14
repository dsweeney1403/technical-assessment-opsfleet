# OpsFleet Analyst: build instructions

A data analysis chat agent for a retail company's executives. LangGraph, Gemini, BigQuery (`bigquery-public-data.thelook_ecommerce`). Prototype must implement R2 (safety and PII), R3 (reports with delete oversight), R5 (resilience) and R7 (observability). Everything else is design only.

## Planning docs (local only, never commit)

All design and planning lives in `.planning/`, which is gitignored. Read files there by explicit path.

- `.planning/build-plan.md`: the plan. Phases 0 to 7, tasks with acceptance criteria, repo layout, config reference. **Read this first, every session.**
- `.planning/design-log.md`: the source of truth for every decision (D1.1 to D8.1). Large. Don't read it end to end; look up the decision IDs the current task cites. Interfaces sections hold the graph, state, tool signatures, error codes and trace shape.
- `.planning/data-facts.md`: facts about the dataset. Use these instead of assuming anything about the data.
- `.planning/client-clarifications.md`: answers from the client.
- `.planning/brief.pdf`: the original assignment.

Ignore anything called a "draft" outside `build-plan.md`. Older drafts describe rejected designs (Langfuse, a `search_trios` tool, typed "delete 3" confirmations).

## How to work

- One phase at a time, in order. Start each phase in plan mode: list the tasks, the decision IDs they implement, and the files you'll touch. Wait for approval before writing code.
- Decisions in the design log are binding. If one looks wrong, unclear or impossible to build, stop and ask. Don't quietly change the design. Daniel owns every decision and has to defend it.
- Prototype inclusion rule (D2.15): build a feature only if it is a small code change and demonstrates a brief requirement. Otherwise it's a production note, not code.
- Safety is enforced in code, never in prompts. The model proposes; code decides (scope, PII, SELECT-only, output check, delete gate).
- Tick a task in `.planning/build-plan.md` only when its acceptance criteria pass. Write tests alongside the code, not after.
- Commit at the end of each task with a plain message. Small commits.

## Rules

- Python, pinned `requirements.txt`, package under `src/analyst/`, run as `python -m analyst`.
- Never commit: `.planning/`, `traces/`, `reports/`, the SQLite file, `.env`, keys.
- No em dashes in docs, README, prompts or any user-facing string. Short, direct sentences.
- Say "Gemini Enterprise Agent Platform", not "Vertex AI", and "Agent Runtime", not "Agent Engine".
- Model IDs come from `config/settings.yaml`, never hard-coded.
