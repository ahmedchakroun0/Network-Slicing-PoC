# PoC v2 — corrections

All changes are minimal and keep the architecture unchanged (Monitoring -> RIC agents -> Trust & Governance).

| # | Problem in v1 | Fix in v2 | Files |
|---|---|---|---|
| 1 | Non-RT prompt told the agent to use `propose_policy_action`, a tool that no longer exists (likely cause of the Non-RT `error` results in the pilot logs) | Prompt lists the three real policy tools with their limits; mentions the persistence rule | `prompts/non_rt_ric.txt` |
| 2 | Governance never checked that a tool belongs to the calling agent (`TOOL_OWNERS` defined but unused): a Near-RT invocation could execute a Non-RT policy | Role binding in the policy/trust stage; unknown caller is rejected (fail closed) | `agents/ran_trust_governance.py` |
| 3 | `NON_RT_MIN_PERSISTENCE = 3` defined but never enforced | SLA stage rejects Non-RT policy actions unless a *violated* KPI persisted >= 3 records | `agents/ran_trust_governance.py` |
| 4 | Persistence counted with one global counter per KPI over the whole stream: interleaved slices/cells reset each other | Counters kept per (cell, slice) stream | `scripts/run_dataset.py` |
| 5 | No cell identifier: every record used target `cell-01` | Optional `Cell ID` column / `TelemetryRecord.cell_id`, used as target when present | `src/schema/monitoring.py`, `scripts/run_dataset.py` |
| 6 | Monitoring Agent could not see persistence, although routing to Non-RT depends on it | `MonitoringAgent.run(record, persistence=...)`; prompt shows it | `agents/monitoring_agent.py`, `prompts/monitoring.txt` |
| 7 | Working memory unbounded: whole history injected in every RIC prompt (>128k tokens after ~150 records), quadratic cost; `MEMORY_MAX_RECORDS`, `MEMORY_TTL_SECONDS`, `REDIS_URL` in `.env.example` were ignored | Memory keeps the last `MEMORY_MAX_RECORDS` (default 30) executions, optional TTL, `REDIS_URL` honoured | `src/memory/working_memory.py` |
| 8 | Audit bug (found by the new end-to-end test): Monitoring decision stored in MongoDB *before* the deterministic override, so the audit showed agents "selected" that were never invoked | Effective decision stored after the override | `agents/monitoring_agent.py` |
| 9 | Knowledge base path `app/data/ran_knowledge.txt` does not exist (only worked via `run_dataset.py`) | Path resolved from the project root | `tools/near_rt_ric_tools.py` |
| 10 | Importing `scripts/run_dataset.py` switched on verbose LangChain tracing globally | Opt-in `--debug` flag | `scripts/run_dataset.py` |
| 11 | `tests/test_ric_and_tools.py` could not be imported (removed tool), and a PRB tool test missed required fields | Tests updated | `tests/` |
| 12 | `.env` with live API keys shipped in the archive | `.gitignore` (`.env`, caches, `dump.rdb`); `.env` not included in v2 — **rotate the keys that were shared** | `.gitignore` |

| 13 | No way to run the PoC on a real-stack dataset | `scripts/download_coloran.sh` + `scripts/prepare_coloran.py` convert ColO-RAN into the exact telemetry CSV format (adds `Cell ID`); `scripts/demo_run_scripted.py` dry-runs the pipeline without model/Redis/Mongo | `scripts/`, `README.md` |
| 14 | `python scripts/run_dataset.py` failed with `No module named 'agents'` unless PYTHONPATH was set | Script adds the project root to `sys.path` | `scripts/run_dataset.py` |

New tests: `tests/test_v2_fixes.py` (11 regression tests) and `tests/test_end_to_end.py`
(full `run_dataset.py` pipeline — Monitoring, LangChain RIC agents, middleware, governance incl.
LLM review, MongoDB audit, Redis memory — with a scripted tool-calling LLM, fakeredis and mongomock).

Result: `poetry run pytest` -> 33 passed (incl. `tests/test_prepare_coloran.py`) (v1: 14 passed, 1 test module failed to import).

Still to do on your side: run the agents with a real model
(`poetry run python scripts/run_dataset.py --max-rows 10`, Ollama/LiteLLM + Redis + MongoDB running);
this could not be done in the evaluation sandbox (no model endpoint reachable).
