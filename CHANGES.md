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

# v3 — fixes from the live-LLM runs (results/run-6..8)

Baseline: `Network-Slicing-PoC-ahmed` as received (commit 0f72dc0). Found by auditing the
MongoDB exports of the live runs and by probing the gate (paper, Sec. IV-E):

1. **Safety stage is default-deny.** `optimize_slice_capacity_policy` had no envelope:
   min 900 % / max -4 % passed every deterministic stage. It now shares the slice-policy
   envelope (5<=min<=80, 20<=max<=100, min<=max); any tool without an envelope
   (`adjust_qos_parameters`, `adjust_link_adaptation`: free-form dictionaries) is rejected
   at `resource_safety` until one is defined.
2. **Governance argument models mirror the tool signatures.** The QoS and link-adaptation
   models lacked `slice_id`, and the link model used `link_adaptation_parameters` while the
   tool declares `link_adaptation_params`: both tools were always rejected at `schema`.
3. **Non-RT persistence gate restored** (removed in the uploaded revision), per (cell, slice)
   stream, passed to the Monitoring prompt and the trusted context. `NON_RT_MIN_PERSISTENCE`
   environment variable (default 3; `1` disables the gate).
4. **Duplicate agent selections are merged** by the Monitoring override. In runs 6 and 7 the
   model selected `near_rt_ric` twice; `agent_executions` is keyed by agent name, so the
   second invocation overwrote the first one's audit (6 invocations, 4 audited).
5. **Invocation limits in the middleware:** at most one executed call per tool per
   invocation and `MAX_PROTECTED_CALLS_PER_INVOCATION` (default 6) protected calls. Run 7:
   one Non-RT invocation made 23 calls and executed `update_slice_resource_policy` three
   times with conflicting values (min 5 %, then 60 % twice).
6. RIC abstention is reported as `no_action` again (it was reported as `error`).
7. Tests use the native policy tool names (the `no_action` carrier was removed);
   `tests/test_v3_live_run_fixes.py` adds 7 regression tests. **40 tests pass.**

Not fixed in code (reported): the Monitoring LLM justified its selections with "recent
successful executions" right after a memory reset (fabricated history; harmless for
actuation because the context is injected by code); the LLM review approved calls whose
expected effect is on a KPI outside the objective and one rationale that misquotes the
measured latency (30.2 ms "above" 100 ms; actual 301.9 ms); `result.retrieved_knowledge` is
never populated, so knowledge-base use cannot be audited; the middleware silently drops
arguments a tool does not declare instead of rejecting them.

# v3.2 — Nemotron runs (results/nvidia-result1.txt = run-23, nvidia-run2.txt = run-24)

Changes made by the PoC team (as received):
- LiteLLM route `nvidia-network-slicing-model` -> `openrouter/nvidia/nemotron-3.5-lightning:free`.
- `MAX_PROTECTED_CALLS_PER_INVOCATION` default 6 -> 20; governance decisions are logged.
- RIC prompts: the "call no action tool only if none of your tools can help" rule was
  removed, and the Non-RT prompt no longer asks for a knowledge search first.

Fix added after auditing runs 23-24:
- A record with verified SLA violations is never reported as `situation="normal"`
  (run 24: `normal` with severity `high`); it becomes `warning`, and the LLM's choice
  between warning and critical is kept. New regression test (41 tests pass).

Security: the received archive contained `.env` with live API keys (HF, xAI, Gemini,
OpenRouter, LiteLLM) and `.env.example` repeated the LiteLLM keys and the database
password. Both were removed / replaced by placeholders here; the keys should be rotated.
