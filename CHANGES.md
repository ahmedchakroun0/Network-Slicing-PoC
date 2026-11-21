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

# v3.3 — native ColO-RAN telemetry, SLA profiles, efficient memory and results store

Telemetry and SLA:
- `TelemetryRecord` is ColO-RAN native: one record per base station, slice and 1 s window, in native units. The CSV columns are its fields (no aliases or unit conversion in the runner). `scripts/prepare_coloran.py` was rewritten; it trims the start-up and tail of each run with a sustained-traffic rule. The synthetic dataset format is no longer supported.
- SLA budgets moved out of the data into per-slice profiles (`src/schema/sla.py`: throughput per UE, queueing delay). `SLAAnalysis` is a dict of KPI -> status, and the LLM no longer generates `sla_analysis`.
- The RIC agents get a trusted `slice_state` (PRB share, utilisation, spare PRBs in the cell, scheduler).
- `scripts/explore_coloran.py` profiles every raw column (report in `data/exploration/`).

Working memory (Redis):
- The single JSON blob that was rewritten on every append is replaced by keys per base station and slice: HASH meta and persistence, LISTs capped with LTRIM for records and tool calls, and pipelined reads and writes.
- Agents see only their cell and slice. Facts only: no LLM text is fed back into prompts.
- The persistence counters moved from the runner into Redis. A new configuration of a base station clears its memory.

Results store (MongoDB):
- The one-document-per-run design, replaced 5 times per record, is gone. There is now `runs` (one small document, `$inc` counters) and `records` (one document per record, updated with `$set` / `$push` at each agent step), plus indexes and no in-process cache.
- Each fact is stored once: no invocation context, no `proposal.invocation_context`, no duplicated typed result (`selected_action` points to the action). A record with two agents drops from ~18 KB to a few KB.
- The middleware's `AuditSink` writes each tool call when it happens.

Agent reasoning in the results store (for expert evaluation):
- Every action stores the RIC agent's stated `reasoning` (rationale: why this tool and these values; expected_effect, evidence_sources, confidence) and its `agent_message`: the visible text of the model turn that emitted the call, and `forced` when the middleware produced the call because the model did not.
- `governance.llm_review` keeps the review's three verdicts (goal_aligned, reasoning_sound, behavior_acceptable) and its `reason` (why it accepts or rejects, citing the claim, parameters and KPI values), for approved and rejected actions. Deterministic stages keep their exact reason in `governance.reason`.
- `agents.<agent>.knowledge` keeps each knowledge lookup and what it returned; `agents.<agent>.final_message` keeps the agent's closing turn.
- `monitoring` keeps the decision's `reason`; when the deterministic override changed the decision, `overridden: true` and `llm_output` hold what the model decided.
- Only output tokens are stored (the reasons the agents state), never hidden thinking tokens.
- Redis action events get the review verdicts as booleans only (`review`); no LLM text goes into working memory.

Tools and governance gate:
- Removed `update_traffic_steering_policy` and `update_admission_control_policy` (nothing in ColO-RAN reflects them), and `optimize_slice_capacity_policy` (a duplicate of `update_slice_resource_policy`). The Non-RT RIC keeps `update_slice_resource_policy`.
- `optimize_scheduler` accepts the testbed's schedulers only (round_robin, waterfilling, proportional_fair). They are defined once (`src.schema.monitoring.Scheduler`) for the telemetry, the tool and governance.
- `adjust_qos_parameters` (priority_level 1-10, packet_delay_budget_ms 5-300, guaranteed_bitrate_mbps 0-10) and `adjust_link_adaptation` (mcs_offset -5..5, target_bler 0.001-0.2) now take typed parameters with safety envelopes instead of free-form dictionaries, which governance always rejected.
- The gate no longer drops arguments a tool does not declare: they reach governance and are rejected at the schema stage with the argument's name.
- The one-execution-per-invocation rule applies per controlled resource (`RESOURCE_OF`), not per tool name.
- Forced tool calls are unchanged.

Records without an SLA violation:
- The runner skips them before the Monitoring Agent (no LLM call): Redis still gets the cell configuration, the reset persistence counters and a healthy entry in the slice history (not in the cell history); MongoDB counts them in `summary.rows_without_violation` and stores no record document.

Tests (68 pass, no external service needed):
- `conftest`: `memory` is the real `RedisWorkingMemory` on fakeredis and `mongo` the real `MongoRepository` on mongomock (the hand-written fake memory is gone); the LLM is still faked. No test touches a local Redis any more (the old ones ran `flushdb()` on it).
- New: `test_working_memory.py` (keys, reconfiguration, persistence, view, caps, TTL, reset) and `test_mongo_repository.py` (targeted updates, no overwrite, failures, indexes).
- Updated: Monitoring (decision and `llm_output` stored, agent errors in MongoDB), RIC agent (actions, selected action, review verdicts in Redis), typed QoS / link-adaptation / scheduler envelopes, undeclared-argument rejection, per-resource limit, end-to-end (healthy records skipped, Redis persistence gate).
- `scripts/demo_run_scripted.py 12 data/coloran_prepared_sample.csv`: a record with two agents is ~4 KB (was ~18 KB).

Routing rules and model-call robustness:
- The Non-RT RIC is invoked only when a violated KPI has persisted for `NON_RT_MIN_PERSISTENCE` (3) consecutive records of the cell and slice. Below that, the Monitoring Agent drops it from the LLM's selection and notes why in `monitoring.orchestration_notes`; `llm_output` keeps the LLM's own choice for evaluation. Governance still checks persistence (defence in depth).
- A record with violations is never left without an agent: when the LLM selects none (or only a Non-RT that is dropped), the Near-RT RIC is invoked with an objective built from the violated KPIs, and the fallback is noted. `action_required` always follows the selected agents.
- Model calls retry transient errors (rate limit, timeout, connection, API) with exponential backoff (`LLM_MAX_RETRIES`, default 6). One request budget is shared by every agent (`LLM_REQUESTS_PER_MINUTE`, 0 = unlimited). The governance review timeout includes retries and waits (`GOVERNANCE_LLM_TIMEOUT_S`, default 180 s), so a rate limit does not turn into a spurious rejection.
- 75 tests pass.

Record workflow as a LangGraph graph:
- `agents/record_workflow.py`: one `StateGraph` run per telemetry record, `ingest -> (skip_healthy | monitoring -> near_rt_ric? -> non_rt_ric? -> finalize)`. Its typed `RecordState` (record, SLA analysis, persistence, slice state, decision, agent results) is shared by the nodes of that record only; memory across records stays in Redis and results in MongoDB.
- The trusted context of the RIC agents is built from the state (`trusted_context(state)`) instead of a mutable dict shared by every call in `run_dataset.py`.
- The selected RIC agents run in a fixed order, Near-RT first, whatever order the LLM listed them in: the Non-RT RIC sees this record's Near-RT calls in its working memory.
- `MonitoringAgent.run` is split into `decide` (LLM + deterministic override + stores) and `invoke_agent` (one RIC call; a missing agent or a failure is recorded and returns None). `run()` still chains them for direct use.
- Each graph run is named `record <record_id>` with tags (cell, slice) and metadata (run_id, record_id, seq, cell_id, slice_id): one LangSmith trace per record when `LANGSMITH_TRACING=true`.
- `RecordWorkflow.draw_mermaid()` generates the workflow diagram from the graph.
- New `tests/test_record_workflow.py`; 80 tests pass.

LangSmith tracing:
- Every protected tool call is one `governance <tool>` span (tag `governance`) around the whole gate, whatever its outcome: a rejected tool never runs, so it would otherwise leave no span. The LLM review and the executed tool are nested inside it.
- `langsmith>=0.14.9` and `langchain-core>=1.6.11`: with `LANGSMITH_AGENT_ID` set, older versions send both the agent address and the default project, and the server rejects every trace with a 400.
- Tests and `scripts/demo_run_scripted.py` force `LANGSMITH_TRACING=false`: scripted data is never sent, whatever `.env` says.
- 81 tests pass at this point.

Evaluation:
- `scripts/evaluate_run.py RUN_ID [RUN_ID ...]` computes the metrics of one or more live runs from MongoDB (read-only) and writes a snapshot to `results/eval-<run>/`: `metrics.json`, `records.csv`, `actions.csv`, `expert_rating.csv` (empty expert columns) and `e4_table.tex`. It also checks the gate's invariants on the stored results, counts review rejections overturned by rewording, and finds executed PRB shares outside a slice policy.
- The branch after `ingest` is a named function: the lambda appeared as an `Unnamed` span in LangSmith.
- New `tests/test_evaluate_run.py`; 88 tests pass.

Governance fixes from the first live runs:
- The agents and the LLM review now get `active_ues`, `dl_cqi`, `dl_mcs` and `ul_sinr_db` in `slice_state`: a rationale citing them was rejected as ungrounded because the reviewer could not see them.
- The same tool with the same parameters cannot be resubmitted after an LLM review rejection in the same invocation (`invocation_limit`): rewording the rationale no longer gets an action through.
- New Redis hash `wm:slice:{cell}:{slice}:controls` keeps the PRB controls in force (last executed `prb_percentage`, `policy_min`, `policy_max`). The workflow passes them to the RIC agents and governance as the trusted `active_controls`, and the resource-safety stage rejects a PRB share outside the slice policy and a policy that excludes the PRB share in force.

Results files:
- `run_dataset.py` writes the result files to `results/<run_id>/` at the end of every run (`--results-dir` changes the parent). `scripts/evaluate_run.py` uses the same directory, and `results/<run1>+<run2>/` when runs are pooled.

Rate limits and provider fallback:
- `litellm_config.yaml` `router_settings`: each model is retried twice, then put on a 60 s cooldown, and its calls go to the next model of its chain (`nvidia-network-slicing-model` <-> `gemini-network-slicing-model`). Checked with a mock proxy: a rate-limited primary is answered by the backup.
- `src/core/llm.py` `PROVIDER_GUARD` (a callback on every model call): a rate limit that reaches the client means the whole chain is exhausted. Every later call then fails at once with `ModelProviderStopped` instead of being sent, and `run_dataset.py` marks the current record `failed` and stops the run (`runs.status = "stopped"`, `runs.stop_reason`). The result files are still written.
- The client no longer retries by default (`LLM_MAX_RETRIES=1`): the gateway does, and client retries would replay the whole fallback chain.
- Each record stores the models that actually answered its calls (`records.models`), and `metrics.json` has `models_served`, so a run that switched provider is visible.
- `scripts/evaluate_run.py` evaluates completed records only; `dataset.records_not_evaluated` counts the others.
- 102 tests pass.

Model usage per record:
- `PROVIDER_GUARD` also counts, per record and per caller (`monitoring`, `near_rt_ric`, `non_rt_ric`, `governance_review`), the model calls, failed calls, and the input and output tokens reported by the provider. The caller is the record-graph node (LangGraph namespace) or the governance gate tag.
- Stored on the record (`records.llm`, with `by_caller`) and added to the run (`runs.summary.llm`). Records without violation make no model call (skipped before the Monitoring Agent), so they count 0.
- `metrics.json` `llm_usage`: mean, median, p95 and max calls and tokens per record with violation, the mean per caller, the 0 of records without violation, and the mean over all rows. `records.csv` has `llm_calls`, `input_tokens` and `output_tokens`.
- 104 tests pass.

Evaluation of the agents (E4):
- `scripts/agent_checks.py`: deterministic checks of every record and call, no model involved. SLA recomputation; the Monitoring LLM's own decision (before the override) against the computed severity, the routing rule and the violated KPIs; the tool against a telemetry diagnosis (PRB-limited, radio-limited, unclear), PRB direction, feasibility, no-ops; quoted numbers traceable to the context (with rounding and simple derivations); evidence sources supported by the context or the retrieved knowledge and tool descriptions. Their columns are in `records.csv` / `actions.csv`, their aggregates in `metrics.json` `agent_quality`, plus `sla_detection` (no violated dataset row skipped).
- `scripts/stability.py RUN_IDS`: agreement of the decisions (situation, agents, tools, executed parameters, governance outcome) and spread of the cost over repeated runs of the same records. `run_dataset.py --record-ids a,b,c` processes only these records.
- `scripts/compare_models.py RUN_IDS`: the E4 metrics of each model's runs side by side.
- The three scripts write a LaTeX table (`--tex PATH`, `--label` / `--labels` for column names). The paper's E4 section `\input`s `tables/e4_live.tex`, and `tables/e4_stability.tex` / `tables/e4_models.tex` when they exist.
- 117 tests pass.

Token reduction:
- Forced action in one call: when the RIC agent has not called an action tool yet (after a knowledge search), the middleware makes one model call offering only the action tools with `tool_choice="any"`, so the model still chooses the tool and its arguments. The two-call structured-output forcing is kept only as a fallback for models that ignore tool calling.
- The invocation is sent once (system prompt), not also in the user message, as compact JSON.
- Compact working memory in prompts (`MemoryView.compact()`): 5 events per list instead of 10, no empty fields or lists, no record ids, no slice name inside the slice's own lists; compact JSON.
- Measured on a real record (claude-run4 #31): the fixed part of a Near-RT call goes from 5.0k to 3.5k tokens (invocation 1.0k -> 0.4k, memory 1.4k -> 0.45k). With `MAX_PROTECTED_CALLS_PER_INVOCATION=2` the Near-RT loop is also shorter.
- Reasoning tokens are recorded apart (`reasoning_tokens`, part of `output_tokens`) when the provider reports them.
- 119 tests pass.

Paper tables:
- `scripts/paper_tables.py --group "LABEL=run1,run2" ... --tex PATH`: one column per named group of runs, with the summary rows of `evaluate_run.py` (now also "Calls rejected by code"); the metrics go to a `.json` next to the table. Reasoning tokens are averaged only over records that recorded them ("--" otherwise). Table captions no longer carry an experiment number.
