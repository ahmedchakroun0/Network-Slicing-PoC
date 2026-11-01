"""E5 - Working-memory growth and orchestration overhead (LLM stubbed).

Runs the real MonitoringAgent and the real RedisWorkingMemory (a local
redis-server must be running) over the first N dataset records. LLM calls
are replaced by a scripted model that returns a fixed MonitoringDecision
selecting both RIC agents, and the RIC agents are replaced by stubs that
return a RICResult shaped like the one logged in run-1. This isolates the
cost of the agentic plumbing (memory I/O, prompt assembly, routing) from
model inference, which is NOT measured here.

Reported per record index n:
  * size of the working-memory blob that BaseRICAgent injects verbatim into
    every RIC system prompt (the whole WorkingMemory dump);
  * wall time of MonitoringAgent.run (excluding inference).
A bounded variant that keeps only the last W tool executions (a proposed
mitigation, not part of the PoC) is measured with the same harness.
"""

from __future__ import annotations

import asyncio
import json
import time

import pandas as pd

import common
from common import DATA_DIR, base_parser, dump_json, use_project


async def main_async(args) -> None:
    from redis.asyncio import Redis
    from agents.monitoring_agent import MonitoringAgent
    from src.memory.working_memory import RedisWorkingMemory
    from src.schema.monitoring import MonitoringDecision
    from scripts.run_dataset import _row_to_telemetry_record
    from langchain_core.globals import set_debug
    set_debug(False)  # run_dataset.py enables verbose tracing on import

    run1 = json.loads((common.HERE.parent / "data" / "e4_replay_run1.json").read_text())
    ric_result = {
        "status": "rejected",
        "proposal": {"tool_name": "optimize_scheduler", "parameters": {"slice_id": "umMTC", "scheduler": "proportional_fair"},
                     "target": "cell-01", "rationale": "KPI violations: latency (1342635.550778835 ns), jitter (2812229.169688584 ns), and packet loss (0.0004989734617149). Reducing latency and jitter should improve these KPIs.",
                     "expected_effect": "improve packet loss", "evidence_sources": [], "confidence": 0.0},
        "governance": {"approved": False, "reason": run1[3]["trusted_context"]["reason"],
                       "failed_stage": "policy_trust", "checks_passed": ["schema"]},
        "execution": None, "retrieved_knowledge": [],
    }

    class ScriptedLLM:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _prompt):
            return MonitoringDecision.model_validate({
                "situation": "critical", "severity": "high", "global_situation": "critical",
                "action_required": True,
                "sla_analysis": {k: {"status": "ok"} for k in ("latency", "jitter", "packet_loss", "throughput")},
                "selected_agents": [
                    {"agent": "near_rt_ric", "responsibility": "Perform near-real-time RAN optimization for packet loss reduction",
                     "objective": "Reduce packet loss to within budget", "priority": "high",
                     "reason": "Near-RT RIC is responsible for fast, local RAN actions."},
                    {"agent": "non_rt_ric", "responsibility": "Optimize RAN configuration for latency and jitter reduction",
                     "objective": "Reduce latency and jitter to within budget", "priority": "high",
                     "reason": "Non-RT RIC is responsible for policy-level RAN optimization."}],
                "reason": "SLA violations detected: latency, jitter, and packet loss."})

    class Bounded(RedisWorkingMemory):
        W = args.window

        async def save(self, memory):
            memory.tools_executed = memory.tools_executed[-self.W:]
            await super().save(memory)

    df = pd.read_csv(args.dataset).head(args.n)
    out = []
    for variant, cls in (("unbounded (PoC)", RedisWorkingMemory), (f"bounded W={args.window} (proposed)", Bounded)):
        redis = Redis.from_url("redis://localhost:6379/15", decode_responses=True)
        await redis.flushdb()
        mem = cls(redis)
        prompt_sizes = []

        last_wm: dict = {}

        async def stub(inv):
            # What BaseRICAgent.invoke does before prompting: read the whole
            # memory and dump it. Its size is measured AFTER timing (below),
            # so the measurement itself is not charged to the run time.
            last_wm["wm"] = (await mem.get()).model_dump(mode="json")
            return ric_result

        agent = MonitoringAgent(llm=ScriptedLLM(), memory=mem, sub_agents={"near_rt_ric": stub, "non_rt_ric": stub})
        for i, (_, row) in enumerate(df.iterrows()):
            rec = _row_to_telemetry_record(row, row["record_id"])
            t0 = time.perf_counter()
            await agent.run(rec)
            dt = (time.perf_counter() - t0) * 1e3
            blob = await redis.strlen(RedisWorkingMemory.KEY)
            prompt_sizes.append(len(json.dumps(last_wm["wm"], default=str)))
            out.append({"variant": variant, "n": i + 1, "run_ms": dt, "redis_blob_bytes": blob,
                        "ric_context_chars": prompt_sizes[-1]})
        await redis.flushdb()
        await redis.aclose()

    res = pd.DataFrame(out)
    res.to_csv(DATA_DIR / "e5_memory_scaling.csv", index=False)
    summ = {}
    for v, g in res.groupby("variant"):
        summ[v] = {
            f"n={n}": {"ric_context_kchars": float(g[g.n == n].ric_context_chars.iloc[0] / 1e3),
                       "run_ms_median_window": float(g[(g.n > n - 50) & (g.n <= n)].run_ms.median())}
            for n in (50, 500, 1000, args.n)
        }
        summ[v]["chars_per_record_slope"] = float((g.ric_context_chars.iloc[-1] - g.ric_context_chars.iloc[99]) / (len(g) - 100))
    dump_json(summ, "e5_summary.json")
    print(json.dumps(summ, indent=2))


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--dataset", default="data/prepared_network_slicing_dataset.csv", help="CSV in the PoC prepared-dataset format (relative to --project or absolute)")
    p.add_argument("--outdir", default=None, help="sub-folder of data/ for the outputs")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--window", type=int, default=30)
    args = p.parse_args()
    args.dataset = str(args.dataset)
    common.set_outdir(args.outdir, globals())
    use_project(args.project)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
