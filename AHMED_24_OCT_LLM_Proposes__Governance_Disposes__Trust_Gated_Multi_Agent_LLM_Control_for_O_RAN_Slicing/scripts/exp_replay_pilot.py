"""E4 - Pilot-run analysis and replay of real LLM-generated tool calls.

(a) Parses the Mongo dumps shipped with the PoC (results/*.txt; pilot runs
    with Llama-3.2-3B served by Ollama through LiteLLM, temperature 0) and
    tallies Monitoring decisions and RIC outcomes.
(b) Replays the tool calls that the RIC agents actually emitted in run-1
    (copied verbatim from results/run-1.txt) through the CURRENT governance
    code, both with the context available at the time (no trusted context)
    and with the trusted context that scripts/run_dataset.py now injects.
    Each deterministic stage is also evaluated independently, to show every
    defect present in a proposal, not only the first.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from common import base_parser, dump_json, use_project


def parse_mongo_dump(path: Path) -> list[dict]:
    """Line-oriented parser for `mongosh` pretty output of run documents."""
    runs, run, rec, agent = [], None, None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s{4}_id: '([^']+)'", line) or re.match(r"^\s{4}_id: '([^']+)'", line)
        if re.match(r"^\s*_id: '", line) and line.startswith("    _id") or re.match(r"^\{?\s*_id: '", line):
            rid = re.search(r"_id: '([^']+)'", line).group(1)
            run = {"run_id": rid, "records": {}}
            runs.append(run)
            continue
        if run is None:
            continue
        m = re.match(r"^\s{6}(record_\d+): \{", line)
        if m:
            rec = {"situation": None, "severity": None, "agents": {}}
            run["records"][m.group(1)] = rec
            agent = None
            continue
        if rec is None:
            continue
        m = re.match(r"^\s+situation: '(\w+)'", line)
        if m and rec["situation"] is None:
            rec["situation"] = m.group(1)
        m = re.match(r"^\s+severity: '(\w+)'", line)
        if m and rec["severity"] is None:
            rec["severity"] = m.group(1)
        m = re.match(r"^\s{10}(near_rt_ric|non_rt_ric): \{", line)
        if m:
            agent = m.group(1)
            rec["agents"][agent] = None
        m = re.match(r"^\s{14}status: '(\w+)'", line)
        if m and agent and rec["agents"].get(agent) is None:
            rec["agents"][agent] = m.group(1)
        if re.match(r"^\s{12}result: null", line) and agent:
            rec["agents"][agent] = "not_recorded"
    return runs


# Tool calls emitted by the RIC agents in run-1 (results/run-1.txt).
# Rationale/expected_effect/confidence are only logged for calls that
# reached governance; the two calls blocked by the (since removed)
# per-invocation action limit have no logged reasoning.
RUN1_CALLS = [
    {"id": "call_f4arreud", "agent": "non_rt_ric", "tool": "update_slice_resource_policy",
     "args": {"slice_id": "run-1", "min_prb_ratio_pct": 80, "max_prb_ratio_pct": 5, "target": "record_00001",
              "rationale": "KPI violations: latency (1342635.550778835 ns), jitter (2812229.169688584 ns), and packet loss (0.0004989734617149). Reducing latency and jitter should improve these KPIs.",
              "expected_effect": "improve latency and jitter", "evidence_sources": [], "confidence": 0.8},
     "reasoning_logged": True},
    {"id": "call_bz93ejca", "agent": "non_rt_ric", "tool": "update_traffic_steering_policy",
     "args": {"load_balancing_threshold": 0.7, "load_balancing_mode": "round_robin", "slice_id": "run-1",
              "target": "record_00001"}, "reasoning_logged": False},
    {"id": "call_cy438tkk", "agent": "non_rt_ric", "tool": "update_admission_control_policy",
     "args": {"slice_id": "run-1", "max_slice_utilization_pct": 90, "target": "record_00001"},
     "reasoning_logged": False},
    {"id": "call_y8fm84lz", "agent": "near_rt_ric", "tool": "optimize_scheduler",
     "args": {"slice_id": "run-1", "scheduler": "proportional_fair", "target": "record_00001",
              "rationale": "KPI violations: latency (1342635.550778835 ns), jitter (2812229.169688584 ns), and packet loss (0.0004989734617149). Reducing latency and jitter should improve these KPIs.",
              "expected_effect": "improve packet loss", "evidence_sources": [], "confidence": 0.0},
     "reasoning_logged": True},
]


async def main_async(project: Path) -> None:
    from agents.ran_trust_governance import (ARG_MODELS, RANTrustGovernanceAgent, _safety_violation)
    from agents.monitoring_agent import MonitoringAgent
    from middleware.ran_authorisation import (RAN_AUDIT, RANAuthorizationMiddleware, build_submission)
    from scripts.run_dataset import _row_to_telemetry_record
    from langchain_core.globals import set_debug
    set_debug(False)  # run_dataset.py enables verbose tracing on import
    import pandas as pd

    # ---------------- (a) pilot tallies
    runs = []
    for f in ("results/mongo_content_result.txt", "results/run-1.txt"):
        runs += parse_mongo_dump(Path(f))
    decisions, ric = [], []
    for r in runs:
        for rid, rec in r["records"].items():
            if rec["situation"]:
                decisions.append({"run": r["run_id"][:8], "record": rid, "situation": rec["situation"],
                                  "severity": rec["severity"], "agents": sorted(rec["agents"])})
            for a, st in rec["agents"].items():
                ric.append({"run": r["run_id"][:8], "record": rid, "agent": a, "status": st})
    pilot = {"decisions": decisions, "ric_outcomes": ric}
    dump_json(pilot, "e4_pilot_runs.json")

    # ---------------- (b) replay
    df = pd.read_csv("data/prepared_network_slicing_dataset.csv")
    row = df[df.record_id == "record_00001"].iloc[0]
    rec = _row_to_telemetry_record(row, "record_00001")
    sla = MonitoringAgent.analyze_sla(rec).model_dump(mode="json")
    trusted = {"target": "cell-01", "slice_id": rec.slice_type, "sla_analysis": sla,
               "persistence": {"latency": 1, "jitter": 1, "packet_loss": 1, "throughput": 0},
               "record_id": "record_00001", "run_id": "run-1"}
    legacy = {"record_id": "record_00001", "run_id": "run-1"}

    gov = RANTrustGovernanceAgent(use_llm=False)
    mw = RANAuthorizationMiddleware(trust_governance_agent=gov)

    class Req:
        def __init__(self, c, ctx):
            self.tool_call = {"name": c["tool"], "args": dict(c["args"]), "id": c["id"]}
            self.state = {"agent_invocation": {"agent": c["agent"], "responsibility": "r", "objective": "o",
                                               "priority": "high", "reason": "x", "context": ctx}}

    async def handler(_):
        return {"status": "success"}

    replay = []
    for c in RUN1_CALLS:
        out = {"call": c["id"], "agent": c["agent"], "tool": c["tool"], "reasoning_logged": c["reasoning_logged"]}
        for name, ctx in (("legacy_context", legacy), ("trusted_context", trusted)):
            audit = []
            tok = RAN_AUDIT.set(audit)
            try:
                await mw.awrap_tool_call(Req(c, ctx), handler)
            finally:
                RAN_AUDIT.reset(tok)
            g = audit[0]["governance"]
            out[name] = {"approved": g["approved"], "failed_stage": g.get("failed_stage"), "reason": g["reason"]}
        # independent per-stage defects (trusted context)
        _, arguments = build_submission(c["tool"], c["args"])
        defects = []
        try:
            a = ARG_MODELS[c["tool"]].model_validate(arguments).model_dump()
        except Exception as exc:
            defects.append("schema: " + "; ".join(e["loc"][0] + " " + e["type"] for e in exc.errors()))
            a = None
        if a is not None:
            if a["target"] != trusted["target"]:
                defects.append(f"policy_trust: target {a['target']!r} != {trusted['target']!r}")
            if "slice_id" in a and a["slice_id"] != trusted["slice_id"]:
                defects.append(f"policy_trust: slice_id {a['slice_id']!r} != {trusted['slice_id']!r}")
            s = _safety_violation(c["tool"], a)
            if s:
                defects.append("resource_safety: " + s)
        else:
            if arguments.get("target") != trusted["target"]:
                defects.append(f"policy_trust: target {arguments.get('target')!r} != {trusted['target']!r}")
            if arguments.get("slice_id") not in (None, trusted["slice_id"]):
                defects.append(f"policy_trust: slice_id {arguments.get('slice_id')!r} != {trusted['slice_id']!r}")
        out["all_defects"] = defects
        replay.append(out)
    dump_json(replay, "e4_replay_run1.json")
    print(json.dumps({"pilot": pilot, "replay": replay}, indent=2))


def main() -> None:
    args = base_parser(__doc__).parse_args()
    use_project(args.project)
    asyncio.run(main_async(args.project))


if __name__ == "__main__":
    main()
