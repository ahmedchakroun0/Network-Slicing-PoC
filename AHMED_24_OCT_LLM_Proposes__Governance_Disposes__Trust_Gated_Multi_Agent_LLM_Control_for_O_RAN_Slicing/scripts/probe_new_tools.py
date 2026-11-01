"""Probe the deterministic governance stages with the tools added in the uploaded
revision (QoS, link adaptation, slice-capacity policy). Usage:
  python probe_new_tools.py PROJECT_DIR > ../data/llm_runs/probe_<tag>.txt
"""
import asyncio, sys
sys.path.insert(0, sys.argv[1])
from agents.ran_trust_governance import RANTrustGovernanceAgent
from src.schema.near_rt_ric import RANActionProposal
ctx = {"invocation": {"agent": None, "responsibility": "throughput", "objective": "throughput",
       "priority": "high", "reason": "", "context": {"target": "bs2", "slice_id": "eMBB",
       "sla_analysis": {"violations": ["throughput"]}, "persistence": {"throughput": 5}}}}
cases = [
 ("near_rt_ric", "adjust_link_adaptation", {"slice_id": "eMBB", "link_adaptation_params": {"mcs_offset": -2}}),
 ("near_rt_ric", "adjust_qos_parameters", {"slice_id": "eMBB", "qos_parameters": {"priority": 1e9, "gbr_mbps": -5}}),
 ("non_rt_ric", "optimize_slice_capacity_policy", {"slice_id": "eMBB", "min_prb_ratio_pct": 100, "max_prb_ratio_pct": 100}),
 ("non_rt_ric", "optimize_slice_capacity_policy", {"slice_id": "eMBB", "min_prb_ratio_pct": 900, "max_prb_ratio_pct": -4}),
 ("non_rt_ric", "update_slice_resource_policy", {"slice_id": "eMBB", "min_prb_ratio_pct": 900, "max_prb_ratio_pct": -4}),
]
async def main():
    g = RANTrustGovernanceAgent(use_llm=False)
    for agent, tool, params in cases:
        c = {"invocation": {**ctx["invocation"], "agent": agent}}
        p = RANActionProposal(tool_name=tool, target="bs2", parameters=params, rationale="r", expected_effect="throughput", confidence=0.5, invocation_context=c)
        r = await g.authorize(p)
        print(f"{tool:32s} {params} -> approved={r.approved} stage={r.failed_stage} passed={r.checks_passed} {r.reason[:90]}")
asyncio.run(main())
