"""Regression tests for the issues found in the live-LLM runs (results/run-6..8).

1. A tool without a safety envelope must never be executed (default deny);
   optimize_slice_capacity_policy gets the same envelope as the slice policy.
2. Governance argument models must mirror the tool signatures
   (adjust_qos_parameters / adjust_link_adaptation were always rejected).
3. A duplicate agent selection must not overwrite the first invocation's audit.
4. One executed call per tool per invocation, and a bounded number of calls.
"""

import json

import pytest
from langchain_core.messages import ToolMessage

from agents.monitoring_agent import MonitoringAgent
from agents.ran_trust_governance import RANTrustGovernanceAgent
from middleware.ran_authorisation import MAX_PROTECTED_CALLS, RAN_AUDIT, RANAuthorizationMiddleware
from src.schema.monitoring import AgentInvocation
from src.schema.near_rt_ric import RANActionProposal

CTX = {
    "target": "bs2",
    "slice_id": "eMBB",
    "sla_analysis": {"violations": ["throughput"]},
    "persistence": {"throughput": 5},
}


def _proposal(tool, params, agent):
    return RANActionProposal(
        tool_name=tool,
        target="bs2",
        parameters=params,
        rationale="throughput 3.9 Mbps < 7.2 Mbps budget",
        expected_effect="throughput increases",
        confidence=0.5,
        invocation_context={"invocation": {"agent": agent, "context": CTX}},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("lo,hi,ok", [(900, -4, False), (100, 100, False), (20, 60, True)])
async def test_capacity_policy_has_safety_envelope(lo, hi, ok):
    gov = RANTrustGovernanceAgent(use_llm=False)
    d = await gov.authorize(_proposal(
        "optimize_slice_capacity_policy",
        {"slice_id": "eMBB", "min_prb_ratio_pct": lo, "max_prb_ratio_pct": hi}, "non_rt_ric"))
    assert d.approved is ok
    if not ok:
        assert d.failed_stage == "resource_safety"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,params", [
    ("adjust_qos_parameters", {"slice_id": "eMBB", "qos_parameters": {"priority": 1e9}}),
    ("adjust_link_adaptation", {"slice_id": "eMBB", "link_adaptation_params": {"mcs_offset": -2}}),
])
async def test_free_form_tools_pass_schema_but_are_denied(tool, params):
    gov = RANTrustGovernanceAgent(use_llm=False)
    d = await gov.authorize(_proposal(tool, params, "near_rt_ric"))
    assert d.approved is False
    assert "schema" in d.checks_passed          # signature now matches the tool
    assert d.failed_stage == "resource_safety"  # default deny: no envelope
    assert "No safety envelope" in d.reason


def test_duplicate_agent_selection_is_merged():
    sel = [
        AgentInvocation(agent="near_rt_ric", responsibility="latency", objective="latency < 100 ms",
                        priority="medium", reason="a", context={}),
        AgentInvocation(agent="near_rt_ric", responsibility="jitter", objective="jitter < 50 ms",
                        priority="high", reason="b", context={}),
        AgentInvocation(agent="non_rt_ric", responsibility="throughput", objective="throughput",
                        priority="low", reason="c", context={}),
    ]
    out = MonitoringAgent._merge_duplicate_agents(sel)
    assert [i.agent for i in out] == ["near_rt_ric", "non_rt_ric"]
    assert "latency" in out[0].objective and "jitter" in out[0].objective
    assert out[0].priority == "high"


class _Req:
    def __init__(self, name, args, call_id):
        self.tool_call = {"name": name, "args": args, "id": call_id}
        self.tool = None
        self.state = {"agent_invocation": {"agent": "non_rt_ric", "context": CTX}}


@pytest.mark.asyncio
async def test_one_execution_per_tool_and_call_budget():
    mw = RANAuthorizationMiddleware(RANTrustGovernanceAgent(use_llm=False))
    executed = []

    async def handler(req):
        executed.append(req.tool_call["args"])
        return ToolMessage(content=json.dumps({"status": "success"}), tool_call_id=req.tool_call["id"])

    reasoning = {"rationale": "throughput below budget", "expected_effect": "throughput up",
                 "confidence": 0.6}
    audit = []
    token = RAN_AUDIT.set(audit)
    try:
        for i, (lo, hi) in enumerate([(5, 80), (60, 80), (60, 80)]):
            args = {"target": "bs2", "slice_id": "eMBB", "min_prb_ratio_pct": lo,
                    "max_prb_ratio_pct": hi, **reasoning}
            await mw.awrap_tool_call(_Req("update_slice_resource_policy", args, f"c{i}"), handler)
        assert len(executed) == 1  # the run-7 sequence executed 3 times
        assert [a["governance"]["failed_stage"] for a in audit] == [None, "invocation_limit", "invocation_limit"]
        for i in range(MAX_PROTECTED_CALLS + 2):
            await mw.awrap_tool_call(_Req("update_traffic_steering_policy",
                                          {"target": "bs2", "steering_mode": "bogus"}, f"s{i}"), handler)
        assert len(audit) == 3 + MAX_PROTECTED_CALLS + 2
        assert audit[-1]["governance"]["failed_stage"] == "invocation_limit"
        assert "protected calls" in audit[-1]["governance"]["reason"]
    finally:
        RAN_AUDIT.reset(token)



@pytest.mark.asyncio
async def test_violation_is_never_reported_as_normal(make_monitoring_agent, fake_llm, make_record):
    """run-24: the LLM returned situation='normal' (severity high) for a violating record."""
    from src.schema.monitoring import MonitoringDecision

    record = make_record(record_id="rec-v", slice_latency_ns=2000.0)
    analysis = MonitoringAgent.analyze_sla(record)
    assert analysis.violations
    fake_llm.queue(MonitoringDecision(
        situation="normal", severity="none", global_situation="violations_detected",
        action_required=False, sla_analysis=analysis, selected_agents=[], reason="r",
        record_id="rec-v"))
    result = await make_monitoring_agent(sub_agents={}).run(record)
    assert result.situation == "warning"
    assert result.severity != "none"
