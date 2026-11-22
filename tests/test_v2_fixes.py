"""Regression tests for the v2 fixes (role binding, Non-RT persistence gate,
knowledge-base path, persistence in the Monitoring
input)."""

import pytest

from agents.ran_trust_governance import NON_RT_MIN_PERSISTENCE, RANTrustGovernanceAgent
from ric_tools import knowledge
from src.schema.near_rt_ric import RANActionProposal


def _proposal(tool_name, parameters, agent, context):
    return RANActionProposal(
        tool_name=tool_name,
        target="cell-1",
        parameters=parameters,
        rationale="Persistent latency violation on this slice justifies the change.",
        expected_effect="Reduce latency for the affected slice.",
        evidence_sources=["latency"],
        confidence=0.8,
        invocation_context={"invocation": {"agent": agent, "context": context}},
    )


CTX = {
    "target": "cell-1",
    "slice_id": "eMBB",
    "sla_analysis": {"violations": ["latency"]},
}
POLICY = {
    "slice_id": "eMBB",
    "min_prb_ratio_pct": 20,
    "max_prb_ratio_pct": 60,
}


@pytest.mark.asyncio
async def test_role_binding_rejects_cross_role_tool():
    gov = RANTrustGovernanceAgent(use_llm=False)
    p = _proposal("update_slice_resource_policy", POLICY, "near_rt_ric", {**CTX, "persistence": {"latency": 5}})
    d = await gov.authorize(p)
    assert d.approved is False
    assert d.failed_stage == "policy_trust"
    assert "not owned" in d.reason


@pytest.mark.asyncio
async def test_role_binding_rejects_unknown_agent():
    gov = RANTrustGovernanceAgent(use_llm=False)
    p = _proposal("adjust_prb_allocation", {"slice_id": "eMBB", "prb_percentage": 50}, None, CTX)
    d = await gov.authorize(p)
    assert d.approved is False and d.failed_stage == "policy_trust"


@pytest.mark.asyncio
@pytest.mark.parametrize("persistence,approved", [
    (None, False),
    ({"latency": NON_RT_MIN_PERSISTENCE - 1}, False),
    ({"latency": NON_RT_MIN_PERSISTENCE}, True),
    ({"latency": 1, "jitter": 9}, False),   # only VIOLATED KPIs count
])
async def test_non_rt_persistence_gate(persistence, approved):
    gov = RANTrustGovernanceAgent(use_llm=False)
    ctx = dict(CTX)
    if persistence is not None:
        ctx["persistence"] = persistence
    d = await gov.authorize(_proposal("update_slice_resource_policy", POLICY, "non_rt_ric", ctx))
    assert d.approved is approved
    if not approved:
        assert d.failed_stage == "sla"


@pytest.mark.asyncio
async def test_near_rt_actions_do_not_need_persistence():
    gov = RANTrustGovernanceAgent(use_llm=False)
    p = _proposal("adjust_prb_allocation", {"slice_id": "eMBB", "prb_percentage": 50}, "near_rt_ric", CTX)
    assert (await gov.authorize(p)).approved is True


def test_knowledge_file_resolves_from_project_root():
    assert knowledge.KNOWLEDGE_FILE.is_file()


@pytest.mark.asyncio
async def test_monitoring_prompt_receives_persistence(make_monitoring_agent, fake_llm, make_record):
    from src.schema.monitoring import MonitoringDecision

    record = make_record(queueing_delay_ms=400.0)
    agent = make_monitoring_agent()
    analysis = agent.analyze_sla(record)
    fake_llm.queue(MonitoringDecision(
        situation="warning", severity="low", global_situation="queueing delay",
        action_required=False, sla_analysis=analysis, reason="test"))
    await agent.run(record, persistence={"queueing_delay": 4})
    assert "'queueing_delay': 4" in fake_llm.last_prompt
