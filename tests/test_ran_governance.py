import pytest

from agents.ran_trust_governance import RANTrustGovernanceAgent
from src.schema.near_rt_ric import GovernanceResult, RANActionProposal


@pytest.mark.asyncio
async def test_authorize_approves_valid_near_rt_action():
    governance = RANTrustGovernanceAgent(use_llm=False)

    proposal = RANActionProposal(
        tool_name="adjust_prb_allocation",
        target="cell-1",
        parameters={"slice_id": "slice-1", "prb_percentage": 55},
        rationale="Latency is exceeding budget, so a targeted PRB increase is justified.",
        expected_effect="Reduce latency for the affected slice.",
        evidence_sources=["latency_budget", "slice_latency"],
        confidence=0.9,
        invocation_context={
            "invocation": {
                "agent": "near_rt_ric",
                "context": {
                    "target": "cell-1",
                    "slice_id": "slice-1",
                    "sla_analysis": {"violations": ["latency"]},
                },
            }
        },
    )

    decision = await governance.authorize(proposal)

    assert isinstance(decision, GovernanceResult)
    assert decision.approved is True
    assert "schema" in decision.checks_passed
    assert "policy_trust" in decision.checks_passed
    assert "sla" in decision.checks_passed
    assert "resource_safety" in decision.checks_passed


@pytest.mark.asyncio
async def test_authorize_rejects_target_mismatch():
    governance = RANTrustGovernanceAgent(use_llm=False)

    proposal = RANActionProposal(
        tool_name="adjust_prb_allocation",
        target="cell-2",
        parameters={"slice_id": "slice-1", "prb_percentage": 55},
        rationale="Targeted increase is required.",
        expected_effect="Reduce latency.",
        evidence_sources=["latency"],
        confidence=0.8,
        invocation_context={
            "invocation": {
                "agent": "near_rt_ric",
                "context": {
                    "target": "cell-1",
                    "slice_id": "slice-1",
                    "sla_analysis": {"violations": ["latency"]},
                },
            }
        },
    )

    decision = await governance.authorize(proposal)

    assert decision.approved is False
    assert decision.failed_stage == "policy_trust"
    assert "Target 'cell-2'" in decision.reason


@pytest.mark.asyncio
async def test_authorize_rejects_out_of_range_resource_change():
    governance = RANTrustGovernanceAgent(use_llm=False)

    proposal = RANActionProposal(
        tool_name="adjust_power_control",
        target="cell-1",
        parameters={"power_offset_db": 9.0},
        rationale="This offset is needed for coverage.",
        expected_effect="Improve signal quality.",
        evidence_sources=["latency"],
        confidence=0.7,
        invocation_context={
            "invocation": {
                "agent": "near_rt_ric",
                "context": {
                    "target": "cell-1",
                    "slice_id": "slice-1",
                    "sla_analysis": {"violations": ["latency"]},
                },
            }
        },
    )

    decision = await governance.authorize(proposal)

    assert decision.approved is False
    assert decision.failed_stage == "resource_safety"
    assert "power_offset_db" in decision.reason
