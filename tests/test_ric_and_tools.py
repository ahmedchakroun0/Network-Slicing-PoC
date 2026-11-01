import pytest

from agents.base_ric import BaseRICAgent
from src.schema.near_rt_ric import (
    GovernanceResult,
    RANActionProposal,
    RANToolResult,
)
from tools.near_rt_ric_tools import adjust_prb_allocation, search_ran_knowledge
from tools.near_rt_ric_tools import optimize_scheduler
from tools.non_rt_ric_tools import update_slice_resource_policy


def test_base_ric_build_typed_result_maps_completed_status():
    audit = [
        {
            "proposal": {
                "tool_name": "adjust_prb_allocation",
                "parameters": {"slice_id": "slice-1", "prb_percentage": 60},
                "target": "cell-1",
                "rationale": "Improve latency.",
                "expected_effect": "Lower latency.",
                "evidence_sources": ["latency"],
                "confidence": 0.9,
                "invocation_context": {},
            },
            "governance": {
                "approved": True,
                "reason": "Approved.",
                "failed_stage": None,
                "checks_passed": ["schema", "policy_trust", "sla", "resource_safety"],
            },
            "execution": {
                "tool_name": "adjust_prb_allocation",
                "status": "executed",
                "target": "cell-1",
                "parameters": {"slice_id": "slice-1", "prb_percentage": 60},
                "result": {"status": "success"},
                "message": "Execution succeeded.",
            },
        }
    ]

    result = BaseRICAgent._build_typed_result(audit, messages=[])

    assert result.status == "completed"
    assert isinstance(result.proposal, RANActionProposal)
    assert isinstance(result.governance, GovernanceResult)
    assert isinstance(result.execution, RANToolResult)


def test_base_ric_build_typed_result_without_audit_returns_no_action():
    result = BaseRICAgent._build_typed_result([], messages=[])

    assert result.status == "no_action"
    assert result.proposal is None
    assert result.governance is None
    assert result.execution is None


@pytest.mark.asyncio
async def test_search_ran_knowledge_returns_relevant_chunk(tmp_path, monkeypatch):
    content = "Latency optimization guidance\n\nScheduler fairness for slices\n\nCoverage tuning."
    knowledge_file = tmp_path / "ran_knowledge.txt"
    knowledge_file.write_text(content, encoding="utf-8")
    monkeypatch.setattr("tools.near_rt_ric_tools.KNOWLEDGE_FILE", knowledge_file)

    result = await search_ran_knowledge.ainvoke({"query": "latency scheduler"})

    assert (
        "Latency optimization guidance" in result
        or "Scheduler fairness for slices" in result
    )


@pytest.mark.asyncio
async def test_near_rt_prb_tool_returns_success_payload():
    result = await adjust_prb_allocation.ainvoke(
        {
            "target": "cell-1",
            "slice_id": "slice-1",
            "prb_percentage": 60,
            "rationale": "Latency exceeds budget; raise the slice PRB share to 60%.",
            "expected_effect": "Lower slice latency.",
            "confidence": 0.8,
        }
    )

    assert result["status"] == "success"
    assert result["tool"] == "adjust_prb_allocation"
    assert result["target"] == "cell-1"
    assert result["slice_id"] == "slice-1"


@pytest.mark.asyncio
async def test_non_rt_policy_tool_returns_simulated_policy_result():
    result = await update_slice_resource_policy.ainvoke(
        {
            "target": "cell-1",
            "slice_id": "eMBB",
            "min_prb_ratio_pct": 40,
            "max_prb_ratio_pct": 80,
            "rationale": "Throughput is persistently below budget; guarantee 40 % of the PRBs.",
            "expected_effect": "Higher throughput per UE.",
            "confidence": 0.85,
        }
    )

    assert result["status"] == "success"
    assert result["tool"] == "update_slice_resource_policy"
    assert result["channel"] == "A1"
    assert result["min_prb_ratio_pct"] == 40


def test_scheduler_tool_offers_the_testbed_schedulers():
    assert optimize_scheduler.args["scheduler"]["enum"] == [
        "round_robin", "waterfilling", "proportional_fair"]
