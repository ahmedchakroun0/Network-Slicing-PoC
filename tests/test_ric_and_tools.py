import pytest

from agents.base_ric import BaseRICAgent
from ric_tools.knowledge import search_ran_knowledge
from src.schema.near_rt_ric import (
    GovernanceResult,
    RANActionProposal,
    RANToolResult,
)


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
    monkeypatch.setattr("ric_tools.knowledge.KNOWLEDGE_FILE", knowledge_file)

    result = await search_ran_knowledge.ainvoke({"query": "latency scheduler"})

    assert (
        "Latency optimization guidance" in result
        or "Scheduler fairness for slices" in result
    )


# The action tools run on the MCP servers: tests/test_mcp_servers.py.


@pytest.mark.asyncio
async def test_scheduler_tool_offers_the_testbed_schedulers(bank, make_scenario):
    import httpx
    from mcp import Client

    from ran_emulator.api import create_app
    from ran_emulator.client import EmulatorClient
    from ric_tools.server import build_server

    emulator = EmulatorClient(base_url="http://emu", transport=httpx.ASGITransport(
        app=create_app(bank, make_scenario())))
    async with Client(build_server("near_rt_ric", emulator)) as client:
        tool = next(t for t in (await client.list_tools()).tools
                    if t.name == "optimize_scheduler")
    assert tool.input_schema["properties"]["scheduler"]["enum"] == [
        "round_robin", "waterfilling", "proportional_fair"]
