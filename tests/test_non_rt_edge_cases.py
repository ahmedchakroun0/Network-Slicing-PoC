import pytest

from agents.non_rt_ric_agent import NonRTRICAgent
from src.schema.monitoring import AgentInvocation


@pytest.mark.asyncio
async def test_non_rt_initialize_requires_governance_agent(memory):
    agent = NonRTRICAgent(memory=memory)

    with pytest.raises(RuntimeError, match="requires a RAN Trust & Governance Agent"):
        await agent.initialize()


@pytest.mark.asyncio
async def test_non_rt_invoke_rejects_wrong_agent_name(memory):
    agent = NonRTRICAgent(memory=memory)
    invocation = AgentInvocation(
        agent="near_rt_ric",
        responsibility="Wrong target agent for this test",
        objective="Should fail fast",
        priority="low",
        reason="Agent mismatch.",
        context={"target": "cell-1", "slice_id": "slice-1"},
    )

    with pytest.raises(ValueError, match="cannot handle agent='near_rt_ric'"):
        await agent.invoke(invocation)
