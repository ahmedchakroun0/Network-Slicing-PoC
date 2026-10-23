import pytest
from redis.asyncio import Redis

from agents.non_rt_ric_agent import NonRTRICAgent
from middleware.ran_authorisation import RAN_AUDIT
from src.memory.working_memory import RedisWorkingMemory
from src.schema.monitoring import AgentInvocation
from src.schema.near_rt_ric import GovernanceResult, RANActionProposal, RANToolResult


@pytest.fixture
async def redis_memory():
    redis = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
    await redis.flushdb()
    memory = RedisWorkingMemory(redis)
    yield memory
    await redis.flushdb()
    await redis.aclose()


@pytest.mark.asyncio
async def test_non_rt_ric_agent_invoke_returns_completed_status(redis_memory):
    class FakeAgent:
        async def ainvoke(self, payload):
            audit = RAN_AUDIT.get()
            audit.append(
                {
                    "proposal": RANActionProposal(
                        tool_name="update_traffic_steering_policy",
                        parameters={
                            "steering_mode": "latency_aware",
                            "load_balance_threshold_pct": 70,
                        },
                        target="cell-1",
                        rationale="Latency is elevated and should be reduced with a low-latency steering policy.",
                        expected_effect="Lower latency and improve slice fairness.",
                        evidence_sources=["latency", "throughput"],
                        confidence=0.85,
                        invocation_context={},
                    ).model_dump(mode="json"),
                    "governance": GovernanceResult(
                        approved=True,
                        reason="Approved by governance.",
                        checks_passed=[
                            "schema",
                            "policy_trust",
                            "sla",
                            "resource_safety",
                        ],
                    ).model_dump(mode="json"),
                    "execution": RANToolResult(
                        tool_name="update_traffic_steering_policy",
                        status="executed",
                        target="cell-1",
                        parameters={
                            "steering_mode": "latency_aware",
                            "load_balance_threshold_pct": 70,
                        },
                        result={"channel": "A1", "applied": True, "simulated": True},
                        message="Policy applied.",
                    ).model_dump(mode="json"),
                }
            )
            return {"messages": [{"tool_calls": [{"name": "update_traffic_steering_policy"}]}]}

    agent = NonRTRICAgent(trust_governance_agent=object(), memory=redis_memory)
    agent.agent = FakeAgent()

    invocation = AgentInvocation(
        agent="non_rt_ric",
        responsibility="Perform long-term policy optimization.",
        objective="Reduce persistent latency degradation.",
        priority="high",
        reason="Persistent KPI degradation requires policy action.",
        context={"target": "cell-1", "slice_id": "slice-1"},
    )

    result = await agent.invoke(invocation)

    assert result.status == "completed"
    assert result.proposal is not None
    assert result.governance is not None
    assert result.execution is not None
    assert result.execution.status == "executed"


@pytest.mark.asyncio
async def test_non_rt_ric_agent_invoke_returns_rejected_status_on_governance_failure(
    redis_memory,
):
    class FakeAgent:
        async def ainvoke(self, payload):
            audit = RAN_AUDIT.get()
            audit.append(
                {
                    "proposal": RANActionProposal(
                        tool_name="update_admission_control_policy",
                        parameters={
                            "slice_id": "slice-1",
                            "max_utilization_pct": 90,
                        },
                        target="cell-1",
                        rationale="Admission should be tightened.",
                        expected_effect="Lower overload risk.",
                        evidence_sources=["throughput"],
                        confidence=0.7,
                        invocation_context={},
                    ).model_dump(mode="json"),
                    "governance": GovernanceResult(
                        approved=False,
                        reason="Admission policy exceeds the allowed policy guardrails.",
                        failed_stage="policy_trust",
                        checks_passed=["schema"],
                    ).model_dump(mode="json"),
                    "execution": RANToolResult(
                        tool_name="update_admission_control_policy",
                        status="rejected",
                        target="cell-1",
                        parameters={"slice_id": "slice-1", "max_utilization_pct": 90},
                        result={"status": "rejected"},
                        message="Rejected.",
                    ).model_dump(mode="json"),
                }
            )
            return {"messages": [{"tool_calls": [{"name": "update_traffic_steering_policy"}]}]}

    agent = NonRTRICAgent(memory=redis_memory)
    agent.agent = FakeAgent()
    invocation = AgentInvocation(
        agent="non_rt_ric",
        responsibility="Review long-term policy options.",
        objective="Limit persistent overload.",
        priority="medium",
        reason="Policy change needed.",
        context={"target": "cell-1", "slice_id": "slice-1"},
    )

    result = await agent.invoke(invocation)

    assert result.status == "rejected"
    assert result.governance is not None
    assert result.governance.approved is False
    assert result.execution is not None
    assert result.execution.status == "rejected"
