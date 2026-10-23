import pytest

from agents.monitoring_agent import MonitoringAgent
from src.schema.monitoring import AgentInvocation, MonitoringDecision


@pytest.mark.asyncio
async def test_run_marks_warning_when_some_metrics_are_unavailable(
    make_monitoring_agent, fake_llm, memory, make_record
):
    decision = MonitoringDecision(
        situation="critical",
        severity="high",
        global_situation="Partial telemetry received.",
        action_required=True,
        sla_analysis=MonitoringAgent.analyze_sla(
            make_record(jitter_budget_ns=None, slice_jitter_ns=None)
        ),
        selected_agents=[
            AgentInvocation(
                agent="near_rt_ric",
                responsibility="Mitigate latency",
                objective="Reduce latency without overreacting to missing jitter data",
                priority="high",
                reason="Latency violation is present.",
                context={"target": "cell-1", "slice_id": "slice-1"},
            )
        ],
        reason="Initial LLM decision.",
        record_id="rec-warning",
    )
    fake_llm.queue(decision)

    agent = make_monitoring_agent(sub_agents={})
    record = make_record(
        record_id="rec-warning",
        jitter_budget_ns=None,
        slice_jitter_ns=None,
        slice_latency_ns=500.0,
        latency_budget_ns=1000.0,
    )

    result = await agent.run(record)

    assert result.situation == "warning"
    assert result.action_required is False
    assert result.selected_agents == []
    assert "monitoring data is incomplete" in result.reason.lower()
    assert result.sla_analysis.unavailable_metrics == ["jitter"]


@pytest.mark.asyncio
async def test_run_records_missing_subagent_as_error(
    make_monitoring_agent, fake_llm, memory, make_record
):
    decision = MonitoringDecision(
        situation="critical",
        severity="high",
        global_situation="Near-RT intervention needed.",
        action_required=True,
        sla_analysis=MonitoringAgent.analyze_sla(
            make_record(slice_latency_ns=3000.0, latency_budget_ns=1000.0)
        ),
        selected_agents=[
            AgentInvocation(
                agent="near_rt_ric",
                responsibility="Mitigate latency",
                objective="Reduce slice latency",
                priority="high",
                reason="Latency violation detected.",
                context={"target": "cell-1", "slice_id": "slice-1"},
            )
        ],
        reason="Invoke near-RT RIC.",
        record_id="rec-missing-agent",
    )
    fake_llm.queue(decision)

    agent = make_monitoring_agent(sub_agents={})
    result = await agent.run(
        make_record(
            record_id="rec-missing-agent",
            slice_latency_ns=3000.0,
            latency_budget_ns=1000.0,
        )
    )

    assert result.action_required is True
    assert memory.executions_of("near_rt_ric")[-1].tool_name == "agent_invocation"
    assert memory.executions_of("near_rt_ric")[-1].result["status"] == "error"
    assert (
        "No implementation registered"
        in memory.executions_of("near_rt_ric")[-1].result["message"]
    )


@pytest.mark.asyncio
async def test_run_records_subagent_exception_as_error(
    make_monitoring_agent, fake_llm, memory, make_record, make_sub_agent
):
    failing_agent = make_sub_agent(error=RuntimeError("policy engine unavailable"))
    decision = MonitoringDecision(
        situation="critical",
        severity="high",
        global_situation="Policy-level action needed.",
        action_required=True,
        sla_analysis=MonitoringAgent.analyze_sla(
            make_record(
                slice_available_transfer_rate_gbps=3.0, data_rate_budget_gbps=10.0
            )
        ),
        selected_agents=[
            AgentInvocation(
                agent="non_rt_ric",
                responsibility="Plan longer-term policy change",
                objective="Address persistent throughput degradation",
                priority="high",
                reason="Throughput violation detected.",
                context={"target": "cell-9", "slice_id": "slice-9"},
            )
        ],
        reason="Invoke non-RT RIC.",
        record_id="rec-subagent-error",
    )
    fake_llm.queue(decision)

    agent = make_monitoring_agent(sub_agents={"non_rt_ric": failing_agent})
    result = await agent.run(
        make_record(
            record_id="rec-subagent-error",
            slice_available_transfer_rate_gbps=3.0,
            data_rate_budget_gbps=10.0,
        )
    )

    assert result.action_required is True
    assert failing_agent.calls[0].agent == "non_rt_ric"
    assert memory.executions_of("non_rt_ric")[-1].result["status"] == "error"
    assert memory.executions_of("non_rt_ric")[-1].result["error_type"] == "RuntimeError"
