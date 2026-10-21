import pytest

from agents.monitoring_agent import MonitoringAgent
from src.schema.monitoring import AgentInvocation, MonitoringDecision


@pytest.mark.asyncio
async def test_analyze_sla_identifies_all_violations(make_record):
    record = make_record(
        slice_latency_ns=2000.0,
        jitter_budget_ns=80.0,
        slice_jitter_ns=95.0,
        slice_packet_loss=0.02,
        packet_loss_budget=0.01,
        slice_available_transfer_rate_gbps=4.0,
        data_rate_budget_gbps=10.0,
    )

    analysis = MonitoringAgent.analyze_sla(record)

    assert analysis.violations == [
        "latency",
        "jitter",
        "packet_loss",
        "throughput",
    ]
    assert analysis.latency.status == "violation"
    assert analysis.jitter.status == "violation"
    assert analysis.packet_loss.status == "violation"
    assert analysis.throughput.status == "violation"


@pytest.mark.asyncio
async def test_run_no_action_when_kpis_are_ok(
    make_monitoring_agent, fake_llm, memory, make_record
):
    decision = MonitoringDecision(
        situation="normal",
        severity="none",
        global_situation="All monitored KPIs are within budget.",
        action_required=False,
        sla_analysis=MonitoringAgent.analyze_sla(make_record()),
        selected_agents=[],
        reason="No intervention required.",
        record_id="rec-1",
    )
    fake_llm.queue(decision)

    agent = make_monitoring_agent(sub_agents={})
    record = make_record(record_id="rec-1")

    result = await agent.run(record)

    assert result.action_required is False
    assert result.selected_agents == []
    assert memory.state.network_status == decision.global_situation
    assert memory.state.tools_executed[0].agent == "monitoring"


@pytest.mark.asyncio
async def test_run_invokes_only_selected_agents(
    make_monitoring_agent, fake_llm, memory, make_record, make_sub_agent
):
    near_agent = make_sub_agent(result={"status": "ok"})
    non_agent = make_sub_agent(result={"status": "ok"})

    selected = [
        AgentInvocation(
            agent="near_rt_ric",
            responsibility="Mitigate latency",
            objective="Reduce slice latency",
            priority="high",
            reason="Latency exceeds budget",
            context={"target": "cell-1", "slice_id": "slice-1"},
        ),
        AgentInvocation(
            agent="non_rt_ric",
            responsibility="Review policy",
            objective="Reduce persistent degradation",
            priority="medium",
            reason="Persistent throughput loss",
            context={"target": "cell-1", "slice_id": "slice-1"},
        ),
    ]

    decision = MonitoringDecision(
        situation="critical",
        severity="high",
        global_situation="High latency and throughput degradation detected.",
        action_required=True,
        sla_analysis=MonitoringAgent.analyze_sla(
            make_record(
                slice_latency_ns=5000.0,
                data_rate_budget_gbps=20.0,
                slice_available_transfer_rate_gbps=8.0,
                packet_loss_budget=0.01,
                slice_packet_loss=0.03,
            )
        ),
        selected_agents=selected,
        reason="Intervention required.",
        record_id="rec-2",
    )
    fake_llm.queue(decision)

    agent = make_monitoring_agent(
        sub_agents={"near_rt_ric": near_agent, "non_rt_ric": non_agent}
    )
    record = make_record(
        record_id="rec-2",
        slice_latency_ns=5000.0,
        data_rate_budget_gbps=20.0,
        slice_available_transfer_rate_gbps=8.0,
        packet_loss_budget=0.01,
        slice_packet_loss=0.03,
        jitter_budget_ns=80.0,
        slice_jitter_ns=95.0,
    )
    result = await agent.run(record)

    assert result.action_required is True
    assert near_agent.calls[0].agent == "near_rt_ric"
    assert non_agent.calls[0].agent == "non_rt_ric"
    assert len(memory.state.tools_executed) >= 3
    assert any(
        item.agent == "near_rt_ric" and item.tool_name == "agent_invocation"
        for item in memory.state.tools_executed
    )
    assert any(
        item.agent == "non_rt_ric" and item.tool_name == "agent_invocation"
        for item in memory.state.tools_executed
    )
