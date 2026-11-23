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
        sla_analysis=MonitoringAgent.analyze_sla(make_record(queueing_delay_ms=None)),
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
    record = make_record(record_id="rec-warning", queueing_delay_ms=None)

    result = await agent.run(record)

    assert result.situation == "warning"
    assert result.action_required is False
    assert result.selected_agents == []
    assert "monitoring data is incomplete" in result.reason.lower()
    assert result.sla_analysis.unavailable_metrics == ["queueing_delay"]


def _insert(mongo, record):
    mongo.insert_record(
        run_id="run1",
        record_id=record.record_id,
        seq=1,
        telemetry=record.model_dump(mode="json"),
        sla={},
        persistence={},
    )


@pytest.mark.asyncio
async def test_run_records_missing_subagent_as_error(
    make_monitoring_agent, fake_llm, mongo, make_record
):
    decision = MonitoringDecision(
        situation="critical",
        severity="high",
        global_situation="Near-RT intervention needed.",
        action_required=True,
        sla_analysis=MonitoringAgent.analyze_sla(make_record(queueing_delay_ms=300.0)),
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

    record = make_record(record_id="rec-missing-agent", queueing_delay_ms=300.0)
    _insert(mongo, record)
    agent = make_monitoring_agent(sub_agents={}, mongo=mongo, run_id="run1")
    result = await agent.run(record)

    assert result.action_required is True
    stored = mongo.records.find_one({"_id": "run1:rec-missing-agent"})["agents"]["near_rt_ric"]
    assert stored["status"] == "error"
    assert "No implementation registered" in stored["error"]["message"]


@pytest.mark.asyncio
async def test_run_records_subagent_exception_as_error(
    make_monitoring_agent, fake_llm, mongo, make_record, make_sub_agent
):
    failing_agent = make_sub_agent(error=RuntimeError("policy engine unavailable"))
    decision = MonitoringDecision(
        situation="critical",
        severity="high",
        global_situation="Policy-level action needed.",
        action_required=True,
        sla_analysis=MonitoringAgent.analyze_sla(make_record(dl_throughput_per_ue_mbps=1.5)),
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

    record = make_record(record_id="rec-subagent-error", dl_throughput_per_ue_mbps=1.5)
    _insert(mongo, record)
    agent = make_monitoring_agent(
        sub_agents={"non_rt_ric": failing_agent}, mongo=mongo, run_id="run1"
    )
    result = await agent.run(record, persistence={"throughput": 3})

    assert result.action_required is True
    assert failing_agent.calls[0].agent == "non_rt_ric"
    stored = mongo.records.find_one({"_id": "run1:rec-subagent-error"})["agents"]["non_rt_ric"]
    assert stored["status"] == "error" and stored["error"]["type"] == "RuntimeError"
