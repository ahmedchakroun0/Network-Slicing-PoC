import pytest

from agents.monitoring_agent import MonitoringAgent
from src.schema.monitoring import AgentInvocation, MonitoringDecision


@pytest.mark.asyncio
async def test_analyze_sla_identifies_all_violations(make_record):
    record = make_record(dl_throughput_per_ue_mbps=1.8, queueing_delay_ms=800.0)

    analysis = MonitoringAgent.analyze_sla(record)

    assert analysis.slice_id == "eMBB"
    assert analysis.violations == ["throughput", "queueing_delay"]
    assert analysis.kpis["throughput"].budget == 3.6
    assert analysis.kpis["throughput"].bound == "min"
    assert analysis.kpis["queueing_delay"].status == "violation"


def test_analyze_sla_uses_the_slice_profile(make_record):
    # 20 ms is fine for eMBB (<= 100 ms) but violates URLLC (<= 10 ms).
    assert MonitoringAgent.analyze_sla(make_record()).violations == []
    urllc = make_record(slice_id="URLLC", dl_throughput_per_ue_mbps=0.12)
    assert MonitoringAgent.analyze_sla(urllc).violations == ["queueing_delay"]


def test_severity_is_relative_to_the_bound(make_record):
    # throughput 1.0 vs min 3.6 -> 3.6x short -> high; delay 120 vs max 100 -> low
    sev = MonitoringAgent._severity(MonitoringAgent.analyze_sla(
        make_record(dl_throughput_per_ue_mbps=1.0)))
    assert sev == "high"
    sev = MonitoringAgent._severity(MonitoringAgent.analyze_sla(
        make_record(queueing_delay_ms=120.0)))
    assert sev == "low"


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
    history = (await memory.view("bs1", "eMBB")).slice_records
    assert [(e.rec, e.violations, e.severity) for e in history] == [("rec-1", [], "none")]


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
            make_record(queueing_delay_ms=500.0, dl_throughput_per_ue_mbps=2.0)
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
        record_id="rec-2", queueing_delay_ms=500.0, dl_throughput_per_ue_mbps=2.0
    )
    result = await agent.run(record, persistence={"throughput": 3, "queueing_delay": 3})

    assert result.action_required is True
    assert near_agent.calls[0].agent == "near_rt_ric"
    assert non_agent.calls[0].agent == "non_rt_ric"
    assert near_agent.calls[0].context["record_id"] == "rec-2"
    history = (await memory.view("bs1", "eMBB")).slice_records
    assert history[0].agents == ["near_rt_ric", "non_rt_ric"]
    assert history[0].violations == ["throughput", "queueing_delay"]


@pytest.mark.asyncio
async def test_effective_decision_and_llm_output_are_stored(
    make_monitoring_agent, fake_llm, mongo, make_record, make_decision
):
    record = make_record(record_id="rec-3", dl_throughput_per_ue_mbps=1.0)
    mongo.insert_record(run_id="run1", record_id="rec-3", seq=1,
                        telemetry=record.model_dump(mode="json"), sla={}, persistence={})
    # The LLM calls the situation "normal": the override makes it "warning".
    fake_llm.queue(make_decision(situation="normal", selected_agents=[]))

    await make_monitoring_agent(mongo=mongo, run_id="run1").run(record)

    stored = mongo.records.find_one({"_id": "run1:rec-3"})["monitoring"]
    assert stored["situation"] == "warning" and stored["severity"] == "high"
    assert stored["overridden"] is True
    assert stored["llm_output"]["situation"] == "normal"
    assert "sla_analysis" not in stored                 # stored once, on the record


@pytest.mark.asyncio
async def test_llm_output_is_not_duplicated_when_not_overridden(
    make_monitoring_agent, fake_llm, mongo, make_record, make_decision, make_invocation
):
    record = make_record(record_id="rec-4", dl_throughput_per_ue_mbps=1.0)
    mongo.insert_record(run_id="run1", record_id="rec-4", seq=1,
                        telemetry=record.model_dump(mode="json"), sla={}, persistence={})
    fake_llm.queue(make_decision(situation="critical", severity="high",
                                 selected_agents=[make_invocation(context={})]))

    await make_monitoring_agent(mongo=mongo, run_id="run1").run(record)

    stored = mongo.records.find_one({"_id": "run1:rec-4"})["monitoring"]
    assert stored["overridden"] is False and "llm_output" not in stored
    assert "context" not in stored["selected_agents"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("persistence,invoked", [
    (None, ["near_rt_ric"]),
    ({"throughput": 2, "queueing_delay": 0}, ["near_rt_ric"]),
    ({"throughput": 3, "queueing_delay": 0}, ["near_rt_ric", "non_rt_ric"]),
])
async def test_non_rt_is_invoked_only_for_a_persistent_violation(
    make_monitoring_agent, fake_llm, mongo, make_record, make_decision, make_invocation,
    make_sub_agent, persistence, invoked,
):
    record = make_record(record_id="rec-p", dl_throughput_per_ue_mbps=1.0)
    mongo.insert_record(run_id="run1", record_id="rec-p", seq=1,
                        telemetry=record.model_dump(mode="json"), sla={}, persistence={})
    fake_llm.queue(make_decision(selected_agents=[make_invocation("near_rt_ric"),
                                                  make_invocation("non_rt_ric")]))
    near, non = make_sub_agent(), make_sub_agent()

    result = await make_monitoring_agent(
        sub_agents={"near_rt_ric": near, "non_rt_ric": non}, mongo=mongo, run_id="run1"
    ).run(record, persistence=persistence)

    assert [i.agent for i in result.selected_agents] == invoked
    assert bool(non.calls) is ("non_rt_ric" in invoked)
    stored = mongo.records.find_one({"_id": "run1:rec-p"})["monitoring"]
    if "non_rt_ric" not in invoked:
        assert "non_rt_ric not invoked" in stored["orchestration_notes"][0]
        assert stored["overridden"] is True
        assert [a["agent"] for a in stored["llm_output"]["selected_agents"]] == [
            "near_rt_ric", "non_rt_ric"]          # the LLM's choice stays visible
    else:
        assert stored["orchestration_notes"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_agents", [[], ["non_rt_ric"]])
async def test_a_violation_is_never_left_without_an_agent(
    make_monitoring_agent, fake_llm, make_record, make_decision, make_invocation, make_sub_agent,
    llm_agents,
):
    record = make_record(dl_throughput_per_ue_mbps=1.0, queueing_delay_ms=400.0)
    fake_llm.queue(make_decision(action_required=False,
                                 selected_agents=[make_invocation(a) for a in llm_agents]))
    near = make_sub_agent()

    result = await make_monitoring_agent(sub_agents={"near_rt_ric": near}).run(
        record, persistence={"throughput": 1, "queueing_delay": 1})

    assert result.action_required is True
    assert [i.agent for i in result.selected_agents] == ["near_rt_ric"]
    objective = near.calls[0].objective
    assert "throughput (1.0 Mbps vs min 3.6)" in objective and "queueing_delay" in objective
    assert any("fallback" in n for n in result.orchestration_notes)
