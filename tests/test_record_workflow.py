"""The record workflow graph (agents/record_workflow.py): routing between
nodes, the shared state, and the trusted context given to the RIC agents."""

import pytest

from agents.record_workflow import RecordWorkflow
from src.memory.working_memory import slice_key
from src.schema.working_memory import ActionEvent

# eMBB budget is 3.6 Mbps per UE: 1.0 violates it.
VIOLATION = {"dl_throughput_per_ue_mbps": 1.0}


@pytest.fixture
def make_workflow(make_monitoring_agent, memory):
    def _make(sub_agents=None, mongo=None, run_id=None):
        monitoring = make_monitoring_agent(sub_agents=sub_agents, mongo=mongo, run_id=run_id)
        return RecordWorkflow(monitoring, memory, mongo_repository=mongo, run_id=run_id)

    return _make


@pytest.mark.asyncio
async def test_healthy_record_skips_monitoring(make_workflow, make_record, fake_llm, mongo, redis):
    workflow = make_workflow(mongo=mongo, run_id="run1")
    mongo.start_run(run_id="run1", config={}, summary={})

    state = await workflow.run(make_record(), seq=1)

    assert state["outcome"] == "skipped"
    assert "decision" not in state
    assert fake_llm.calls == []
    assert mongo.records.count_documents({}) == 0
    assert mongo.runs.find_one({"_id": "run1"})["summary"]["rows_without_violation"] == 1
    assert await redis.llen(slice_key("bs1", "eMBB", "records")) == 1


@pytest.mark.asyncio
async def test_violation_runs_near_rt_with_trusted_context(
    make_workflow, make_record, make_decision, make_invocation, make_sub_agent, fake_llm, mongo
):
    near, non = make_sub_agent(), make_sub_agent()
    # The LLM selects both agents and puts a wrong target in the context.
    fake_llm.queue(make_decision(selected_agents=[
        make_invocation("non_rt_ric"),
        make_invocation("near_rt_ric", context={"target": "bs999", "slice_id": "URLLC"}),
    ]))
    workflow = make_workflow({"near_rt_ric": near, "non_rt_ric": non}, mongo, "run1")
    mongo.start_run(run_id="run1", config={}, summary={})

    state = await workflow.run(make_record(**VIOLATION), seq=7)

    # First violated record: persistence 1, so the Non-RT RIC is dropped.
    assert state["persistence"]["throughput"] == 1
    assert [i.agent for i in state["decision"].selected_agents] == ["near_rt_ric"]
    assert non.calls == []
    assert set(state["agent_results"]) == {"near_rt_ric"}

    context = near.calls[0].context
    assert context["target"] == "bs1" and context["slice_id"] == "eMBB"
    assert context["record_id"] == "rec-1" and context["run_id"] == "run1"
    assert context["persistence"] == {"throughput": 1, "queueing_delay": 0}
    assert context["sla_analysis"]["violations"] == ["throughput"]
    assert context["slice_state"]["slice_prb_share_pct"] == 84.0
    assert context["slice_state"]["dl_cqi"] == 8.5 and context["slice_state"]["dl_mcs"] == 15.0
    assert context["active_controls"] == {}

    doc = mongo.records.find_one({"_id": "run1:rec-1"})
    assert doc["status"] == "done" and doc["seq"] == 7
    summary = mongo.runs.find_one({"_id": "run1"})["summary"]
    assert summary["rows_processed"] == 1
    assert summary["agent_invocations"] == {"near_rt_ric": 1}
    assert summary["ric_results"] == {"near_rt_ric": {"ok": 1}}


@pytest.mark.asyncio
async def test_persistent_violation_runs_near_rt_then_non_rt(
    make_workflow, make_record, make_decision, make_invocation, fake_llm
):
    order: list[str] = []

    def agent(name):
        async def _invoke(invocation):
            order.append(name)
            return {"status": "completed"}
        return _invoke

    workflow = make_workflow({"near_rt_ric": agent("near_rt_ric"),
                              "non_rt_ric": agent("non_rt_ric")})
    for i in range(3):
        # LLM order is Non-RT first; the graph runs Near-RT first.
        fake_llm.queue(make_decision(selected_agents=[make_invocation("non_rt_ric"),
                                                      make_invocation("near_rt_ric")]))
        state = await workflow.run(make_record(record_id=f"rec-{i}", window_start_s=60.0 + i,
                                               **VIOLATION))

    assert state["persistence"]["throughput"] == 3
    assert order == ["near_rt_ric", "near_rt_ric", "near_rt_ric", "non_rt_ric"]
    assert set(state["agent_results"]) == {"near_rt_ric", "non_rt_ric"}


@pytest.mark.asyncio
async def test_non_rt_sees_the_controls_near_rt_just_executed(
    make_workflow, make_record, make_decision, make_invocation, make_sub_agent, fake_llm, memory
):
    async def near(invocation):
        await memory.record_actions("bs1", [ActionEvent(
            rec="rec-2", t=62.0, slice="eMBB", agent="near_rt_ric", tool="adjust_prb_allocation",
            params={"prb_percentage": 80}, approved=True, executed=True)])
        return {"status": "completed"}

    non = make_sub_agent()
    workflow = make_workflow({"near_rt_ric": near, "non_rt_ric": non})
    for i in range(3):
        fake_llm.queue(make_decision(selected_agents=[make_invocation("near_rt_ric"),
                                                      make_invocation("non_rt_ric")]))
        await workflow.run(make_record(record_id=f"rec-{i}", window_start_s=60.0 + i, **VIOLATION))

    assert non.calls[0].context["active_controls"] == {"prb_percentage": 80}


@pytest.mark.asyncio
async def test_failed_agent_does_not_stop_the_record(
    make_workflow, make_record, make_decision, make_sub_agent, fake_llm, mongo
):
    fake_llm.queue(make_decision())
    workflow = make_workflow({"near_rt_ric": make_sub_agent(error=RuntimeError("boom"))},
                             mongo, "run1")
    mongo.start_run(run_id="run1", config={}, summary={})

    state = await workflow.run(make_record(**VIOLATION))

    assert state["outcome"] == "done"
    assert state["agent_results"] == {"near_rt_ric": None}
    doc = mongo.records.find_one({"_id": "run1:rec-1"})
    assert doc["agents"]["near_rt_ric"]["error"]["message"] == "boom"
    summary = mongo.runs.find_one({"_id": "run1"})["summary"]
    assert summary["ric_results"] == {"near_rt_ric": {"error": 1}}


def test_graph_diagram(make_workflow):
    diagram = make_workflow().draw_mermaid()
    for node in ("ingest", "skip_healthy", "monitoring", "near_rt_ric", "non_rt_ric", "finalize"):
        assert node in diagram
