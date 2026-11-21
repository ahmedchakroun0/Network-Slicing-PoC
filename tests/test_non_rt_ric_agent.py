"""BaseRICAgent.invoke with a fake LangChain agent that plays the middleware:
it records protected calls in the real AuditSink, so the results store
(mongomock) and the working memory (fakeredis) are exercised for real."""

import pytest

from agents.non_rt_ric_agent import NonRTRICAgent
from middleware.ran_authorisation import RAN_AUDIT
from src.schema.monitoring import AgentInvocation
from src.schema.near_rt_ric import GovernanceResult, LLMReview, RANActionProposal, RANToolResult
from src.schema.working_memory import ActionEvent

CONTEXT = {"target": "bs1", "slice_id": "eMBB", "record_id": "r1", "run_id": "run1",
           "window_start_s": 60.0}
POLICY = {"slice_id": "eMBB", "min_prb_ratio_pct": 40, "max_prb_ratio_pct": 80}
REASONING = {"rationale": "throughput 1.8 Mbps < 3.6 Mbps for 4 records; guarantee 40 % PRB.",
             "expected_effect": "throughput per UE rises towards 3.6 Mbps.", "confidence": 0.8}


def _call(call_id, approved, stage=None, reason="Approved.", review=None):
    """What the middleware records for one protected call."""
    proposal = RANActionProposal(tool_name="update_slice_resource_policy", parameters=POLICY,
                                 target="bs1", **REASONING)
    decision = GovernanceResult(approved=approved, reason=reason, failed_stage=stage,
                                llm_review=review)
    execution = RANToolResult(tool_name="update_slice_resource_policy",
                              status="executed" if approved else "rejected", target="bs1",
                              parameters={**POLICY, "target": "bs1"}, result={"status": "success"})
    submitted = {"tool_name": "update_slice_resource_policy", "target": "bs1",
                 "parameters": POLICY, **REASONING}
    return call_id, submitted, proposal, decision, execution


class FakeAgent:
    """Records the given calls in the sink, then ends with a closing message."""

    def __init__(self, calls):
        self.calls = calls

    async def ainvoke(self, payload):
        self.payload = payload
        sink = RAN_AUDIT.get()
        for call in self.calls:
            sink.add(*call)
        return {"messages": []}


def _invocation():
    return AgentInvocation(agent="non_rt_ric", responsibility="PRB policy",
                           objective="throughput 1.8 < 3.6 Mbps", priority="high",
                           reason="persistent throughput violation", context=CONTEXT)


@pytest.mark.asyncio
async def test_completed_invocation_is_stored_and_remembered(memory, mongo):
    mongo.insert_record(run_id="run1", record_id="r1", seq=1, telemetry={}, sla={}, persistence={})
    review = LLMReview(goal_aligned=True, reasoning_sound=False, behavior_acceptable=True,
                       reason="Rationale cites a value not in the context.")
    agent = NonRTRICAgent(trust_governance_agent=object(), memory=memory, mongo_repository=mongo)
    agent.agent = FakeAgent([
        _call("c0", False, "llm_review", "rejected by review", review),
        _call("c1", True),
    ])

    result = await agent.invoke(_invocation())

    assert result.status == "completed"
    assert result.execution.status == "executed"
    stored = mongo.records.find_one({"_id": "run1:r1"})["agents"]["non_rt_ric"]
    assert stored["status"] == "completed" and stored["selected_action"] == 1
    first = stored["actions"][0]
    assert first["reasoning"]["rationale"] == REASONING["rationale"]
    assert first["governance"]["llm_review"]["reason"] == review.reason
    assert first["params"] == POLICY                       # target kept apart, not duplicated

    actions = (await memory.view("bs1", "eMBB")).slice_actions
    assert [(a.approved, a.stage) for a in actions] == [(True, None), (False, "llm_review")]
    assert actions[1].review == {"goal_aligned": True, "reasoning_sound": False,
                                 "behavior_acceptable": True}
    assert actions[1].why is None                           # no LLM text in working memory


@pytest.mark.asyncio
async def test_rejected_invocation_and_memory_view_in_prompt_state(memory, mongo, make_record):
    mongo.insert_record(run_id="run1", record_id="r1", seq=1, telemetry={}, sla={}, persistence={})
    await memory.observe(make_record())
    await memory.record_actions("bs1", [ActionEvent(
        rec="r0", t=59.0, slice="eMBB", agent="near_rt_ric", tool="adjust_prb_allocation",
        params={"prb_percentage": 60}, approved=True, executed=True)])
    agent = NonRTRICAgent(trust_governance_agent=object(), memory=memory, mongo_repository=mongo)
    fake = FakeAgent([_call("c0", False, "sla", "[sla] Policy actions require a violation "
                                                 "persisting for at least 3 consecutive records")])
    agent.agent = fake

    result = await agent.invoke(_invocation())

    assert result.status == "rejected"
    assert result.governance.failed_stage == "sla"
    # compact view: no empty lists, no record ids, no empty fields
    view = fake.payload["working_memory"]
    assert set(view) == {"cell", "slice_actions"}
    assert view["slice_actions"] == [{"t": 59.0, "agent": "near_rt_ric", "tool": "adjust_prb_allocation",
                                      "params": {"prb_percentage": 60}, "approved": True,
                                      "executed": True}]
    stored = mongo.records.find_one({"_id": "run1:r1"})["agents"]["non_rt_ric"]
    assert stored["status"] == "rejected"
    why = (await memory.view("bs1", "eMBB")).slice_actions[0].why
    assert why.startswith("[sla] Policy actions require")   # deterministic reason is kept


@pytest.mark.asyncio
async def test_failure_is_recorded_as_error(memory, mongo):
    mongo.insert_record(run_id="run1", record_id="r1", seq=1, telemetry={}, sla={}, persistence={})

    class Broken:
        async def ainvoke(self, payload):
            raise RuntimeError("model endpoint down")

    agent = NonRTRICAgent(trust_governance_agent=object(), memory=memory, mongo_repository=mongo)
    agent.agent = Broken()

    result = await agent.invoke(_invocation())

    assert result.status == "error"
    stored = mongo.records.find_one({"_id": "run1:r1"})["agents"]["non_rt_ric"]
    assert stored["status"] == "error" and stored["error"]["type"] == "RuntimeError"
