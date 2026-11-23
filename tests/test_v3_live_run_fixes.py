"""Regression tests for the issues found in the live-LLM runs (results/run-6..8).

1. The slice policy has a safety envelope; a tool without one is never
   executed (default deny).
2. Governance argument models mirror the tool signatures: QoS and link
   adaptation take typed parameters with envelopes (v3.3).
3. A duplicate agent selection must not overwrite the first invocation's audit.
4. One executed change per resource per invocation, and a bounded number of calls.
5. An argument the tool does not declare is rejected, not dropped (v3.3).
"""

import json

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tracers.context import collect_runs

from agents.monitoring_agent import MonitoringAgent
from agents.ran_trust_governance import RANTrustGovernanceAgent
from middleware.ran_authorisation import (
    MAX_PROTECTED_CALLS,
    RAN_AUDIT,
    AuditSink,
    RANAuthorizationMiddleware,
)
from src.schema.monitoring import AgentInvocation
from src.schema.near_rt_ric import GovernanceResult, RANActionProposal

CTX = {
    "target": "bs2",
    "slice_id": "eMBB",
    "sla_analysis": {"violations": ["throughput"]},
    "persistence": {"throughput": 5},
}


def _proposal(tool, params, agent):
    return RANActionProposal(
        tool_name=tool,
        target="bs2",
        parameters=params,
        rationale="throughput 3.9 Mbps < 7.2 Mbps budget",
        expected_effect="throughput increases",
        confidence=0.5,
        invocation_context={"invocation": {"agent": agent, "context": CTX}},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("lo,hi,ok", [(900, -4, False), (100, 100, False), (20, 60, True)])
async def test_slice_policy_has_safety_envelope(lo, hi, ok):
    gov = RANTrustGovernanceAgent(use_llm=False)
    d = await gov.authorize(
        _proposal(
            "update_slice_resource_policy",
            {"slice_id": "eMBB", "min_prb_ratio_pct": lo, "max_prb_ratio_pct": hi},
            "non_rt_ric",
        )
    )
    assert d.approved is ok
    if not ok:
        assert d.failed_stage == "resource_safety"


QOS = {"slice_id": "eMBB", "priority_level": 2}
LINK = {"slice_id": "eMBB", "mcs_offset": 2, "target_bler": 0.1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool,params,ok,field",
    [
        ("adjust_qos_parameters", QOS, True, None),
        ("adjust_qos_parameters", {**QOS, "priority_level": 0}, False, "priority_level"),
        ("adjust_link_adaptation", LINK, True, None),
        ("adjust_link_adaptation", {**LINK, "mcs_offset": 9}, False, "mcs_offset"),
        ("adjust_link_adaptation", {**LINK, "target_bler": 0.5}, False, "target_bler"),
        ("optimize_scheduler", {"scheduler": "waterfilling"}, True, None),
    ],
)
async def test_typed_tools_have_envelopes(tool, params, ok, field):
    gov = RANTrustGovernanceAgent(use_llm=False)
    d = await gov.authorize(_proposal(tool, params, "near_rt_ric"))
    assert d.approved is ok
    if not ok:
        assert d.failed_stage == "resource_safety" and field in d.reason


@pytest.mark.asyncio
async def test_unknown_scheduler_is_rejected_at_schema():
    """The scheduler argument is typed: only the testbed's schedulers pass."""
    gov = RANTrustGovernanceAgent(use_llm=False)
    d = await gov.authorize(
        _proposal("optimize_scheduler", {"scheduler": "max_cqi"}, "near_rt_ric")
    )
    assert d.failed_stage == "schema" and "scheduler" in d.reason


@pytest.mark.asyncio
async def test_removed_arguments_are_rejected_at_schema():
    """Arguments the tools no longer take are refused, not silently dropped."""
    gov = RANTrustGovernanceAgent(use_llm=False)
    for tool, params in (
        ("optimize_scheduler", {"slice_id": "eMBB", "scheduler": "waterfilling"}),
        ("adjust_qos_parameters", {**QOS, "packet_delay_budget_ms": 100}),
    ):
        d = await gov.authorize(_proposal(tool, params, "near_rt_ric"))
        assert d.failed_stage == "schema" and "Extra inputs" in d.reason, tool


@pytest.mark.asyncio
async def test_old_free_form_arguments_are_rejected_at_schema():
    gov = RANTrustGovernanceAgent(use_llm=False)
    d = await gov.authorize(
        _proposal(
            "adjust_qos_parameters",
            {"slice_id": "eMBB", "qos_parameters": {"priority": 1}},
            "near_rt_ric",
        )
    )
    assert d.approved is False and d.failed_stage == "schema"
    assert "qos_parameters" in d.reason


def test_duplicate_agent_selection_is_merged():
    sel = [
        AgentInvocation(
            agent="near_rt_ric",
            responsibility="latency",
            objective="latency < 100 ms",
            priority="medium",
            reason="a",
            context={},
        ),
        AgentInvocation(
            agent="near_rt_ric",
            responsibility="jitter",
            objective="jitter < 50 ms",
            priority="high",
            reason="b",
            context={},
        ),
        AgentInvocation(
            agent="non_rt_ric",
            responsibility="throughput",
            objective="throughput",
            priority="low",
            reason="c",
            context={},
        ),
    ]
    out = MonitoringAgent._merge_duplicate_agents(sel)
    assert [i.agent for i in out] == ["near_rt_ric", "non_rt_ric"]
    assert "latency" in out[0].objective and "jitter" in out[0].objective
    assert out[0].priority == "high"


class _Req:
    def __init__(self, name, args, call_id, agent="non_rt_ric"):
        self.tool_call = {"name": name, "args": args, "id": call_id}
        self.tool = None
        self.state = {"agent_invocation": {"agent": agent, "context": CTX}}


@pytest.mark.asyncio
async def test_one_execution_per_resource_and_call_budget():
    mw = RANAuthorizationMiddleware(RANTrustGovernanceAgent(use_llm=False))
    executed = []

    async def handler(req):
        executed.append(req.tool_call["args"])
        return ToolMessage(
            content=json.dumps({"status": "success"}), tool_call_id=req.tool_call["id"]
        )

    reasoning = {
        "rationale": "throughput below budget",
        "expected_effect": "throughput up",
        "confidence": 0.6,
    }
    audit = AuditSink()
    token = RAN_AUDIT.set(audit)
    try:
        for i, (lo, hi) in enumerate([(5, 80), (60, 80), (60, 80)]):
            args = {
                "target": "bs2",
                "slice_id": "eMBB",
                "min_prb_ratio_pct": lo,
                "max_prb_ratio_pct": hi,
                **reasoning,
            }
            await mw.awrap_tool_call(_Req("update_slice_resource_policy", args, f"c{i}"), handler)
        assert len(executed) == 1  # the run-7 sequence executed 3 times
        stages = [a["governance"]["failed_stage"] for a in audit.entries]
        assert stages == [None, "invocation_limit", "invocation_limit"]
        assert "prb_policy" in audit.entries[1]["governance"]["reason"]
        for i in range(MAX_PROTECTED_CALLS + 2):
            await mw.awrap_tool_call(
                _Req(
                    "optimize_scheduler",
                    {"target": "bs2", "slice_id": "eMBB", "scheduler": "bogus", **reasoning},
                    f"s{i}",
                ),
                handler,
            )
        assert len(audit) == 3 + MAX_PROTECTED_CALLS + 2
        assert audit.entries[-1]["governance"]["failed_stage"] == "invocation_limit"
        assert "protected calls" in audit.entries[-1]["governance"]["reason"]
    finally:
        RAN_AUDIT.reset(token)


@pytest.mark.asyncio
async def test_undeclared_argument_is_rejected_not_dropped():
    mw = RANAuthorizationMiddleware(RANTrustGovernanceAgent(use_llm=False))
    executed = []

    async def handler(req):
        executed.append(req.tool_call["args"])
        return ToolMessage(content="{}", tool_call_id=req.tool_call["id"])

    audit = AuditSink()
    token = RAN_AUDIT.set(audit)
    try:
        args = {
            "target": "bs2",
            "slice_id": "eMBB",
            "prb_percentage": 60,
            "boost": True,
            "rationale": "throughput below budget",
            "expected_effect": "up",
            "confidence": 0.6,
        }
        reply = await mw.awrap_tool_call(
            _Req("adjust_prb_allocation", args, "x1", "near_rt_ric"), handler
        )
    finally:
        RAN_AUDIT.reset(token)

    assert executed == []
    governance = audit.entries[0]["governance"]
    assert governance["failed_stage"] == "schema" and "boost" in governance["reason"]
    assert audit.actions[0].params["boost"] is True  # the trail shows what was sent
    assert "boost" in json.loads(reply.content)["reason"]


class _ReviewRejectsFirst:
    """Governance double: the review rejects the first call, approves the rest."""

    def __init__(self):
        self.calls = 0

    async def authorize(self, proposal):
        self.calls += 1
        approved = self.calls > 1
        return GovernanceResult(
            approved=approved, reason="review", failed_stage=None if approved else "llm_review"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("second,executed", [(60, 0), (70, 1)])
async def test_rewording_cannot_overturn_a_review_rejection(second, executed):
    gov = _ReviewRejectsFirst()
    mw = RANAuthorizationMiddleware(gov)
    ran = []

    async def handler(req):
        ran.append(req.tool_call["id"])
        return ToolMessage(content='{"status": "success"}', tool_call_id=req.tool_call["id"])

    base = {"target": "bs2", "slice_id": "eMBB", "expected_effect": "up", "confidence": 0.6}
    audit = AuditSink()
    token = RAN_AUDIT.set(audit)
    try:
        await mw.awrap_tool_call(
            _Req(
                "adjust_prb_allocation",
                {**base, "prb_percentage": 60, "rationale": "first wording"},
                "c1",
                "near_rt_ric",
            ),
            handler,
        )
        await mw.awrap_tool_call(
            _Req(
                "adjust_prb_allocation",
                {**base, "prb_percentage": second, "rationale": "new wording"},
                "c2",
                "near_rt_ric",
            ),
            handler,
        )
    finally:
        RAN_AUDIT.reset(token)

    assert len(ran) == executed
    if not executed:
        assert gov.calls == 1  # the review was not asked again
        assert audit.entries[1]["governance"]["failed_stage"] == "invocation_limit"
        assert "rewording" in audit.entries[1]["governance"]["reason"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool,params,controls,ok",
    [
        (
            "adjust_prb_allocation",
            {"slice_id": "eMBB", "prb_percentage": 80},
            {"policy_min": 50, "policy_max": 70},
            False,
        ),
        (
            "adjust_prb_allocation",
            {"slice_id": "eMBB", "prb_percentage": 60},
            {"policy_min": 50, "policy_max": 70},
            True,
        ),
        (
            "update_slice_resource_policy",
            {"slice_id": "eMBB", "min_prb_ratio_pct": 50, "max_prb_ratio_pct": 70},
            {"prb_percentage": 80},
            False,
        ),
        (
            "update_slice_resource_policy",
            {"slice_id": "eMBB", "min_prb_ratio_pct": 50, "max_prb_ratio_pct": 85},
            {"prb_percentage": 80},
            True,
        ),
    ],
)
async def test_prb_share_and_slice_policy_must_agree(tool, params, controls, ok):
    agent = "non_rt_ric" if tool == "update_slice_resource_policy" else "near_rt_ric"
    proposal = _proposal(tool, params, agent)
    proposal.invocation_context["invocation"]["context"] = {**CTX, "active_controls": controls}

    decision = await RANTrustGovernanceAgent(use_llm=False).authorize(proposal)

    assert decision.approved is ok
    if not ok:
        assert decision.failed_stage == "resource_safety"


@pytest.mark.asyncio
async def test_rejected_call_leaves_a_governance_span():
    """A rejected tool never runs: its span is the only trace of the call."""
    mw = RANAuthorizationMiddleware(RANTrustGovernanceAgent(use_llm=False))

    async def handler(req):
        raise AssertionError("a rejected tool must not run")

    token = RAN_AUDIT.set(AuditSink())
    try:
        args = {
            "target": "bs2",
            "slice_id": "eMBB",
            "prb_percentage": 60,
            "boost": True,
            "rationale": "throughput below budget",
            "expected_effect": "up",
            "confidence": 0.6,
        }
        with collect_runs() as cb:
            await mw.awrap_tool_call(
                _Req("adjust_prb_allocation", args, "x1", "near_rt_ric"), handler
            )
    finally:
        RAN_AUDIT.reset(token)

    span = next(r for r in cb.traced_runs if r.name == "governance adjust_prb_allocation")
    assert "governance" in span.tags
    assert "rejected" in span.outputs["output"].content


@pytest.mark.asyncio
async def test_violation_is_never_reported_as_normal(make_monitoring_agent, fake_llm, make_record):
    """run-24: the LLM returned situation='normal' (severity high) for a violating record."""
    from src.schema.monitoring import MonitoringDecision

    record = make_record(record_id="rec-v", queueing_delay_ms=400.0)
    analysis = MonitoringAgent.analyze_sla(record)
    assert analysis.violations
    fake_llm.queue(
        MonitoringDecision(
            situation="normal",
            severity="none",
            global_situation="violations_detected",
            action_required=False,
            sla_analysis=analysis,
            selected_agents=[],
            reason="r",
            record_id="rec-v",
        )
    )
    result = await make_monitoring_agent(sub_agents={}).run(record)
    assert result.situation == "warning"
    assert result.severity != "none"


class _ModelRequest:
    """Just what awrap_model_call reads from a ModelRequest."""

    def __init__(self, messages, tools, tool_choice=None):
        self.messages, self.tools, self.tool_choice = messages, tools, tool_choice

    def override(self, **changes):
        return _ModelRequest(
            changes.get("messages", self.messages),
            changes.get("tools", self.tools),
            changes.get("tool_choice", self.tool_choice),
        )


def _tool(name):
    from types import SimpleNamespace

    return SimpleNamespace(name=name)


@pytest.mark.asyncio
@pytest.mark.parametrize("answers_with_action", [True, False])
async def test_action_is_forced_with_one_call_offering_only_action_tools(
    monkeypatch, answers_with_action
):
    from langchain.agents.middleware import ModelResponse
    from langchain_core.messages import AIMessage

    from middleware.ran_authorisation import FORCED_FLAG

    mw = RANAuthorizationMiddleware(RANTrustGovernanceAgent(use_llm=False))
    searched = AIMessage(
        content="",
        tool_calls=[{"name": "search_ran_knowledge", "args": {"query": "q"}, "id": "s1"}],
    )
    request = _ModelRequest(
        [searched], [_tool("search_ran_knowledge"), _tool("adjust_prb_allocation")]
    )
    seen = []

    async def handler(req):
        seen.append(req)
        calls = (
            [{"name": "adjust_prb_allocation", "args": {}, "id": "a1"}]
            if answers_with_action
            else []
        )
        return ModelResponse(result=[AIMessage(content="", tool_calls=calls)])

    async def two_call_fallback(req):
        return ModelResponse(result=[AIMessage(content="fallback")])

    monkeypatch.setattr(mw, "_force_action", two_call_fallback)
    response = await mw.awrap_model_call(request, handler)

    assert len(seen) == 1  # one model call
    assert [t.name for t in seen[0].tools] == ["adjust_prb_allocation"]
    assert seen[0].tool_choice == "any"
    message = response.result[0]
    if answers_with_action:
        assert message.additional_kwargs[FORCED_FLAG] is True
    else:
        assert message.content == "fallback"  # models that ignore tools
