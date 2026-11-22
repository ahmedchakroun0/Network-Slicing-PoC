"""Closed loop: deterministic verification of executed actions, the controls
the RAN actually applied, and the guarantees around them."""

import json

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException

from agents.monitoring_agent import MonitoringAgent
from middleware.ran_authorisation import RAN_AUDIT, AuditSink, RANAuthorizationMiddleware
from ran_emulator.engine import Emulator
from src.memory.verification import verify
from src.schema.near_rt_ric import GovernanceResult
from src.schema.working_memory import ActionEvent, KPIAtAction, PendingCheck

# eMBB budgets: throughput >= 3.6 Mbps/UE, queueing delay <= 100 ms.
CHECK = PendingCheck(
    rec="r0", t=0.0, slice="eMBB", agent="near_rt_ric", i=0, tool="adjust_prb_allocation",
    due_step=1, kpis={"throughput": KPIAtAction(measured=1.8, budget=3.6, bound="min"),
                      "queueing_delay": KPIAtAction(measured=300.0, budget=100.0, bound="max")})


@pytest.mark.parametrize("throughput, delay, expected", [
    (4.0, 50.0, "resolved"),     # both back within budget
    (2.7, 200.0, "improved"),    # both closer to budget
    (1.8, 300.0, "unchanged"),
    (1.81, 301.0, "unchanged"),  # within the noise tolerance
    (1.8, 400.0, "worse"),       # delay further from budget
    (3.0, 350.0, "worse"),       # one better, one worse: not an improvement
])
def test_outcome_classes(make_record, throughput, delay, expected):
    record = make_record(dl_throughput_per_ue_mbps=throughput, queueing_delay_ms=delay,
                         record_id="r1")
    outcome = verify(CHECK, record, MonitoringAgent.analyze_sla(record))
    assert outcome.outcome == expected
    assert outcome.checked_rec == "r1" and outcome.i == 0
    assert outcome.kpis["queueing_delay"] == {"before": 300.0, "after": delay, "budget": 100.0}


def test_unmeasurable_kpi_is_unknown(make_record):
    from src.schema.monitoring import KPIStatus

    record = make_record(queueing_delay_ms=50.0)
    analysis = MonitoringAgent.analyze_sla(record)
    analysis.kpis["throughput"] = KPIStatus(status="unavailable")
    assert verify(CHECK, record, analysis).outcome == "unknown"


def test_zero_throughput_recovering_counts_as_improved(make_record):
    check = CHECK.model_copy(update={"kpis": {
        "throughput": KPIAtAction(measured=0.0, budget=3.6, bound="min")}})
    record = make_record(dl_throughput_per_ue_mbps=1.0)
    assert verify(check, record, MonitoringAgent.analyze_sla(record)).outcome == "improved"


async def test_pending_checks_are_due_at_their_step(memory):
    await memory.add_pending("bs2", [CHECK, CHECK.model_copy(update={"due_step": 3, "i": 1})])
    assert await memory.pop_due("bs2", "eMBB", 0) == []
    assert [c.i for c in await memory.pop_due("bs2", "eMBB", 1)] == [0]
    assert await memory.pop_due("bs2", "eMBB", 2) == []
    assert [c.i for c in await memory.pop_due("bs2", "eMBB", 5)] == [1]
    assert await memory.pop_due("bs2", "eMBB", 9) == []


def _event(tool, params, applied=None, executed=True, slice_id="eMBB"):
    return ActionEvent(rec="r0", t=0.0, slice=slice_id, agent="near_rt_ric", tool=tool,
                       params=params, approved=True, executed=executed, applied=applied)


async def test_controls_in_force_are_the_applied_split_of_every_slice(memory):
    """The RAN snapped 70 % to 36 PRBs (72 %), and MTC gave up PRBs."""
    await memory.record_actions("bs2", [_event(
        "adjust_prb_allocation", {"slice_id": "eMBB", "prb_percentage": 70},
        applied={"current": {"eMBB": 36, "MTC": 3, "URLLC": 11}, "applied_pct": 72.0})])
    assert await memory.controls("bs2", "eMBB") == {"prb_percentage": 72.0}
    assert await memory.controls("bs2", "MTC") == {"prb_percentage": 6.0}


async def test_policy_records_the_policy_and_the_split_it_moved_to(memory):
    await memory.record_actions("bs2", [_event(
        "update_slice_resource_policy",
        {"slice_id": "eMBB", "min_prb_ratio_pct": 60, "max_prb_ratio_pct": 90},
        applied={"current": {"min_prb_ratio_pct": 60.0, "max_prb_ratio_pct": 90.0},
                 "prb": {"eMBB": 36, "MTC": 3, "URLLC": 11}})])
    assert await memory.controls("bs2", "eMBB") == {
        "prb_percentage": 72.0, "policy_min": 60.0, "policy_max": 90.0}


async def test_failed_or_unreported_calls(memory):
    await memory.record_actions("bs2", [
        _event("adjust_prb_allocation", {"prb_percentage": 50}, executed=False),
        _event("adjust_prb_allocation", {"prb_percentage": 40}, slice_id="URLLC"),
    ])
    assert await memory.controls("bs2", "eMBB") == {}
    # No RAN report: the requested value is the best knowledge.
    assert await memory.controls("bs2", "URLLC") == {"prb_percentage": 40.0}


def test_emulator_run_is_stable_across_a_trace_switch(bank, make_scenario):
    """Working memory clears a cell whose `run` changes: a control must not."""
    emu = Emulator(bank, make_scenario(tr=9))
    before = emu.step().records[0].run
    emu.set_prb("bs2", "eMBB", 72)
    after = emu.step()
    assert after.records[0].run == before
    assert after.meta["bs2"]["source_run"] != "sched0/tr9/exp1"


async def test_observe_keeps_memory_across_a_control(bank, make_scenario, memory):
    emu = Emulator(bank, make_scenario(tr=9))
    await memory.observe(emu.step().records[0])
    await memory.record_actions("bs2", [_event("adjust_power_control", {"power_offset_db": 2})])
    emu.set_prb("bs2", "eMBB", 72)
    await memory.observe(emu.step().records[0])
    assert len((await memory.view("bs2", "eMBB")).slice_actions) == 1


# ------------------------------------------------- middleware: failed tools


class _Approve:
    async def authorize(self, proposal, state=None):
        return GovernanceResult(approved=True, reason="ok", checks_passed=["schema"])


class _Request:
    def __init__(self):
        self.tool_call = {"name": "adjust_power_control", "id": "c1", "args": {
            "target": "bs2", "power_offset_db": 20.0, "rationale": "r",
            "expected_effect": "e", "confidence": 0.5}}
        self.state = {"messages": [], "agent_invocation": {"agent": "near_rt_ric"}}


@pytest.mark.parametrize("failure", ["message", "exception"])
async def test_a_refused_control_is_audited_as_error_not_executed(failure):
    async def handler(_request):
        if failure == "exception":
            raise ToolException("the RAN refused the control (422): out of range")
        return ToolMessage(content="the RAN refused the control (422): out of range",
                           tool_call_id="c1", status="error")

    audit = AuditSink()
    token = RAN_AUDIT.set(audit)
    try:
        response = await RANAuthorizationMiddleware(_Approve())._gate(_Request(), handler)
    finally:
        RAN_AUDIT.reset(token)

    (entry,) = audit.entries
    assert entry["governance"]["approved"] is True
    assert entry["execution"]["status"] == "error"
    assert "refused" in entry["execution"]["message"]
    if failure == "exception":
        assert json.loads(response.content)["status"] == "error"
