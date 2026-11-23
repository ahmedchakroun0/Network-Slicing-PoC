"""scripts/agent_checks.py: the automatic checks of interpretations and decisions."""

import pytest

from scripts.agent_checks import (
    check_action,
    check_record,
    diagnosis,
    evidence_check,
    expected_agents,
    number_check,
    traceable,
)

TELEMETRY = {
    "record_id": "r1",
    "cell_id": "bs2",
    "slice_id": "eMBB",
    "run": "sched0/tr2/exp1",
    "window_start_s": 51.0,
    "window_s": 1.0,
    "active_ues": 2,
    "slice_prb": 18,
    "cell_prb": 50,
    "scheduler": "round_robin",
    "dl_throughput_mbps": 5.42,
    "dl_throughput_per_ue_mbps": 2.712318,
    "dl_buffer_bytes": 9000.0,
    "queueing_delay_ms": 507.088633,
    "prb_utilisation": 0.999667,
    "cell_spare_prb": 29.914,
    "dl_cqi": 9.1,
    "dl_mcs": 16.0,
    "ul_sinr_db": None,
}
SLA = {
    "violations": ["throughput", "queueing_delay"],
    "kpis": {
        "throughput": {"measured": 2.712318, "budget": 3.6, "status": "violation"},
        "queueing_delay": {"measured": 507.088633, "budget": 100.0, "status": "violation"},
    },
}


def _doc(**monitoring):
    return {
        "run_id": "run1",
        "seq": 4,
        "telemetry": TELEMETRY,
        "sla": SLA,
        "persistence": {"throughput": 1, "queueing_delay": 1},
        "monitoring": {
            "situation": "warning",
            "severity": "high",
            "action_required": True,
            "reason": "throughput 2.71 Mbps vs 3.6 Mbps",
            "selected_agents": [
                {"agent": "near_rt_ric", "objective": "fix throughput and queueing_delay"}
            ],
            **monitoring,
        },
        "agents": {},
    }


def _action(tool, params, rationale="throughput 2.71 Mbps vs 3.6", evidence=None):
    return {
        "tool": tool,
        "params": params,
        "reasoning": {
            "rationale": rationale,
            "expected_effect": "up",
            "evidence_sources": evidence,
        },
    }


def test_quoted_numbers_allow_rounding_and_simple_derivations():
    context = [2.712318, 3.6, 507.088633, 100.0, 100 * 2.712318 / 3.6, 507.088633 / 100]
    assert traceable("2.71", context) and traceable("75.3", context) and traceable("5.07", context)
    assert not traceable("4.2", context)
    assert traceable("0.999", [0.999667])  # truncated
    assert not traceable("27.6", [27.5, 28.14])  # 2.587 vs 3.6 is a 28.1% deficit
    result = number_check(
        ["2.71 Mbps (75.3% of budget), 5.07x over; A1 and 6G ignored, 42 not"], context
    )
    assert result["quoted"] == 4 and result["untraced_values"] == ["42"]


def test_diagnosis_and_routing_rule():
    assert diagnosis({"prb_utilisation": 0.95, "dl_cqi": 4}) == "prb_limited"
    assert diagnosis({"prb_utilisation": 0.5, "dl_cqi": 4}) == "radio_limited"
    assert diagnosis({"prb_utilisation": 0.5, "dl_cqi": 9}) == "unclear"
    assert expected_agents(["throughput"], {"throughput": 2}) == {"near_rt_ric"}
    assert expected_agents(["throughput"], {"throughput": 3}) == {"near_rt_ric", "non_rt_ric"}


def test_monitoring_checks_use_the_llm_decision_before_the_override():
    doc = _doc(
        llm_output={
            "situation": "normal",
            "severity": "low",
            "action_required": True,
            "selected_agents": [{"agent": "near_rt_ric"}, {"agent": "non_rt_ric"}],
        }
    )
    checks = check_record(doc, [])
    assert checks["sla_recomputed_agrees"] and checks["llm_flags_violation"]
    assert not checks["llm_situation_ok"]  # "normal" with two violations
    assert not checks["llm_severity_agrees"]  # low vs computed high
    assert not checks["llm_routing_agrees"]  # Non-RT at persistence 1
    assert checks["objective_names_kpis"] and checks["monitoring_numbers_untraced"] == 0


@pytest.mark.parametrize(
    "tool,params,cause,direction,feasible,no_op",
    [
        ("adjust_prb_allocation", {"prb_percentage": 50}, "consistent", True, True, False),
        ("adjust_prb_allocation", {"prb_percentage": 30}, "consistent", False, True, False),
        ("adjust_prb_allocation", {"prb_percentage": 98}, "consistent", True, False, False),
        ("adjust_link_adaptation", {"mcs_offset": 1}, "inconsistent", None, None, None),
        ("optimize_scheduler", {"scheduler": "round_robin"}, "neutral", None, None, True),
        ("optimize_handover", {"handover_margin_db": 2}, "inconsistent", None, None, None),
    ],
)
def test_tool_choice_against_the_diagnosis(tool, params, cause, direction, feasible, no_op):
    checks = check_action(_action(tool, params), "near_rt_ric", _doc(), [])
    assert checks["diagnosis"] == "prb_limited"
    assert (checks["tool_cause"], checks["direction_ok"], checks["feasible"], checks["no_op"]) == (
        cause,
        direction,
        feasible,
        no_op,
    )


def test_evidence_needs_retrieved_knowledge_or_traceable_context():
    knowledge = [
        "Pattern B: Throughput shortfall. Possible response: inspect slice allocation, "
        "load, radio conditions and other bottleneck evidence."
    ]
    result = evidence_check(
        [
            "Pattern B: Throughput shortfall - inspect slice allocation, load, radio conditions",
            "slice_state: prb_utilisation 0.999667, slice_prb 18 (invocation context)",
            "slice_state: prb_utilisation 0.42 (invocation context)",
            "waterfilling allocates resources based on channel quality",
        ],
        knowledge,
        [0.999667, 18.0],
    )
    assert result["evidence_sources"] == 4 and result["evidence_supported"] == 2
    assert (
        "0.42" in result["evidence_unsupported"]
        and "waterfilling" in result["evidence_unsupported"]
    )
    assert evidence_check(["Pattern B: Throughput shortfall"], [], [])["evidence_supported"] == 0
    # what a tool's description says counts as given knowledge
    assert (
        evidence_check(["adjust_link_adaptation: mcs_offset range"], [], [5.0])[
            "evidence_supported"
        ]
        == 1
    )
