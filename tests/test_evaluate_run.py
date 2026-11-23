"""scripts/evaluate_run.py: metrics computed from stored records."""

import json

import pytest

from scripts.evaluate_run import (
    action_rows,
    compute_metrics,
    latex_table,
    prb_policy_conflicts,
    wilson,
    write_outputs,
)


def _action(tool, params, approved=True, stage=None, target="bs2", at="t1", forced=False):
    return {
        "tool": tool,
        "target": target,
        "at": at,
        "params": params,
        "reasoning": {"rationale": "why", "expected_effect": "effect", "confidence": 0.7},
        "agent_message": {"forced": forced},
        "governance": {
            "approved": approved,
            "failed_stage": stage,
            "reason": "because",
            "llm_review": {"reasoning_sound": approved},
        },
        "execution": {"status": "executed" if approved else "rejected"},
    }


def _record(seq, agents, persistence=3):
    return {
        "run_id": "run1",
        "seq": seq,
        "status": "done",
        "telemetry": {
            "record_id": f"r{seq}",
            "cell_id": "bs2",
            "slice_id": "eMBB",
            "run": "sched0/tr2/exp1",
            "window_start_s": 50.0 + seq,
            "window_s": 1.0,
            "active_ues": 2,
            "slice_prb": 18,
            "cell_prb": 50,
            "scheduler": "round_robin",
            "dl_throughput_mbps": 5.0,
            "dl_throughput_per_ue_mbps": 2.5,
            "dl_buffer_bytes": 1000.0,
            "queueing_delay_ms": 50.0,
            "prb_utilisation": 1.0,
            "cell_spare_prb": 30.0,
            "dl_cqi": 9.0,
            "dl_mcs": 15.0,
            "ul_sinr_db": None,
        },
        "sla": {"violations": ["throughput"]},
        "persistence": {"throughput": persistence, "queueing_delay": 0},
        "monitoring": {
            "situation": "critical",
            "overridden": False,
            "orchestration_notes": [],
            "selected_agents": [{"agent": a} for a in agents],
        },
        "agents": {a: {"status": "completed", "actions": actions} for a, actions in agents.items()},
    }


PRB = {"slice_id": "eMBB", "prb_percentage": 80}
SCHEDULER = {"slice_id": "eMBB", "scheduler": "proportional_fair"}
POLICY = {"slice_id": "eMBB", "min_prb_ratio_pct": 50, "max_prb_ratio_pct": 70}


def _metrics(records):
    run = {"_id": "run1", "config": {"model": "m"}, "summary": {"rows_selected": 10}}
    return compute_metrics([run], records, action_rows(records))


def test_wilson_interval():
    assert wilson(0, 0) is None
    low, high = wilson(8, 10)
    assert low == pytest.approx(0.490, abs=0.001) and high == pytest.approx(0.943, abs=0.001)


def test_rewording_after_a_review_rejection_is_counted():
    metrics = _metrics(
        [
            _record(
                1,
                {
                    "near_rt_ric": [
                        _action("optimize_scheduler", SCHEDULER, False, "llm_review", at="t1"),
                        _action(
                            "optimize_scheduler", {"slice_id": "eMBB"}, False, "schema", at="t2"
                        ),
                        _action("optimize_scheduler", SCHEDULER, at="t3"),
                    ]
                },
            )
        ]
    )

    assert metrics["governance"]["rejected_by_stage"] == {"llm_review": 1, "schema": 1}
    assert metrics["governance"]["approved"]["count"] == 1
    assert metrics["behaviour"]["resubmissions"] == {
        "rejected_then_same_tool_approved": 2,
        "review_rejection_overturned_by_rewording": 1,
        "rejected_never_approved": 0,
    }


def test_clean_run_has_no_invariant_violation():
    metrics = _metrics([_record(1, {"near_rt_ric": [_action("adjust_prb_allocation", PRB)]})])
    assert set(metrics["invariant_violations"].values()) == {0}


def test_invariants_catch_a_gate_failure():
    leaked = _action("adjust_prb_allocation", PRB, False, "resource_safety")
    leaked["execution"]["status"] = "executed"
    early = _action("update_slice_resource_policy", POLICY)
    wrong_cell = _action("optimize_scheduler", SCHEDULER, target="bs9")
    metrics = _metrics(
        [_record(1, {"near_rt_ric": [leaked, wrong_cell], "non_rt_ric": [early]}, persistence=1)]
    )

    assert metrics["invariant_violations"]["rejected_but_executed"] == 1
    assert metrics["invariant_violations"]["wrong_cell_executed"] == 1
    assert metrics["invariant_violations"]["non_rt_below_persistence"] == 1


def test_prb_share_outside_the_slice_policy_is_a_conflict():
    records = [
        _record(
            10,
            {
                "near_rt_ric": [_action("adjust_prb_allocation", PRB, at="t1")],
                "non_rt_ric": [_action("update_slice_resource_policy", POLICY, at="t2")],
            },
        )
    ]
    conflicts = prb_policy_conflicts(action_rows(records))
    assert conflicts == [
        {
            "run_id": "run1",
            "cell_id": "bs2",
            "slice_id": "eMBB",
            "prb_percentage": 80,
            "prb_seq": 10,
            "policy_min": 50,
            "policy_max": 70,
            "policy_seq": 10,
        }
    ]


def test_rejected_prb_change_is_not_a_conflict():
    records = [
        _record(
            10,
            {
                "near_rt_ric": [_action("adjust_prb_allocation", PRB, False, "llm_review")],
                "non_rt_ric": [_action("update_slice_resource_policy", POLICY)],
            },
        )
    ]
    assert prb_policy_conflicts(action_rows(records)) == []


def test_outputs_are_written(tmp_path):
    records = [_record(1, {"near_rt_ric": [_action("adjust_prb_allocation", PRB)]})]
    metrics = _metrics(records)
    write_outputs(tmp_path, metrics, records, action_rows(records))

    assert json.loads((tmp_path / "metrics.json").read_text())["governance"]["proposals"] == 1
    assert "expert_action_appropriate_1to5" in (tmp_path / "expert_rating.csv").read_text()
    assert "\\begin{tabular}" in latex_table({"m": metrics}, "caption", "tab:x")
    assert (tmp_path / "records.csv").exists() and (tmp_path / "actions.csv").exists()


def test_llm_usage_per_record_with_and_without_violation():
    def usage(calls, tokens_in, tokens_out):
        return {
            "calls": calls,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "by_caller": {
                "monitoring": {"calls": 1, "input_tokens": 100, "output_tokens": 10},
                "near_rt_ric": {
                    "calls": calls - 1,
                    "input_tokens": tokens_in - 100,
                    "output_tokens": tokens_out - 10,
                },
            },
        }

    first, second, legacy = _record(1, {}), _record(2, {}), _record(3, {})
    first["llm"], second["llm"] = usage(4, 1000, 100), usage(6, 3000, 300)
    run = {
        "_id": "run1",
        "config": {},
        "summary": {"rows_selected": 10, "rows_without_violation": 7},
    }

    result = compute_metrics([run], [first, second, legacy], [])["llm_usage"]

    assert result["records_measured"] == 2 and result["records_without_usage_field"] == 1
    assert result["with_violation"]["calls"]["mean"] == 5
    assert result["with_violation"]["input_tokens"]["mean"] == 2000
    assert result["with_violation"]["total_tokens"]["mean"] == 2200
    assert result["with_violation_mean_per_caller"]["monitoring"] == {
        "calls": 1,
        "input_tokens": 100,
        "output_tokens": 10,
        "reasoning_tokens": 0,
    }
    assert result["without_violation"] == {
        "records": 7,
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }
    assert result["mean_per_row_all"]["calls"] == 10 / 9  # 2 measured + 7 skipped rows
