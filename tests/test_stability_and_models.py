"""scripts/stability.py and scripts/compare_models.py on stored runs."""

import pytest

from scripts.compare_models import compare, group_by_model
from scripts.stability import stability
from tests.test_evaluate_run import _action, _record

PRB = {"slice_id": "eMBB", "prb_percentage": 60}


def _run_record(run_id, prb, tokens):
    doc = _record(
        4,
        {
            "near_rt_ric": [
                _action("adjust_prb_allocation", {**PRB, "prb_percentage": prb}),
                _action("optimize_scheduler", {"slice_id": "eMBB", "scheduler": "waterfilling"}),
            ]
        },
    )
    doc.update(
        run_id=run_id,
        _id=f"{run_id}:r4",
        llm={"calls": 10, "input_tokens": tokens, "output_tokens": 0},
    )
    return doc


def test_stability_counts_agreement_per_dimension():
    records = [_run_record("s1", 60, 1000), _run_record("s2", 60, 1200), _run_record("s3", 70, 800)]
    result = stability(records)

    summary = result["summary"]
    assert result["records"] == 1
    assert summary["tools_executed"]["mean_modal_agreement"] == 1.0  # same tools
    assert summary["actions_executed"]["mean_modal_agreement"] == pytest.approx(2 / 3)
    assert summary["actions_executed"]["mean_distinct"] == 2
    record = result["per_record"]["r4"]
    assert (
        record["executed_prb_percentage"]["min"] == 60
        and record["executed_prb_percentage"]["max"] == 70
    )
    assert record["total_tokens"]["mean"] == 1000 and record["total_tokens"]["cv"] == pytest.approx(
        0.2
    )


def test_failed_records_are_left_out():
    failed = _run_record("s4", 90, 5000)
    failed["status"] = "failed"
    result = stability([_run_record("s1", 60, 1000), failed])
    assert result["per_record"]["r4"]["runs"] == 1


def test_models_are_compared_side_by_side(mongo, tmp_path):
    for run_id, model in (("a1", "nemotron"), ("a2", "nemotron"), ("b1", "gemini")):
        mongo.runs.insert_one(
            {
                "_id": run_id,
                "config": {"model": model},
                "summary": {"rows_selected": 3, "rows_without_violation": 2},
            }
        )
        mongo.records.insert_one(_run_record(run_id, 60, 1000))

    assert group_by_model(mongo, ["a1", "a2", "b1"]) == {"gemini": ["b1"], "nemotron": ["a1", "a2"]}
    columns = compare(mongo, ["a1", "a2", "b1"], tmp_path)

    assert columns["nemotron"]["dataset"]["records_with_violation"] == 2
    assert columns["gemini"]["dataset"]["records_with_violation"] == 1
    table = (tmp_path / "comparison_table.tex").read_text()
    assert "gemini & nemotron" in table and "Tokens per record, total (k)" in table
    with pytest.raises(ValueError):
        group_by_model(mongo, ["missing"])
