"""Evaluation of live runs, computed from the MongoDB results store (read-only).

    poetry run python scripts/evaluate_run.py RUN_ID [RUN_ID ...] [--out DIR]

MongoDB stays the source of truth; this writes a snapshot of one or more runs
to results/<run_id>/ so the paper's numbers can be regenerated and archived
(run_dataset.py does this at the end of every run; several runs pooled go to
results/<run1>+<run2>/):

    metrics.json        every metric below
    records.csv         one row per record that had a violation
    actions.csv         one row per protected tool call, with the stated reasons
    expert_rating.csv   actions.csv plus empty columns for the expert reviewers
    e4_table.tex        LaTeX table of the governance outcomes

Only records whose workflow completed (status "done") are evaluated: a record
that failed, e.g. when every model was rate-limited and the run stopped, is
counted but its calls are not, since the gate failed closed on them.

The tools do not act on a network and the telemetry is replayed, so these
metrics describe the decisions and the gate. They say nothing about whether
an action would have fixed the violation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from dotenv import load_dotenv

from agents.monitoring_agent import MonitoringAgent
from agents.ran_trust_governance import NON_RT_MIN_PERSISTENCE
from scripts.agent_checks import annotate
from src.repositories.mongo_repository import MongoRepository
from src.schema.monitoring import TelemetryRecord

load_dotenv()

REVIEW_VERDICTS = ("goal_aligned", "reasoning_sound", "behavior_acceptable")
# Below this, a rate is an anecdote: reported, but flagged.
MIN_PROPOSALS = 30
RATING_COLUMNS = ("expert_action_appropriate_1to5", "expert_reasoning_grounded_1to5",
                  "expert_agrees_with_governance_yes_no", "expert_notes")


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson interval of a proportion. Treats calls as independent, which
    consecutive records of one slice are not: read it as a lower bound on
    the uncertainty."""
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(successes: int, n: int) -> dict[str, Any]:
    interval = wilson(successes, n)
    return {"count": successes, "n": n, "rate": successes / n if n else None,
            "ci95": list(interval) if interval else None}


def _seconds(start: Any, end: Any) -> float | None:
    if start is None or end is None:
        return None
    return (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()


def _distribution(values: list[float]) -> dict[str, Any]:
    series = pd.Series(values, dtype=float)
    if series.empty:
        return {"n": 0}
    return {"n": len(series), "mean": series.mean(), "median": series.median(),
            "p95": series.quantile(0.95), "max": series.max()}


# --------------------------------------------------------------------------
# MongoDB documents -> tables
# --------------------------------------------------------------------------


def record_rows(records: list[dict[str, Any]],
                rows_by_run: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    checks = annotate(records, rows_by_run)[0]
    rows = []
    for doc in records:
        telemetry, monitoring = doc["telemetry"], doc.get("monitoring") or {}
        agents = doc.get("agents") or {}
        finished = [a["finished_at"] for a in agents.values() if a.get("finished_at")]
        rows.append({
            "run_id": doc["run_id"], "seq": doc["seq"], "record_id": telemetry["record_id"],
            "status": doc.get("status"), "cell_id": telemetry["cell_id"],
            "slice_id": telemetry["slice_id"], "experiment": telemetry["run"],
            "slice_prb": telemetry["slice_prb"], "cell_prb": telemetry["cell_prb"],
            "scheduler": telemetry["scheduler"],
            "throughput_per_ue_mbps": telemetry["dl_throughput_per_ue_mbps"],
            "queueing_delay_ms": telemetry["queueing_delay_ms"],
            "prb_utilisation": telemetry["prb_utilisation"],
            "cell_spare_prb": telemetry["cell_spare_prb"],
            "violations": ",".join(doc["sla"].get("violations", [])),
            "max_persistence": max((doc.get("persistence") or {}).values(), default=0),
            "situation": monitoring.get("situation"), "severity": monitoring.get("severity"),
            "overridden": monitoring.get("overridden"),
            "agents_selected": ",".join(i["agent"] for i in monitoring.get("selected_agents", [])),
            "notes": " | ".join(monitoring.get("orchestration_notes", [])),
            "ric_seconds": _seconds(monitoring.get("decided_at"), max(finished, default=None)),
            "llm_calls": (doc.get("llm") or {}).get("calls"),
            "input_tokens": (doc.get("llm") or {}).get("input_tokens"),
            "output_tokens": (doc.get("llm") or {}).get("output_tokens"),
            **checks[(doc["run_id"], doc["seq"])],
        })
    return rows


def action_rows(records: list[dict[str, Any]],
                rows_by_run: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    checks = annotate(records, rows_by_run)[1]
    rows = []
    for doc in records:
        telemetry = doc["telemetry"]
        for agent, run in (doc.get("agents") or {}).items():
            for position, action in enumerate(run.get("actions", [])):
                governance = action.get("governance") or {}
                reasoning = action.get("reasoning") or {}
                review = governance.get("llm_review") or {}
                rows.append({
                    "run_id": doc["run_id"], "seq": doc["seq"], "at": str(action.get("at")),
                    "cell_id": telemetry["cell_id"], "slice_id": telemetry["slice_id"],
                    "experiment": telemetry["run"],
                    "violations": ",".join(doc["sla"].get("violations", [])),
                    "throughput_per_ue_mbps": telemetry["dl_throughput_per_ue_mbps"],
                    "queueing_delay_ms": telemetry["queueing_delay_ms"],
                    "slice_prb_share_pct": round(100 * telemetry["slice_prb"] / telemetry["cell_prb"], 1),
                    "max_persistence": max((doc.get("persistence") or {}).values(), default=0),
                    "agent": agent, "tool": action["tool"], "target": action.get("target"),
                    "params": json.dumps(action.get("params") or {}, sort_keys=True),
                    "forced": bool((action.get("agent_message") or {}).get("forced")),
                    "confidence": reasoning.get("confidence"),
                    "rationale": reasoning.get("rationale"),
                    "expected_effect": reasoning.get("expected_effect"),
                    "approved": bool(governance.get("approved")),
                    "failed_stage": governance.get("failed_stage"),
                    "governance_reason": governance.get("reason"),
                    **{verdict: review.get(verdict) for verdict in REVIEW_VERDICTS},
                    "execution": (action.get("execution") or {}).get("status"),
                    **checks[(doc["run_id"], doc["seq"], agent, action.get("i", position))],
                })
    return sorted(rows, key=lambda r: (r["run_id"], r["seq"], r["at"]))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def invariant_violations(actions: list[dict[str, Any]]) -> dict[str, int]:
    """What the gate guarantees, checked on the stored results: all must be 0."""
    return {
        "rejected_but_executed": sum(not a["approved"] and a["execution"] == "executed"
                                     for a in actions),
        "approved_but_not_executed": sum(a["approved"] and a["execution"] != "executed"
                                         for a in actions),
        "non_rt_below_persistence": sum(
            a["approved"] and a["agent"] == "non_rt_ric"
            and a["max_persistence"] < NON_RT_MIN_PERSISTENCE for a in actions),
        "wrong_cell_executed": sum(a["execution"] == "executed" and a["target"] != a["cell_id"]
                                   for a in actions),
        "wrong_slice_executed": sum(
            a["execution"] == "executed"
            and json.loads(a["params"]).get("slice_id", a["slice_id"]) != a["slice_id"]
            for a in actions),
    }


def resubmissions(actions: list[dict[str, Any]]) -> dict[str, int]:
    """What the agents did after a rejection, within one invocation. An LLM
    review rejection followed by approval of the SAME parameters means only
    the wording changed."""
    recovered = reworded = abandoned = 0
    keys = {(a["run_id"], a["seq"], a["agent"], a["tool"]) for a in actions}
    for key in keys:
        calls = [a for a in actions if (a["run_id"], a["seq"], a["agent"], a["tool"]) == key]
        for index, call in enumerate(calls):
            if call["approved"]:
                continue
            later = [c for c in calls[index + 1:] if c["approved"]]
            if not later:
                abandoned += 1
                continue
            recovered += 1
            reworded += (call["failed_stage"] == "llm_review"
                         and any(c["params"] == call["params"] for c in later))
    return {"rejected_then_same_tool_approved": recovered,
            "review_rejection_overturned_by_rewording": reworded,
            "rejected_never_approved": abandoned}


def prb_policy_conflicts(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Executed PRB shares outside the slice policy executed on the same slice
    (same run, cell, slice and experiment), in either order. Governance checks
    each call on its own, so it does not see these."""
    allocation: dict[tuple, tuple[int, int]] = {}
    policy: dict[tuple, tuple[int, int, int]] = {}
    conflicts = []
    for action in actions:
        if action["execution"] != "executed":
            continue
        stream = (action["run_id"], action["cell_id"], action["slice_id"], action["experiment"])
        params = json.loads(action["params"])
        if action["tool"] == "adjust_prb_allocation":
            allocation[stream] = (params["prb_percentage"], action["seq"])
        elif action["tool"] == "update_slice_resource_policy":
            policy[stream] = (params["min_prb_ratio_pct"], params["max_prb_ratio_pct"], action["seq"])
        else:
            continue
        if stream in allocation and stream in policy:
            share, share_seq = allocation[stream]
            low, high, policy_seq = policy[stream]
            if not low <= share <= high:
                conflicts.append({"run_id": stream[0], "cell_id": stream[1], "slice_id": stream[2],
                                  "prb_percentage": share, "prb_seq": share_seq,
                                  "policy_min": low, "policy_max": high, "policy_seq": policy_seq})
    return conflicts


def llm_usage(records: list[dict[str, Any]], rows_without_violation: int) -> dict[str, Any]:
    """Model calls and tokens per record. A record without violation is
    skipped before the Monitoring Agent: it makes no model call at all.
    Records stored before usage was measured (no `llm` field) are left out."""
    measured = [doc["llm"] for doc in records if doc.get("llm")]
    keys = ("calls", "input_tokens", "output_tokens", "reasoning_tokens")
    # A key a record does not have was not measured (e.g. reasoning tokens
    # before they were recorded): left out, not counted as 0.
    with_violation = {key: _distribution([u[key] for u in measured if key in u]) for key in keys}
    with_violation["total_tokens"] = _distribution(
        [u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in measured])
    callers = sorted({caller for u in measured for caller in u.get("by_caller", {})})
    per_caller = {
        caller: {key: sum(u.get("by_caller", {}).get(caller, {}).get(key, 0) for u in measured)
                 / len(measured) for key in keys}
        for caller in callers
    }
    rows = len(measured) + rows_without_violation
    totals = {key: sum(u.get(key, 0) for u in measured) for key in keys}
    return {
        "records_measured": len(measured),
        "records_without_usage_field": len(records) - len(measured),
        "with_violation": with_violation,
        "with_violation_mean_per_caller": per_caller,
        "without_violation": {"records": rows_without_violation, "calls": 0,
                              "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
        # Unknown, not 0, when no record of the runs was measured.
        "mean_per_row_all": {key: totals[key] / rows if measured else None for key in keys},
        "totals": totals,
        "failed_calls": sum(u.get("failed_calls", 0) for u in measured),
        "calls_without_usage": sum(u.get("calls_without_usage", 0) for u in measured),
    }


def _bool_rate(values: list[Any]) -> dict[str, Any]:
    """Rate of True among the values that apply (None = not applicable)."""
    applicable = [v for v in values if v is not None]
    return _rate(sum(bool(v) for v in applicable), len(applicable))


def agent_quality(rows: list[dict[str, Any]], actions: list[dict[str, Any]]) -> dict[str, Any]:
    """The automatic checks of scripts/agent_checks.py, aggregated."""
    quoted = sum(a["numbers_quoted"] for a in actions)
    untraced = sum(a["numbers_untraced"] for a in actions)
    monitoring_quoted = sum(r["monitoring_numbers_quoted"] for r in rows)
    monitoring_untraced = sum(r["monitoring_numbers_untraced"] for r in rows)
    sources = sum(a["evidence_sources"] for a in actions)
    causes = Counter(a["tool_cause"] for a in actions)
    judged = causes["consistent"] + causes["inconsistent"]
    return {
        "monitoring": {
            key: _bool_rate([r[key] for r in rows])
            for key in ("sla_recomputed_agrees", "llm_flags_violation", "llm_situation_ok",
                        "llm_severity_agrees", "llm_routing_agrees", "objective_names_kpis")
        } | {"numbers_traceable": _rate(monitoring_quoted - monitoring_untraced,
                                        monitoring_quoted)},
        "tool_choice": {
            "diagnosis": dict(Counter(a["diagnosis"] for a in actions)),
            "tool_cause": dict(causes),
            "consistent_among_judged": _rate(causes["consistent"], judged),
            "direction_ok": _bool_rate([a["direction_ok"] for a in actions]),
            "feasible": _bool_rate([a["feasible"] for a in actions]),
            "no_op": sum(bool(a["no_op"]) for a in actions),
        },
        "grounding": {
            "numbers_traceable": _rate(quoted - untraced, quoted),
            "actions_all_numbers_traceable": _rate(
                sum(a["numbers_untraced"] == 0 for a in actions), len(actions)),
            "evidence_supported": _rate(sum(a["evidence_supported"] for a in actions), sources),
            "actions_without_evidence": sum(a["evidence_sources"] == 0 for a in actions),
            "knowledge_cited_without_retrieval": sum(
                a["evidence_knowledge"] > 0 and a["retrieved_knowledge"] == 0 for a in actions),
        },
        # The review against the rules: proxies for false approvals / rejections.
        "governance_vs_rules": {
            "approved_but_rule_inconsistent": sum(
                a["approved"] and a["tool_cause"] == "inconsistent" for a in actions),
            "review_rejected_but_rule_consistent": sum(
                a["failed_stage"] == "llm_review" and a["tool_cause"] == "consistent"
                for a in actions),
        },
    }


def processed_rows(run: dict[str, Any], records: list[dict[str, Any]]) -> list[TelemetryRecord] | None:
    """The dataset rows a run processed (None when the dataset file is gone):
    its selection, cut at the last stored record when the run stopped."""
    config = run.get("config") or {}
    path = Path(config.get("dataset") or "")
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if config.get("record_ids"):
        frame = frame[frame["record_id"].isin(config["record_ids"])]
    else:
        start = max((config.get("start_row") or 1) - 1, 0)
        stop = None if config.get("max_rows") is None else start + config["max_rows"]
        frame = frame.iloc[start:stop]
    if run.get("status") == "stopped":
        last = max((d["seq"] for d in records if d["run_id"] == run["_id"]), default=0)
        frame = frame[frame.index + 1 <= last]
    fields = set(TelemetryRecord.model_fields)
    return [TelemetryRecord.model_validate(
        {k: (None if pd.isna(v) else v) for k, v in row.items() if k in fields})
        for _, row in frame.iterrows()]


def sla_detection(runs: list[dict[str, Any]], records: list[dict[str, Any]],
                  rows: dict[str, list[TelemetryRecord] | None]) -> dict[str, Any]:
    """Every violated row of the dataset reached the agents: recompute the SLA
    analysis on the rows each run processed and compare with the records it
    stored (a skipped row has no record)."""
    stored = {(d["run_id"], (d.get("telemetry") or {}).get("record_id") or d["_id"].split(":", 1)[1])
              for d in records}
    result = {"rows_checked": 0, "violated_rows": 0, "missed": 0, "runs_without_dataset": []}
    for run in runs:
        if rows.get(run["_id"]) is None:
            result["runs_without_dataset"].append(run["_id"])
            continue
        for record in rows[run["_id"]]:
            result["rows_checked"] += 1
            if MonitoringAgent.analyze_sla(record).violations:
                result["violated_rows"] += 1
                result["missed"] += (run["_id"], record.record_id) not in stored
    return result


def compute_metrics(runs: list[dict[str, Any]], records: list[dict[str, Any]],
                    actions: list[dict[str, Any]],
                    rows_by_run: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    failed = [doc for doc in records if doc.get("status") != "done"]
    records = [doc for doc in records if doc.get("status") == "done"]
    rows = record_rows(records, rows_by_run)
    approved = sum(a["approved"] for a in actions)
    summaries = [r.get("summary") or {} for r in runs]
    notes = [row["notes"] for row in rows]
    durations: dict[str, list[float]] = {}
    for doc in records:
        for agent, run in (doc.get("agents") or {}).items():
            seconds = _seconds(run.get("started_at"), run.get("finished_at"))
            if seconds is not None:
                durations.setdefault(agent, []).append(seconds)

    per_tool: dict[str, dict[str, int]] = {}
    for action in actions:
        entry = per_tool.setdefault(action["tool"], {"proposed": 0, "approved": 0, "rejected": 0})
        entry["proposed"] += 1
        entry["approved" if action["approved"] else "rejected"] += 1

    return {
        "run_ids": [r["_id"] for r in runs],
        "models": sorted({str((r.get("config") or {}).get("model")) for r in runs}),
        "models_served": dict(sum((Counter({m["model"]: m["calls"] for m in doc.get("models", [])})
                                   for doc in records), Counter())),
        "stopped": {r["_id"]: r["stop_reason"] for r in runs if r.get("stop_reason")},
        "dataset": {
            "rows_selected": sum(s.get("rows_selected", 0) for s in summaries),
            "records_with_violation": len(rows),
            "rows_without_violation": sum(s.get("rows_without_violation", 0) for s in summaries),
            "rows_failed": sum(s.get("rows_failed", 0) for s in summaries),
            "records_not_evaluated": len(failed),
            "by_slice": dict(Counter(row["slice_id"] for row in rows)),
            "cells": len({row["cell_id"] for row in rows}),
            "max_persistence": max((row["max_persistence"] for row in rows), default=0),
        },
        "monitoring": {
            "decision_overridden_by_code": sum(bool(row["overridden"]) for row in rows),
            "non_rt_dropped_below_threshold": sum("not invoked" in n for n in notes),
            "near_rt_fallback": sum("by fallback" in n for n in notes),
            "agent_selected": dict(Counter(
                agent for row in rows for agent in row["agents_selected"].split(",") if agent)),
            "situation": dict(Counter(row["situation"] for row in rows)),
        },
        "governance": {
            "proposals": len(actions),
            "approved": _rate(approved, len(actions)),
            "rejected_by_stage": dict(Counter(
                a["failed_stage"] for a in actions if not a["approved"])),
            "review_verdict_false": {
                verdict: sum(a[verdict] is False for a in actions) for verdict in REVIEW_VERDICTS},
            "per_tool": per_tool,
            "per_agent": dict(Counter(a["agent"] for a in actions)),
        },
        "behaviour": {
            "calls_per_record": _distribution(
                list(Counter((a["run_id"], a["seq"]) for a in actions).values())),
            "forced_calls": _rate(sum(a["forced"] for a in actions), len(actions)),
            "resubmissions": resubmissions(actions),
            "mean_confidence": {
                "approved": _distribution([a["confidence"] for a in actions
                                           if a["approved"] and a["confidence"] is not None]).get("mean"),
                "rejected": _distribution([a["confidence"] for a in actions
                                           if not a["approved"] and a["confidence"] is not None]).get("mean"),
            },
        },
        "invariant_violations": invariant_violations(actions),
        "cross_agent_conflicts": prb_policy_conflicts(actions),
        "latency_s": {
            **{agent: _distribution(values) for agent, values in durations.items()},
            "ric_phase_per_record": _distribution(
                [row["ric_seconds"] for row in rows if row["ric_seconds"] is not None]),
        },
        "llm_usage": llm_usage(records, sum(s.get("rows_without_violation", 0)
                                            for s in summaries)),
        "agent_quality": agent_quality(rows, actions),
        "enough_data": len(actions) >= MIN_PROPOSALS,
    }


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


def _percent(rate: dict[str, Any] | None) -> str:
    if not rate or rate.get("rate") is None:
        return "--"
    return f"{100 * rate['rate']:.0f} ({rate['count']}/{rate['n']})"


def _mean(distribution: dict[str, Any], scale: float = 1.0, digits: int = 1) -> str:
    mean = distribution.get("mean")
    return "--" if mean is None else f"{mean / scale:.{digits}f}"


def summary_rows(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    """The paper's E4 metrics: one (label, value) per row, shared by the
    per-run table and the model comparison."""
    governance, behaviour = metrics["governance"], metrics["behaviour"]
    quality, usage = metrics["agent_quality"], metrics["llm_usage"]["with_violation"]
    proposals = governance["proposals"]
    review_rejected = governance["rejected_by_stage"].get("llm_review", 0)
    return [
        ("Records with a violation", str(metrics["dataset"]["records_with_violation"])),
        ("Violations flagged by the Monitoring LLM (\\%)",
         _percent(quality["monitoring"]["llm_flags_violation"])),
        ("Situation not called normal (\\%)", _percent(quality["monitoring"]["llm_situation_ok"])),
        ("Severity agrees with code (\\%)", _percent(quality["monitoring"]["llm_severity_agrees"])),
        ("Routing agrees with the rule (\\%)", _percent(quality["monitoring"]["llm_routing_agrees"])),
        ("Protected tool calls per record", _mean(behaviour["calls_per_record"])),
        ("Tool consistent with the diagnosis (\\%)",
         _percent(quality["tool_choice"]["consistent_among_judged"])),
        ("PRB change in the right direction (\\%)", _percent(quality["tool_choice"]["direction_ok"])),
        ("Quoted numbers traceable (\\%)", _percent(quality["grounding"]["numbers_traceable"])),
        ("Evidence sources supported (\\%)", _percent(quality["grounding"]["evidence_supported"])),
        ("Calls approved (\\%)", _percent(governance["approved"])),
        ("Calls rejected by code (\\%)",
         _percent({"count": proposals - governance["approved"]["count"] - review_rejected,
                   "n": proposals,
                   "rate": (proposals - governance["approved"]["count"] - review_rejected) / proposals
                   if proposals else None})),
        ("Calls rejected by the LLM review (\\%)",
         _percent({"count": review_rejected, "n": proposals,
                   "rate": review_rejected / proposals if proposals else None})),
        ("Review rejection overturned by rewording",
         str(behaviour["resubmissions"]["review_rejection_overturned_by_rewording"])),
        ("Gate invariant violations", str(sum(metrics["invariant_violations"].values()))),
        ("PRB share outside the slice policy", str(len(metrics["cross_agent_conflicts"]))),
        ("Model calls per record", _mean(usage["calls"])),
        ("Tokens per record, input (k)", _mean(usage["input_tokens"], 1000)),
        ("Tokens per record, output (k)", _mean(usage["output_tokens"], 1000)),
        ("\\quad of which reasoning (k)", _mean(usage["reasoning_tokens"], 1000)),
        ("Tokens per record, total (k)", _mean(usage["total_tokens"], 1000)),
        ("RIC phase per record (s)", _mean(metrics["latency_s"]["ric_phase_per_record"], 1, 0)),
    ]


def latex_table(columns: dict[str, dict[str, Any]], caption: str, label: str) -> str:
    """A booktabs table: one row per summary metric, one column per entry
    (a run, a set of runs or a model)."""
    names = list(columns)
    rows = [summary_rows(m) for m in columns.values()]
    body = "\n".join(
        " & ".join([label_] + [r[index][1] for r in rows]) + " \\\\"
        for index, (label_, _) in enumerate(rows[0]))
    header = " & ".join(["Metric"] + [n.replace("_", "\\_") for n in names]) + " \\\\"
    return ("\\begin{table}[!t]\n\\centering\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\n\\footnotesize\n"
            "\\setlength{\\tabcolsep}{3pt}\n"
            f"\\begin{{tabular}}{{@{{}}l{'r' * len(names)}@{{}}}}\n\\toprule\n{header}\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


def write_outputs(out: Path, metrics: dict[str, Any], records: list[dict[str, Any]],
                  actions: list[dict[str, Any]],
                  rows_by_run: dict[str, list[dict[str, Any]]] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(record_rows(records, rows_by_run)).to_csv(out / "records.csv", index=False)
    table = pd.DataFrame(actions)
    table.to_csv(out / "actions.csv", index=False)
    rating = table.copy()
    for column in RATING_COLUMNS:
        rating[column] = ""
    rating.to_csv(out / "expert_rating.csv", index=False)
    model = ", ".join(metrics["models"])
    (out / "e4_table.tex").write_text(latex_table(
        {model: metrics}, "Live LLM agents behind the gate", "tab:live"), encoding="utf-8")


def collect(repository: MongoRepository, run_ids: list[str]) -> dict[str, Any]:
    """Metrics, completed records, actions and processed dataset rows of runs."""
    runs = list(repository.runs.find({"_id": {"$in": run_ids}}))
    missing = sorted(set(run_ids) - {r["_id"] for r in runs})
    if missing:
        raise ValueError(f"unknown run id(s): {', '.join(missing)}")
    records = list(repository.records.find({"run_id": {"$in": run_ids}}).sort(
        [("run_id", 1), ("seq", 1)]))
    rows = {run["_id"]: processed_rows(run, records) for run in runs}
    rows_by_run = {run_id: [r.model_dump() for r in found] for run_id, found in rows.items() if found}
    done = [doc for doc in records if doc.get("status") == "done"]
    actions = action_rows(done, rows_by_run)
    metrics = compute_metrics(runs, records, actions, rows_by_run)
    metrics["sla_detection"] = sla_detection(runs, records, rows)
    return {"metrics": metrics, "records": done, "actions": actions, "rows_by_run": rows_by_run}


def evaluate(repository: MongoRepository, run_ids: list[str], out: Path | None = None,
             results_dir: Path = Path("results")) -> tuple[Path, dict[str, Any]]:
    """Compute the metrics of the runs and write their result files."""
    found = collect(repository, run_ids)
    out = out or results_dir / "+".join(run_ids)
    write_outputs(out, found["metrics"], found["records"], found["actions"], found["rows_by_run"])
    return out, found["metrics"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate live runs stored in MongoDB.")
    parser.add_argument("run_ids", nargs="+", help="One or more run ids (pooled together).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: results/<run id>).")
    parser.add_argument("--tex", type=Path, default=None,
                        help="Also write the LaTeX table here (e.g. the paper's tables/).")
    parser.add_argument("--label", default=None,
                        help="Column name of the LaTeX table (default: the configured model).")
    args = parser.parse_args()

    try:
        out, metrics = evaluate(MongoRepository(), args.run_ids, args.out)
    except ValueError as exc:
        parser.error(str(exc))
    if args.tex:
        args.tex.parent.mkdir(parents=True, exist_ok=True)
        label = args.label or ", ".join(metrics["models"])
        args.tex.write_text(latex_table({label: metrics}, "Live LLM agents behind the gate",
                                        "tab:live"), encoding="utf-8")

    print(json.dumps({k: metrics[k] for k in ("dataset", "governance", "invariant_violations",
                                              "cross_agent_conflicts", "llm_usage",
                                              "agent_quality", "sla_detection")},
                     indent=2, default=float))
    print(f"\nWritten to {out}/")
    if not metrics["enough_data"]:
        print(f"Only {metrics['governance']['proposals']} tool calls: too few for rates "
              f"(need at least {MIN_PROPOSALS}).")


if __name__ == "__main__":
    main()
