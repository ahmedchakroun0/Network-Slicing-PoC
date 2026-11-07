"""Stability of the agentic system over repeated runs of the same records.

    poetry run python -m scripts.run_dataset --record-ids A,B,C --run-id stab-1   (x N)
    poetry run python scripts/stability.py stab-1 stab-2 ... [--name NAME]

For each record and each run, the decision is reduced to signatures; a
record is stable on a dimension when every run gives the same signature.
Per dimension: the mean modal agreement (share of runs that agree with the
most frequent outcome, 1 = always the same) and the mean number of distinct
outcomes. Cost (calls, tokens, time) gets its mean, standard deviation and
coefficient of variation per record; executed PRB shares get their spread.

Each run starts from an empty working memory (run_dataset resets it), so the
runs are independent repetitions.

Writes results/stability-<name>/: stability.json, signatures.csv,
stability_table.tex.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from scripts.evaluate_run import _seconds
from src.repositories.mongo_repository import MongoRepository

DIMENSIONS = {
    "situation_severity": "Situation and severity",
    "llm_agents": "Agents chosen by the Monitoring LLM",
    "agents": "Agents invoked",
    "tools_proposed": "Tools proposed",
    "tools_executed": "Tools executed",
    "actions_executed": "Executed tools and parameters",
    "governance": "Governance outcome per call",
}


def _calls(doc: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(agent, action) for agent, run in sorted((doc.get("agents") or {}).items())
            for action in run.get("actions", [])]


def signatures(doc: dict[str, Any]) -> dict[str, str]:
    monitoring = doc.get("monitoring") or {}
    llm = monitoring.get("llm_output") or monitoring
    calls = _calls(doc)
    executed = [(agent, a) for agent, a in calls
                if (a.get("execution") or {}).get("status") == "executed"]
    return {
        "situation_severity": f"{monitoring.get('situation')}/{monitoring.get('severity')}",
        "llm_agents": ",".join(sorted(i["agent"] for i in llm.get("selected_agents", []))),
        "agents": ",".join(sorted(i["agent"] for i in monitoring.get("selected_agents", []))),
        "tools_proposed": ",".join(sorted(f"{agent}:{a['tool']}" for agent, a in calls)),
        "tools_executed": ",".join(sorted(f"{agent}:{a['tool']}" for agent, a in executed)),
        "actions_executed": ";".join(sorted(
            f"{a['tool']}{json.dumps(a.get('params') or {}, sort_keys=True)}" for _, a in executed)),
        "governance": ";".join(
            f"{a['tool']}:{(a.get('governance') or {}).get('failed_stage') or 'approved'}"
            for _, a in calls),
    }


def _spread(values: list[float]) -> dict[str, Any]:
    series = pd.Series([v for v in values if v is not None], dtype=float)
    if series.empty:
        return {"n": 0}
    mean = series.mean()
    std = series.std(ddof=1) if len(series) > 1 else 0.0
    return {"n": len(series), "mean": mean, "std": std, "min": series.min(),
            "max": series.max(), "cv": std / mean if mean else None}


def stability(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_record: dict[str, list[dict[str, Any]]] = {}
    for doc in records:
        if doc.get("status") == "done":
            by_record.setdefault(doc["telemetry"]["record_id"], []).append(doc)

    per_record, rows = {}, []
    for record_id, docs in sorted(by_record.items()):
        sigs = [signatures(doc) for doc in docs]
        for doc, sig in zip(docs, sigs):
            rows.append({"record_id": record_id, "run_id": doc["run_id"], **sig})
        agreement = {}
        for dimension in DIMENSIONS:
            counts = Counter(sig[dimension] for sig in sigs)
            agreement[dimension] = {"modal_agreement": counts.most_common(1)[0][1] / len(sigs),
                                    "distinct": len(counts)}
        llm = [doc.get("llm") or {} for doc in docs]
        prb = [a["params"]["prb_percentage"] for doc in docs for _, a in _calls(doc)
               if a["tool"] == "adjust_prb_allocation"
               and (a.get("execution") or {}).get("status") == "executed"]
        finished = [[r.get("finished_at") for r in (doc.get("agents") or {}).values()
                     if r.get("finished_at")] for doc in docs]
        per_record[record_id] = {
            "runs": len(docs),
            "agreement": agreement,
            "calls": _spread([u.get("calls") for u in llm]),
            "total_tokens": _spread([(u.get("input_tokens") or 0) + (u.get("output_tokens") or 0)
                                     if u else None for u in llm]),
            "ric_seconds": _spread([
                _seconds((doc.get("monitoring") or {}).get("decided_at"), max(f, default=None))
                for doc, f in zip(docs, finished)]),
            "executed_prb_percentage": _spread(prb),
        }

    summary = {
        dimension: {
            "mean_modal_agreement": _mean([r["agreement"][dimension]["modal_agreement"]
                                           for r in per_record.values()]),
            "mean_distinct": _mean([r["agreement"][dimension]["distinct"]
                                    for r in per_record.values()]),
            "records_fully_stable": sum(r["agreement"][dimension]["distinct"] == 1
                                        for r in per_record.values()),
        }
        for dimension in DIMENSIONS
    }
    return {"records": len(per_record), "runs": sorted({d["run_id"] for d in records}),
            "summary": summary, "per_record": per_record, "rows": rows}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def latex(result: dict[str, Any]) -> str:
    runs = max((r["runs"] for r in result["per_record"].values()), default=0)
    lines = [f"{DIMENSIONS[d]} & {100 * s['mean_modal_agreement']:.0f} & "
             f"{s['mean_distinct']:.1f} & {s['records_fully_stable']}/{result['records']} \\\\"
             for d, s in result["summary"].items() if s["mean_modal_agreement"] is not None]
    cost = [r["total_tokens"] for r in result["per_record"].values() if r["total_tokens"]["n"]]
    if cost:
        cv = _mean([c["cv"] for c in cost if c.get("cv") is not None])
        lines.append("\\midrule")
        lines.append(f"Tokens per record, CV & \\multicolumn{{3}}{{r}}{{{cv:.2f}}} \\\\"
                     if cv is not None else "Tokens per record, CV & \\multicolumn{3}{r}{--} \\\\")
    body = "\n".join(lines)
    return ("\\begin{table}[!t]\n\\centering\n"
            f"\\caption{{Stability over {runs} runs of {result['records']} records}}"
            "\\label{tab:stability}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n"
            "Decision & Agreement (\\%) & Distinct & Stable \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stability over repeated runs.")
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--name", default="runs", help="results/stability-<name>/")
    parser.add_argument("--tex", type=Path, default=None,
                        help="Also write the LaTeX table here (e.g. the paper's tables/).")
    args = parser.parse_args()
    records = list(MongoRepository().records.find({"run_id": {"$in": args.run_ids}}))
    result = stability(records)
    out = Path("results") / f"stability-{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.pop("rows")).to_csv(out / "signatures.csv", index=False)
    (out / "stability.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    (out / "stability_table.tex").write_text(latex(result), encoding="utf-8")
    if args.tex:
        args.tex.parent.mkdir(parents=True, exist_ok=True)
        args.tex.write_text(latex(result), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nWritten to {out}/")


if __name__ == "__main__":
    main()
