"""P1 - Protocol for the LLM-in-the-loop evaluation (NOT executed for the paper).

The paper reports only deterministic experiments plus pilot traces. This
script is the harness for the end-to-end study described in Sec. VI-B: run
the PoC with each model configured in litellm_config.yaml, R repetitions
per model, on a stratified sample of records, then compute the metrics
below directly from the MongoDB run documents written by the PoC.

Step 1 (per model m, repetition r), from the PoC root:
    MODEL_NAME=<m> poetry run python scripts/run_dataset.py \
        --run-id <m>-r<r> --start-row 1 --max-rows 500
Step 2:
    python protocol_llm_eval.py --runs <m>-r1 <m>-r2 ... --out ../data/p1_<m>.json

Metrics (per model):
  routing_agreement   share of records whose selected agent set equals the
                      reference policy  {near} if max persistence < k else
                      {near, non}  (k = NON_RT_MIN_PERSISTENCE), and {} when
                      there is no violation;
  no_violation_silence  share of violation-free records with no invocation;
  severity_consistency  share of records with identical (situation,
                      severity) across repetitions;
  ric_status          distribution of completed / rejected / no_action / error;
  proposal_validity   share of protected calls that pass the schema stage;
  approval_rate       approved / audited calls;
  first_failing_stage distribution over audited calls;
  hallucinated_ids    share of calls whose target/slice differ from context.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from pymongo import MongoClient


def reference_agents(analysis: dict, persistence: dict | None, k: int) -> set[str]:
    if not analysis.get("violations"):
        return set()
    if persistence and max(persistence.values(), default=0) >= k:
        return {"near_rt_ric", "non_rt_ric"}
    return {"near_rt_ric"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    col = MongoClient(a.mongo_uri)["network_slicing_poc"]["dataset_runs"]
    route_ok = route_n = silent_ok = silent_n = 0
    sev = defaultdict(set)
    ric = Counter()
    stages = Counter()
    calls = valid_schema = approved = halluc = 0

    for run_id in a.runs:
        doc = col.find_one({"_id": run_id})
        if doc is None:
            raise SystemExit(f"run {run_id} not found")
        persistence: dict[str, int] = defaultdict(int)
        for rid, rec in sorted(doc.get("records", {}).items()):
            dec = rec.get("monitoring_decision")
            ana = rec.get("computed_analysis") or {}
            for q in ("latency", "jitter", "packet_loss", "throughput"):
                persistence[q] = persistence[q] + 1 if q in ana.get("violations", []) else 0
            if not dec:
                continue
            chosen = {i["agent"] for i in dec.get("selected_agents", [])}
            ref = reference_agents(ana, dict(persistence), a.k)
            route_n += 1
            route_ok += int(chosen == ref)
            if not ana.get("violations"):
                silent_n += 1
                silent_ok += int(not chosen)
            sev[rid].add((dec.get("situation"), dec.get("severity")))
            for agent, ex in (rec.get("agent_executions") or {}).items():
                ric[(agent, ((ex or {}).get("result") or {}).get("status", "missing"))] += 1
                inv_ctx = ((ex or {}).get("invocation") or {}).get("context") or {}
                for item in (ex or {}).get("middleware_audit") or []:
                    calls += 1
                    g = item.get("governance") or {}
                    st = g.get("failed_stage") or "approved"
                    stages[st] += 1
                    approved += int(bool(g.get("approved")))
                    valid_schema += int(st not in ("schema", "invalid_proposal"))
                    args = (item.get("execution") or {}).get("parameters") or {}
                    if args.get("target") != inv_ctx.get("target") or (
                        "slice_id" in args and args.get("slice_id") != inv_ctx.get("slice_id")):
                        halluc += 1

    out = {
        "runs": a.runs,
        "routing_agreement": route_ok / route_n if route_n else None,
        "no_violation_silence": silent_ok / silent_n if silent_n else None,
        "severity_consistency": (sum(len(v) == 1 for v in sev.values()) / len(sev)) if sev else None,
        "ric_status": {f"{k[0]}:{k[1]}": v for k, v in ric.items()},
        "protected_calls": calls,
        "proposal_validity": valid_schema / calls if calls else None,
        "approval_rate": approved / calls if calls else None,
        "first_failing_stage": dict(stages),
        "hallucinated_ids": halluc / calls if calls else None,
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
