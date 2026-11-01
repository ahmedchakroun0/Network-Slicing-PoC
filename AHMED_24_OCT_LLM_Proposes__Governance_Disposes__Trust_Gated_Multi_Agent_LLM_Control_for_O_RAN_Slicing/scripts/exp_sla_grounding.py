"""E1 - Deterministic SLA grounding on the full telemetry dataset.

Runs the PoC's own comparator (MonitoringAgent.analyze_sla) on every record
and reports:
  * agreement with the dataset's per-KPI violation labels;
  * violation prevalence per slice type under two budget semantics:
      S1 literal   - a budget of 0 is a hard bound (the PoC's behaviour);
      S2 unspecified - a budget of 0 means "no requirement" (KPI marked
                       unavailable, as the PoC does for missing budgets);
  * the routing load each semantics implies (records with >=1 violation);
  * consecutive-violation persistence in dataset order (the quantity the
    run script passes to the Non-RT RIC agent);
  * comparator cost per record.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import pandas as pd

from common import KPIS, base_parser, dump_json, use_project, DATA_DIR


def main() -> None:
    args = base_parser(__doc__).parse_args()
    use_project(args.project)

    from agents.monitoring_agent import MonitoringAgent
    from scripts.run_dataset import _row_to_telemetry_record, _compare_ground_truth
    from langchain_core.globals import set_debug
    set_debug(False)  # run_dataset.py enables verbose tracing on import

    df = pd.read_csv("data/prepared_network_slicing_dataset.csv")
    n = len(df)

    rows = []
    t_ns = []
    agree = Counter()
    for i, row in df.iterrows():
        key = row["record_id"]
        rec = _row_to_telemetry_record(row, key)
        t0 = time.perf_counter_ns()
        a = MonitoringAgent.analyze_sla(rec)
        t_ns.append(time.perf_counter_ns() - t0)
        cmp = _compare_ground_truth(rec, row, key)["comparison"]
        for k in KPIS:
            agree[k] += int(cmp["metric_matches"][k])
        agree["set"] += int(cmp["sla_violations_match"])

        # S2: a zero budget is read as "unspecified" -> unavailable.
        rec2 = rec.model_copy(
            update={
                "latency_budget_ns": rec.latency_budget_ns or None,
                "jitter_budget_ns": rec.jitter_budget_ns or None,
                "packet_loss_budget": rec.packet_loss_budget or None,
                "data_rate_budget_gbps": rec.data_rate_budget_gbps or None,
            }
        )
        a2 = MonitoringAgent.analyze_sla(rec2)
        rows.append(
            {
                "record_id": key,
                "slice": rec.slice_type,
                "use_case": rec.use_case_type,
                **{f"s1_{k}": k in a.violations for k in KPIS},
                **{f"s2_{k}": k in a2.violations for k in KPIS},
                **{f"s2_unavail_{k}": k in a2.unavailable_metrics for k in KPIS},
                "s1_any": bool(a.violations),
                "s2_any": bool(a2.violations),
                "s1_set": "+".join(a.violations) or "none",
                "s2_set": "+".join(a2.violations) or "none",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "e1_sla_per_record.csv", index=False)

    # Per-slice prevalence table (both semantics)
    prev = []
    for sem in ("s1", "s2"):
        for sl, g in out.groupby("slice"):
            r = {"semantics": sem, "slice": sl, "n": len(g)}
            for k in KPIS:
                r[k] = g[f"{sem}_{k}"].mean()
            r["any"] = g[f"{sem}_any"].mean()
            prev.append(r)
    pd.DataFrame(prev).to_csv(DATA_DIR / "e1_prevalence_by_slice.csv", index=False)

    # Persistence in dataset order (per KPI, run script semantics)
    persist = {}
    for sem in ("s1", "s2"):
        for k in KPIS:
            p, seq = 0, []
            for v in out[f"{sem}_{k}"]:
                p = p + 1 if v else 0
                seq.append(p)
            persist[(sem, k)] = np.array(seq)
    kgrid = list(range(1, 21))
    ccdf = []
    for (sem, k), seq in persist.items():
        for kk in kgrid:
            ccdf.append({"semantics": sem, "kpi": k, "k": kk, "frac_ge_k": float((seq >= kk).mean())})
    # "any KPI persistent >= k": the condition an enforced Non-RT gate would test
    for sem in ("s1", "s2"):
        mx = np.max(np.vstack([persist[(sem, k)] for k in KPIS]), axis=0)
        for kk in kgrid:
            ccdf.append({"semantics": sem, "kpi": "any", "k": kk, "frac_ge_k": float((mx >= kk).mean())})
    pd.DataFrame(ccdf).to_csv(DATA_DIR / "e1_persistence_ccdf.csv", index=False)

    t = np.array(t_ns) / 1e3
    summary = {
        "n_records": n,
        "label_agreement": {k: agree[k] / n for k in (*KPIS, "set")},
        "s1_records_with_violation": float(out.s1_any.mean()),
        "s2_records_with_violation": float(out.s2_any.mean()),
        "s1_violation_sets": out.s1_set.value_counts().to_dict(),
        "s2_violation_sets": out.s2_set.value_counts().to_dict(),
        "zero_budget_share": {
            "latency": float((df["Latency Budget (ns)"] == 0).mean()),
            "jitter": float((df["Jitter Budget (ns)"] == 0).mean()),
            "packet_loss": float((df["Packet Loss Budget"] == 0).mean()),
            "throughput": float((df["Data Rate Budget (Gbps)"] == 0).mean()),
        },
        "s1_kpi_prevalence": {k: float(out[f"s1_{k}"].mean()) for k in KPIS},
        "s2_kpi_prevalence": {k: float(out[f"s2_{k}"].mean()) for k in KPIS},
        "s2_kpi_unavailable": {k: float(out[f"s2_unavail_{k}"].mean()) for k in KPIS},
        "s1_nonrt_eligible_k3": float(
            next(r["frac_ge_k"] for r in ccdf if r["semantics"] == "s1" and r["kpi"] == "any" and r["k"] == 3)
        ),
        "s2_nonrt_eligible_k3": float(
            next(r["frac_ge_k"] for r in ccdf if r["semantics"] == "s2" and r["kpi"] == "any" and r["k"] == 3)
        ),
        "comparator_us": {
            "median": float(np.median(t)),
            "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)),
        },
    }
    dump_json(summary, "e1_summary.json")
    import json
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
