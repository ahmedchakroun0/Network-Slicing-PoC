"""E1 (ColO-RAN) - SLA grounding and persistence on real-stack traces.

Runs the PoC's TelemetryRecord + MonitoringAgent.analyze_sla on every
1-s slice window produced by coloran_adapter.py and reports:
  * agreement with the adapter's labels (implementation consistency);
  * per-slice violation prevalence vs. the RBGs given to the slice;
  * persistence: per-stream (BS, slice) consecutive-violation counts vs.
    the GLOBAL counter used by the PoC's run_dataset.py when the three
    slices of a BS are interleaved in one telemetry stream;
  * comparator cost per record.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import common
from common import KPIS, base_parser, dump_json, use_project

ALIAS = ["Use Case Type", "Packet Loss Budget", "Latency Budget (ns)", "Jitter Budget (ns)",
         "Data Rate Budget (Gbps)", "Required Mobility", "Required Connectivity",
         "Slice Available Transfer Rate (Gbps)", "Slice Latency (ns)", "Slice Packet Loss",
         "Slice Jitter (ns)", "Slice Type", "Slice Handover"]


def main() -> None:
    p = base_parser(__doc__)
    a = p.parse_args()
    common.set_outdir("coloran", globals())
    df = pd.read_csv(common.HERE.parent / "data" / "coloran_windows.csv.gz")
    use_project(a.project)
    from agents.monitoring_agent import MonitoringAgent
    from src.schema.monitoring import TelemetryRecord

    recs = df[ALIAS].to_dict("records")
    viol = np.zeros((len(df), 4), dtype=bool)
    t_us = []
    for i, r in enumerate(recs):
        tr = TelemetryRecord.model_validate(r)
        t0 = time.perf_counter_ns()
        an = MonitoringAgent.analyze_sla(tr)
        if i % 50 == 0:
            t_us.append((time.perf_counter_ns() - t0) / 1e3)
        for j, k in enumerate(KPIS):
            viol[i, j] = k in an.violations
    lab = df[["Latency Violation", "Jitter Violation", "Packet Loss Violation", "Throughput Violation"]].values
    agree = {k: float((viol[:, j] == lab[:, j]).mean()) for j, k in enumerate(KPIS)}
    df["any"] = viol.any(axis=1)
    for j, k in enumerate(KPIS):
        df[f"v_{k}"] = viol[:, j]

    names = {0: "eMBB", 1: "MTC", 2: "URLLC"}
    prev = (df.groupby(["slice", "rbg_slice"])[["any"] + [f"v_{k}" for k in KPIS]].mean()
            .reset_index())
    prev["slice_name"] = prev.slice.map(names)
    prev.to_csv(common.DATA_DIR / "e1_violation_vs_rbg.csv", index=False)

    # persistence: per stream vs global (interleaved) counter, PoC semantics per KPI, "any KPI" max
    df = df.sort_values(["sched", "tr", "exp", "bs", "w", "slice"]).reset_index(drop=True)
    glob = {k: 0 for k in KPIS}
    g_any = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        for k in KPIS:
            glob[k] = glob[k] + 1 if df.at[i, f"v_{k}"] else 0
        g_any[i] = max(glob.values())
    df["rho_global"] = g_any
    stream = df.groupby(["sched", "tr", "exp", "bs", "slice"], sort=False)
    per = np.zeros(len(df), dtype=int)
    for _, idx in stream.groups.items():
        idx = sorted(idx, key=lambda i: df.at[i, "w"])
        c = {k: 0 for k in KPIS}
        for i in idx:
            for k in KPIS:
                c[k] = c[k] + 1 if df.at[i, f"v_{k}"] else 0
            per[i] = max(c.values())
    df["rho_stream"] = per
    cc = []
    for kk in range(1, 21):
        cc.append({"k": kk, "global": float((df.rho_global >= kk).mean()), "stream": float((df.rho_stream >= kk).mean())})
    pd.DataFrame(cc).to_csv(common.DATA_DIR / "e1_persistence_ccdf.csv", index=False)
    k3_disagree = float(((df.rho_global >= 3) != (df.rho_stream >= 3)).mean())

    summary = {
        "n_windows": int(len(df)),
        "n_streams": int(stream.ngroups),
        "label_agreement": agree,
        "violation_rate_by_slice": {names[s]: float(v) for s, v in df.groupby("slice")["any"].mean().items()},
        "kpi_rate_by_slice": {names[s]: {k: float(g[f"v_{k}"].mean()) for k in KPIS} for s, g in df.groupby("slice")},
        "records_with_violation": float(df["any"].mean()),
        "nonrt_eligible_k3_stream": float((df.rho_stream >= 3).mean()),
        "nonrt_eligible_k3_global": float((df.rho_global >= 3).mean()),
        "k3_decision_disagreement": k3_disagree,
        "comparator_us": {"median": float(np.median(t_us)), "p99": float(np.percentile(t_us, 99))},
    }
    dump_json(summary, "e1_summary.json")
    import json
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
