"""Print every number quoted in main.tex, recomputed from data/.

Run after the experiments to cross-check the manuscript:  python paper_numbers.py
"""

import json

import pandas as pd

from common import DATA_DIR

# Student-t 97.5% quantiles (two-sided 95% CI) for the episode counts used
T975 = {19: 2.093, 9: 2.262, 4: 2.776}


def main() -> None:
    C = DATA_DIR / "coloran"
    e1 = json.loads((C / "e1_summary.json").read_text())
    e2 = json.loads((C / "e2_e3_summary.json").read_text())
    e5 = pd.read_csv(DATA_DIR / "e5_memory_scaling.csv")  # PoC-dataset records (paper E5)
    ep = pd.read_csv(DATA_DIR / "e6_closed_loop_episodes.csv")
    meta = json.loads((DATA_DIR / "e6_meta.json").read_text())
    fb = pd.read_csv(DATA_DIR / "e6_fault_breakdown.csv", index_col=0)
    u = e5[e5.variant.str.contains("unbounded")]
    w = e5[~e5.variant.str.contains("unbounded")]
    def ci(x):
        return round(100 * x.mean(), 1), round(100 * T975[len(x) - 1] * x.std() / len(x) ** 0.5, 1)
    cl = {f"{a} eps={e}": ci(g.ok_all) for (a, e), g in ep.groupby(["arm", "eps"])}
    g05 = ep[(ep.arm == "gov") & (ep.eps == 0.5)]
    n05 = ep[(ep.arm == "nogov") & (ep.eps == 0.5)]
    g0 = ep[(ep.arm == "gov") & (ep.eps == 0)]
    d = (g05.set_index("episode").ok_all - n05.set_index("episode").ok_all)
    out = {
        "E1 windows / streams": (e1["n_windows"], e1["n_streams"]),
        "E1 comparator median/p99 (us)": (round(e1["comparator_us"]["median"], 1), round(e1["comparator_us"]["p99"], 1)),
        "E1 violation share overall (%)": round(100 * e1["records_with_violation"], 1),
        "E1 violation share by slice (%)": {k: round(100 * v, 1) for k, v in e1["violation_rate_by_slice"].items()},
        "E1 rho>=3 stream/global (%)": (round(100 * e1["nonrt_eligible_k3_stream"], 1), round(100 * e1["nonrt_eligible_k3_global"], 1)),
        "E1 k=3 decision disagreement (%)": round(100 * e1["k3_decision_disagreement"], 1),
        "E2 all-slice compliance mean,CI95 (%)": cl,
        "E2 hindsight-best static (%)": round(100 * meta["oracle_static_mean"], 1),
        "E2 paired gov-nogov eps=0.5 (pts, t-CI95)": (round(100 * d.mean(), 1), round(100 * T975[len(d) - 1] * d.std() / len(d) ** 0.5, 1)),
        "E2 eps=0 calls / reconfigs per episode": (g0.calls.mean(), g0.reconfigs.mean()),
        "E2 eps=0.5 faulty exec/ep gov vs nogov": (g05.faulty_executed.mean(), n05.faulty_executed.mean()),
        "E2 eps=0.5 reconfigs/ep gov vs nogov": (g05.reconfigs.mean(), n05.reconfigs.mean()),
        "E2 governed faults generated/executed": fb.loc["gov"].to_dict(),
        "E3 detection (%)": {g: round(100 * v["detection"], 1) for g, v in e2["overall"].items()},
        "E3 false rejection (%)": {g: round(100 * v["false_rejection"], 1) for g, v in e2["overall"].items()},
        "E3 G3 latency median/p99 (us)": (round(e2["latency_us"]["G3_us"]["median"], 1), round(e2["latency_us"]["G3_us"]["p99"], 1)),
        "E3 G1 latency median (us)": round(e2["latency_us"]["G1_us"]["median"], 1),
        "E3 latency over repeated runs: G3 median min/max, G3 p99 max, G1 median min/max (us)": (
            lambda L: (round(min(x["G3_us"]["median"] for x in L)), round(max(x["G3_us"]["median"] for x in L)),
                       round(max(x["G3_us"]["p99"] for x in L)), round(min(x["G1_us"]["median"] for x in L), 1),
                       round(max(x["G1_us"]["median"] for x in L), 1))
        )([json.loads((DATA_DIR / d / "e2_e3_summary.json").read_text())["latency_us"]
           for d in ("coloran", "coloran_rep1", "coloran_rep2", "coloran_rep3")]),
        "E3 on corrected PoC v2: G3 detection / FRR (%)": (
            lambda v: (round(100 * v["overall"]["G3"]["detection"], 1), round(100 * v["overall"]["G3"]["false_rejection"], 1))
        )(json.loads((DATA_DIR / "coloran_v2" / "e2_e3_summary.json").read_text())),
        "E3 reviewer share of proposals (%)": round(100 * e2["e3_llm_review"]["approve"]["reviewer_calls"] / e2["n_cases"], 1),
        "E5 chars/record (k)": round((u.ric_context_chars.iloc[-1] - u.ric_context_chars.iloc[99]) / (len(u) - 100) / 1e3, 1),
        "E5 first record > 128k tokens": int(u[u.ric_context_chars / 4 > 128_000].n.min()),
        "E5 context at 2000 (M chars)": round(u.ric_context_chars.iloc[-1] / 1e6, 1),
        "E5 time first 10 records, median (ms)": round(u[u.n <= 10].run_ms.median(), 1),
        "E5 time last 50 records, median (s)": round(u[u.n > len(u) - 50].run_ms.median() / 1e3, 2),
        "E5 window context (k chars)": round(w.ric_context_chars.median() / 1e3, 1),
        "E5 window time median (ms)": round(w.run_ms.median(), 1),
    }
    L = DATA_DIR / "llm_runs"
    if (L / "llm_calls.csv").exists():
        calls = pd.read_csv(L / "llm_calls.csv")
        recs = pd.read_csv(L / "llm_records.csv")
        rv3 = json.loads((L / "replay_v3.json").read_text())
        rup = json.loads((L / "replay_uploaded.json").read_text())
        calls["outcome"] = calls.failed_stage.fillna("approved")
        out.update({
            "E4 decisions / violating / label+routing consistent": (
                len(recs), int((recs.violations.fillna("") != "").sum()),
                int((recs.label_ok & recs.routing_consistent).sum())),
            "E4 invocations selected / audited": (int(recs.invocations_selected.sum()), int(recs.invocations_logged.sum())),
            "E4 Table II outcome by agent": calls.groupby(["agent", "outcome"]).size().unstack(fill_value=0).to_dict("index"),
            "E4 approved off-objective by agent": calls[calls.approved & (calls.addresses_objective == False)].groupby("agent").size().to_dict(),  # noqa: E712
            "E4 malformed (invalid+schema) / total": (int(calls.outcome.isin(["invalid_proposal", "schema"]).sum()), len(calls)),
            "E4 target+slice correct / with evidence": (int((calls.target_ok & calls.slice_ok).sum()), int((calls.evidence_sources > 0).sum())),
            "E4 calls per invocation": calls.groupby(["run", "agent"]).size().to_dict(),
            "E4 replay of uploaded gate: mismatches": sum(r["original"] != r["replayed"] for r in rup["rows"]),
            "E4 replay through fixed gate: executions, by agent": (
                rv3["executed_replayed"], {a: v["replayed"] for a, v in rv3["by_agent"].items()}),
            "E4 fixed revision (v3) E3 G3 detection / FRR (%)": (
                lambda v: (round(100 * v["overall"]["G3"]["detection"], 1), round(100 * v["overall"]["G3"]["false_rejection"], 1))
            )(json.loads((DATA_DIR / "coloran_v3" / "e2_e3_summary.json").read_text())),
        })
    B = DATA_DIR / "llm_runs_nemotron"
    if (B / "llm_calls.csv").exists():
        cb = pd.read_csv(B / "llm_calls.csv")
        rb = pd.read_csv(B / "llm_records.csv")
        rpb = json.loads((B / "replay_v3_2.json").read_text())
        ev = json.loads((B / "evidence_check.json").read_text())
        ac = json.loads((B / "action_check.json").read_text())
        out.update({
            "E4-B decisions / violating / label+routing consistent": (
                len(rb), int((rb.violations.fillna("") != "").sum()), int((rb.label_ok & rb.routing_consistent).sum())),
            "E4-B selections": rb.selected.fillna("").tolist(),
            "E4-B situations": rb.situation.tolist(),
            "E4-B calls / approved / off-objective / misquoted": (
                len(cb), int(cb.approved.sum()), int((cb.approved & (cb.addresses_objective == False)).sum()),  # noqa: E712
                int(cb.misquoted_numbers.fillna("").astype(bool).sum())),
            "E4-B executed actions": cb[["run", "tool", "params"]].values.tolist(),
            "E4-B replay mismatches (v3.2 gate)": sum(r["original"] != r["replayed"] for r in rpb["rows"]),
            "E4-B evidence items / untraceable": (len(ev), sum(e["verdict"] == "not in KB" for e in ev)),
            "E4 trace check current split bs2": ac["current"]["bs2"],
            "E4 trace check proposals (run, %, eMBB RBG, [eMBB viol, all-slice ok] per split at bs2)": [
                (q["run"], q["prb_percentage"], q["nearest_eMBB_rbg"],
                 [(s["bs2"]["split"], s["bs2"]["viol_eMBB"], s["bs2"]["all_slice_ok"]) for s in q["splits"]])
                for q in ac["proposals"]],
        })
    for k, v in out.items():
        print(f"{k:45s} {v}")


if __name__ == "__main__":
    main()
