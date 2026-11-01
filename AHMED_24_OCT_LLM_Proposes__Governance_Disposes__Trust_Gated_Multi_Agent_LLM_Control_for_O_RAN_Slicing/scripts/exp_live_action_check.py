"""What would the PRB shares proposed by the live LLM agents give in the traces?

The live runs act on record coloran_0000000: bs2, round-robin, split 6/9/2 RBGs
(eMBB/MTC/URLLC). For every executed adjust_prb_allocation call, map the
requested eMBB share to RBGs (share x 17), take the dataset splits with the
nearest eMBB RBG count under the same scheduler, and report per-slice violation
rates and all-slice compliance at bs2 (and over all BSs) from the 730,053
ColO-RAN windows, next to the current split. The scheduler changes cannot be
checked: priority_based / proportional_fair per slice do not exist in ColO-RAN.

Usage: python exp_live_action_check.py [RUNS_DIR ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from coloran_adapter import TR_RBG
from common import DATA_DIR

N_RBG = 17


def stats(d: pd.DataFrame, tr: int, bs: int | None) -> dict:
    g = d[(d.sched == 0) & (d.tr == tr)]
    if bs is not None:
        g = g[g.bs == bs]
    viol = g.groupby("slice").viol.mean()
    ok = (~g.groupby(["bs", "w"]).viol.any()).mean()
    return dict(split="/".join(map(str, TR_RBG[tr])),
                viol_eMBB=round(float(viol[0]), 3), viol_MTC=round(float(viol[1]), 3),
                viol_URLLC=round(float(viol[2]), 3), all_slice_ok=round(float(ok), 3),
                windows=int(len(g)))


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or [DATA_DIR / "llm_runs", DATA_DIR / "llm_runs_nemotron"]
    d = pd.read_csv(DATA_DIR / "coloran_windows.csv.gz",
                    usecols=["sched", "tr", "bs", "slice", "w", "SLA Evaluation Status"])
    d["viol"] = d["SLA Evaluation Status"].eq("violation")
    current_tr = TR_RBG.index((6, 9, 2))
    out = {"current": {"bs2": stats(d, current_tr, 2), "all": stats(d, current_tr, None)}, "proposals": []}
    for rd in dirs:
        for f in sorted(rd.glob("run-*.json")):
            doc = json.loads(f.read_text())
            for rec in doc["records"].values():
                for ex in (rec.get("agent_executions") or {}).values():
                    for a in ex.get("middleware_audit") or []:
                        e = a.get("execution") or {}
                        if e.get("tool_name") != "adjust_prb_allocation" or e.get("status") != "executed":
                            continue
                        pct = e["parameters"]["prb_percentage"]
                        rbg = pct / 100 * N_RBG
                        cand = sorted({s[0] for s in TR_RBG}, key=lambda r: abs(r - rbg))[0]
                        trs = [i for i, s in enumerate(TR_RBG) if s[0] == cand]
                        out["proposals"].append(dict(
                            run=doc["run_id"], prb_percentage=pct, rbg=round(rbg, 2), nearest_eMBB_rbg=cand,
                            splits=[{"bs2": stats(d, t, 2), "all": stats(d, t, None)} for t in trs]))
    (DATA_DIR / "llm_runs_nemotron" / "action_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
