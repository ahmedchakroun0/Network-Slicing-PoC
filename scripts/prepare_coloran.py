"""Prepare the ColO-RAN dataset as telemetry for the Monitoring Agent.

Input : raw ColO-RAN per-UE slice traces downloaded by
        `scripts/download_coloran.sh` (data/raw/coloran/rome_static_medium/
        sched{0,1,2}/tr{0..27}/exp*/bs{1..7}/slices_bs*/<imsi>_metrics.csv,
        one row every 250 ms per UE).
Output: a CSV with exactly the columns of data/prepared_network_slicing_dataset.csv
        (so `TelemetryRecord`, `MonitoringAgent.analyze_sla` and
        `scripts/run_dataset.py` work unchanged), plus "Cell ID" and a few
        metadata columns that the agents ignore.

    poetry run python scripts/prepare_coloran.py                  # all runs
    poetry run python scripts/prepare_coloran.py --runs 50        # 50 random cell runs
    poetry run python scripts/run_dataset.py data/coloran_prepared_dataset.csv --max-rows 30

Dataset (Polese et al., IEEE TMC 2023): srsRAN-based stack on the Colosseum
emulator; 7 BSs (10 MHz, 50 PRBs), 3 slices per BS (slice 0 eMBB: 4 Mbps CBR
per UE; slice 1 MTC: Poisson 30 pkt/s x 125 B; slice 2 URLLC: Poisson
10 pkt/s x 125 B), 2 UEs per slice and BS, 3 schedulers (0 RR, 1 WF, 2 PF)
x 28 RBG splits of 17 RBGs.

Each run is trimmed to start at the first window in which all three slices
carry traffic (UE attach / traffic start-up takes ~50 s, up to ~80 s).
One output record = one slice of one BS over one window (default 1 s):
  Slice Available Transfer Rate (Gbps) = sum over the slice's UEs of the mean DL rate
  Slice Latency (ns)                   = mean over UEs of the queueing-delay
                                         estimate dl_buffer*8 / DL rate (capped 2 s)
  Slice Jitter (ns)                    = std of that estimate inside the window
  Slice Packet Loss                    = mean DL transmission-error ratio
SLA budgets per slice are fixed a priori from the traffic classes (BUDGETS).
Violation labels are computed with the PoC's own MonitoringAgent.analyze_sla.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # allow `python scripts/prepare_coloran.py`
    sys.path.insert(0, str(ROOT))

from agents.monitoring_agent import MonitoringAgent  # noqa: E402
from src.schema.monitoring import TelemetryRecord  # noqa: E402

SLICE_NAME = {0: "eMBB", 1: "MTC", 2: "URLLC"}
USE_CASE = {0: "eMBB video (CBR 4 Mbps/UE)", 1: "MTC sensing (Poisson 30 pkt/s)",
            2: "URLLC control (Poisson 10 pkt/s)"}
SCHEDULER = {0: "round_robin", 1: "waterfilling", 2: "proportional_fair"}
UES_PER_SLICE = 2
OFFERED_MBPS = {0: 4.0 * UES_PER_SLICE,
                1: 30 * 125 * 8 / 1e6 * UES_PER_SLICE,
                2: 10 * 125 * 8 / 1e6 * UES_PER_SLICE}
# A-priori SLA budgets (not tuned on results):
#   throughput >= 90 % of offered load (CBR eMBB) / >= 50 % (Poisson slices)
#   queueing delay / jitter <= 100/50 ms (eMBB, MTC), 10/5 ms (URLLC)
#   DL error ratio <= 1e-2 (eMBB, MTC), 1e-3 (URLLC)
BUDGETS = {
    0: dict(thr_mbps=0.9 * OFFERED_MBPS[0], lat_ms=100.0, jit_ms=50.0, loss=1e-2),
    1: dict(thr_mbps=0.5 * OFFERED_MBPS[1], lat_ms=100.0, jit_ms=50.0, loss=1e-2),
    2: dict(thr_mbps=0.5 * OFFERED_MBPS[2], lat_ms=10.0, jit_ms=5.0, loss=1e-3),
}
# RBGs per slice for training configurations tr0..tr27 (dataset README)
TR_RBG = [(2, 13, 2), (4, 11, 2), (6, 9, 2), (8, 7, 2), (10, 5, 2), (12, 3, 2), (14, 1, 2),
          (2, 11, 4), (4, 9, 4), (6, 7, 4), (8, 5, 4), (10, 3, 4), (12, 1, 4),
          (2, 9, 6), (4, 7, 6), (6, 5, 6), (8, 3, 6), (10, 1, 6),
          (2, 7, 8), (4, 5, 8), (6, 3, 8), (8, 1, 8),
          (2, 5, 10), (4, 3, 10), (6, 1, 10),
          (2, 3, 12), (4, 1, 12), (2, 1, 14)]
DELAY_CAP_MS = 2000.0
FILE_RE = re.compile(r"sched(\d)/tr(\d+)/(exp\d+)/bs(\d+)/slices_bs\d+/(\d+)_metrics\.csv$")

# Columns of data/prepared_network_slicing_dataset.csv (TelemetryRecord aliases + labels)
POC_COLUMNS = ["record_id", "Use Case Type", "Packet Loss Budget", "Latency Budget (ns)",
               "Jitter Budget (ns)", "Data Rate Budget (Gbps)", "Required Mobility",
               "Required Connectivity", "Slice Available Transfer Rate (Gbps)", "Slice Latency (ns)",
               "Slice Packet Loss", "Slice Jitter (ns)", "Slice Type", "Slice Handover",
               "Latency Violation", "Jitter Violation", "Packet Loss Violation",
               "Throughput Violation", "SLA Violations", "SLA Evaluation Status"]
EXTRA_COLUMNS = ["Cell ID", "Scheduler", "RBG Split", "Run", "Window Start (s)"]


def _load_ue(path: Path, t0_ms: float, warmup_s: float) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d.loc[:, ~d.columns.str.startswith("Unnamed")]
    d = d.rename(columns={"tx_brate downlink [Mbps]": "rate", "dl_buffer [bytes]": "buf",
                          "tx_errors downlink (%)": "err"})
    d = d[["Timestamp", "slice_id", "rate", "buf", "err"]].copy()
    d["t"] = (d.Timestamp - t0_ms) / 1000.0          # common clock of the BS run
    d = d[d.t >= warmup_s]
    with np.errstate(divide="ignore", invalid="ignore"):
        delay = np.where(d.buf <= 0, 0.0, d.buf * 8 / (d.rate * 1e6) * 1e3)
    d["delay_ms"] = np.clip(np.nan_to_num(delay, nan=DELAY_CAP_MS, posinf=DELAY_CAP_MS), 0, DELAY_CAP_MS)
    return d


def discover(root: Path, runs: list[str] | None) -> dict[tuple, list[Path]]:
    """Group trace files by (scheduler, split, run, BS)."""
    groups: dict[tuple, list[Path]] = {}
    for f in sorted(root.rglob("*_metrics.csv")):
        m = FILE_RE.search(f.as_posix())
        if not m or (runs and m.group(3) not in runs):
            continue
        key = (int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4)))
        groups.setdefault(key, []).append(f)
    return groups


def build_cell_run(key: tuple, files: list[Path], window_s: float, warmup_s: float) -> list[dict]:
    """All slice-window records of one (scheduler, split, run, BS)."""
    sched, tr, run, bs = key
    t0 = min(float(pd.read_csv(f, usecols=["Timestamp"], nrows=1).Timestamp.iloc[0]) for f in files)
    by_slice: dict[int, list[pd.DataFrame]] = {}
    for f in files:
        d = _load_ue(f, t0, warmup_s)
        if d.empty:
            continue
        d["w"] = (d.t // window_s).astype(int)
        by_slice.setdefault(int(d.slice_id.mode().iloc[0]), []).append(d)
    if set(by_slice) != {0, 1, 2}:
        return []

    agg = {}
    for sl, ues in by_slice.items():
        per_ue = [u.groupby("w").agg(rate=("rate", "mean"), delay=("delay_ms", "mean"), err=("err", "mean"))
                  for u in ues]
        common = set.intersection(*(set(p.index) for p in per_ue))
        jit = pd.concat(ues).groupby("w").delay_ms.std(ddof=0)
        agg[sl] = {w: (sum(p.loc[w, "rate"] for p in per_ue),
                       float(np.mean([p.loc[w, "delay"] for p in per_ue])),
                       float(np.mean([p.loc[w, "err"] for p in per_ue])) / 100.0,
                       float(jit.get(w, 0.0)))
                   for w in common}
    windows = sorted(set.intersection(*(set(a) for a in agg.values())))   # all 3 slices present
    # Start-up: UEs attach and traffic generators start ~50 s into each run
    # (up to ~80 s). Drop the windows before every slice carries traffic, so
    # idle start-up periods are not reported as SLA violations.
    active = [w for w in windows if all(agg[sl][w][0] > 0 for sl in (0, 1, 2))]
    if not active:
        return []
    windows = [w for w in windows if w >= active[0]]

    out = []
    for w in windows:
        for sl in (0, 1, 2):
            rate, lat_ms, loss, jit_ms = agg[sl][w]
            b = BUDGETS[sl]
            out.append({
                "Use Case Type": USE_CASE[sl],
                "Packet Loss Budget": b["loss"],
                "Latency Budget (ns)": b["lat_ms"] * 1e6,
                "Jitter Budget (ns)": b["jit_ms"] * 1e6,
                "Data Rate Budget (Gbps)": b["thr_mbps"] / 1e3,
                "Required Mobility": "no",          # static UEs in this scenario
                "Required Connectivity": "yes",
                "Slice Available Transfer Rate (Gbps)": rate / 1e3,
                "Slice Latency (ns)": lat_ms * 1e6,
                "Slice Packet Loss": loss,
                "Slice Jitter (ns)": jit_ms * 1e6,
                "Slice Type": SLICE_NAME[sl],
                "Slice Handover": "no",
                "Cell ID": f"bs{bs}",
                "Scheduler": SCHEDULER[sched],
                "RBG Split": "/".join(map(str, TR_RBG[tr])),
                "Run": f"sched{sched}/tr{tr}/{run}",
                "Window Start (s)": w * window_s,
            })
    return out


def label(df: pd.DataFrame) -> pd.DataFrame:
    """Validate every row as a TelemetryRecord and label it with the PoC's
    own deterministic SLA analyzer."""
    alias_cols = [c for c in POC_COLUMNS if c not in ("record_id",) and "Violation" not in c
                  and c not in ("SLA Violations", "SLA Evaluation Status")]
    flags = {k: [] for k in ("latency", "jitter", "packet_loss", "throughput")}
    sets, status = [], []
    for row in df[alias_cols].to_dict("records"):
        analysis = MonitoringAgent.analyze_sla(TelemetryRecord.model_validate(row))
        for k in flags:
            flags[k].append(k in analysis.violations)
        sets.append(str(analysis.violations))
        status.append("violation" if analysis.violations else "ok")
    df["Latency Violation"] = flags["latency"]
    df["Jitter Violation"] = flags["jitter"]
    df["Packet Loss Violation"] = flags["packet_loss"]
    df["Throughput Violation"] = flags["throughput"]
    df["SLA Violations"] = sets
    df["SLA Evaluation Status"] = status
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT / "data/raw/coloran/rome_static_medium")
    p.add_argument("--out", type=Path, default=ROOT / "data/coloran_prepared_dataset.csv")
    p.add_argument("--window", type=float, default=1.0, help="seconds per record (default 1)")
    p.add_argument("--warmup", type=float, default=20.0, help="seconds dropped at the start of each run")
    p.add_argument("--experiments", nargs="*", default=None, help="e.g. exp1 exp2 (default: all downloaded)")
    p.add_argument("--runs", type=int, default=None, help="keep N random (scheduler, split, BS) runs")
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()

    if not a.root.is_dir():
        raise SystemExit(f"{a.root} not found: run `bash scripts/download_coloran.sh` first.")
    groups = discover(a.root, a.experiments)
    keys = sorted(groups)
    if a.runs is not None and a.runs < len(keys):
        rng = np.random.default_rng(a.seed)
        keys = sorted(keys[i] for i in rng.choice(len(keys), size=a.runs, replace=False))
    print(f"{len(keys)} cell runs from {len(groups)} available ({sum(len(groups[k]) for k in keys)} UE traces)")

    rows = []
    for i, key in enumerate(keys, 1):
        rows += build_cell_run(key, groups[key], a.window, a.warmup)
        if i % 100 == 0:
            print(f"  {i}/{len(keys)} runs, {len(rows)} records", flush=True)
    if not rows:
        raise SystemExit("no complete records produced")

    df = label(pd.DataFrame(rows))
    df.insert(0, "record_id", [f"coloran_{i:07d}" for i in range(len(df))])
    df = df[POC_COLUMNS[:14] + ["Cell ID"] + POC_COLUMNS[14:] + EXTRA_COLUMNS[1:]]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    summary = {
        "records": int(len(df)), "cell_runs": len(keys), "window_s": a.window,
        "violation_rate_by_slice": df.groupby("Slice Type")["SLA Evaluation Status"]
                                     .apply(lambda s: round(float((s == "violation").mean()), 4)).to_dict(),
        "budgets": {SLICE_NAME[k]: v for k, v in BUDGETS.items()},
        "source": "ColO-RAN (Polese et al., IEEE TMC 2023), GPL-3.0",
    }
    a.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {a.out} ({len(df)} records)")
    print(json.dumps(summary["violation_rate_by_slice"]))


if __name__ == "__main__":
    main()
