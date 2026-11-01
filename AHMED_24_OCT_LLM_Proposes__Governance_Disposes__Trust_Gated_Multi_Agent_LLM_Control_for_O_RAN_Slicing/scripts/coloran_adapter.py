"""ColO-RAN -> PoC TelemetryRecord adapter (no change to the PoC code).

Converts the per-UE slice traces of the ColO-RAN dataset (Polese et al.,
IEEE TMC 2022; srsRAN stack on the Colosseum RF emulator, 7 BSs, 3 slices,
3 schedulers x 28 RBG splits) into records with EXACTLY the columns of the
PoC's prepared_network_slicing_dataset.csv, so TelemetryRecord,
MonitoringAgent.analyze_sla and every downstream agent run unmodified.

One record = one slice of one BS over a W-second window (default 1 s =
4 samples of 250 ms). KPI mapping (documented in the paper):
  throughput  = sum over the slice's UEs of mean DL tx_brate      [Gbps]
  latency     = mean over UEs of the queueing-delay estimate
                dl_buffer*8 / tx_brate                             [ns]
  jitter      = std of the per-sample queueing-delay estimate      [ns]
  packet loss = mean DL tx_errors(%) / 100
SLA budgets are fixed a priori from the traffic classes of the dataset
(eMBB CBR 4 Mbps/UE, MTC Poisson 30 pkt/s x 125 B, URLLC Poisson
10 pkt/s x 125 B, 2 UEs per slice per BS); see BUDGETS below.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA_DIR

SLICE_NAME = {0: "eMBB", 1: "MTC", 2: "URLLC"}
USE_CASE = {0: "eMBB video (CBR 4 Mbps/UE)", 1: "MTC sensing (Poisson 30 pkt/s)",
            2: "URLLC control (Poisson 10 pkt/s)"}
UES_PER_SLICE = 2
OFFERED_MBPS = {0: 4.0 * UES_PER_SLICE, 1: 30 * 125 * 8 / 1e6 * UES_PER_SLICE,
                2: 10 * 125 * 8 / 1e6 * UES_PER_SLICE}
# A-priori SLA budgets per slice (fixed before any controller was run).
#   throughput: >= 90 % of offered load for CBR eMBB, >= 50 % for Poisson
#               slices (1-s windows of Poisson traffic fluctuate);
#   queueing delay / jitter: 100/50 ms (eMBB, MTC), 10/5 ms (URLLC);
#   loss (DL transmission errors): 1e-2 (eMBB, MTC), 1e-3 (URLLC).
BUDGETS = {
    0: dict(thr_mbps=0.9 * OFFERED_MBPS[0], lat_ms=100.0, jit_ms=50.0, loss=1e-2),
    1: dict(thr_mbps=0.5 * OFFERED_MBPS[1], lat_ms=100.0, jit_ms=50.0, loss=1e-2),
    2: dict(thr_mbps=0.5 * OFFERED_MBPS[2], lat_ms=10.0, jit_ms=5.0, loss=1e-3),
}
DELAY_CAP_MS = 2000.0     # buffer with ~zero service rate -> capped delay
WARMUP_S = 20.0
# RBGs per slice for tr0..tr27 (dataset README).
TR_RBG = [(2, 13, 2), (4, 11, 2), (6, 9, 2), (8, 7, 2), (10, 5, 2), (12, 3, 2), (14, 1, 2),
          (2, 11, 4), (4, 9, 4), (6, 7, 4), (8, 5, 4), (10, 3, 4), (12, 1, 4),
          (2, 9, 6), (4, 7, 6), (6, 5, 6), (8, 3, 6), (10, 1, 6),
          (2, 7, 8), (4, 5, 8), (6, 3, 8), (8, 1, 8),
          (2, 5, 10), (4, 3, 10), (6, 1, 10),
          (2, 3, 12), (4, 1, 12), (2, 1, 14)]


def load_ue(path: Path, t0_ms: float) -> pd.DataFrame:
    """Load one UE trace; time is measured from t0_ms, the earliest log line of
    ANY UE of the same (scheduler, split, run, BS), so all UEs and slices of a
    cell share one clock."""
    d = pd.read_csv(path)
    d = d.loc[:, ~d.columns.str.startswith("Unnamed")]
    d = d.rename(columns={"tx_brate downlink [Mbps]": "brate", "dl_buffer [bytes]": "buf",
                          "tx_errors downlink (%)": "err"})
    d = d[["Timestamp", "slice_id", "slice_prb", "scheduling_policy", "brate", "buf", "err"]].copy()
    d["t"] = (d.Timestamp - t0_ms) / 1000.0
    d = d[d.t >= WARMUP_S]
    with np.errstate(divide="ignore", invalid="ignore"):
        delay = np.where(d.buf <= 0, 0.0, d.buf * 8 / (d.brate * 1e6) * 1e3)
    d["delay_ms"] = np.clip(np.nan_to_num(delay, nan=DELAY_CAP_MS, posinf=DELAY_CAP_MS), 0, DELAY_CAP_MS)
    return d


def build(root: Path, window_s: float) -> pd.DataFrame:
    pat = re.compile(r"sched(\d)/tr(\d+)/(exp\d+)/bs(\d+)/slices_bs\d+/(\d+)_metrics\.csv$")
    groups: dict[tuple, list[pd.DataFrame]] = {}
    files = []
    t0: dict[tuple, float] = {}
    for f in sorted(root.rglob("*_metrics.csv")):
        m = pat.search(f.as_posix())
        if not m:
            continue
        run = (int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4)))
        first = float(pd.read_csv(f, usecols=["Timestamp"], nrows=1).Timestamp.iloc[0])
        t0[run] = min(t0.get(run, first), first)
        files.append((f, run))
    for f, run in files:
        sched, tr, exp, bs = run
        d = load_ue(f, t0[run])
        if d.empty:
            continue
        sl = int(d.slice_id.mode().iloc[0])
        d["w"] = (d.t // window_s).astype(int)
        groups.setdefault((sched, tr, exp, bs, sl), []).append(d)

    rows = []
    for (sched, tr, exp, bs, sl), ues in groups.items():
        per_ue = [u.groupby("w").agg(brate=("brate", "mean"), delay=("delay_ms", "mean"),
                                     err=("err", "mean")) for u in ues]
        common = sorted(set.intersection(*(set(p.index) for p in per_ue)))
        allsamp = pd.concat(ues)
        jit = allsamp.groupby("w").delay_ms.std(ddof=0)
        b = BUDGETS[sl]
        for w in common:
            thr = sum(p.loc[w, "brate"] for p in per_ue)
            lat = float(np.mean([p.loc[w, "delay"] for p in per_ue]))
            loss = float(np.mean([p.loc[w, "err"] for p in per_ue])) / 100.0
            rows.append({
                "sched": sched, "tr": tr, "exp": exp, "bs": bs, "slice": sl, "w": w,
                "rbg_slice": TR_RBG[tr][sl], "n_ues": len(per_ue),
                "Use Case Type": USE_CASE[sl],
                "Packet Loss Budget": b["loss"],
                "Latency Budget (ns)": b["lat_ms"] * 1e6,
                "Jitter Budget (ns)": b["jit_ms"] * 1e6,
                "Data Rate Budget (Gbps)": b["thr_mbps"] / 1e3,
                "Required Mobility": "no", "Required Connectivity": "yes",
                "Slice Available Transfer Rate (Gbps)": thr / 1e3,
                "Slice Latency (ns)": lat * 1e6,
                "Slice Packet Loss": loss,
                "Slice Jitter (ns)": float(jit.get(w, 0.0)) * 1e6,
                "Slice Type": SLICE_NAME[sl], "Slice Handover": "no",
                "Cell ID": f"bs{bs}",
            })
    df = pd.DataFrame(rows)
    # Start-up trimming (same rule as the PoC's scripts/prepare_coloran.py):
    # UEs attach and traffic starts ~50 s (up to ~80 s) into each run; drop
    # every window of a (scheduler, split, run, BS) before the first window in
    # which all three slices carry traffic, so idle start-up is not counted
    # as SLA violation.
    rate = "Slice Available Transfer Rate (Gbps)"
    key = ["sched", "tr", "exp", "bs"]
    full = df.groupby(key + ["w"])["slice"].transform("nunique") == 3
    act = df[full].groupby(key + ["w"])[rate].transform(lambda r: (r > 0).all())
    starts = df[full][act].groupby(key)["w"].min().rename("w_start")
    df = df.merge(starts, left_on=key, right_index=True, how="inner")
    df = df[df.w >= df.w_start].drop(columns="w_start").reset_index(drop=True)
    df["Latency Violation"] = df["Slice Latency (ns)"] > df["Latency Budget (ns)"]
    df["Jitter Violation"] = df["Slice Jitter (ns)"] > df["Jitter Budget (ns)"]
    df["Packet Loss Violation"] = df["Slice Packet Loss"] > df["Packet Loss Budget"]
    df["Throughput Violation"] = df["Slice Available Transfer Rate (Gbps)"] < df["Data Rate Budget (Gbps)"]
    names = ["latency", "jitter", "packet_loss", "throughput"]
    flags = df[["Latency Violation", "Jitter Violation", "Packet Loss Violation", "Throughput Violation"]].values
    df["SLA Violations"] = [str([n for n, f in zip(names, r) if f]) for r in flags]
    df["SLA Evaluation Status"] = np.where(flags.any(axis=1), "violation", "ok")
    df = df.sort_values(["sched", "tr", "exp", "bs", "w", "slice"]).reset_index(drop=True)
    df.insert(0, "record_id", [f"coloran_{i:06d}" for i in range(len(df))])
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/home/claude/coloran_data/coloran/rome_static_medium"))
    p.add_argument("--window", type=float, default=1.0)
    a = p.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = build(a.root, a.window)
    df.to_csv(DATA_DIR / "coloran_windows.csv.gz", index=False)
    poc_cols = ["record_id", "Use Case Type", "Packet Loss Budget", "Latency Budget (ns)", "Jitter Budget (ns)",
                "Data Rate Budget (Gbps)", "Required Mobility", "Required Connectivity",
                "Slice Available Transfer Rate (Gbps)", "Slice Latency (ns)", "Slice Packet Loss",
                "Slice Jitter (ns)", "Slice Type", "Slice Handover", "Cell ID", "Latency Violation", "Jitter Violation",
                "Packet Loss Violation", "Throughput Violation", "SLA Violations", "SLA Evaluation Status"]
    # Drop-in replacement for data/prepared_network_slicing_dataset.csv:
    # one BS-time stream (3 slices interleaved) per (scheduler, RBG split).
    rng = np.random.default_rng(2026)
    keys = df[["sched", "tr", "bs"]].drop_duplicates().sample(frac=1, random_state=2026)
    parts, n = [], 0
    for _, k in keys.iterrows():
        g = df[(df.sched == k.sched) & (df.tr == k.tr) & (df.bs == k.bs)].head(60 * 3)
        parts.append(g)
        n += len(g)
        if n >= 10_000:
            break
    replay = pd.concat(parts).head(10_000)
    replay[poc_cols].to_csv(DATA_DIR / "coloran_prepared_dataset.csv", index=False)
    print(df.shape, "windows;", len(replay), "records in the PoC-format replay file")
    print(df.groupby(["slice"])[["Latency Violation", "Jitter Violation", "Packet Loss Violation",
                                  "Throughput Violation"]].mean().round(3))
    print("any violation by slice:\n", (df["SLA Evaluation Status"] == "violation").groupby(df.slice).mean().round(3))
    _ = rng


if __name__ == "__main__":
    main()
