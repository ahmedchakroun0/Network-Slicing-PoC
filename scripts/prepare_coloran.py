"""Prepare the ColO-RAN dataset as slice telemetry for the Monitoring Agent.

Input : raw ColO-RAN per-UE KPM traces downloaded by
        `scripts/download_coloran.sh` (data/raw/coloran/rome_static_medium/
        sched{0,1,2}/tr{0..27}/exp*/bs{1..7}/slices_bs*/<imsi>_metrics.csv,
        one report every 250 ms per UE).
Output: a CSV whose columns are exactly the fields of
        src.schema.monitoring.TelemetryRecord (one row = one slice of one BS
        over one window), plus a .summary.json. No SLA labels: the Monitoring
        Agent evaluates the SLA profiles (src/schema/sla.py) at runtime.

    poetry run python scripts/prepare_coloran.py                  # all cell runs
    poetry run python scripts/prepare_coloran.py --runs 21        # 21 random cell runs
    poetry run python scripts/run_dataset.py data/coloran_prepared_dataset.csv --max-rows 30

Dataset (Polese et al., IEEE TMC 2023): srsRAN stack on the Colosseum
emulator; 7 BSs (10 MHz, 50 PRBs), 3 slices per BS (0 eMBB, 1 MTC, 2 URLLC),
2 static UEs per slice and BS, 3 schedulers x 28 PRB splits. The PRB split
and the scheduler never change inside a run (scripts/explore_coloran.py).

Per window and slice:
  dl_throughput_mbps         sum over UEs of the mean DL rate
  dl_throughput_per_ue_mbps  mean over UEs of the mean DL rate
  dl_buffer_bytes            mean over UEs of the mean DL backlog
  queueing_delay_ms          mean over UEs of backlog * 8 / rate; a UE with a
                             backlog and no delivery counts as the full window
  prb_utilisation            sum of granted PRBs / (slice_prb * TTIs); the KPM
                             granted_prbs is per UE, summed over the report's TTIs
  cell_spare_prb             sum over the OTHER slices of slice_prb * (1 - util)
  dl_cqi, dl_mcs             mean over reports with DL transmissions
  ul_sinr_db                 mean over reports with UL activity (else empty)
Only the window in which all three slices carry sustained traffic is kept
(UEs attach and traffic starts ~50 s into a run and stops before its end).
Columns that are constant in the whole dataset (dl_err_pct, ul_rssi,
power_multiplier, ...) are not used.
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

from src.schema.monitoring import TelemetryRecord  # noqa: E402

SLICE_NAME = {0: "eMBB", 1: "MTC", 2: "URLLC"}
SCHEDULER = {0: "round_robin", 1: "waterfilling", 2: "proportional_fair"}
CELL_PRB = 50
TTI_PER_S = 1000
# Sustained traffic: per-second slice DL rate above ACTIVE_FRACTION of the
# run's 95th percentile for ACTIVE_HOLD_S consecutive seconds.
ACTIVE_FRACTION, ACTIVE_HOLD_S = 0.2, 5
FILE_RE = re.compile(r"sched(\d)/tr(\d+)/(exp\d+)/bs(\d+)/slices_bs\d+/(\d+)_metrics\.csv$")
COLUMNS = list(TelemetryRecord.model_fields)

RAW = {"Timestamp": "ts_ms", "slice_id": "slice", "slice_prb": "slice_prb",
       "scheduling_policy": "sched", "tx_brate downlink [Mbps]": "rate",
       "dl_buffer [bytes]": "buf", "sum_granted_prbs": "granted", "dl_cqi": "cqi",
       "dl_mcs": "mcs", "dl_n_samples": "dl_n", "ul_sinr": "sinr", "ul_n_samples": "ul_n"}


def discover(root: Path, experiments: list[str] | None) -> dict[tuple, list[Path]]:
    """Group trace files by (scheduler, split, experiment, BS)."""
    groups: dict[tuple, list[Path]] = {}
    for f in sorted(root.rglob("*_metrics.csv")):
        m = FILE_RE.search(f.as_posix())
        if not m or (experiments and m.group(3) not in experiments):
            continue
        key = (int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4)))
        groups.setdefault(key, []).append(f)
    return groups


def _load_ue(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, usecols=lambda c: c in RAW).rename(columns=RAW)
    d["imsi"] = path.name.split("_")[0]
    # A zero means "no transmission in this report", not a measurement.
    d.loc[d.dl_n == 0, "mcs"] = np.nan
    d.loc[d.ul_n == 0, "sinr"] = np.nan
    return d


def active_window(per_s: pd.DataFrame) -> tuple[float, float] | None:
    """[start, end) in seconds where every slice (column) carries sustained traffic."""
    starts, ends = [], []
    for rate in per_s.T.to_numpy():
        on = rate > ACTIVE_FRACTION * np.quantile(rate, .95)
        held = np.convolve(on, np.ones(ACTIVE_HOLD_S, dtype=int), "valid") == ACTIVE_HOLD_S
        secs = per_s.index.to_numpy()[: len(held)][held]
        if len(secs) == 0:
            return None
        starts.append(secs.min())
        ends.append(secs.max() + ACTIVE_HOLD_S)
    start, end = max(starts), min(ends)
    return (float(start), float(end)) if end > start else None


def slice_windows(d: pd.DataFrame, slice_prb: pd.Series, window_s: float) -> pd.DataFrame:
    """Aggregate the reports (column w = window index) per (window, slice)."""
    ue = d.groupby(["w", "slice", "imsi"]).agg(rate=("rate", "mean"), buf=("buf", "mean"))
    cap_ms = window_s * 1e3
    with np.errstate(divide="ignore", invalid="ignore"):
        delay = np.where(ue.buf <= 0, 0.0,
                         np.where(ue.rate > 0, ue.buf * 8 / (ue.rate * 1e6) * 1e3, cap_ms))
    ue["delay"] = np.minimum(delay, cap_ms)
    out = ue.groupby(["w", "slice"]).agg(
        active_ues=("rate", "size"), dl_throughput_mbps=("rate", "sum"),
        dl_throughput_per_ue_mbps=("rate", "mean"), dl_buffer_bytes=("buf", "mean"),
        queueing_delay_ms=("delay", "mean"))
    rep = d.groupby(["w", "slice"]).agg(granted=("granted", "sum"), dl_cqi=("cqi", "mean"),
                                        dl_mcs=("mcs", "mean"), ul_sinr_db=("sinr", "mean"))
    out = out.join(rep)
    prb = slice_prb.reindex(out.index.get_level_values("slice")).to_numpy()
    out["slice_prb"] = prb
    out["prb_utilisation"] = np.minimum(out.pop("granted") / (prb * window_s * TTI_PER_S), 1.0)
    idle = (out.slice_prb * (1 - out.prb_utilisation)).groupby("w").transform("sum")
    out["cell_spare_prb"] = idle - out.slice_prb * (1 - out.prb_utilisation)   # other slices only
    complete = out.groupby("w").active_ues.transform("size") == 3              # all three slices
    return out[complete].reset_index()


def build_cell_run(key: tuple, files: list[Path], window_s: float) -> list[dict]:
    """All slice-window records of one (scheduler, split, experiment, BS)."""
    sched, tr, exp, bs = key
    d = pd.concat([_load_ue(f) for f in files], ignore_index=True)
    if set(d.slice.unique()) != {0, 1, 2}:
        return []
    d["t"] = (d.ts_ms - d.ts_ms.min()) / 1000.0

    per_s = d.assign(sec=d.t.astype(int)).groupby(["sec", "slice"]).rate.sum().unstack(fill_value=0.0)
    window = active_window(per_s)
    if window is None:
        return []
    start, end = window
    n_windows = int((end - start) // window_s)
    d = d[(d.t >= start) & (d.t < start + n_windows * window_s)]
    d["w"] = ((d.t - start) // window_s).astype(int)
    slice_prb = d.groupby("slice").slice_prb.first().astype(int)
    run = f"sched{sched}/tr{tr}/{exp}"

    out = slice_windows(d, slice_prb, window_s)
    out["window_start_s"] = start + out.w * window_s
    out["slice_id"] = out.slice.map(SLICE_NAME)
    out["record_id"] = [f"{run.replace('/', '-')}-bs{bs}-t{t:06.1f}-{s}"
                        for t, s in zip(out.window_start_s, out.slice_id)]
    out = out.assign(cell_id=f"bs{bs}", run=run, window_s=window_s, cell_prb=CELL_PRB,
                     scheduler=SCHEDULER[sched], slice_prb=out.slice_prb.astype(int),
                     active_ues=out.active_ues.astype(int))
    out = out[COLUMNS].round(6)
    out = out.astype(object).where(out.notna(), None)                    # NaN -> None
    return out.to_dict("records")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT / "data/raw/coloran/rome_static_medium")
    p.add_argument("--out", type=Path, default=ROOT / "data/coloran_prepared_dataset.csv")
    p.add_argument("--window", type=float, default=1.0, help="seconds per record (default 1)")
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

    rows: list[dict] = []
    for i, key in enumerate(keys, 1):
        rows += build_cell_run(key, groups[key], a.window)
        if i % 50 == 0:
            print(f"  {i}/{len(keys)} runs, {len(rows)} records", flush=True)
    if not rows:
        raise SystemExit("no complete records produced")

    for row in rows:                    # every row must be a valid TelemetryRecord
        TelemetryRecord.model_validate(row)
    df = pd.DataFrame(rows, columns=COLUMNS)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    kpis = ["dl_throughput_per_ue_mbps", "queueing_delay_ms", "prb_utilisation", "cell_spare_prb"]
    summary = {
        "records": int(len(df)), "cell_runs": len(keys), "window_s": a.window,
        "records_by_slice": df.slice_id.value_counts().to_dict(),
        "kpi_median_by_slice": df.groupby("slice_id")[kpis].median().round(4).to_dict("index"),
        "source": "ColO-RAN (Polese et al., IEEE TMC 2023), GPL-3.0",
    }
    a.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {a.out} ({len(df)} records)")
    print(json.dumps(summary["kpi_median_by_slice"], indent=2))


if __name__ == "__main__":
    main()
