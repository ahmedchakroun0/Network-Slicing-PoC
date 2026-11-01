"""Explore the raw ColO-RAN dataset (every column) before designing the
telemetry model fed to the Monitoring Agent.

Input : data/raw/coloran/rome_static_medium/
        sched{0,1,2}/tr{0..27}/exp*/bs{1..7}/slices_bs*/<imsi>_metrics.csv
        (one file per UE, one row per ~250 ms KPM report, 31 columns).
Output: a sectioned report on stdout, the tables behind it as CSV and the
        figures as PNG in data/exploration/ (git-ignored).

    poetry run python scripts/explore_coloran.py                 # all 84 configs
    poetry run python scripts/explore_coloran.py --max-configs 6 # quick look

Sections:
   1. Coverage      which (scheduler, RBG split, experiment, BS) runs exist
   2. Integrity     separator columns, IMSI vs file name, timestamp order
   3. Columns       profile of EVERY column and its role (id / config / telemetry)
   4. By slice      every column per slice
   5. Topology      IMSIs, slices, RNTIs and UE counts vs the documentation
   6. Time base     sampling period, trace length, clock alignment, gaps
   7. Config        slice_prb / scheduling_policy / power vs directory names
   8. Start-up      when each slice starts carrying traffic
   9. Correlation   rank correlation between all varying columns
  10. Response      every telemetry column vs PRB share and scheduler
  11. Per BS        radio conditions per base station
  12. Delay         is dl_buffer / DL rate a usable latency proxy?
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data/raw/coloran/rome_static_medium"
DEFAULT_OUT = ROOT / "data/exploration"

FILE_RE = re.compile(r"sched(\d)/tr(\d+)/(exp\d+)/bs(\d+)/slices_bs\d+/(\d+)_metrics\.csv$")

# From the dataset README.
SLICE_NAME = {0: "eMBB", 1: "MTC", 2: "URLLC"}
SCHEDULER = {0: "RR", 1: "WF", 2: "PF"}
DOC_UES_BY_SLICE = {
    0: {3, 6, 10, 13, 17, 20, 24, 27, 31, 34, 38, 41, 45, 48},
    1: {4, 7, 11, 14, 18, 21, 25, 28, 32, 35, 39, 42, 46, 49},
    2: {2, 5, 9, 12, 16, 19, 23, 26, 30, 33, 37, 40, 44, 47},
}
TR_RBG = [(2, 13, 2), (4, 11, 2), (6, 9, 2), (8, 7, 2), (10, 5, 2), (12, 3, 2), (14, 1, 2),
          (2, 11, 4), (4, 9, 4), (6, 7, 4), (8, 5, 4), (10, 3, 4), (12, 1, 4),
          (2, 9, 6), (4, 7, 6), (6, 5, 6), (8, 3, 6), (10, 1, 6),
          (2, 7, 8), (4, 5, 8), (6, 3, 8), (8, 1, 8),
          (2, 5, 10), (4, 3, 10), (6, 1, 10),
          (2, 3, 12), (4, 1, 12), (2, 1, 14)]
OFFERED_MBPS_PER_UE = {0: 4.0, 1: 30 * 125 * 8 / 1e6, 2: 10 * 125 * 8 / 1e6}
TTI_PER_S = 1000
# UL columns are 0 whenever the UE had no UL grant in the report: their
# statistics are taken only over reports with UL activity (ul_n_samples > 0).
UL_COLS = ["ul_mcs", "ul_buffer", "ul_mbps", "ul_pkts", "ul_err_pct", "ul_sinr", "phr", "ul_turbo_iters"]
# Active window: traffic is "sustained" when a slice's per-second DL rate stays
# above ACTIVE_FRACTION of its run's 95th percentile for ACTIVE_HOLD_S seconds.
ACTIVE_FRACTION, ACTIVE_HOLD_S = 0.2, 5

# Short names for the 31 named raw columns. The header also has 5 unnamed
# separator columns (",,"); section 2 checks that they are really empty.
RENAME = {
    "Timestamp": "ts_ms", "num_ues": "num_ues", "IMSI": "imsi", "RNTI": "rnti",
    "slicing_enabled": "slicing_enabled", "slice_id": "slice_id",
    "slice_prb": "slice_prb", "power_multiplier": "power_multiplier",
    "scheduling_policy": "scheduling_policy",
    "dl_mcs": "dl_mcs", "dl_n_samples": "dl_n_samples", "dl_buffer [bytes]": "dl_buffer",
    "tx_brate downlink [Mbps]": "dl_mbps", "tx_pkts downlink": "dl_pkts",
    "tx_errors downlink (%)": "dl_err_pct", "dl_cqi": "dl_cqi",
    "ul_mcs": "ul_mcs", "ul_n_samples": "ul_n_samples", "ul_buffer [bytes]": "ul_buffer",
    "rx_brate uplink [Mbps]": "ul_mbps", "rx_pkts uplink": "ul_pkts",
    "rx_errors uplink (%)": "ul_err_pct", "ul_rssi": "ul_rssi", "ul_sinr": "ul_sinr",
    "phr": "phr", "sum_requested_prbs": "req_prbs", "sum_granted_prbs": "granted_prbs",
    "dl_pmi": "dl_pmi", "dl_ri": "dl_ri", "ul_n": "ul_n", "ul_turbo_iters": "ul_turbo_iters",
}
COLUMNS = [c for c in RENAME.values() if c != "ts_ms"]
RUN_KEY = ["sched", "tr", "exp", "bs"]
TRACE_KEY = RUN_KEY + ["file_imsi"]

# Chart tokens (dataviz reference palette, light surface). Three categorical
# slots: they stay distinguishable for colour-blind readers in every pairing.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 9, "axes.labelsize": 8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 8, "legend.frameon": False, "lines.linewidth": 1.5, "font.size": 8,
})


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show(df: pd.DataFrame | pd.Series, out_dir: Path, name: str) -> None:
    """Print a table and save it next to the report."""
    with pd.option_context("display.width", 200, "display.max_columns", 60,
                           "display.max_rows", 200, "display.float_format", "{:,.4g}".format):
        print(df)
    df.to_csv(out_dir / f"{name}.csv")


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / "figures" / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path.relative_to(ROOT)}")


def grid(n: int, ncols: int = 5, w: float = 3.0, h: float = 2.2):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * w, nrows * h), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def slice_legend(fig: plt.Figure, labels: list[str], title: str | None = None) -> None:
    handles = [plt.Line2D([], [], color=SERIES[i], lw=2) for i in range(len(labels))]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), title=title,
               bbox_to_anchor=(0.5, 1.0))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def discover(raw: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(raw.rglob("*_metrics.csv")):
        m = FILE_RE.search(f.as_posix())
        if m:
            rows.append(dict(sched=int(m[1]), tr=int(m[2]), exp=m[3], bs=int(m[4]),
                             imsi=int(m[5]), path=f))
    return pd.DataFrame(rows)


def load(files: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All UE traces in one long frame (every named column, float32), plus
    per-file integrity facts."""
    frames, checks = [], []
    for i, f in enumerate(files.itertuples(), 1):
        d = pd.read_csv(f.path)
        sep = d.columns[d.columns.str.startswith("Unnamed")]
        d = d.drop(columns=sep).rename(columns=RENAME)
        ts = d.pop("ts_ms").astype("int64")
        imsi = d.pop("imsi").astype("int64")   # 13 digits: float32 would round it
        checks.append(dict(
            path=f.path.relative_to(DEFAULT_RAW).as_posix() if f.path.is_relative_to(DEFAULT_RAW) else str(f.path),
            rows=len(d), separator_cols=len(sep), separator_non_null=int(d.shape[0] and 0),
            imsi_matches_file=bool((imsi == f.imsi).all()),
            ts_monotonic=bool(ts.is_monotonic_increasing), ts_duplicates=int(ts.duplicated().sum()),
            nan_cells=int(d.isna().sum().sum())))
        sep_non_null = int(pd.read_csv(f.path, usecols=list(sep)).notna().sum().sum()) if len(sep) else 0
        checks[-1]["separator_non_null"] = sep_non_null
        d = d.astype("float32")
        d.insert(0, "ts_ms", ts)
        d.insert(1, "imsi", imsi)
        d["sched"], d["tr"], d["bs"], d["file_imsi"] = f.sched, f.tr, f.bs, f.imsi
        d["exp"] = f.exp
        frames.append(d)
        if i % 500 == 0:
            print(f"  loaded {i}/{len(files)} files", flush=True)
    df = pd.concat(frames, ignore_index=True)
    for col in ("sched", "tr", "bs"):
        df[col] = df[col].astype("int8")
    df["exp"] = df["exp"].astype("category")
    df["ue"] = (df["file_imsi"] % 1000).astype("int16")   # 1010123456028 -> UE 28
    run_t0 = df.groupby(RUN_KEY, observed=True).ts_ms.transform("min")
    df["t_s"] = ((df.ts_ms - run_t0) / 1000).astype("float32")
    return df, pd.DataFrame(checks)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def coverage(files: pd.DataFrame, out: Path) -> None:
    section("1. COVERAGE - files per scheduler / RBG split / BS")
    print(f"files: {len(files)}   configs (sched x tr x exp): "
          f"{files.groupby(['sched', 'tr', 'exp']).ngroups}   "
          f"cell runs (x bs): {files.groupby(RUN_KEY).ngroups}")
    per_cfg = files.groupby(["sched", "tr", "exp"]).agg(bs=("bs", "nunique"), ues=("imsi", "nunique"))
    print("\nBSs and UE files per config (how many configs have each combination):")
    show(per_cfg.value_counts().rename("configs").to_frame(), out, "01_coverage_per_config")
    print("\nUE files per BS over all configs:")
    show(files.groupby("bs").imsi.agg(files="count", distinct_ues="nunique"), out, "01_coverage_per_bs")


def integrity(checks: pd.DataFrame, out: Path) -> None:
    section("2. INTEGRITY - per-file checks")
    summary = pd.Series({
        "files": len(checks),
        "rows (total)": int(checks.rows.sum()),
        "rows per file (min / median / max)": f"{checks.rows.min()} / {int(checks.rows.median())} / {checks.rows.max()}",
        "separator columns per file": sorted(checks.separator_cols.unique().tolist()),
        "files with data in a separator column": int((checks.separator_non_null > 0).sum()),
        "files whose IMSI column != file name": int((~checks.imsi_matches_file).sum()),
        "files with non-monotonic timestamps": int((~checks.ts_monotonic).sum()),
        "files with duplicate timestamps": int((checks.ts_duplicates > 0).sum()),
        "files with NaN cells": int((checks.nan_cells > 0).sum()),
    }, name="value")
    show(summary.to_frame(), out, "02_integrity_summary")
    checks.to_csv(out / "02_integrity_per_file.csv", index=False)
    bad = checks[(checks.separator_non_null > 0) | ~checks.imsi_matches_file | ~checks.ts_monotonic
                 | (checks.ts_duplicates > 0) | (checks.nan_cells > 0)]
    if len(bad):
        print("\nfiles with at least one problem (first 20):")
        show(bad.head(20).set_index("path"), out, "02_integrity_problems")


def column_profile(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    section("3. COLUMNS - profile of every column")
    per_trace = df.groupby(TRACE_KEY, observed=True)[COLUMNS].nunique()
    per_run = df.groupby(RUN_KEY + ["slice_id"], observed=True)[COLUMNS].nunique()
    rows = []
    for c in COLUMNS:
        s = df[c]
        q = s.quantile([0, .01, .5, .99, 1]).to_numpy()
        finite = s.dropna()
        rows.append(dict(
            column=c, null_pct=100 * s.isna().mean(), zero_pct=100 * (s == 0).mean(),
            neg_pct=100 * (s < 0).mean(), n_unique=s.nunique(),
            integer_valued=bool((finite == np.round(finite)).all()),
            min=q[0], p01=q[1], median=q[2], p99=q[3], max=q[4], mean=s.mean(), std=s.std(),
            const_in_trace_pct=100 * (per_trace[c] <= 1).mean(),
            const_in_run_slice_pct=100 * (per_run[c] <= 1).mean()))
    t = pd.DataFrame(rows).set_index("column")

    def role(r: pd.Series) -> str:
        if r.n_unique <= 1:
            return "constant"
        if r.name in ("imsi", "rnti", "num_ues"):
            return "identifier/topology"
        if r.const_in_run_slice_pct >= 99:
            return "configuration"
        if r.zero_pct >= 99:
            return "telemetry (almost always 0)"
        return "telemetry"

    t["role"] = t.apply(role, axis=1)
    show(t, out, "03_column_profile")
    print("\ncolumns by role:")
    for r, cols in t.groupby("role").groups.items():
        print(f"  {r:30s} {list(cols)}")

    print("\nmost frequent values of low-cardinality columns (<= 12 distinct):")
    for c in t.index[t.n_unique <= 12]:
        vc = df[c].value_counts(normalize=True, dropna=False).mul(100).round(2)
        print(f"  {c:18s} {dict(vc)}")
    return t


def by_slice(df: pd.DataFrame, profile: pd.DataFrame, out: Path) -> None:
    section("4. BY SLICE - every varying column per slice (all reports)")
    varying = profile.index[profile.role.str.startswith("telemetry") | (profile.role == "configuration")]
    g = df.groupby("slice_id")[list(varying)]
    t = pd.concat({"median": g.median(), "p95": g.quantile(.95),
                   "zero%": g.agg(lambda s: 100 * (s == 0).mean())}, axis=1)
    t.index = [SLICE_NAME[int(i)] for i in t.index]
    show(t.T.sort_index(), out, "04_by_slice")

    tele = list(profile.index[profile.role.str.startswith("telemetry")])
    fig, axes = grid(len(tele))
    for ax, c in zip(axes, tele):
        lo, hi = df[c].quantile([.005, .995])
        bins = np.linspace(lo, hi if hi > lo else lo + 1, 40)
        for i, (sl, name) in enumerate(SLICE_NAME.items()):
            v = df.loc[df.slice_id == sl, c].clip(lo, hi)
            ax.hist(v, bins=bins, histtype="step", density=True, color=SERIES[i], lw=1.5)
        ax.set_title(f"{c}  (zeros {profile.at[c, 'zero_pct']:.0f}%)")
        ax.set_yscale("log")
        ax.set_yticklabels([])
    slice_legend(fig, list(SLICE_NAME.values()), "slice (density, log scale, 0.5-99.5 pct range)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "04_distributions_by_slice")


def topology(df: pd.DataFrame, out: Path) -> None:
    section("5. TOPOLOGY - UE / slice / BS mapping against the documentation")
    ue_slice = df.groupby("ue").slice_id.agg(lambda s: sorted(s.dropna().unique().astype(int)))
    mismatch = []
    for ue, slices in ue_slice.items():
        doc = next((k for k, v in DOC_UES_BY_SLICE.items() if ue in v), None)
        if slices != [doc]:
            mismatch.append(dict(ue=ue, observed_slices=slices, documented_slice=doc))
    print(f"UEs seen: {len(ue_slice)}   UEs whose slice_id differs from the README: {len(mismatch)}")
    if mismatch:
        show(pd.DataFrame(mismatch).set_index("ue"), out, "05_topology_slice_mismatch")
    ue_bs = df.groupby("ue").bs.unique()
    print(f"UEs seen at more than one BS across configs: {int((ue_bs.str.len() > 1).sum())}")

    print("\nUEs per BS and slice:")
    t = (df.groupby(["bs", "slice_id"]).ue.unique()
         .apply(lambda u: ",".join(map(str, sorted(u)))).unstack())
    t.columns = [SLICE_NAME[int(c)] for c in t.columns]
    show(t, out, "05_topology_ues_per_bs_slice")

    print("\nnum_ues (reported by the BS) vs UE files present, per cell run:")
    files_per_run = df.groupby(RUN_KEY, observed=True).ue.nunique().rename("ue_files")
    rep = df.groupby(RUN_KEY, observed=True).num_ues.agg(num_ues_min="min", num_ues_max="max")
    show(pd.concat([files_per_run, rep], axis=1).value_counts().rename("cell_runs").to_frame(),
         out, "05_topology_num_ues")
    rnti = df.groupby(TRACE_KEY, observed=True).rnti.nunique()
    print(f"\nUE traces whose RNTI changes during the trace (re-attach): {int((rnti > 1).sum())} / {len(rnti)}")


def time_base(df: pd.DataFrame, out: Path) -> None:
    section("6. TIME BASE - sampling period, trace length, alignment, gaps")
    dt = df.groupby(TRACE_KEY, observed=True).ts_ms.diff().dropna()
    print("period between consecutive reports of one UE (ms):")
    show(dt.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).rename("dt_ms"), out, "06_time_dt")
    print(f"\ngaps > 1 s: {100 * (dt > 1000).mean():.3f} %   > 5 s: {100 * (dt > 5000).mean():.3f} %   "
          f"dt <= 0: {100 * (dt <= 0).mean():.3f} %")

    trace = df.groupby(TRACE_KEY, observed=True).ts_ms.agg(["min", "max", "count"])
    trace["duration_s"] = (trace["max"] - trace["min"]) / 1000
    print("\nUE trace duration (s) and reports per trace:")
    show(trace[["duration_s", "count"]].describe(), out, "06_time_trace_length")

    run = trace.groupby(RUN_KEY, observed=True).agg(
        ue_start_spread_s=("min", lambda s: (s.max() - s.min()) / 1000),
        ue_end_spread_s=("max", lambda s: (s.max() - s.min()) / 1000))
    print("\nhow far apart the UEs of one cell run start / end (s):")
    show(run.describe(), out, "06_time_ue_alignment")
    cfg = trace.groupby(["sched", "tr", "exp"], observed=True)["min"].agg(lambda s: (s.max() - s.min()) / 1000)
    print("\nhow far apart the 7 BSs of one config start (s) - same experiment clock?")
    show(cfg.describe().rename("bs_start_spread_s"), out, "06_time_bs_alignment")
    first = pd.to_datetime(df.ts_ms.min(), unit="ms")
    last = pd.to_datetime(df.ts_ms.max(), unit="ms")
    print(f"\nwall-clock span of the whole dataset: {first} -> {last}")


def configuration(df: pd.DataFrame, out: Path) -> None:
    section("7. CONFIG - logged control state vs directory names")
    pol = df.groupby("sched").scheduling_policy.value_counts(normalize=True).mul(100).rename("share %")
    print("scheduling_policy logged per sched directory:")
    show(pol.to_frame(), out, "07_config_scheduling_policy")

    obs = df.groupby(["tr", "slice_id"]).slice_prb.agg(lambda s: s.mode().iloc[0]).unstack()
    obs.columns = [SLICE_NAME[int(c)] for c in obs.columns]
    doc = pd.DataFrame(TR_RBG, columns=list(SLICE_NAME.values())).rename_axis("tr")
    print("\nslice_prb logged (mode) vs README RBGs per slice, and PRBs per RBG:")
    t = pd.concat({"slice_prb": obs, "rbg_doc": doc, "prb_per_rbg": (obs / doc).round(2)}, axis=1)
    t[("slice_prb", "sum")] = obs.sum(axis=1)
    show(t, out, "07_config_slice_prb")

    within = df.groupby(RUN_KEY + ["slice_id"], observed=True)[
        ["slice_prb", "scheduling_policy", "power_multiplier", "slicing_enabled"]].nunique()
    print("\ncell-run slices in which a configuration value changes during the run:")
    show((within > 1).sum().rename("runs").to_frame(), out, "07_config_changes_in_run")


def _sustained_window(rate: pd.Series) -> pd.Series:
    """First and last second of sustained traffic in one slice's per-second rate."""
    on = (rate > ACTIVE_FRACTION * rate.quantile(.95)).astype(int)
    held = on.rolling(ACTIVE_HOLD_S).min().shift(-(ACTIVE_HOLD_S - 1)) == 1   # on for the next N s
    secs = rate.index.get_level_values("sec")[held.to_numpy()]
    if len(secs) == 0:
        return pd.Series({"start_s": np.nan, "end_s": np.nan})
    return pd.Series({"start_s": secs.min(), "end_s": secs.max() + ACTIVE_HOLD_S})


def startup(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Return, per cell run, the window in which every slice carries sustained traffic."""
    section("8. ACTIVE WINDOW - when does sustained traffic start and stop?")
    first_blip = (df[df.dl_mbps > 0].groupby(RUN_KEY + ["slice_id"], observed=True).t_s.min().unstack())
    first_blip.columns = [SLICE_NAME[int(c)] for c in first_blip.columns]
    print("first non-zero DL rate (s) - attach-time blips, NOT the traffic start:")
    show(first_blip.describe(), out, "08_first_nonzero_rate")

    per_s = (df.assign(sec=df.t_s.astype("int32"))
             .groupby(RUN_KEY + ["slice_id", "sec"], observed=True).dl_mbps.sum())
    win = per_s.groupby(RUN_KEY + ["slice_id"], observed=True).apply(_sustained_window).unstack()
    print(f"\nsustained traffic (> {ACTIVE_FRACTION:.0%} of the run's p95 rate for "
          f"{ACTIVE_HOLD_S} s) per slice, seconds after the first report:")
    t = win.unstack("slice_id")
    t.columns = [f"{a}_{SLICE_NAME[int(s)]}" for a, s in t.columns]
    show(t.describe(), out, "08_active_window_per_slice")

    run = pd.DataFrame({"active_from_s": win.start_s.groupby(RUN_KEY, observed=True).max(),
                        "active_to_s": win.end_s.groupby(RUN_KEY, observed=True).min()})
    run["active_s"] = run.active_to_s - run.active_from_s
    trace_end = df.groupby(RUN_KEY, observed=True).t_s.max()
    run["idle_tail_s"] = trace_end - run.active_to_s
    print("\ncell-run active window (all three slices sustained):")
    show(run.describe(), out, "08_active_window_per_run")
    print(f"\ncell runs without a common active window: {int((run.active_s.isna() | (run.active_s <= 0)).sum())}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, col, label in ((axes[0], "start_s", "start of sustained traffic"),
                           (axes[1], "end_s", "end of sustained traffic")):
        v = win[col].unstack("slice_id")
        bins = np.linspace(np.nanmin(v.to_numpy()), np.nanmax(v.to_numpy()) + 1, 40)
        for i, sl in enumerate(v.columns):
            ax.hist(v[sl].dropna(), bins=bins, histtype="step", color=SERIES[i], lw=1.5,
                    label=SLICE_NAME[int(sl)])
        ax.set_xlabel(f"{label} (s after the first report)")
        ax.set_ylabel("cell runs")
    axes[0].legend()
    fig.suptitle("Sustained-traffic window per slice", color=INK)
    fig.tight_layout()
    save(fig, out, "08_active_window")
    return run[["active_from_s", "active_to_s"]]


def example_run(df: pd.DataFrame, profile: pd.DataFrame, out: Path) -> None:
    """Every telemetry column over time for one cell run (slice means per second)."""
    key = df[RUN_KEY].drop_duplicates().sort_values(RUN_KEY).iloc[0]
    d = df[(df[RUN_KEY] == key.values).all(axis=1)]
    tele = list(profile.index[profile.role.str.startswith("telemetry")])
    per_s = d.assign(sec=d.t_s.astype(int)).groupby(["slice_id", "sec"])[tele].mean()
    fig, axes = grid(len(tele), ncols=4, w=3.6)
    for ax, c in zip(axes, tele):
        for i, sl in enumerate(SLICE_NAME):
            if sl in per_s.index.get_level_values(0):
                s = per_s.loc[sl, c]
                ax.plot(s.index, s.values, color=SERIES[i], lw=1)
        ax.set_title(c)
    slice_legend(fig, list(SLICE_NAME.values()),
                 f"one cell run: sched{key.sched}/tr{key.tr}/{key.exp}/bs{key.bs} "
                 f"(mean per slice per second; x = seconds)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "08_example_run")


def ul_active(df: pd.DataFrame, out: Path) -> None:
    section("8b. UPLINK - UL columns over reports that carry UL (active window)")
    act = df[df.active]
    print(f"active reports with UL activity: {100 * act.ul_n_samples.gt(0).mean():.1f} %")
    t = act.groupby("slice_id")[UL_COLS].quantile([.05, .5, .95]).unstack()
    t.index = [SLICE_NAME[int(i)] for i in t.index]
    show(t.T, out, "08b_uplink_quantiles")


def correlation(df: pd.DataFrame, profile: pd.DataFrame, out: Path) -> None:
    section("9. CORRELATION - Spearman rank correlation (active reports, sampled; UL only where reported)")
    cols = list(profile.index[profile.role != "constant"].drop(["imsi", "rnti"], errors="ignore"))
    sample = df.loc[df.active, cols].sample(n=min(400_000, int(df.active.sum())), random_state=0)
    corr = sample.rank().corr()
    show(corr, out, "09_correlation")
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    print("\nstrongest pairs (|rho| >= 0.7) - candidates for redundancy:")
    show(pairs[pairs.abs() >= .7].sort_values(key=abs, ascending=False).rename("rho").to_frame(),
         out, "09_correlation_strong_pairs")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90)
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(corr.to_numpy()):
        if i != j and abs(v) >= .5:
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.5, color=INK)
    fig.colorbar(im, ax=ax, shrink=.7, label="Spearman rho")
    ax.set_title("Rank correlation between columns (labels where |rho| >= 0.5)")
    save(fig, out, "09_correlation")


def response(df: pd.DataFrame, profile: pd.DataFrame, out: Path) -> None:
    section("10. RESPONSE - every telemetry column vs slice PRB share and scheduler")
    tele = list(profile.index[profile.role.str.startswith("telemetry")])
    act = df[df.active]
    run_slice = act.groupby(RUN_KEY + ["slice_id"], observed=True)[tele + ["slice_prb"]].mean().reset_index()
    run_slice["slice_prb"] = run_slice.slice_prb.round()
    table = run_slice.groupby(["slice_id", "slice_prb", "sched"])[tele].median()
    table.index = table.index.set_levels([[SLICE_NAME[int(i)] for i in table.index.levels[0]],
                                          table.index.levels[1],
                                          [SCHEDULER[int(i)] for i in table.index.levels[2]]])
    table.to_csv(out / "10_response_table.csv")
    print("per-slice medians over cell runs (full table in 10_response_table.csv); DL rate per UE:")
    show(table["dl_mbps"].unstack(), out, "10_response_dl_mbps")

    print("\nhow much each column moves with the PRB share (Spearman rho over cell-run slices):")
    sens = run_slice.groupby("slice_id").apply(
        lambda g: g[tele].rank().corrwith(g.slice_prb.rank()), include_groups=False).T
    sens.columns = [SLICE_NAME[int(c)] for c in sens.columns]
    show(sens.sort_values("eMBB", key=abs, ascending=False), out, "10_response_sensitivity_prb")
    print("\nhow much each column moves with the scheduler (Kruskal-style ratio: "
          "spread of scheduler medians / overall IQR, per slice):")
    rows = {}
    for sl, name in SLICE_NAME.items():
        g = run_slice[run_slice.slice_id == sl]
        iqr = g[tele].quantile(.75) - g[tele].quantile(.25)
        med = g.groupby("sched")[tele].median()
        rows[name] = (med.max() - med.min()) / iqr.replace(0, np.nan)
    show(pd.DataFrame(rows).sort_values("eMBB", ascending=False), out, "10_response_sensitivity_sched")

    for sl, name in SLICE_NAME.items():
        fig, axes = grid(len(tele))
        g = run_slice[run_slice.slice_id == sl]
        for ax, c in zip(axes, tele):
            m = g.groupby(["slice_prb", "sched"])[c].median().unstack()
            for i, s in enumerate(m.columns):
                ax.plot(m.index, m[s], color=SERIES[i], marker="o", ms=3)
            if c == "dl_mbps":
                ax.axhline(OFFERED_MBPS_PER_UE[sl], color=MUTED, lw=1, ls="--")
                ax.annotate("offered", (m.index.max(), OFFERED_MBPS_PER_UE[sl]), color=MUTED,
                            fontsize=7, ha="right", va="bottom")
            ax.set_title(c)
        slice_legend(fig, [f"{SCHEDULER[s]}" for s in sorted(SCHEDULER)],
                     f"{name} slice: median over cell runs vs slice_prb (x), per scheduler")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save(fig, out, f"10_response_{name}")


def utilisation(df: pd.DataFrame, out: Path) -> None:
    """granted_prbs is per UE, summed over the TTIs of one report, so the slice
    PRB utilisation of one second is sum(granted) / (slice_prb * 1000 TTIs)."""
    section("10b. UTILISATION - share of the slice's PRBs actually granted")
    act = df[df.active]
    per_s = (act.assign(sec=act.t_s.astype("int32"))
             .groupby(RUN_KEY + ["slice_id", "sec"], observed=True)
             .agg(granted=("granted_prbs", "sum"), slice_prb=("slice_prb", "first")))
    per_s["util"] = per_s.granted / (per_s.slice_prb * TTI_PER_S)
    print("per-second slice utilisation, all active seconds:")
    q = per_s.groupby("slice_id").util.quantile([.05, .5, .95, .99]).unstack()
    q.index = [SLICE_NAME[int(i)] for i in q.index]
    show(q, out, "10b_utilisation_quantiles")
    print(f"\nseconds with utilisation > 1.05 (check of the per-UE / per-TTI reading): "
          f"{100 * (per_s.util > 1.05).mean():.2f} %")

    run = per_s.groupby(RUN_KEY + ["slice_id"], observed=True)[["util", "slice_prb"]].mean().reset_index()
    t = run.groupby(["slice_id", "slice_prb", "sched"]).util.median().unstack()
    t.columns = [SCHEDULER[int(c)] for c in t.columns]
    show(t, out, "10b_utilisation_by_prb_sched")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for ax, (sl, name) in zip(axes, SLICE_NAME.items()):
        m = run[run.slice_id == sl].groupby(["slice_prb", "sched"]).util.median().unstack()
        for i, s in enumerate(m.columns):
            ax.plot(m.index, m[s], color=SERIES[i], marker="o", ms=3, label=SCHEDULER[int(s)])
        ax.set_title(name)
        ax.set_xlabel("slice_prb")
    axes[0].set_ylabel("PRB utilisation (median over cell runs)")
    axes[0].legend()
    fig.suptitle("Share of the slice's PRBs granted, by slice size and scheduler", color=INK)
    fig.tight_layout()
    save(fig, out, "10b_utilisation")


def per_bs(df: pd.DataFrame, out: Path) -> None:
    section("11. PER BS - radio conditions per base station (active reports)")
    cols = ["dl_cqi", "dl_mcs", "ul_sinr", "ul_rssi", "phr", "dl_err_pct", "ul_err_pct", "dl_mbps"]
    act = df[df.active]
    t = act.groupby("bs")[cols].median()
    show(t, out, "11_per_bs_median")
    ue = act.groupby(["bs", "ue"])[["dl_cqi", "ul_sinr"]].median()
    print("\nper-UE median CQI / SINR (static UEs: is each UE's channel fixed?):")
    show(ue.unstack(0).round(1), out, "11_per_ue_radio")

    fig, axes = grid(len(cols), ncols=4, w=3.2)
    for ax, c in zip(axes, cols):
        values = [act.loc[act.bs == b, c].dropna() for b in sorted(act.bs.unique())]
        data = [v.sample(n=min(20000, len(v)), random_state=0) for v in values]
        ax.boxplot(data, showfliers=False, widths=.5,
                   medianprops=dict(color=SERIES[0], lw=2), boxprops=dict(color=INK_2),
                   whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        ax.set_xticks(range(1, len(data) + 1), [f"bs{b}" for b in sorted(act.bs.unique())])
        ax.set_title(c)
    fig.suptitle("Radio and error KPIs per base station (active reports, no outliers)", color=INK)
    fig.tight_layout()
    save(fig, out, "11_per_bs")


def delay_proxy(df: pd.DataFrame, out: Path) -> None:
    section("12. DELAY - dl_buffer / DL rate as a queueing-delay proxy")
    act = df[df.active]
    undefined = (act.dl_buffer > 0) & (act.dl_mbps == 0)
    t = pd.DataFrame({
        "buffer>0 & rate==0 %": undefined.groupby(act.slice_id).mean() * 100,
        "buffer==0 %": (act.dl_buffer == 0).groupby(act.slice_id).mean() * 100,
    })
    ok = act[(act.dl_buffer > 0) & (act.dl_mbps > 0)]
    delay_ms = ok.dl_buffer * 8 / (ok.dl_mbps * 1e6) * 1e3
    q = delay_ms.groupby(ok.slice_id).quantile([.5, .9, .99]).unstack()
    q.columns = [f"delay p{int(c * 100)} ms" for c in q.columns]
    t = t.join(q)
    t.index = [SLICE_NAME[int(i)] for i in t.index]
    show(t, out, "12_delay_proxy")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-configs", type=int, default=None,
                   help="only load N random (scheduler, split, experiment) configs")
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()

    if not a.root.is_dir():
        raise SystemExit(f"{a.root} not found: run `bash scripts/download_coloran.sh` first.")
    (a.out / "figures").mkdir(parents=True, exist_ok=True)

    files = discover(a.root)
    coverage(files, a.out)
    if a.max_configs is not None:
        cfgs = files[["sched", "tr", "exp"]].drop_duplicates()
        files = files.merge(cfgs.sample(n=min(a.max_configs, len(cfgs)), random_state=a.seed))
    print(f"\nloading {len(files)} UE traces ...")
    df, checks = load(files)
    print(f"{len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e6:,.0f} MB in memory")

    integrity(checks, a.out)
    profile = column_profile(df, a.out)
    by_slice(df, profile, a.out)
    topology(df, a.out)
    time_base(df, a.out)
    configuration(df, a.out)
    window = startup(df, a.out)
    df = df.merge(window.reset_index(), on=RUN_KEY, how="left")
    df["active"] = (df.t_s >= df.active_from_s) & (df.t_s <= df.active_to_s)
    print(f"\nreports inside the active window: {100 * df.active.mean():.1f} %")
    example_run(df, profile, a.out)

    # From here on, UL zeros mean "no UL grant in this report", not a measurement.
    df[UL_COLS] = df[UL_COLS].where(df.ul_n_samples > 0)
    ul_active(df, a.out)
    correlation(df, profile, a.out)
    response(df, profile, a.out)
    utilisation(df, a.out)
    per_bs(df, a.out)
    delay_proxy(df, a.out)
    print(f"\ntables and figures written to {a.out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
