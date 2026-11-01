"""Regenerate every data figure of the paper from data/*.csv|json.

Usage:  python plot_all.py         (reads ../data, writes ../figures/*.pdf)
Only matplotlib/numpy/pandas are required; no PoC import.
"""

from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA_DIR, FIG_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)
# Times-metric TrueType face (embeds cleanly as Type 42 for IEEE PDF eXpress).
for _f in mpl.font_manager.findSystemFonts():
    if "LiberationSerif" in _f:
        mpl.font_manager.fontManager.addfont(_f)

# Validated categorical slots (light mode) + neutral ink; see README.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COL_W, FULL_W = 3.45, 7.0

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

KPI_LABEL = {"latency": "Latency", "jitter": "Jitter", "packet_loss": "Pkt. loss", "throughput": "Throughput"}


def fig_sla_grounding() -> None:
    s = json.loads((DATA_DIR / "e1_summary.json").read_text())
    cc = pd.read_csv(DATA_DIR / "e1_persistence_ccdf.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.75), gridspec_kw={"width_ratios": [1.1, 1]})
    kpis = list(KPI_LABEL)
    short = {"latency": "Lat.", "jitter": "Jitter", "packet_loss": "Loss", "throughput": "Thr."}
    x = np.arange(len(kpis))
    w = 0.38
    v1 = [100 * s["s1_kpi_prevalence"][k] for k in kpis]
    v2 = [100 * s["s2_kpi_prevalence"][k] for k in kpis]
    vu = [100 * s["s2_kpi_unavailable"][k] for k in kpis]
    h1 = a.bar(x - w / 2 - 0.02, v1, w, color=C1, zorder=3)
    h2 = a.bar(x + w / 2 + 0.02, v2, w, color=C2, zorder=3)
    h3 = a.bar(x + w / 2 + 0.02, vu, w, bottom=v2, color="white", edgecolor=C2, hatch="//////", lw=0.6, zorder=3)
    a.set_xticks(x, [short[k] for k in kpis])
    a.set_ylabel("Records (%)")
    a.set_ylim(0, 102)
    a.set_title("(a) KPI status", loc="left")
    for sem, col in (("s1", C1), ("s2", C2)):
        g = cc[(cc.semantics == sem) & (cc.kpi == "any")]
        b.plot(g.k, 100 * g.frac_ge_k, color=col, marker="o", ms=2.4)
    b.axvline(3, color=INK2, lw=0.7, ls="--")
    b.text(3.4, 62, "$k{=}3$", color=INK2, fontsize=7)
    b.set_xlabel("Persistence threshold $k$")
    b.set_ylabel(r"Records, $\rho\geq k$ (%)")
    b.set_xlim(1, 20)
    b.set_ylim(0, 102)
    b.set_xticks([1, 5, 10, 15, 20])
    b.set_title("(b) Non-RT eligibility", loc="left")
    fig.legend([h1, h2, h3], ["S1: literal budgets (violation)", "S2: zero = unspecified (violation)",
                              "S2: KPI unavailable"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.13), handlelength=1.3, columnspacing=1.0)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG_DIR / "fig_sla_grounding.pdf")
    plt.close(fig)


FAULT_ORDER = ["renamed_arg", "wrong_type", "extra_arg", "empty_rationale", "bad_confidence",
               "target_spoof", "slice_spoof", "missing_context", "no_violation",
               "out_of_range", "bad_enum", "inverted_policy", "injection_oor", "cross_role", "valid"]
FAULT_SHORT = {"renamed_arg": "Hallucinated param. name", "wrong_type": "Wrong arg. type",
               "extra_arg": "Unexpected extra arg.", "empty_rationale": "Empty rationale",
               "bad_confidence": "Confidence $\\notin[0,1]$", "target_spoof": "Untrusted target",
               "slice_spoof": "Untrusted slice_id", "missing_context": "No trusted context",
               "no_violation": "No active SLA violation", "out_of_range": "Out-of-range param.",
               "bad_enum": "Disallowed mode", "inverted_policy": "min PRB > max PRB",
               "injection_oor": "Injection + out-of-range", "cross_role": "Cross-role tool call",
               "valid": "Valid (false-rejection)"}
GUARDS = [("G0", "G0\nnone"), ("G1", "G1\nschema"), ("G2", "G2\nctx-free"), ("G3", "G3\nours"),
          ("G3_no_policy_trust", "$-$trust"), ("G3_no_sla", "$-$SLA"), ("G3_no_resource_safety", "$-$safety"), ("G3_role_fix", "$+$role\nfix")]


def fig_governance_heatmap(sub: str = "") -> None:
    t = pd.read_csv(DATA_DIR / sub / "e2_detection_table.csv")
    M = np.zeros((len(FAULT_ORDER), len(GUARDS)))
    for i, f in enumerate(FAULT_ORDER):
        for j, (g, _) in enumerate(GUARDS):
            M[i, j] = 100 * t[(t.fault == f) & (t.guard == g)].rate.iloc[0]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("seq", ["#f4f3f0", "#a9c7ee", C1, "#123a6b"])
    fig, ax = plt.subplots(figsize=(COL_W, 3.05))
    ax.grid(False)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.3,
                    color="white" if v > 60 else INK)
    ax.set_xticks(range(len(GUARDS)), [g[1] for g in GUARDS], fontsize=6.5)
    ax.set_yticks(range(len(FAULT_ORDER)), [FAULT_SHORT[f] for f in FAULT_ORDER], fontsize=6.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(3.5, color="white", lw=2.5)
    ax.axhline(len(FAULT_ORDER) - 1.5, color="white", lw=2.5)
    ax.axvline(6.5, color="white", lw=2.5)
    ax.text(5, -1.55, "ablations of G3", ha="center", fontsize=6.5, color=INK2)
    fig.savefig(FIG_DIR / f"fig_governance_heatmap{'_' + sub if sub else ''}.pdf")
    plt.close(fig)


def fig_latency_cdf() -> None:
    d = pd.read_csv(DATA_DIR / "e2_latency_samples.csv")
    fig, ax = plt.subplots(figsize=(COL_W, 1.35))
    for col, c, lab in (("G1_us", C2, "G1 schema-only"), ("G3_us", C1, "G3 full deterministic pipeline")):
        x = np.sort(d[col].values)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel(r"Authorization latency per tool call ($\mu$s)")
    ax.set_ylabel("CDF")
    ax.legend(loc="lower right")
    fig.savefig(FIG_DIR / "fig_latency_cdf.pdf")
    plt.close(fig)


def fig_memory() -> None:
    d = pd.read_csv(DATA_DIR / "e5_memory_scaling.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.55))
    for v, g in d.groupby("variant"):
        unb = "unbounded" in v
        c, lab = (C1, "PoC (unbounded)") if unb else (C2, "Window $W{=}30$")
        a.plot(g.n, g.ric_context_chars / 4e3, color=c, label=lab)
        b.plot(g.n, g.run_ms.rolling(25, center=True, min_periods=1).median(), color=c, label=lab)
    a.set_yscale("log")
    a.axhline(128, color=INK2, ls="--", lw=0.7)
    a.text(1950, 160, "128k tokens", fontsize=6.5, color=INK2, ha="right")
    a.set_xlabel("Records processed")
    a.set_ylabel("RIC memory context\n(k tokens $\\approx$ chars/4)")
    a.set_title("(a) Prompt context", loc="left")
    b.set_xlabel("Records processed")
    b.set_ylabel("Orchestration time\nper record (ms)")
    b.set_title("(b) Non-inference overhead", loc="left")
    h, l = b.get_legend_handles_labels()
    fig.tight_layout(w_pad=1.2)
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.09), fontsize=7)
    fig.savefig(FIG_DIR / "fig_memory_scaling.pdf")
    plt.close(fig)


SL_COL = {"eMBB": C1, "MTC": C3, "URLLC": C2}


def fig_coloran_char() -> None:
    d = DATA_DIR / "coloran"
    v = pd.read_csv(d / "e1_violation_vs_rbg.csv")
    cc = pd.read_csv(d / "e1_persistence_ccdf.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.6))
    for name in ("eMBB", "URLLC", "MTC"):
        g = v[v.slice_name == name].sort_values("rbg_slice")
        a.plot(g.rbg_slice, 100 * g["any"], color=SL_COL[name], marker="o", ms=2.6, label=name)
    a.set_xlabel("RBGs allocated to the slice (of 17)")
    a.set_ylabel("Windows with SLA\nviolation (%)")
    a.set_ylim(0, 102)
    a.set_title("(a) Violation vs. share", loc="left")
    a.legend(loc="center left", bbox_to_anchor=(0.0, 0.66), fontsize=6.5)
    b.plot(cc.k, 100 * cc.stream, color=C1, marker="o", ms=2.4, label="per (BS, slice) stream")
    b.plot(cc.k, 100 * cc["global"], color=C2, marker="s", ms=2.4, label="global (PoC run script)")
    b.axvline(3, color=INK2, lw=0.7, ls="--")
    b.set_xlabel("Persistence threshold $k$")
    b.set_ylabel(r"Records, $\rho\geq k$ (%)")
    b.set_xlim(1, 20)
    b.set_xticks([1, 5, 10, 15, 20])
    b.set_title("(b) Non-RT eligibility", loc="left")
    b.set_ylim(0, 48)
    b.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), fontsize=6.3)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG_DIR / "fig_coloran_char.pdf")
    plt.close(fig)


def fig_closed_loop() -> None:
    ts = pd.read_csv(DATA_DIR / "e6_closed_loop_timeseries.csv")
    ep = pd.read_csv(DATA_DIR / "e6_closed_loop_episodes.csv")
    meta = json.loads((DATA_DIR / "e6_meta.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.75), gridspec_kw={"width_ratios": [1.25, 1]})
    D = meta["D_s"]
    def line(arm, eps, col, lab, ls="-"):
        g = ts[(ts.arm == arm) & (np.isclose(ts.eps, eps))].sort_values("step")
        a.plot(g.step * D, 100 * g.ok_all.rolling(5, min_periods=1).mean(), color=col, ls=ls, label=lab, lw=1.2)
    line("static", 0.0, INK2, "No control (random start)")
    line("gov", 0.0, C1, "Agentic, governed, $\\epsilon{=}0$")
    line("nogov", 0.5, C2, "Agentic, ungoverned, $\\epsilon{=}0.5$", ls="--")
    line("gov", 0.5, C3, "Agentic, governed, $\\epsilon{=}0.5$", ls=":")
    a.axhline(100 * meta["oracle_static_mean"], color=INK, lw=0.8, ls=(0, (1, 1)))
    a.text(3, 100 * meta["oracle_static_mean"] + 3, "hindsight-best static", fontsize=6.3, color=INK)
    a.set_xlabel("Time (s)")
    a.set_ylabel("All-slice SLA\ncompliance (%)")
    a.set_ylim(0, 70)
    a.set_title("(a) Closed loop on ColO-RAN", loc="left")
    eps = sorted(ep.eps.unique())
    x = np.arange(len(eps))
    w = 0.38
    for off, arm, col, lab in ((-w / 2, "nogov", C2, "ungoverned"), (w / 2, "gov", C1, "governed")):
        m = [ep[(ep.arm == arm) & np.isclose(ep.eps, e)].faulty_executed.mean() for e in eps]
        b.bar(x + off, m, w * 0.95, color=col, label=lab, zorder=3)
    b.set_xticks(x, [f"{e:.1f}" for e in eps])
    b.set_xlabel("Proposer fault rate $\\epsilon$")
    b.set_ylabel("Faulty calls executed\nper episode")
    b.set_title("(b) Faults reaching RAN", loc="left")
    b.legend(loc="upper left", fontsize=6.5)
    h, l = a.get_legend_handles_labels()
    fig.tight_layout(w_pad=0.8)
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16), fontsize=6.5, handlelength=2.2)
    fig.savefig(FIG_DIR / "fig_closed_loop.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_sla_grounding()
    fig_governance_heatmap()
    if (DATA_DIR / "coloran" / "e2_detection_table.csv").exists():
        fig_governance_heatmap("coloran")
    if (DATA_DIR / "coloran" / "e1_violation_vs_rbg.csv").exists():
        fig_coloran_char()
    if (DATA_DIR / "e6_closed_loop_timeseries.csv").exists():
        fig_closed_loop()
    fig_latency_cdf()
    if (DATA_DIR / "e5_memory_scaling.csv").exists():
        fig_memory()
    print("figures written to", FIG_DIR)
