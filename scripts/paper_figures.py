"""The paper's live-LLM figures (pgfplots bodies that main.tex \\input's).

    poetry run python scripts/paper_figures.py \\
        --metrics <paper>/tables/live.json --tex <paper>/figures/live_gov.tex \\
        --timeline "R1=claude-run2,claude-run4" --timeline "R2=claude-run5" \\
        --timeline-tex <paper>/figures/live_timeline.tex

Gate decisions (from the metrics of scripts/paper_tables.py):
(a) outcome of every protected call per group of runs: approved, or the stage
    that rejected it; (b) calls proposed per tool over all groups.

Timeline (from results/<run_id>/records.csv and actions.csv): (a) throughput
per UE of each violated record against the SLA budget, with its persistence;
(b) every protected call at its record, per tool, coloured by its outcome.
Open-loop replay: the calls do not alter the trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Rejection stages in gate order, with the legend label and colour of each.
STAGES = [
    ("approved", "Approved", "okline!75"),
    ("schema", "Malformed ($g_S$)", "detline!35"),
    ("invalid_proposal", "Malformed ($g_S$)", "detline!35"),
    ("invocation_limit", "Guards", "detline!60"),
    ("policy_trust", "Role/scope ($g_P$)", "detline!80"),
    ("sla", "Not justified ($g_L$)", "detline"),
    ("resource_safety", "Unsafe ($g_R$)", "rejline"),
    ("llm_review", "LLM review ($g_V$)", "llmfill!60!ink"),
]

TOOLS = [
    ("adjust_prb_allocation", "PRB share"),
    ("optimize_scheduler", "Scheduler"),
    ("adjust_qos_parameters", "QoS"),
    ("adjust_link_adaptation", "Link adapt."),
    ("adjust_handover", "Handover"),
    ("adjust_power", "Power"),
    ("update_slice_resource_policy", "Slice policy"),
]


def outcome_series(metrics: dict) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    """Legend label -> {group: number of calls} (nonzero labels only), and colours."""
    series: dict[str, dict[str, int]] = {}
    colours: dict[str, str] = {}
    for group, m in metrics.items():
        gov = m["governance"]
        counts = {"approved": gov["approved"]["count"], **gov["rejected_by_stage"]}
        for key, label, colour in STAGES:
            colours[label] = colour
            series.setdefault(label, {g: 0 for g in metrics})[group] += counts.get(key, 0)
    return {label: v for label, v in series.items() if any(v.values())}, colours


def tool_counts(metrics: dict) -> list[tuple[str, int, int]]:
    """(tool label, approved, rejected) over all groups, tools ever proposed."""
    rows = []
    for key, label in TOOLS:
        approved = sum(m["governance"]["per_tool"].get(key, {}).get("approved", 0)
                       for m in metrics.values())
        rejected = sum(m["governance"]["per_tool"].get(key, {}).get("rejected", 0)
                       for m in metrics.values())
        if approved or rejected:
            rows.append((label, approved, rejected))
    return rows


def figure(metrics: dict) -> str:
    groups = list(metrics)
    series, colours = outcome_series(metrics)
    tools = tool_counts(metrics)
    max_calls = max(sum(v[g] for v in series.values()) for g in groups)
    max_tool = max(a + r for _, a, r in tools)

    common = ("xbar stacked, bar width=7pt, axis x line*=bottom, axis y line*=left, "
              "tick label style={font=\\scriptsize}, label style={font=\\scriptsize}, "
              "legend style={font=\\tiny, draw=none, fill=none, legend columns=-1, "
              "/tikz/every even column/.append style={column sep=3pt}}, "
              "legend image code/.code={\\draw[#1] (0,-0.08) rectangle (0.18,0.1);}, "
              "nodes near coords={\\pgfmathfloatifflags{\\pgfplotspointmeta}{0}{}"
              "{\\pgfmathprintnumber{\\pgfplotspointmeta}}}, "
              "point meta=explicit, every node near coord/.append style={font=\\tiny}, "
              "nodes near coords align={center}")

    a = [f"\\begin{{axis}}[{common}, width=\\columnwidth, height=2.9cm,",
         f"  xmin=0, xmax={max_calls + 1}, xlabel={{Protected tool calls}},",
         f"  symbolic y coords={{{','.join(reversed(groups))}}}, ytick=data, enlarge y limits=0.45,",
         "  legend to name=legOutcome,",
         "  title={\\scriptsize (a) Outcome of each call}]"]
    for label, values in series.items():
        coords = " ".join(f"({values[g]},{g}) [{values[g]}]" for g in groups)
        a.append(f"\\addplot[fill={colours[label]}, draw=ink, line width=0.3pt] "
                 f"coordinates {{{coords}}};")
        a.append(f"\\addlegendentry{{{label}}}")
    a.append("\\end{axis}")

    names = [t[0] for t in tools]
    b = [f"\\begin{{axis}}[{common}, width=\\columnwidth, height=3.6cm,",
         f"  xmin=0, xmax={max_tool + 1}, xlabel={{Calls proposed (all runs)}},",
         f"  symbolic y coords={{{','.join(reversed(names))}}}, ytick=data, enlarge y limits=0.12,",
         "  legend to name=legTool,",
         "  title={\\scriptsize (b) Calls per tool}]"]
    for idx, (label, colour) in enumerate([("Approved", "okline!75"),
                                           ("Rejected", "detline!60")]):
        coords = " ".join(f"({t[1 + idx]},{t[0]}) [{t[1 + idx]}]" for t in tools)
        b.append(f"\\addplot[fill={colour}, draw=ink, line width=0.3pt] coordinates {{{coords}}};")
        b.append(f"\\addlegendentry{{{label}}}")
    b.append("\\end{axis}")

    return "\n".join([
        "% Generated by scripts/paper_figures.py from tables/live.json -- do not edit.",
        "\\begin{tikzpicture}", *a, "\\end{tikzpicture}\\\\[-1mm]",
        "\\ref{legOutcome}\\\\[1mm]",
        "\\begin{tikzpicture}", *b, "\\end{tikzpicture}\\\\[-1mm]",
        "\\ref{legTool}",
        "",
    ])


# Timeline: tools from bottom to top, and the three outcome classes.
TIMELINE_TOOLS = ["adjust_link_adaptation", "adjust_qos_parameters", "optimize_scheduler",
                  "adjust_prb_allocation", "update_slice_resource_policy"]
OUTCOMES = [
    ("approved", "Approved", "mark=*, mark size=1.5pt, color=okline"),
    ("code", "Rejected by code", "mark=x, mark size=2pt, thick, color=detline"),
    ("review", "Rejected by $g_V$", "mark=square*, mark size=1.3pt, color=llmfill!45!ink"),
]


def _outcome(action: dict) -> str:
    if action["approved"] == "True":
        return "approved"
    return "review" if action["failed_stage"] == "llm_review" else "code"


def _second(record_id: str) -> int:
    return int(float(re.search(r"-t(\d+(?:\.\d+)?)-", record_id).group(1)))


def timeline_data(groups: list[tuple[str, list[str]]], results: Path = RESULTS):
    """Records [(x, t, throughput, persistence, run)], calls [(x, tool, outcome)],
    and the x span of each group; the groups follow each other with a gap."""
    records, calls, spans = [], [], []
    x0 = 0
    for label, run_ids in groups:
        rows, acts = [], []
        for run in run_ids:
            by_seq = {}
            for r in csv.DictReader(open(results / run / "records.csv", encoding="utf-8")):
                t = _second(r["record_id"])
                by_seq[r["seq"]] = t
                rows.append((t, float(r["throughput_per_ue_mbps"]), int(r["max_persistence"]), run))
            for a in csv.DictReader(open(results / run / "actions.csv", encoding="utf-8")):
                if a["seq"] in by_seq:
                    acts.append((by_seq[a["seq"]], a["tool"], _outcome(a)))
        if not rows:
            continue
        t0 = min(t for t, *_ in rows)
        pos = lambda t: x0 + t - t0  # noqa: E731, B023 (used in this iteration)
        records += [(pos(t), t, thr, rho, run) for t, thr, rho, run in sorted(rows)]
        calls += [(pos(t), tool, outcome) for t, tool, outcome in acts]
        spans.append((label, x0, pos(max(t for t, *_ in rows))))
        x0 = spans[-1][2] + 2
    return records, calls, spans


def timeline_figure(groups: list[tuple[str, list[str]]], budget: float = 3.6,
                    results: Path = RESULTS) -> str:
    records, calls, spans = timeline_data(groups, results)
    xmin, xmax = -0.6, spans[-1][2] + 0.6
    ticks = ",".join(str(x) for x, *_ in records)
    labels = ",".join(str(t) for _, t, *_ in records)
    axis = (f"scale only axis, width=0.8\\columnwidth, xmin={xmin}, xmax={xmax}, "
            f"xtick={{{ticks}}}, axis x line*=bottom, axis y line*=left, "
            "tick label style={font=\\tiny}, label style={font=\\scriptsize}")
    seps = [(spans[i][2] + spans[i + 1][1]) / 2 for i in range(len(spans) - 1)]

    out = ["% Generated by scripts/paper_figures.py from results/<run>/{records,actions}.csv"
           " -- do not edit.", "\\begin{tikzpicture}"]
    # (a) throughput against the budget, persistence above each point
    out += [f"\\begin{{axis}}[{axis}, name=thr, height=1.9cm, ymin=0, ymax=4.6, "
            "ytick={0,2,3.6}, xticklabels={}, ylabel={Thr./UE (Mbps)}, "
            "title={\\scriptsize (a) Violated eMBB records of cell bs2}, "
            "title style={yshift=-1mm}]",
            f"\\addplot[densely dashed, draw=detline, line width=0.6pt] coordinates "
            f"{{({xmin},{budget}) ({xmax},{budget})}};",
            f"\\node[font=\\tiny, text=detline, anchor=south east] at (axis cs:{xmax},{budget}) "
            "{SLA budget};"]
    runs = list(dict.fromkeys(r[4] for r in records))
    for run in runs:  # one line per run: the persistence restarts with each run
        pts = " ".join(f"({x},{thr:.3f}) [{rho}]" for x, _, thr, rho, r in records if r == run)
        out.append("\\addplot[mark=*, mark size=1.2pt, color=ink, line width=0.5pt, "
                   "point meta=explicit symbolic, nodes near coords={\\tiny$\\rho{=}"
                   "\\pgfplotspointmeta$}, every node near coord/.append style={yshift=-1pt}] "
                   f"coordinates {{{pts}}};")
    for label, lo, hi in spans:
        out.append(f"\\node[font=\\scriptsize\\bfseries, text=inkmuted, anchor=north] at "
                   f"(axis cs:{(lo + hi) / 2},1.6) {{{label}}};")
    for s in seps:
        out.append(f"\\draw[densely dotted, inkmuted] (axis cs:{s},0) -- (axis cs:{s},4.6);")
    out.append("\\end{axis}")

    # (b) one marker per call, side by side when a record has several calls of a tool
    order = {key: i + 1 for i, key in enumerate(TIMELINE_TOOLS)}
    names = [dict(TOOLS)[key] for key in TIMELINE_TOOLS]
    out += [f"\\begin{{axis}}[{axis}, at={{(thr.south west)}}, anchor=north west, yshift=-6mm, "
            f"height=2.5cm, ymin=0.4, ymax={len(TIMELINE_TOOLS) + 0.6}, "
            f"ytick={{{','.join(str(i + 1) for i in range(len(names)))}}}, "
            f"yticklabels={{{','.join(names)}}}, xticklabels={{{labels}}}, "
            "xlabel={Trace time $t$ (s)}, ymajorgrids, grid style={inkmuted!25}, "
            "title={\\scriptsize (b) Protected tool calls}, title style={yshift=-1mm}, "
            "legend style={font=\\tiny, draw=none, legend columns=-1, at={(0.5,-0.3)}, "
            "anchor=north, /tikz/every even column/.append style={column sep=4pt}}]"]
    slots = defaultdict(list)
    for x, tool, outcome in calls:
        slots[(x, tool)].append(outcome)
    placed = defaultdict(list)
    for (x, tool), outs in slots.items():
        n = len(outs)
        for i, outcome in enumerate(outs):
            placed[outcome].append((x + (i - (n - 1) / 2) * 0.17, order[tool]))
    for key, label, style in OUTCOMES:
        pts = " ".join(f"({x:.3f},{y})" for x, y in placed[key])
        out.append(f"\\addplot[only marks, {style}] coordinates {{{pts}}};")
        out.append(f"\\addlegendentry{{{label}}}")
    for s in seps:
        out.append(f"\\draw[densely dotted, inkmuted] (axis cs:{s},0.4) -- "
                   f"(axis cs:{s},{len(TIMELINE_TOOLS) + 0.6});")
    out += ["\\end{axis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


# Gate latency (scripts/gate_latency.py): CDF per stage and scaling with n.
LATENCY_SERIES = [
    ("g_S", "$g_S$", "detline, densely dashed"),
    ("g_P", "$g_P$", "okline, densely dotted"),
    ("g_L", "$g_L$", "llmfill!45!ink, dashdotted"),
    ("g_R", "$g_R$", "rejline, loosely dashed"),
    ("gate", "Gate (all)", "ink, line width=1.1pt"),
]


def _cdf(values: list[float], points: int = 120) -> list[tuple[float, float]]:
    values = sorted(values)
    n = len(values)
    idx = sorted({min(n - 1, round(i * (n - 1) / (points - 1))) for i in range(points)})
    return [(values[i], (i + 1) / n) for i in idx]


def latency_figure(data: dict, live: dict | None = None) -> str:
    """live: results/gate_latency_llm.json (LangSmith), adds the gate with g_V."""
    samples = data["samples_us"]
    xmax = 3e8 if live else 3e4
    rows = data["scaling"]["rows"]
    common = ("scale only axis, height=2.6cm, "
              "axis x line*=bottom, axis y line*=left, tick label style={font=\\tiny}, "
              "label style={font=\\scriptsize}, title style={font=\\scriptsize, yshift=-1mm}, "
              "grid style={inkmuted!25}, ymajorgrids")
    out = ["% Generated by scripts/paper_figures.py from results/gate_latency.json -- do not edit.",
           "\\begin{tikzpicture}",
           f"\\begin{{axis}}[{common}, width=0.5\\columnwidth, name=cdf, xmode=log, xmin=0.3, xmax={xmax:g}, ymin=0, ymax=1.02,",
           "  xtick={1,100,10000,1000000,100000000}, xlabel={Decision latency ($\\mu$s)}, ylabel={CDF},",
           "  title={(a) Latency per call},",
           "  legend to name=legLatency, legend columns=3, legend cell align=left,",
           "  legend style={font=\\tiny, draw=none, /tikz/every even column/.append style={column sep=5pt}},",
           "  xminorticks=false]"]
    for key, label, style in LATENCY_SERIES:
        pts = " ".join(f"({x:.3f},{y:.4f})" for x, y in _cdf(samples[key]))
        out.append(f"\\addplot[{style}, no marks] coordinates {{{pts}}};")
        out.append(f"\\addlegendentry{{{label}}}")
    if live:
        reviewed = [c["gate_s"] * 1e6 for c in live["calls"] if c["reviewed"]]
        pts = " ".join(f"({x:.1f},{y:.4f})" for x, y in _cdf(reviewed, len(reviewed)))
        out.append(f"\\addplot[llmfill!40!ink, line width=1.1pt, const plot, no marks] "
                   f"coordinates {{({reviewed and min(reviewed):.1f},0) {pts}}};")
        out.append("\\addlegendentry{Gate + $g_V$ (live)}")
    out += ["\\draw[detline, densely dashed, line width=0.5pt] (axis cs:1e4,0) -- (axis cs:1e4,1.02);",
            "\\node[font=\\tiny, text=detline, rotate=90, anchor=south east, xshift=-0.5pt] at (axis cs:1e4,1.0)"
            " {Near-RT 10\\,ms};",
            "\\end{axis}"]
    band = ([(r["n"], r["p5"]) for r in rows] + [(r["n"], r["p95"]) for r in reversed(rows)])
    top = max(r["p95"] for r in rows)
    out += [f"\\begin{{axis}}[{common}, width=0.27\\columnwidth, at={{(cdf.east)}}, anchor=west, xshift=11mm, "
            f"xmin=1, xmax={rows[-1]['n']}, ymin=0, ymax={10 * (int(top / 10) + 2)},",
            "  xlabel={Calls in the invocation $n$}, ylabel={Latency of call $n$ ($\\mu$s)},",
            "  title={(b) Scaling with $n$},",
            "  legend style={font=\\tiny, draw=none, fill=none, at={(0.98,0.03)}, anchor=south east},"
            " legend cell align=left]",
            "\\addplot[fill=detline!22, draw=none, forget plot] coordinates {"
            + " ".join(f"({n},{v:.2f})" for n, v in band) + "} -- cycle;",
            "\\addplot[ink, line width=1.1pt, no marks] coordinates {"
            + " ".join(f"({r['n']},{r['median']:.2f})" for r in rows) + "};",
            "\\addlegendentry{median}",
            "\\addlegendimage{area legend, fill=detline!22, draw=none}",
            "\\addlegendentry{5th--95th pct.}",
            "\\end{axis}", "\\end{tikzpicture}\\\\[0.5mm]", "\\ref{legLatency}", ""]
    return "\n".join(out)


def parse_group(text: str) -> tuple[str, list[str]]:
    label, _, ids = text.partition("=")
    run_ids = [r.strip() for r in ids.split(",") if r.strip()]
    if not label.strip() or not run_ids:
        raise argparse.ArgumentTypeError(f"expected LABEL=run1,run2: {text!r}")
    return label.strip(), run_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-LLM figures of the paper.")
    parser.add_argument("--metrics", type=Path, help="tables/live.json (gate-decision figure)")
    parser.add_argument("--tex", type=Path, help="output of the gate-decision figure")
    parser.add_argument("--timeline", type=parse_group, action="append",
                        help="LABEL=run1,run2 (repeatable, in time order)")
    parser.add_argument("--timeline-tex", type=Path, help="output of the timeline figure")
    parser.add_argument("--latency", type=Path, help="results/gate_latency.json")
    parser.add_argument("--latency-tex", type=Path, help="output of the gate-latency figure")
    parser.add_argument("--latency-live", type=Path,
                        help="results/gate_latency_llm.json (LangSmith): adds the gate with g_V")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if (bool(args.metrics) != bool(args.tex) or bool(args.timeline) != bool(args.timeline_tex)
            or bool(args.latency) != bool(args.latency_tex)):
        parser.error("--metrics goes with --tex, --timeline with --timeline-tex, "
                     "--latency with --latency-tex")
    if args.metrics:
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        args.tex.write_text(figure(metrics), encoding="utf-8")
        print(f"wrote {args.tex}")
    if args.timeline:
        args.timeline_tex.write_text(timeline_figure(args.timeline, results=args.results),
                                     encoding="utf-8")
        print(f"wrote {args.timeline_tex}")
    if args.latency:
        data = json.loads(args.latency.read_text(encoding="utf-8"))
        live = (json.loads(args.latency_live.read_text(encoding="utf-8"))
                if args.latency_live else None)
        args.latency_tex.write_text(latency_figure(data, live), encoding="utf-8")
        print(f"wrote {args.latency_tex}")


if __name__ == "__main__":
    main()
