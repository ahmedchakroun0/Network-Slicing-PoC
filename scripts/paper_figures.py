"""The paper's live-LLM figure, drawn from the metrics of scripts/paper_tables.py.

    poetry run python scripts/paper_figures.py \\
        --metrics <paper>/tables/live.json --tex <paper>/figures/live_gov.tex

(a) Outcome of every protected call per group of runs: approved, or the stage
    that rejected it.  (b) Calls proposed per tool over all groups, approved or
    rejected.  The output is a pgfplots figure body that main.tex \\input's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-LLM figure of the paper.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--tex", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    args.tex.write_text(figure(metrics), encoding="utf-8")
    print(args.tex.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
