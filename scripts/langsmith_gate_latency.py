"""Gate latency WITH the LLM review (g_V), read from the LangSmith traces of
the live runs.

    poetry run python scripts/langsmith_gate_latency.py \\
        --runs claude-run2,claude-run4,claude-run5 --out results/gate_latency_llm.json

Each protected call of a live run left one span named "governance <tool>"
(tag "governance"): its duration is the whole gate decision with g_V, under
the real model and conditions of the run. The model call(s) inside the span
are g_V; their duration and tokens are reported too. Nothing is re-run.

The key is read from .env (never printed): LANGSMITH_READ_API_KEY, a
LangSmith Personal Access Token (the tracing key may not be allowed to read).
The project is LANGSMITH_PROJECT (default "default"), or --project.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")


def _seconds(run) -> float | None:
    if run.start_time is None or run.end_time is None:
        return None
    return (run.end_time - run.start_time).total_seconds()


def _metadata(run) -> dict[str, Any]:
    return ((run.extra or {}).get("metadata") or {}) if hasattr(run, "extra") else {}


def _tokens(run) -> dict[str, int]:
    usage = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(run, key, None)
        if value:
            usage[key] = int(value)
    return usage


def collect(client, project: str, runs: set[str]) -> list[dict[str, Any]]:
    # One query: the spans and every run nested in them inherit the tag.
    tagged = list(client.list_runs(project_name=project, filter='has(tags, "governance")'))
    spans = [s for s in tagged if s.name.startswith("governance ")]
    models = [r for r in tagged if r.run_type == "llm" and r.dotted_order]
    print(f"{len(spans)} governance spans, {len(models)} model calls inside them, "
          f"in project {project!r}")
    calls = []
    for span in spans:
        run_id = _metadata(span).get("run_id")
        if runs and run_id not in runs:
            continue
        # model calls inside this span = the LLM review g_V
        reviews = [r for r in models if r.dotted_order.startswith(span.dotted_order)]
        outputs = span.outputs or {}
        content = json.dumps(outputs, default=str)
        calls.append({
            "run_id": run_id,
            "record_id": _metadata(span).get("record_id"),
            "tool": span.name.removeprefix("governance "),
            "gate_s": _seconds(span),
            "reviewed": bool(reviews),
            "g_v_s": sum(_seconds(r) or 0 for r in reviews) if reviews else None,
            "tokens": {k: sum(_tokens(r).get(k, 0) for r in reviews)
                       for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
            "outcome": ("rejected" if '"rejected"' in content
                        else "executed" if reviews or "executed" in content else "unknown"),
            "start": span.start_time.isoformat() if span.start_time else None,
        })
    return calls


def match_results(calls: list[dict[str, Any]], results: Path) -> list[dict[str, Any]]:
    """Keep the traced calls that are the stored ones of results/<run>/: per
    (run, record), the latest k spans, k = calls stored for that record. A run
    id reused for an earlier execution, or a record interrupted by the rate
    limit (not stored), leaves traces that are dropped here."""
    import csv
    from collections import Counter, defaultdict

    stored: Counter = Counter()
    for run in {c["run_id"] for c in calls}:
        folder = results / str(run)
        if not (folder / "actions.csv").exists():
            continue
        records = {r["seq"]: r["record_id"] for r in csv.DictReader(open(folder / "records.csv"))}
        for a in csv.DictReader(open(folder / "actions.csv")):
            stored[(run, records.get(a["seq"]))] += 1
    groups: dict[tuple, list] = defaultdict(list)
    for c in calls:
        groups[(c["run_id"], c["record_id"])].append(c)
    kept = []
    for key, group in groups.items():
        k = stored.get(key, 0)
        if k:
            kept += sorted(group, key=lambda c: c["start"] or "")[-k:]
    return kept


def _dist(values: list[float]) -> dict[str, float] | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return {"n": len(values), "median": statistics.median(values),
            "p95": values[min(len(values) - 1, int(0.95 * len(values)))], "max": values[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate latency with g_V from LangSmith traces.")
    parser.add_argument("--runs", default="claude-run2,claude-run4,claude-run5",
                        help="comma-separated dataset run ids (empty = all)")
    parser.add_argument("--project", default=os.getenv("LANGSMITH_PROJECT", "default"))
    parser.add_argument("--match-results", action="store_true",
                        help="keep only the calls stored in results/<run>/actions.csv")
    parser.add_argument("--out", type=Path, default=_ROOT / "results" / "gate_latency_llm.json")
    args = parser.parse_args()

    key = os.getenv("LANGSMITH_READ_API_KEY")
    if not key:
        sys.exit("LANGSMITH_READ_API_KEY is missing from .env (see the docstring).")
    from langsmith import Client

    client = Client(api_key=key, api_url=os.getenv("LANGSMITH_ENDPOINT") or None)
    try:
        calls = collect(client, args.project, {r for r in args.runs.split(",") if r})
    except Exception as exc:
        projects = []
        try:
            projects = [p.name for p in client.list_projects()]
        except Exception:
            pass
        sys.exit(f"Could not read the traces ({type(exc).__name__}: {exc}). "
                 f"Projects visible with this key: {projects or 'none'}")

    if args.match_results:
        traced = len(calls)
        calls = match_results(calls, _ROOT / "results")
        print(f"{len(calls)} of {traced} traced calls match the stored results")
    reviewed = [c for c in calls if c["reviewed"]]
    summary = {
        "calls": len(calls),
        "reviewed_by_g_v": len(reviewed),
        "gate_with_g_v_s": _dist([c["gate_s"] for c in reviewed]),
        "gate_without_g_v_s": _dist([c["gate_s"] for c in calls if not c["reviewed"]]),
        "g_v_s": _dist([c["g_v_s"] for c in reviewed]),
        "g_v_tokens_mean": {k: statistics.mean(c["tokens"][k] for c in reviewed)
                            for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        if reviewed else None,
    }
    args.out.write_text(json.dumps({"project": args.project, "summary": summary, "calls": calls},
                                   indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
