"""Latency of the Trust & Governance gate, measured on the real code.

    poetry run python scripts/gate_latency.py \\
        --group "R1=claude-run2,claude-run4:20" --group "R2=claude-run5:2" \\
        --out results/gate_latency.json

The protected calls of the live runs (results/<run>/actions.csv, the LLM's own
proposals) are replayed through the unmodified RANAuthorizationMiddleware and
RANTrustGovernanceAgent. The LLM review (g_V) is not called (use_llm=False, an
existing option): only the deterministic part of the gate is timed. The
trusted context of each call is rebuilt from the stored record (cell, slice,
violations, persistence) and from the PRB controls executed before it.

1. Fidelity: the replay must reproduce the stored decision of every call
   (g_V's stored verdict stands in for the review). Mismatches are reported.
2. Latency per stage (g_S, g_P, g_L, g_R) and of the whole deterministic gate
   (guards + proposal + stages + audit), each call repeated --reps times.
3. Scaling: latency of the n-th protected call of one invocation, n = 1..N,
   when the n-1 earlier calls are in the audit (the guards scan them).

Nothing in the project is modified; MAX_PROTECTED_CALLS is set on the
imported module for each group (the value the run used), and lifted for 3.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("LANGSMITH_TRACING", "false")
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.messages import ToolMessage

import middleware.ran_authorisation as mw
from agents.ran_trust_governance import ARG_MODELS, RANTrustGovernanceAgent
from src.schema.near_rt_ric import GovernanceResult

RESULTS = _ROOT / "results"
logging.disable(logging.INFO)


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _controls_of(tool: str, params: dict[str, Any]) -> dict[str, int]:
    """Same rule as src/memory/working_memory._controls_of (executed calls)."""
    if tool == "adjust_prb_allocation":
        return {"prb_percentage": params["prb_percentage"]}
    if tool == "update_slice_resource_policy":
        return {
            "policy_min": params["min_prb_ratio_pct"],
            "policy_max": params["max_prb_ratio_pct"],
        }
    return {}


def load_invocations(run: str, results: Path = RESULTS) -> list[dict[str, Any]]:
    """The protected calls of one run, grouped by invocation (record, agent),
    in call order, each invocation with its rebuilt trusted context."""
    records = {r["seq"]: r for r in csv.DictReader(open(results / run / "records.csv"))}
    invocations: list[dict[str, Any]] = []
    controls: dict[str, int] = {}
    for a in csv.DictReader(open(results / run / "actions.csv")):
        rec = records[a["seq"]]
        key = (a["seq"], a["agent"])
        if not invocations or invocations[-1]["key"] != key:
            violations = [v for v in rec["violations"].split(",") if v]
            persistence = int(rec["max_persistence"])
            invocations.append(
                {
                    "key": key,
                    "agent": a["agent"],
                    "invocation": {
                        "agent": a["agent"],
                        "context": {
                            "target": rec["cell_id"],
                            "slice_id": rec["slice_id"],
                            "sla_analysis": {"violations": violations},
                            "persistence": {v: persistence for v in violations},
                            "active_controls": dict(controls),
                        },
                    },
                    "calls": [],
                }
            )
        params = json.loads(a["params"] or "{}")
        args = {**params, "target": a["target"] or None}
        for field in ("rationale", "expected_effect"):  # empty in the CSV = not sent
            if a[field]:
                args[field] = a[field]
        if a.get("confidence"):
            args["confidence"] = float(a["confidence"])
        if a.get("evidence_sources"):
            args["evidence_sources"] = [s for s in a["evidence_sources"].split(" || ") if s]
        approved = _bool(a["approved"])
        invocations[-1]["calls"].append(
            {
                "tool": a["tool"],
                "args": args,
                "params": params,
                "stored": "approved" if approved else a["failed_stage"],
            }
        )
        if approved:
            controls.update(_controls_of(a["tool"], params))
    return invocations


class ReplayedReview:
    """The real deterministic gate; when it approves a call that g_V rejected
    in the live run, the stored verdict is returned (the review is not
    called again). Used only for the fidelity check."""

    def __init__(self) -> None:
        self.gate = RANTrustGovernanceAgent(use_llm=False)
        self.verdict = "approved"

    async def authorize(self, proposal, state=None) -> GovernanceResult:
        result = await self.gate.authorize(proposal)
        if result.approved and self.verdict == "llm_review":
            return GovernanceResult(
                approved=False,
                reason="[llm_review] stored verdict",
                failed_stage="llm_review",
                checks_passed=result.checks_passed,
            )
        return result


async def _executed(request):
    return ToolMessage(content='{"status": "executed"}', tool_call_id=request.tool_call["id"])


def _request(call: dict[str, Any], invocation: dict[str, Any], i: int):
    return SimpleNamespace(
        tool_call={"name": call["tool"], "id": f"call_{i}", "args": dict(call["args"])},
        state={"agent_invocation": invocation, "messages": []},
    )


async def replay(invocation: dict[str, Any], middleware, review=None) -> list[str]:
    """Run the calls of one invocation through the middleware; outcomes."""
    sink = mw.AuditSink()
    token = mw.RAN_AUDIT.set(sink)
    try:
        for i, call in enumerate(invocation["calls"]):
            if review is not None:
                review.verdict = call["stored"]
            await middleware._gate(_request(call, invocation["invocation"], i), _executed)
    finally:
        mw.RAN_AUDIT.reset(token)
    return [
        "approved" if e["governance"]["approved"] else e["governance"]["failed_stage"]
        for e in sink.entries
    ]


async def fidelity(groups) -> dict[str, Any]:
    review = ReplayedReview()
    middleware = mw.RANAuthorizationMiddleware(review)
    checked = mismatches = 0
    details = []
    for _label, runs, cap in groups:
        mw.MAX_PROTECTED_CALLS = cap
        for run in runs:
            for inv in load_invocations(run):
                got = await replay(inv, middleware, review)
                for call, outcome in zip(inv["calls"], got, strict=False):
                    checked += 1
                    if outcome != call["stored"]:
                        mismatches += 1
                        details.append(
                            {
                                "run": run,
                                "record": inv["key"][0],
                                "tool": call["tool"],
                                "stored": call["stored"],
                                "replayed": outcome,
                            }
                        )
    return {"calls": checked, "mismatches": mismatches, "details": details}


def _ns(fn, *args) -> int:
    t0 = time.perf_counter_ns()
    fn(*args)
    return time.perf_counter_ns() - t0


async def stage_latency(groups, reps: int) -> dict[str, list[float]]:
    """Per-stage and whole-gate latency (microseconds) of every replayed call."""
    gate = RANTrustGovernanceAgent(use_llm=False)
    middleware = mw.RANAuthorizationMiddleware(gate)
    out: dict[str, list[float]] = {"g_S": [], "g_P": [], "g_L": [], "g_R": [], "gate": []}
    for _label, runs, cap in groups:
        mw.MAX_PROTECTED_CALLS = cap
        for run in runs:
            for inv in load_invocations(run):
                prefix: list[dict[str, Any]] = []
                for i, call in enumerate(inv["calls"]):
                    submitted, _ = mw.build_submission(call["tool"], call["args"])
                    try:
                        proposal = mw.RANActionProposal.model_validate(submitted).model_copy(
                            update={"invocation_context": {"invocation": inv["invocation"]}}
                        )
                    except Exception:
                        proposal = None
                    model = ARG_MODELS.get(call["tool"])
                    args = None
                    if proposal is not None and model is not None:
                        try:
                            args = model.model_validate(
                                {**proposal.parameters, "target": proposal.target}
                            ).model_dump()
                        except Exception:
                            args = None

                    # g_S as in authorize(): reject = exception. Loop values are
                    # bound as defaults (the closure is timed in this iteration).
                    def schema_stage(model=model, proposal=proposal):
                        try:
                            model.model_validate({**proposal.parameters, "target": proposal.target})
                        except Exception:
                            pass
                        return bool(proposal.rationale.strip() and proposal.expected_effect.strip())

                    for _ in range(reps):
                        if model is not None and proposal is not None:
                            out["g_S"].append(_ns(schema_stage) / 1e3)
                        if args is not None:
                            out["g_P"].append(_ns(gate._check_policy_trust, proposal, args) / 1e3)
                            out["g_L"].append(_ns(gate._check_sla, proposal, args) / 1e3)
                            out["g_R"].append(
                                _ns(gate._check_resource_safety, proposal, args) / 1e3
                            )
                        sink = mw.AuditSink()
                        sink.entries = list(prefix)
                        token = mw.RAN_AUDIT.set(sink)
                        request = _request(call, inv["invocation"], i)
                        t0 = time.perf_counter_ns()
                        await middleware._gate(request, _executed)
                        out["gate"].append((time.perf_counter_ns() - t0) / 1e3)
                        mw.RAN_AUDIT.reset(token)
                    prefix = list(sink.entries)
    return out


async def scaling(groups, n_max: int, reps: int) -> dict[str, Any]:
    """Latency of the n-th call of one invocation with n-1 earlier calls in
    the audit: the last approved PRB call of the runs, after n-1 replayed
    calls of other tools (rejected, so every guard scans the whole audit)."""
    gate = RANTrustGovernanceAgent(use_llm=False)
    middleware = mw.RANAuthorizationMiddleware(gate)
    mw.MAX_PROTECTED_CALLS = 10**6
    invs = [inv for _, runs, _ in groups for run in runs for inv in load_invocations(run)]
    target_inv = next(
        inv
        for inv in reversed(invs)
        for c in inv["calls"]
        if c["tool"] == "adjust_prb_allocation" and c["stored"] == "approved"
    )
    target = next(
        c
        for c in target_inv["calls"]
        if c["tool"] == "adjust_prb_allocation" and c["stored"] == "approved"
    )
    fillers = [
        c
        for inv in invs
        for c in inv["calls"]
        if c["tool"] != "adjust_prb_allocation" and c["stored"] != "approved"
    ]
    rows = []
    for n in range(1, n_max + 1):
        sink = mw.AuditSink()
        token = mw.RAN_AUDIT.set(sink)
        for i in range(n - 1):
            await middleware._gate(
                _request(fillers[i % len(fillers)], target_inv["invocation"], i), _executed
            )
        mw.RAN_AUDIT.reset(token)
        prefix = list(sink.entries)
        samples = []
        for _ in range(reps):
            sink = mw.AuditSink()
            sink.entries = list(prefix)
            token = mw.RAN_AUDIT.set(sink)
            request = _request(target, target_inv["invocation"], n)
            t0 = time.perf_counter_ns()
            await middleware._gate(request, _executed)
            samples.append((time.perf_counter_ns() - t0) / 1e3)
            mw.RAN_AUDIT.reset(token)
        samples.sort()
        q = lambda p: samples[min(len(samples) - 1, int(p * len(samples)))]  # noqa: E731, B023 (used in this iteration)
        rows.append(
            {
                "n": n,
                "p5": q(0.05),
                "p25": q(0.25),
                "median": statistics.median(samples),
                "p75": q(0.75),
                "p95": q(0.95),
            }
        )
    return {"target_tool": target["tool"], "rows": rows}


def parse_group(text: str):
    label, _, rest = text.partition("=")
    runs, _, cap = rest.partition(":")
    return label, [r for r in runs.split(",") if r], int(cap or mw.MAX_PROTECTED_CALLS)


def _cpu() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


async def main_async(args) -> dict[str, Any]:
    groups = args.group
    check = await fidelity(groups)
    print(f"fidelity: {check['calls'] - check['mismatches']}/{check['calls']} decisions reproduced")
    for d in check["details"]:
        print("  mismatch", d)
    stages = await stage_latency(groups, args.reps)
    scale = await scaling(groups, args.n_max, args.reps)
    summary = {
        k: {
            "n": len(v),
            "median_us": statistics.median(v),
            "p99_us": sorted(v)[int(0.99 * len(v)) - 1],
        }
        for k, v in stages.items()
        if v
    }
    for k, s in summary.items():
        print(f"{k:5s} n={s['n']:6d} median={s['median_us']:8.1f} us  p99={s['p99_us']:8.1f} us")
    return {
        "cpu": _cpu(),
        "python": platform.python_version(),
        "reps": args.reps,
        "fidelity": check,
        "summary": summary,
        "samples_us": stages,
        "scaling": scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic gate latency on replayed live calls."
    )
    parser.add_argument(
        "--group",
        type=parse_group,
        action="append",
        required=True,
        help="LABEL=run1,run2:CAP (CAP = protected calls per invocation of the run)",
    )
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--n-max", type=int, default=64)
    parser.add_argument("--out", type=Path, default=RESULTS / "gate_latency.json")
    args = parser.parse_args()
    result = asyncio.run(main_async(args))
    args.out.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {args.out} ({result['cpu']})")


if __name__ == "__main__":
    main()
