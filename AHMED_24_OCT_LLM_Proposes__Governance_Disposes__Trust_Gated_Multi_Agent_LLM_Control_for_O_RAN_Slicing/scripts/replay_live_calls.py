"""Replay the protected tool calls logged in the live-LLM runs through a PoC gate.

Reads the run documents produced by parse_llm_runs.py (data/llm_runs/run-*.json)
and re-submits every logged call, in its original order, to the deterministic
stages of the governance of PROJECT (LLM review disabled), with the invocation
limits of its middleware emulated. The trusted context is rebuilt as the fixed
run script would build it: per-(cell, slice) persistence, which is 1 for the
first window of a stream (all logged runs started at the first record).

A call whose proposal could not even be built (invalid_proposal) is replayed as
invalid. Calls that pass every deterministic stage are reported as reaching the
LLM review; their final outcome would depend on g_V.

Usage: python replay_live_calls.py --project ../../poc_v3 [--runs ../data/llm_runs]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

from common import DATA_DIR


async def replay(project: Path, runs_dir: Path) -> dict:
    sys.path.insert(0, str(project))
    gov_mod = importlib.import_module("agents.ran_trust_governance")
    mw_mod = importlib.import_module("middleware.ran_authorisation")
    schema = importlib.import_module("src.schema.near_rt_ric")
    gov = gov_mod.RANTrustGovernanceAgent(use_llm=False)
    max_calls = getattr(mw_mod, "MAX_PROTECTED_CALLS", None)

    rows = []
    for f in sorted(runs_dir.glob("run-*.json")):
        doc = json.loads(f.read_text())
        for rec_id, rec in doc["records"].items():
            persistence_seen: dict[tuple, dict] = {}
            for agent, ex in (rec.get("agent_executions") or {}).items():
                inv = ex["invocation"]
                ctx = dict(inv["context"])
                key = (ctx["target"], ctx["slice_id"])
                pers = persistence_seen.setdefault(
                    key, {k: (1 if k in ctx["sla_analysis"]["violations"] else 0)
                          for k in ("latency", "jitter", "packet_loss", "throughput")})
                ctx["persistence"] = pers
                executed_tools: set[str] = set()
                n = 0
                for a in ex.get("middleware_audit") or []:
                    orig = (a.get("governance") or {}).get("failed_stage") or "approved"
                    tool = (a.get("execution") or {}).get("tool_name")
                    n += 1
                    if max_calls is not None and n > max_calls:
                        new = "invocation_limit"
                    elif max_calls is not None and tool in executed_tools:
                        new = "invocation_limit"
                    elif a.get("proposal") is None:
                        new = "invalid_proposal"
                    else:
                        p = dict(a["proposal"])
                        p["invocation_context"] = {"invocation": {**inv, "context": ctx}}
                        d = await gov.authorize(schema.RANActionProposal.model_validate(p))
                        new = d.failed_stage or "reaches_llm_review"
                        if new == "reaches_llm_review" and orig in ("approved", "llm_review"):
                            # the logged g_V verdict is the best available estimate
                            new = "approved" if orig == "approved" else "llm_review"
                        if new == "approved":
                            executed_tools.add(tool)
                    rows.append(dict(run=doc["run_id"], record=rec_id, agent=agent, tool=tool,
                                     original=orig, replayed=new))
    out = dict(
        project=str(project),
        calls=len(rows),
        original=dict(Counter(r["original"] for r in rows)),
        replayed=dict(Counter(r["replayed"] for r in rows)),
        executed_original=sum(r["original"] == "approved" for r in rows),
        executed_replayed=sum(r["replayed"] == "approved" for r in rows),
        by_agent={ag: dict(original=dict(Counter(r["original"] for r in rows if r["agent"] == ag)),
                           replayed=dict(Counter(r["replayed"] for r in rows if r["agent"] == ag)))
                  for ag in sorted({r["agent"] for r in rows})},
        rows=rows,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=DATA_DIR / "llm_runs")
    ap.add_argument("--tag", default="v3")
    a = ap.parse_args()
    res = asyncio.run(replay(a.project.resolve(), a.runs))
    (a.runs / f"replay_{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
