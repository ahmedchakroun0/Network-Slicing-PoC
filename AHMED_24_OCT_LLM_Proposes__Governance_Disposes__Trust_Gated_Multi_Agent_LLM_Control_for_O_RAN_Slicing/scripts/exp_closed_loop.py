"""E6 - Trace-driven closed loop on ColO-RAN (harness only; PoC unmodified).

Each of the 7 BSs runs a slicing configuration (scheduler s, RBG split k).
Every decision step (D windows = D seconds) the latest 1-s record of each
slice is read from the REAL ColO-RAN traces of the current (s, k, BS),
turned into a PoC TelemetryRecord, and analysed by the PoC's
MonitoringAgent.analyze_sla. Violated slices are routed (Near-RT; Non-RT
when the per-stream persistence reaches k=3) to a RIC proposer that emits
the PoC's own tool calls (adjust_prb_allocation, optimize_scheduler,
update_slice_resource_policy). In the GOVERNED arm every call goes through
the PoC's RANAuthorizationMiddleware + RANTrustGovernanceAgent (LLM review
off: no model available in the sandbox); only approved calls reach the
environment. In the UNGOVERNED arm calls are applied directly.

The LLM is replaced by a scripted proposer whose calls are corrupted with
probability eps by faults observed in the pilot traces (hallucinated
identifiers, out-of-range share, inverted policy) plus a spurious action
on a healthy slice and a semantically wrong (share-decreasing) call that
is in range and well-formed. eps is swept; it is NOT a measured LLM error
rate.

Environment semantics (documented assumptions):
* adjust_prb_allocation(slice, pct): target RBGs = pct% of 17; move to the
  dataset split whose RBG count for that slice is closest to the target,
  then closest to the current split for the other slices.
* optimize_scheduler: BS-wide (ColO-RAN applies one policy per BS):
  round_robin->RR, proportional_fair->PF, max_cqi->WF (closest
  channel-aware policy in the dataset); priority_based -> no-op.
* update_slice_resource_policy(min,max): the slice share is kept inside
  [min,max] (immediately and for later Near-RT moves). An inverted policy
  (min>max), if executed, pins the slice to `max`.
* A call whose target/slice names another existing BS/slice is applied
  there; an unparseable identifier is a no-op.
"""

from __future__ import annotations

import asyncio
import copy
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import DATA_DIR, base_parser, dump_json, use_project
from coloran_adapter import SLICE_NAME, TR_RBG

SL_ID = {v: k for k, v in SLICE_NAME.items()}
TOTAL_RBG = 17
SCHED_OF = {"round_robin": 0, "max_cqi": 1, "proportional_fair": 2}
SCHED_NAME = {0: "round_robin", 1: "max_cqi", 2: "proportional_fair"}
H0, H1 = 60, 260   # history windows [60 s, 260 s): offline profile (Non-RT knowledge)
T0 = 260           # online evaluation starts at 260 s (never seen by the profile)
FAULTS = ["hallucinated_ids", "out_of_range", "inverted_policy", "spurious_healthy", "wrong_direction"]


# ------------------------------------------------------------------ traces
class Traces:
    def __init__(self, df: pd.DataFrame):
        self.cols = ["Use Case Type", "Packet Loss Budget", "Latency Budget (ns)", "Jitter Budget (ns)",
                     "Data Rate Budget (Gbps)", "Required Mobility", "Required Connectivity",
                     "Slice Available Transfer Rate (Gbps)", "Slice Latency (ns)", "Slice Packet Loss",
                     "Slice Jitter (ns)", "Slice Type", "Slice Handover"]
        self.store: dict[tuple, list[dict]] = {}
        self.wins: dict[tuple, list[int]] = {}
        for key, g in df.groupby(["sched", "tr", "bs", "slice"]):
            g = g.sort_values(["exp", "w"])
            self.store[key] = g[self.cols].to_dict("records")
            self.wins[key] = g["w"].tolist()
        self.bss = sorted(df.bs.unique())
        def enough(key):
            ws = self.wins.get(key, [])
            return sum(H0 <= w < H1 for w in ws) >= 100 and sum(w >= T0 for w in ws) >= 120
        self.valid = {(s, k, b) for (s, k, b, _sl) in self.store
                      if all(enough((s, k, b, x)) for x in (0, 1, 2))}

    def record(self, s, k, b, sl, t):
        seq, ws = self.store[(s, k, b, sl)], self.wins[(s, k, b, sl)]
        online = [i for i, w in enumerate(ws) if w >= T0]
        return seq[online[(t - T0) % len(online)]]


@dataclass
class Cell:
    bs: int
    sched: int
    tr: int
    policy: dict = field(default_factory=dict)   # slice -> (min_rbg, max_rbg)


def nearest_split(cur: tuple, sl: int, target_rbg: float, policy: dict, valid_tr: set,
                  stressed: frozenset = frozenset()) -> int:
    """Split closest to the requested share; resources are taken first from
    slices that currently meet their SLA, then from the largest one."""
    best, best_key = None, None
    for k, split in enumerate(TR_RBG):
        if k not in valid_tr:
            continue
        ok = all(lo <= split[x] <= hi for x, (lo, hi) in policy.items())
        others = sum(abs(split[x] - cur[x]) for x in range(3) if x != sl)
        # tie-break: take resources from the largest other slice first
        taken_from_stressed = sum(max(0, cur[x] - split[x]) for x in stressed if x != sl)
        donor = -max((cur[x] - split[x]) * cur[x] for x in range(3) if x != sl)
        key = (0 if ok else 1, abs(split[sl] - target_rbg), taken_from_stressed, others, donor)
        if best_key is None or key < best_key:
            best, best_key = k, key
    return best


class Env:
    def __init__(self, traces: Traces, starts: dict[int, tuple[int, int]]):
        self.tr = traces
        self.cells = {b: Cell(b, s, k) for b, (s, k) in starts.items()}
        self.t = 0
        self.executed = 0
        self.stressed: dict[int, frozenset] = {b: frozenset() for b in starts}

    def valid_tr(self, cell: Cell) -> set:
        return {k for (s, k, b) in self.tr.valid if s == cell.sched and b == cell.bs}

    def observe(self, b: int) -> list[dict]:
        c = self.cells[b]
        return [self.tr.record(c.sched, c.tr, b, sl, self.t) for sl in (0, 1, 2)]

    # ---- apply a (possibly faulty) tool call; returns True if state changed
    def apply(self, tool: str, args: dict) -> bool:
        self.executed += 1
        tgt = str(args.get("target", ""))
        if not tgt.startswith("bs") or not tgt[2:].isdigit() or int(tgt[2:]) not in self.cells:
            return False
        c = self.cells[int(tgt[2:])]
        sl = SL_ID.get(str(args.get("slice_id", "")))
        before = (c.sched, c.tr)
        cur = TR_RBG[c.tr]
        if tool == "adjust_prb_allocation" and sl is not None:
            pct = float(args.get("prb_percentage", 0))
            target = pct / 100 * TOTAL_RBG
            pol = {x: v for x, v in c.policy.items() if v[0] <= v[1]}
            c.tr = nearest_split(cur, sl, target, pol, self.valid_tr(c), self.stressed[c.bs])
        elif tool == "optimize_scheduler":
            s = SCHED_OF.get(str(args.get("scheduler")))
            if s is not None and (s, c.tr, c.bs) in self.tr.valid:
                c.sched = s
        elif tool == "update_slice_resource_policy" and sl is not None:
            lo = float(args.get("min_prb_ratio_pct", 0)) / 100 * TOTAL_RBG
            hi = float(args.get("max_prb_ratio_pct", 100)) / 100 * TOTAL_RBG
            if lo > hi:                      # inverted policy executed without a gate
                c.policy[sl] = (hi, hi)
                c.tr = nearest_split(cur, sl, hi, {}, self.valid_tr(c))
            else:
                c.policy[sl] = (lo, hi)
                if not lo <= cur[sl] <= hi:
                    c.tr = nearest_split(cur, sl, min(max(cur[sl], lo), hi), {sl: (lo, hi)}, self.valid_tr(c))
        return (c.sched, c.tr) != before


# --------------------------------------------------------------- proposer
PRIOR_PROFILE: dict = {}


class Proposer:
    """Scripted stand-in for the RIC agents (no LLM).

    Knowledge: an offline profile of each (scheduler, split) per BS built
    from the HISTORY part of the traces (Non-RT knowledge), blended with
    the compliance observed online (working memory). When a slice has been
    violated for 2 decisions, it scores every configuration reachable with
    ONE PoC tool call on a violated slice (adjust_prb_allocation to any
    share, or optimize_scheduler) and issues the best one if it beats the
    current configuration by MARGIN. When persistence reaches 3, the
    Non-RT agent sets a slice-class floor policy. Tuned on eps=0 only.
    """


    PRIORITY = (2, 0, 1)
    FLOOR_PCT = {0: 11, 1: 5, 2: 11}   # 1.87, 0.85, 1.87 RBGs of 17 -> floors of 2, 1, 2 RBGs
    PRIOR, N0, MARGIN = 0.5, 10.0, 0.05

    def __init__(self, eps: float, rng: random.Random, bss: list[int], env: "Env | None" = None,
                 prior: dict | None = None):
        self.eps, self.rng, self.bss, self.env = eps, rng, bss, env
        self.prior = prior or {}
        self.mem: dict[tuple, list[float]] = {}   # (bs, sched, tr) -> [ewma score, visits]

    def observe(self, b: int, cell: Cell, ok_all: bool):
        m = self.mem.setdefault((b, cell.sched, cell.tr), [0.0, 0])
        m[0] += ok_all
        m[1] += 1

    def value(self, b, s, k):
        """Bayesian blend of the offline profile (weight N0) and online memory."""
        prior = self.prior.get((s, k, b), self.PRIOR)
        hits, n = self.mem.get((b, s, k), [0.0, 0])
        return (prior * self.N0 + hits) / (self.N0 + n)

    def plan(self, b: int, cell: Cell, viol: list[list[str]], rho: dict, healthy: list[int]):
        cands = [sl for sl in self.PRIORITY if viol[sl] and rho[(b, sl)] >= 2]
        if not cands:
            return []
        cur = TR_RBG[cell.tr]
        vt = self.env.valid_tr(cell)
        pol = {x: v for x, v in cell.policy.items() if v[0] <= v[1]}
        stressed = frozenset(x for x in (0, 1, 2) if viol[x])
        options = []   # (value, slice, call) -- every call moves a VIOLATED slice (slice-scoped governance)
        for sl in cands:
            base = {"target": f"bs{b}", "slice_id": SLICE_NAME[sl],
                    "expected_effect": f"restore {', '.join(viol[sl])} for {SLICE_NAME[sl]}",
                    "evidence_sources": ["working_memory", "history_profile"]}
            for r in range(1, 15):
                if r == cur[sl]:
                    continue
                pct = int(max(5, min(90, round(100 * r / TOTAL_RBG))))
                k = nearest_split(cur, sl, pct / 100 * TOTAL_RBG, pol, vt, stressed)
                if k == cell.tr:
                    continue
                v = self.value(b, cell.sched, k)
                options.append((v, sl, ("near_rt_ric", "adjust_prb_allocation",
                                        {**base, "prb_percentage": pct,
                                         "rationale": f"{SLICE_NAME[sl]} violates {viol[sl]}; moving it to "
                                                      f"{pct}% (split {TR_RBG[k]}) has expected compliance {v:.2f}.",
                                         "confidence": round(min(0.95, max(0.05, v)), 2)})))
            for s2 in (0, 1, 2):
                if s2 != cell.sched and (s2, cell.tr, b) in self.env.tr.valid:
                    v = self.value(b, s2, cell.tr)
                    options.append((v, sl, ("near_rt_ric", "optimize_scheduler",
                                            {**base, "scheduler": SCHED_NAME[s2],
                                             "rationale": f"{SLICE_NAME[sl]} violates {viol[sl]}; scheduler "
                                                          f"{SCHED_NAME[s2]} has expected compliance {v:.2f}.",
                                             "confidence": round(min(0.95, max(0.05, v)), 2)})))
        if not options:
            return []
        best_v, sl, call = max(options, key=lambda o: (o[0], -self.PRIORITY.index(o[1])))
        if best_v <= self.value(b, cell.sched, cell.tr) + self.MARGIN:
            return []
        calls = [call]
        if rho[(b, sl)] == 3:   # persistent: Non-RT guarantees a slice-class floor
            calls.append(("non_rt_ric", "update_slice_resource_policy",
                          {"target": f"bs{b}", "slice_id": SLICE_NAME[sl],
                           "min_prb_ratio_pct": self.FLOOR_PCT[sl], "max_prb_ratio_pct": 100,
                           "rationale": f"{SLICE_NAME[sl]} violated for 3 consecutive decisions; guarantee "
                                        f"its class minimum share.",
                           "expected_effect": f"protect {SLICE_NAME[sl]} from starvation",
                           "confidence": 0.8, "evidence_sources": ["history_profile"]}))
        return [self.corrupt(c, b, sl, healthy) for c in calls]

    def corrupt(self, call, b, sl, healthy):
        agent, tool, args = call
        if self.rng.random() >= self.eps:
            return (*call, None, sl)
        # "spurious_healthy" needs a healthy slice to act on; otherwise draw another fault
        f = self.rng.choice(FAULTS if healthy else [x for x in FAULTS if x != "spurious_healthy"])
        a = copy.deepcopy(args)
        if f == "hallucinated_ids":
            if self.rng.random() < 0.5:   # copies record/run ids (pilot behaviour) -> unparseable
                a["target"], a["slice_id"] = f"coloran_{self.rng.randint(0, 99999):06d}", "run-1"
            else:                          # valid but wrong cell / slice
                a["target"] = f"bs{self.rng.choice([x for x in self.bss if x != b])}"
                a["slice_id"] = SLICE_NAME[self.rng.choice([x for x in (0, 1, 2) if x != sl])]
        elif f == "out_of_range":
            tool, agent = "adjust_prb_allocation", "near_rt_ric"
            a = {k: v for k, v in a.items() if k not in ("scheduler", "min_prb_ratio_pct", "max_prb_ratio_pct")}
            a["prb_percentage"] = self.rng.choice([1, 2, 3, 95, 99])
        elif f == "inverted_policy":
            tool, agent = "update_slice_resource_policy", "non_rt_ric"
            a = {k: v for k, v in a.items() if k not in ("scheduler", "prb_percentage")}
            a["min_prb_ratio_pct"], a["max_prb_ratio_pct"] = self.rng.randint(60, 80), self.rng.randint(20, 30)
        elif f == "spurious_healthy":
            h = self.rng.choice(healthy)
            tool, agent = "adjust_prb_allocation", "near_rt_ric"
            a = {k: v for k, v in a.items() if k not in ("scheduler", "min_prb_ratio_pct", "max_prb_ratio_pct")}
            a["slice_id"], a["prb_percentage"] = SLICE_NAME[h], self.rng.choice([10, 20, 70, 80])
            return (agent, tool, a, f, h)
        elif f == "wrong_direction":
            tool, agent = "adjust_prb_allocation", "near_rt_ric"
            a = {k: v for k, v in a.items() if k not in ("scheduler", "min_prb_ratio_pct", "max_prb_ratio_pct")}
            a["prb_percentage"] = 12      # ~2 RBGs: shrinks the violated slice
        return (agent, tool, a, f, sl)


# ------------------------------------------------------------------ runner
class _Req:
    def __init__(self, tool, args, inv, i):
        self.tool_call = {"name": tool, "args": copy.deepcopy(args), "id": f"c{i}"}
        self.state = {"agent_invocation": inv}


async def run_episode(traces, starts, eps, governed, seed, steps, D, mods):
    MonitoringAgent, TelemetryRecord, mw, RAN_AUDIT = mods
    rng = random.Random(seed)
    env = Env(traces, dict(starts))
    prop = Proposer(eps, rng, traces.bss, env, PRIOR_PROFILE)
    rho = {(b, sl): 0 for b in traces.bss for sl in (0, 1, 2)}
    log, ncall = [], 0
    stats = dict(calls=0, executed=0, faulty=0, faulty_executed=0, blocked=0, valid_blocked=0)
    for step in range(steps):
        env.t = T0 + step * D
        for b in traces.bss:
            recs = env.observe(b)
            anal, viol = [], []
            for sl, r in enumerate(recs):
                tr = TelemetryRecord.model_validate({**r, "record_id": f"bs{b}-s{sl}-t{env.t}"})
                a = MonitoringAgent.analyze_sla(tr)
                anal.append(a)
                viol.append(a.violations)
                rho[(b, sl)] = rho[(b, sl)] + 1 if a.violations else 0
            c = env.cells[b]
            log.append({"step": step, "bs": b, "sched": c.sched, "tr": c.tr,
                        **{f"ok_{SLICE_NAME[sl]}": not viol[sl] for sl in (0, 1, 2)},
                        "ok_all": not any(viol),
                        "embb_mbps": recs[0]["Slice Available Transfer Rate (Gbps)"] * 1e3,
                        "urllc_delay_ms": recs[2]["Slice Latency (ns)"] / 1e6})
            healthy = [sl for sl in (0, 1, 2) if not viol[sl]]
            env.stressed[b] = frozenset(sl for sl in (0, 1, 2) if viol[sl])
            prop.observe(b, c, not any(viol))
            if True:
                for agent, tool, args, fault, ctx_sl in prop.plan(b, c, viol, rho, healthy):
                    sl = ctx_sl
                    stats["calls"] += 1
                    stats["faulty"] += fault is not None
                    if fault is not None:
                        stats[f"n_{fault}"] = stats.get(f"n_{fault}", 0) + 1
                    if not governed:
                        env.apply(tool, args)
                        stats["executed"] += 1
                        stats["faulty_executed"] += fault is not None
                        if fault is not None:
                            stats[f"x_{fault}"] = stats.get(f"x_{fault}", 0) + 1
                        continue
                    ctx = {"target": f"bs{b}", "slice_id": SLICE_NAME[ctx_sl],
                           "sla_analysis": anal[ctx_sl].model_dump(mode="json"),
                           "persistence": {"stream": rho[(b, ctx_sl)]}}
                    inv = {"agent": agent, "responsibility": "restore SLA", "objective": f"fix {viol[ctx_sl]}",
                           "priority": "high", "reason": "deterministic violation", "context": ctx}
                    done = {"x": False}

                    async def handler(_req, tool=tool, args=args):
                        done["x"] = True
                        env.apply(tool, args)
                        return {"status": "success"}

                    audit = []
                    tok = RAN_AUDIT.set(audit)
                    try:
                        ncall += 1
                        await mw.awrap_tool_call(_Req(tool, args, inv, ncall), handler)
                    finally:
                        RAN_AUDIT.reset(tok)
                    if done["x"]:
                        stats["executed"] += 1
                        stats["faulty_executed"] += fault is not None
                        if fault is not None:
                            stats[f"x_{fault}"] = stats.get(f"x_{fault}", 0) + 1
                    else:
                        stats["blocked"] += 1
                        stats["valid_blocked"] += fault is None
    return pd.DataFrame(log), stats


async def main_async(args):
    from agents.monitoring_agent import MonitoringAgent
    from agents.ran_trust_governance import RANTrustGovernanceAgent
    from middleware.ran_authorisation import RANAuthorizationMiddleware, RAN_AUDIT
    from src.schema.monitoring import TelemetryRecord

    df = pd.read_csv(DATA_DIR / "coloran_windows.csv.gz")
    # keep only windows where all three slices of the cell are present, so the
    # slices of a cell stay time-aligned and the oracle compares complete windows
    full = df.groupby(["sched", "tr", "exp", "bs", "w"]).slice.transform("nunique") == 3
    df = df[full].reset_index(drop=True)
    traces = Traces(df)
    mw = RANAuthorizationMiddleware(trust_governance_agent=RANTrustGovernanceAgent(use_llm=False))
    mods = (MonitoringAgent, TelemetryRecord, mw, RAN_AUDIT)

    df["ok"] = df["SLA Evaluation Status"] == "ok"
    allok = df.groupby(["sched", "tr", "bs", "exp", "w"]).ok.all().reset_index()
    # offline profile from HISTORY windows only (Non-RT knowledge)
    allok = allok[[(int(s), int(k), int(b)) in traces.valid for s, k, b in zip(allok.sched, allok.tr, allok.bs)]]
    hist = allok[(allok.w >= H0) & (allok.w < H1)].groupby(["sched", "tr", "bs"]).ok.mean()
    PRIOR_PROFILE.clear()
    PRIOR_PROFILE.update({(int(s), int(k), int(b)): float(v) for (s, k, b), v in hist.items()})
    # oracle: best static (sched, split) per BS in hindsight on the ONLINE windows
    comp = allok[allok.w >= T0].groupby(["sched", "tr", "bs"]).ok.mean()
    oracle = {int(b): float(comp.xs(b, level="bs").max()) for b in traces.bss}
    # profile-only baseline value: online compliance of the historically best split per BS
    best_hist = hist.groupby("bs").idxmax()
    profile_static = {int(b): float(comp.loc[best_hist[b]]) for b in traces.bss}

    rng = np.random.default_rng(args.seed)
    valid = sorted(traces.valid)
    episodes = []
    for e in range(args.episodes):
        starts = {}
        for b in traces.bss:
            opts = [(s, k) for (s, k, bb) in valid if bb == b]
            starts[b] = opts[rng.integers(len(opts))]
        episodes.append(starts)

    rows, ts = [], []
    arms = [("static", None, 0.0)] + [(f"{'gov' if g else 'nogov'}", g, e) for e in args.eps for g in (False, True)]
    for name, governed, eps in arms:
        for ei, starts in enumerate(episodes):
            if governed is None:   # static: no proposer at all
                lg, st = await run_episode(traces, starts, 0.0, True, 10_000 + ei, args.steps, args.D,
                                           (MonitoringAgent, TelemetryRecord, _NoMW(), RAN_AUDIT))
            else:
                lg, st = await run_episode(traces, starts, eps, governed, 10_000 + ei, args.steps, args.D, mods)
            lg["arm"], lg["eps"], lg["episode"] = name, eps, ei
            ts.append(lg)
            rows.append({"arm": name, "eps": eps, "episode": ei, **st,
                         "ok_all": lg.ok_all.mean(), "ok_eMBB": lg.ok_eMBB.mean(), "ok_MTC": lg.ok_MTC.mean(),
                         "ok_URLLC": lg.ok_URLLC.mean(),
                         "ok_all_last50": lg[lg.step >= args.steps // 2].ok_all.mean(),
                         "reconfigs": int((lg.groupby("bs")[["sched", "tr"]].diff().abs().sum(axis=1) > 0).sum())})
        print(name, eps, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR / "e6_closed_loop_episodes.csv", index=False)
    tsd = pd.concat(ts)
    tsd.groupby(["arm", "eps", "step"]).agg(ok_all=("ok_all", "mean"), ok_eMBB=("ok_eMBB", "mean"),
                                           ok_URLLC=("ok_URLLC", "mean"), ok_MTC=("ok_MTC", "mean"),
                                           embb_mbps=("embb_mbps", "mean"),
                                           urllc_delay_ms=("urllc_delay_ms", "median")).reset_index() \
        .to_csv(DATA_DIR / "e6_closed_loop_timeseries.csv", index=False)
    summ = res.groupby(["arm", "eps"]).agg(ok_all=("ok_all", "mean"), ok_all_sd=("ok_all", "std"),
                                          ok_last=("ok_all_last50", "mean"), ok_eMBB=("ok_eMBB", "mean"),
                                          ok_URLLC=("ok_URLLC", "mean"), ok_MTC=("ok_MTC", "mean"),
                                          calls=("calls", "sum"), executed=("executed", "sum"),
                                          faulty=("faulty", "sum"), faulty_executed=("faulty_executed", "sum"),
                                          valid_blocked=("valid_blocked", "sum"),
                                          reconfigs=("reconfigs", "mean")).reset_index()
    summ.to_csv(DATA_DIR / "e6_closed_loop_summary.csv", index=False)
    fcols = [c for c in res.columns if c.startswith(("n_", "x_"))]
    res.fillna({c: 0 for c in fcols}).groupby(["arm"])[fcols].sum().to_csv(DATA_DIR / "e6_fault_breakdown.csv")
    dump_json({"oracle_static_per_bs": oracle, "oracle_static_mean": float(np.mean(list(oracle.values()))),
               "profile_static_per_bs": profile_static,
               "profile_static_mean": float(np.nanmean(list(profile_static.values()))),
               "episodes": args.episodes, "steps": args.steps, "D_s": args.D, "bss": [int(b) for b in traces.bss]},
              "e6_meta.json")
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string())
    print("oracle", np.mean(list(oracle.values())), "profile-static", np.nanmean(list(profile_static.values())))


class _NoMW:
    async def awrap_tool_call(self, request, handler):
        return None


def main():
    p = base_parser(__doc__)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--D", type=int, default=3, help="seconds (1-s windows) between decisions")
    p.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    a = p.parse_args()
    use_project(a.project)
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
