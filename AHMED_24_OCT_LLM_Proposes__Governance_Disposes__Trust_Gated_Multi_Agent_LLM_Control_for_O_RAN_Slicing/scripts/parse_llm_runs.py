"""Parse mongosh dumps of PoC run documents (results/run-*.txt) and audit them.

The files are printed by mongosh (JS object literals: unquoted keys, single-
quoted strings), not JSON. This script tokenizes them without eval(), rebuilds
the run documents, and checks every logged decision against the deterministic
rules of the PoC:

  * SLA labels:   computed_analysis vs ground truth vs re-computation
  * Monitoring:   action_required <=> violations; duplicate agent selection;
                  objectives naming only violated KPIs; claims of "recent
                  execution" when the working memory was empty
  * RIC calls:    every protected tool call (middleware_audit), its first
                  failing governance stage, objective alignment (does the
                  rationale/expected effect address the KPI of the objective?),
                  knowledge use, extreme-but-legal parameters
  * Lost audit:   invocations counted in summary.agent_invocations but absent
                  from agent_executions (keyed by agent name)

Usage: python -I parse_llm_runs.py RESULTS_DIR OUT_DIR
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

TOKEN = re.compile(
    r"""\s*(?:
      (?P<s1>'(?:[^'\\]|\\.)*')
    | (?P<s2>"(?:[^"\\]|\\.)*")
    | (?P<num>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    | (?P<id>[A-Za-z_$][\w$]*)
    | (?P<p>[{}\[\]:,()])
    )""",
    re.X,
)


def tokenize(text: str):
    pos = 0
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m:
            if text[pos:].strip() == "":
                return
            raise ValueError(f"cannot parse at {pos}: {text[pos:pos+40]!r}")
        pos = m.end()
        kind = m.lastgroup
        val = m.group(kind)
        if kind in ("s1", "s2"):
            body = val[1:-1]
            # JS escapes -> python
            body = re.sub(r"\\(.)", lambda x: {"n": "\n", "t": "\t"}.get(x.group(1), x.group(1)), body)
            yield ("str", body)
        elif kind == "num":
            yield ("num", float(val) if re.search(r"[.eE]", val) else int(val))
        elif kind == "id":
            yield ("id", val)
        else:
            yield ("p", val)


class Parser:
    def __init__(self, toks):
        self.t = list(toks)
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, expect=None):
        tok = self.t[self.i]
        self.i += 1
        if expect is not None and tok[1] != expect:
            raise ValueError(f"expected {expect}, got {tok}")
        return tok

    def value(self):
        kind, v = self.take()
        if kind == "p" and v == "{":
            obj = {}
            while self.peek()[1] != "}":
                k_kind, k = self.take()
                self.take(":")
                obj[str(k)] = self.value()
                if self.peek()[1] == ",":
                    self.take(",")
            self.take("}")
            return obj
        if kind == "p" and v == "[":
            arr = []
            while self.peek()[1] != "]":
                arr.append(self.value())
                if self.peek()[1] == ",":
                    self.take(",")
            self.take("]")
            return arr
        if kind in ("str", "num"):
            return v
        if kind == "id":
            if v in ("true", "false"):
                return v == "true"
            if v in ("null", "undefined"):
                return None
            if self.peek()[1] == "(":  # ObjectId('..'), ISODate('..')
                self.take("(")
                inner = self.value()
                self.take(")")
                return f"{v}({inner})"
            return v
        raise ValueError(f"unexpected token {kind} {v}")


def load_docs(path: Path) -> list[dict]:
    text = path.read_text()
    p = Parser(tokenize(text))
    docs = []
    while p.i < len(p.t):
        kind, v = p.peek()
        if v == "{":
            docs.append(p.value())
        else:
            p.take()  # stray '[' / ']' / ',' from a partially copied array
    return docs


KPIS = ("latency", "jitter", "packet_loss", "throughput")
KPI_WORDS = {
    "latency": ("latency", "delay"),
    "jitter": ("jitter",),
    "packet_loss": ("packet loss", "packet_loss", "loss"),
    "throughput": ("throughput", "data rate", "transfer rate"),
}


def kpis_named(text: str) -> set[str]:
    t = (text or "").lower()
    return {k for k, words in KPI_WORDS.items() if any(w in t for w in words)}


def recompute(tr: dict) -> list[str]:
    v = []
    if tr["slice_latency_ns"] > tr["latency_budget_ns"]:
        v.append("latency")
    if tr["slice_jitter_ns"] > tr["jitter_budget_ns"]:
        v.append("jitter")
    if tr["slice_packet_loss"] > tr["packet_loss_budget"]:
        v.append("packet_loss")
    if tr["slice_available_transfer_rate_gbps"] < tr["data_rate_budget_gbps"]:
        v.append("throughput")
    return v


def extreme(tool: str, params: dict) -> str | None:
    """Legal (passes resource_safety) but at the edge of the allowed range."""
    if tool in ("update_slice_resource_policy", "optimize_slice_capacity_policy"):
        if (params.get("min_prb_ratio_pct") or 0) >= 60:
            return f"guaranteed PRB share {params['min_prb_ratio_pct']}% for one slice"
    if tool == "adjust_prb_allocation" and (params.get("prb_percentage") or 0) >= 80:
        return f"PRB share {params['prb_percentage']}%"
    return None


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    runs, calls, records, issues = [], [], [], []
    for f in sorted(src.glob("run-*.txt")):
        for doc in load_docs(f):
            runs.append(doc)
            (out / f"{doc['run_id']}.json").write_text(json.dumps(doc, indent=1))
    for doc in runs:
        rid = doc["run_id"]
        mem_executions = 0  # memory is reset at run start (run_dataset.py)
        inv_logged = Counter()
        for rec_id, rec in doc["records"].items():
            tr = rec["telemetry_record"]
            ca = rec["computed_analysis"]
            gt = rec["ground_truth"]
            md = rec["monitoring_decision"] or {}
            viol = ca["violations"]
            row = dict(run=rid, record=rec_id, slice=tr["slice_type"], cell=tr.get("cell_id"),
                       violations=",".join(viol), label_ok=(sorted(viol) == sorted(recompute(tr)) ==
                                                           sorted(gt["sla_violations"])),
                       action_required=md.get("action_required"), severity=md.get("severity"),
                       situation=md.get("situation"), global_situation=md.get("global_situation"))
            sel = md.get("selected_agents") or []
            row["selected"] = ",".join(f"{a['agent']}:{a['responsibility']}" for a in sel)
            agents = [a["agent"] for a in sel]
            row["duplicate_agent"] = len(agents) != len(set(agents))
            row["routing_consistent"] = bool(viol) == bool(md.get("action_required")) == bool(sel)
            obj_kpis = set().union(*(kpis_named(a["objective"]) for a in sel)) if sel else set()
            row["objective_names_only_violated"] = obj_kpis <= set(viol)
            row["objectives_cover_violations"] = set(viol) <= obj_kpis
            claims = [a for a in sel if re.search(r"recent execution|executed successfully", a.get("reason", ""), re.I)]
            row["claims_recent_execution"] = len(claims)
            row["memory_executions_before"] = mem_executions
            if claims and mem_executions == 0:
                issues.append((rid, rec_id, "monitoring", "fabricated history",
                               f"{len(claims)} selected agent(s) justified by 'Recent execution ... executed successfully' "
                               "although the working memory was reset and empty"))
            if viol and md.get("situation") == "normal":
                issues.append((rid, rec_id, "monitoring", "situation inconsistent",
                               f"situation='normal' with severity={md.get('severity')!r} and violations {viol}"))
            if row["duplicate_agent"]:
                issues.append((rid, rec_id, "monitoring", "duplicate selection",
                               f"selected {agents}; the prompt says never select an agent twice"))
            for a in sel:
                inv_logged[a["agent"]] += 1
            ex = rec.get("agent_executions") or {}
            row["invocations_selected"] = len(sel)
            row["invocations_logged"] = len(ex)
            if len(ex) < len(sel):
                lost = [f"{a['agent']}:{a['responsibility']}" for a in sel
                        if not any(e["invocation"]["responsibility"] == a["responsibility"] and e["agent"] == a["agent"]
                                   for e in ex.values())]
                issues.append((rid, rec_id, "audit", "lost invocation",
                               f"{len(sel)} invocations, {len(ex)} stored (agent_executions is keyed by agent name); "
                               f"lost: {lost}"))
            for agent, e in ex.items():
                inv = e["invocation"]
                obj = kpis_named(inv["objective"])
                res = e.get("result") or {}
                for k, a in enumerate(e.get("middleware_audit") or []):
                    exe = a.get("execution") or {}
                    gov = a.get("governance") or {}
                    prop = a.get("proposal") or {}
                    params = {k2: v for k2, v in (exe.get("parameters") or {}).items() if k2 != "target"}
                    tool = exe.get("tool_name") or prop.get("tool_name")
                    addressed = kpis_named((prop.get("rationale") or "") + " " + (prop.get("expected_effect") or ""))
                    eff = kpis_named(prop.get("expected_effect") or "")
                    c = dict(run=rid, record=rec_id, agent=agent, objective_kpis=",".join(sorted(obj)),
                             call_index=k, tool_call_id=a["tool_call_id"],
                             forced=bool(re.fullmatch(r"call_[0-9a-f]{12}", a["tool_call_id"])),
                             tool=tool, params=json.dumps(params, sort_keys=True),
                             approved=bool(gov.get("approved")), failed_stage=gov.get("failed_stage") or "",
                             reason=(gov.get("reason") or "")[:300],
                             expected_effect=prop.get("expected_effect"),
                             expected_effect_kpis=",".join(sorted(eff)),
                             addresses_objective=bool(eff & obj) if prop else None,
                             target_ok=(exe.get("target") == inv["context"]["target"]),
                             slice_ok=(params.get("slice_id") in (None, inv["context"]["slice_id"])),
                             evidence_sources=len(prop.get("evidence_sources") or []),
                             confidence=prop.get("confidence"),
                             extreme=extreme(tool, params) or "")
                    # numbers quoted in the rationale vs the trusted context
                    # Accepted: a context value (measured/budget) in any unit that is a
                    # power of 1000 away (ns->ms, Gbps->Mbps, "484.28M"), measured/budget
                    # ratios and percentages, rounded to 0.5 %. A dropped or added digit
                    # (factor 10) is NOT accepted.
                    pairs = [(float(v["measured"]), float(v["budget"]))
                             for v in inv["context"]["sla_analysis"].values()
                             if isinstance(v, dict) and isinstance(v.get("measured"), (int, float))
                             and isinstance(v.get("budget"), (int, float)) and v["budget"]]
                    ctx_vals = [s * 1000.0 ** k for m, b in pairs for s in (m, b) for k in range(-4, 5)]
                    for m, b in pairs:
                        if m:
                            ctx_vals += [m / b, b / m, 100 * m / b, 100 * abs(1 - m / b)]
                    if isinstance(prop.get("confidence"), (int, float)):
                        ctx_vals.append(float(prop["confidence"]))  # self-quoted confidence
                    bad_nums = []
                    text = re.sub(r"(?<=\d),(?=\d{3})", "", prop.get("rationale") or "")
                    for tok in re.findall(r"\d+\.\d+|\d{4,}", text):
                        x = float(tok)
                        if not any(abs(x - v) <= 5e-3 * abs(v) for v in ctx_vals if v):
                            bad_nums.append(tok)
                    c["misquoted_numbers"] = ",".join(bad_nums)
                    c["claims_current_value"] = bool(re.search(r"\bfrom \d+(\.\d+)? ?%? to \d+", prop.get("rationale") or ""))
                    calls.append(c)
                    if bad_nums:
                        issues.append((rid, rec_id, agent, "misquoted KPI value" + (" approved" if c["approved"] else ""),
                                       f"{tool}: rationale quotes {bad_nums}, not in the trusted context"))
                    if c["claims_current_value"]:
                        issues.append((rid, rec_id, agent, "claims unknown current value",
                                       f"{tool}: {prop.get('rationale')[:80]!r}"))
                    if c["approved"] and c["addresses_objective"] is False:
                        issues.append((rid, rec_id, agent, "objective mismatch approved",
                                       f"{tool} {params} approved; objective={sorted(obj)}, expected_effect="
                                       f"{prop.get('expected_effect')!r}"))
                    if c["approved"] and c["extreme"]:
                        issues.append((rid, rec_id, agent, "extreme action approved",
                                       f"{tool} {params}: {c['extreme']}; passes the 5<=min<=80 limit and the LLM review"))
                    if c["approved"] and agent == "non_rt_ric":
                        issues.append((rid, rec_id, agent, "policy action on a single record",
                                       f"{tool} executed although the violation was observed in 1 record "
                                       "(v2 persistence gate removed)"))
                    if gov.get("failed_stage") == "schema" and "Unknown" not in gov.get("reason", ""):
                        pass
                if not any((a.get("proposal") or {}).get("evidence_sources") for a in e.get("middleware_audit") or []):
                    # NB: result.retrieved_knowledge is never populated by the PoC code,
                    # so whether search_ran_knowledge was called cannot be read from the logs.
                    issues.append((rid, rec_id, agent, "no evidence cited",
                                   "evidence_sources empty in every call of this invocation"))
                mem_executions += sum(1 for a in e.get("middleware_audit") or []
                                      if (a.get("execution") or {}).get("status") == "executed")
            records.append(row)
        s = (doc.get("summary") or {}).get("agent_invocations")
        if s is None:
            continue
        if dict(inv_logged) != {k: v for k, v in s.items() if v}:
            issues.append((rid, "-", "summary", "count mismatch", f"summary {s} vs selected {dict(inv_logged)}"))

    import csv
    for name, rows in (("llm_records.csv", records), ("llm_calls.csv", calls)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(out / "llm_issues.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "record", "component", "issue", "detail"])
        w.writerows(issues)

    stage = Counter(c["failed_stage"] or "approved" for c in calls)
    summ = dict(
        runs=[d["run_id"] for d in runs],
        records=len(records),
        distinct_records=len({(r["record"]) for r in records}),
        violating_records=sum(bool(r["violations"]) for r in records),
        label_agreement=sum(r["label_ok"] for r in records) / len(records),
        routing_consistent=sum(r["routing_consistent"] for r in records) / len(records),
        invocations_selected=sum(r["invocations_selected"] for r in records),
        invocations_logged=sum(r["invocations_logged"] for r in records),
        protected_calls=len(calls),
        forced_calls=sum(c["forced"] for c in calls),
        outcome=dict(stage),
        approved=sum(c["approved"] for c in calls),
        approved_addressing_objective=sum(1 for c in calls if c["approved"] and c["addresses_objective"]),
        approved_extreme=sum(1 for c in calls if c["approved"] and c["extreme"]),
        target_slice_ok=sum(c["target_ok"] and c["slice_ok"] for c in calls),
        calls_with_evidence=sum(c["evidence_sources"] > 0 for c in calls),
        llm_review_calls=sum(1 for c in calls if c["approved"] or c["failed_stage"] == "llm_review"),
        llm_review_rejections=stage.get("llm_review", 0),
        issues=Counter(i[3] for i in issues),
    )
    (out / "llm_summary.json").write_text(json.dumps(summ, indent=1, default=dict))
    print(json.dumps(summ, indent=1, default=dict))


if __name__ == "__main__":
    main()
