"""Automatic checks of the agents' interpretations and decisions.

Deterministic rules applied to the stored records (no model is involved), so
the scores are reproducible. They play the part of a first expert pass; the
human ratings of expert_rating.csv remain the reference.

Record level (Monitoring Agent, using its own decision before the code
overrode it):
  sla_recomputed_agrees  the stored SLA analysis equals a recomputation from
                         the stored telemetry and the SLA profiles
  llm_flags_violation    the LLM asked for action and selected an agent
  llm_situation_ok       the LLM did not call a violated record "normal"
  llm_severity_agrees    the LLM severity equals the computed severity
  llm_routing_agrees     the LLM's agents equal the routing rule: Near-RT on
                         every violation, plus Non-RT from the persistence
                         threshold on
  objective_names_kpis   every objective names every violated KPI
  monitoring_numbers_*   numbers quoted in the decision traceable to context

Action level (RIC agents):
  diagnosis              from the telemetry: prb_limited (utilisation >=
                         PRB_LIMITED_UTILISATION), radio_limited (lower
                         utilisation and CQI < RADIO_LIMITED_CQI), or unclear
  tool_cause             consistent / neutral / inconsistent with it (TOOL_CAUSE)
  direction_ok           a PRB share or guarantee that raises the slice's
                         current share; None for other tools
  feasible               a PRB share within what the slice has plus the spare
                         PRBs of the other slices
  no_op                  a scheduler equal to the one in use, or the current share
  numbers_*              numbers quoted in the rationale traceable to context
  evidence_*             evidence sources supported by the context (numbers)
                         or by the knowledge actually retrieved (words)

The rules are heuristics, stated so they can be argued with: a number can be
correct yet derived in a way not covered, and word overlap can accept a
distorted paraphrase. Untraceable numbers are listed for manual review.
"""

from __future__ import annotations

import math
import re
from typing import Any

from agents.monitoring_agent import MonitoringAgent
from agents.ran_trust_governance import NON_RT_MIN_PERSISTENCE
import agents.ran_trust_governance as governance
from langchain_core.tools import BaseTool

from src.schema.monitoring import TelemetryRecord
from src.schema.sla import SLA_PROFILES
from tools import near_rt_ric_tools, non_rt_ric_tools

PRB_LIMITED_UTILISATION = 0.9
RADIO_LIMITED_CQI = 7.0
EVIDENCE_WORD_SUPPORT = 0.6

# How each tool relates to the cause of a throughput / delay violation.
# ColO-RAN UEs are static: a handover change cannot address a violation.
TOOL_CAUSE = {
    "adjust_prb_allocation": {"prb_limited": "consistent", "radio_limited": "inconsistent",
                              "unclear": "inconsistent"},
    "update_slice_resource_policy": {"prb_limited": "consistent", "radio_limited": "inconsistent",
                                     "unclear": "inconsistent"},
    "adjust_link_adaptation": {"prb_limited": "inconsistent", "radio_limited": "consistent",
                               "unclear": "neutral"},
    "adjust_power_control": {"prb_limited": "inconsistent", "radio_limited": "consistent",
                             "unclear": "neutral"},
    "optimize_scheduler": {"prb_limited": "neutral", "radio_limited": "neutral",
                           "unclear": "neutral"},
    "adjust_qos_parameters": {"prb_limited": "neutral", "radio_limited": "neutral",
                              "unclear": "neutral"},
    "optimize_handover": {"prb_limited": "inconsistent", "radio_limited": "inconsistent",
                          "unclear": "inconsistent"},
}

KPI_WORDS = {"throughput": ("throughput",),
             "queueing_delay": ("queueing_delay", "queueing delay", "queuing delay", "delay",
                                "latency")}

_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?:x|%)?(?!\w)(?!\.\d)")
_WORD = re.compile(r"[a-z][a-z0-9_]{3,}")
_STOP = {"with", "from", "that", "this", "than", "into", "when", "which", "their", "there",
         "have", "been", "were", "will", "slice", "knowledge", "retrieved", "result",
         "search_ran_knowledge", "pattern", "context", "invocation"}
# What the agents saw of the tools: their descriptions count as knowledge.
_TOOLS = [t for module in (near_rt_ric_tools, non_rt_ric_tools)
          for t in vars(module).values() if isinstance(t, BaseTool)]
TOOL_DESCRIPTIONS = [f"{t.name}: {t.description}" for t in _TOOLS]
# Names that make an evidence source a statement about the context.
_CONTEXT_NAMES = (set(TelemetryRecord.model_fields)
                  | {arg for t in _TOOLS for arg in t.args} | {"budget"}) - {
    "target", "rationale", "expected_effect", "confidence", "evidence_sources"}
# What the working memory showed of a slice's records: its last k records.
MEMORY_RECORDS = 10
MEMORY_FIELDS = ("window_start_s", "slice_prb", "prb_utilisation", "dl_throughput_per_ue_mbps",
                 "queueing_delay_ms")
_CONTEXT_PREFIXES = ("sla", "slice_state", "slice state", "persistence", "working memory",
                     "slice_records", "cell_records", "slice_actions", "cell_actions",
                     "objective", "invocation", "prior ", "previous ", "other slices",
                     "telemetry", "cell:", "cell conditions", "util:", "active_controls",
                     "active controls", "context", "kpi", "measured", "current ", *TOOL_CAUSE)


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------


def numbers_in(text: str | None) -> list[str]:
    return _NUMBER.findall(text or "")


def _numeric_values(data: Any) -> list[float]:
    if isinstance(data, bool) or data is None:
        return []
    if isinstance(data, (int, float)):
        return [] if isinstance(data, float) and math.isnan(data) else [float(data)]
    if isinstance(data, dict):
        return [v for value in data.values() for v in _numeric_values(value)]
    if isinstance(data, (list, tuple)):
        return [v for value in data for v in _numeric_values(value)]
    if isinstance(data, str):
        return [float(n) for n in numbers_in(data)]
    return []


def _constants() -> list[float]:
    values = [float(NON_RT_MIN_PERSISTENCE), PRB_LIMITED_UTILISATION]
    for name in dir(governance):
        if name.endswith(("_RANGE", "_RANGE_PCT", "_RANGE_MS", "_RANGE_DB", "_RANGE_MBPS")):
            values += _numeric_values(getattr(governance, name))
    for profile in SLA_PROFILES.values():
        values += [term.budget for term in profile.values()]
    return values


_CONSTANTS = _constants()


def _derived(telemetry: dict[str, Any], sla: dict[str, Any]) -> list[float]:
    values = []
    for kpi in (sla.get("kpis") or {}).values():
        m, b = kpi.get("measured"), kpi.get("budget")
        if not m or not b:
            continue
        values += [m / b, b / m, 100 * m / b, 100 * b / m, 100 * (1 - m / b), 100 * (m / b - 1),
                   100 * (b - m) / m, m - b, b - m]
    prb, cell, spare = (telemetry.get(k) for k in ("slice_prb", "cell_prb", "cell_spare_prb"))
    if prb and cell:
        values += [100 * prb / cell, cell - prb, 100 * (cell - prb) / cell]
        if spare is not None:
            values += [prb + spare, 100 * (prb + spare) / cell, 100 * spare / cell]
    return values


def traceable(number: str, context: list[float]) -> bool:
    """A quoted number is a context value rounded or truncated to the digits
    quoted, or within 0.1% of it."""
    value = float(number)
    decimals = len(number.split(".")[1]) if "." in number else 0
    scale = 10 ** decimals
    for candidate in context:
        if (abs(round(candidate, decimals) - value) < 1e-9
                or abs(math.trunc(candidate * scale) / scale - value) < 1e-9
                or abs(value - candidate) <= 0.001 * abs(candidate)):
            return True
    return False


def number_check(texts: list[str | None], context: list[float]) -> dict[str, Any]:
    quoted = [n for text in texts for n in numbers_in(text)]
    untraced = [n for n in quoted if not traceable(n, context)]
    return {"quoted": len(quoted), "untraced": len(untraced), "untraced_values": untraced}


# --------------------------------------------------------------------------
# Context the agents had
# --------------------------------------------------------------------------


def _param_values(params: dict[str, Any], telemetry: dict[str, Any]) -> list[float]:
    """A call's parameters, and what they amount to: a PRB share in PRBs and
    in PRBs added, an MCS offset applied to the current MCS."""
    values = _numeric_values(params)
    cell = telemetry.get("cell_prb") or 0
    for key, value in params.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if key.endswith("_pct") or key == "prb_percentage":
                values.append(value * cell / 100)
                values.append(value * cell / 100 - (telemetry.get("slice_prb") or 0))
            if key == "mcs_offset" and telemetry.get("dl_mcs") is not None:
                values.append(telemetry["dl_mcs"] + value)
    return values


def _record_context(doc: dict[str, Any], history: list[dict[str, Any]],
                    rows: list[dict[str, Any]] = ()) -> list[float]:
    """Numbers the Monitoring Agent could see: the record, the run's earlier
    records of the same cell and the dataset rows of the cell up to now
    (working memory, healthy records included), and the constants."""
    telemetry, sla = doc.get("telemetry") or {}, doc.get("sla") or {}
    values = (_numeric_values(telemetry) + _numeric_values(sla) + _derived(telemetry, sla)
              + _numeric_values(doc.get("persistence")) + _CONSTANTS)
    for row in rows:
        values += _numeric_values({key: row.get(key) for key in MEMORY_FIELDS})
    for earlier in history:
        values += _numeric_values(earlier.get("telemetry"))
        for run in (earlier.get("agents") or {}).values():
            values += [v for a in run.get("actions", [])
                       for v in _param_values(a.get("params") or {}, telemetry)]
    return values


def _knowledge(run: dict[str, Any]) -> list[str]:
    """What the agent retrieved (search results)."""
    return [entry.get("result") or "" for entry in run.get("knowledge", [])]


def _agent_context(doc: dict[str, Any], history: list[dict[str, Any]],
                   agent: str, rows: list[dict[str, Any]] = ()) -> list[float]:
    """What a RIC agent could see on top: the objective, its retrieved
    knowledge and the calls already made on this record."""
    values = _record_context(doc, history, rows)
    telemetry = doc.get("telemetry") or {}
    for invocation in (doc.get("monitoring") or {}).get("selected_agents", []):
        values += _numeric_values(invocation.get("objective"))
    for name, run in (doc.get("agents") or {}).items():
        values += [v for a in run.get("actions", [])
                   for v in _param_values(a.get("params") or {}, telemetry)]
        if name == agent:
            values += [v for text in _knowledge(run) for v in _numeric_values(text)]
    return values


# --------------------------------------------------------------------------
# Record checks (Monitoring Agent)
# --------------------------------------------------------------------------


def _mentions(text: str, kpi: str) -> bool:
    lowered = (text or "").lower()
    return any(word in lowered for word in KPI_WORDS.get(kpi, (kpi,)))


def expected_agents(violations: list[str], persistence: dict[str, int]) -> set[str]:
    if not violations:
        return set()
    observed = max((persistence.get(kpi, 0) for kpi in violations), default=0)
    return {"near_rt_ric", "non_rt_ric"} if observed >= NON_RT_MIN_PERSISTENCE else {"near_rt_ric"}


def check_record(doc: dict[str, Any], history: list[dict[str, Any]],
                 rows: list[dict[str, Any]] = ()) -> dict[str, Any]:
    monitoring = doc.get("monitoring") or {}
    llm = monitoring.get("llm_output") or monitoring   # stored only when overridden
    violations = (doc.get("sla") or {}).get("violations", [])
    persistence = doc.get("persistence") or {}

    recomputed = MonitoringAgent.analyze_sla(TelemetryRecord.model_validate(doc["telemetry"]))
    objectives = [i.get("objective") or "" for i in monitoring.get("selected_agents", [])]
    numbers = number_check([monitoring.get("reason"), monitoring.get("global_situation"),
                            *objectives], _record_context(doc, history, rows))
    return {
        "sla_recomputed_agrees": recomputed.violations == violations,
        "llm_flags_violation": bool(llm.get("action_required")) and bool(llm.get("selected_agents")),
        "llm_situation_ok": llm.get("situation") != "normal",
        "llm_severity_agrees": llm.get("severity") == monitoring.get("severity"),
        "llm_routing_agrees": {i["agent"] for i in llm.get("selected_agents", [])}
        == expected_agents(violations, persistence),
        "objective_names_kpis": all(_mentions(o, kpi) for o in objectives for kpi in violations)
        if objectives else False,
        "monitoring_numbers_quoted": numbers["quoted"],
        "monitoring_numbers_untraced": numbers["untraced"],
        "monitoring_untraced_values": " ".join(numbers["untraced_values"]),
    }


# --------------------------------------------------------------------------
# Action checks (RIC agents)
# --------------------------------------------------------------------------


def diagnosis(telemetry: dict[str, Any]) -> str:
    if (telemetry.get("prb_utilisation") or 0) >= PRB_LIMITED_UTILISATION:
        return "prb_limited"
    if (telemetry.get("dl_cqi") or 0) < RADIO_LIMITED_CQI:
        return "radio_limited"
    return "unclear"


def _share(telemetry: dict[str, Any]) -> float:
    return 100 * telemetry["slice_prb"] / telemetry["cell_prb"]


def _evidence_kind(source: str) -> str:
    lowered = source.lower().strip()
    if (lowered.endswith("(invocation context)") or lowered.startswith(_CONTEXT_PREFIXES)
            or any(name in lowered for name in _CONTEXT_NAMES)):
        return "context"
    return "knowledge"


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def evidence_check(sources: list[str] | None, knowledge: list[str],
                   context: list[float]) -> dict[str, Any]:
    """Context sources must carry traceable numbers; knowledge sources must
    share most of their words with what the agent retrieved or the tool
    descriptions it was given."""
    sources = [s for s in (sources or []) if isinstance(s, str) and s.strip()]
    retrieved = _words(" ".join(knowledge + TOOL_DESCRIPTIONS))
    supported, unsupported = 0, []
    for source in sources:
        if _evidence_kind(source) == "context":
            ok = number_check([source], context)["untraced"] == 0
        else:
            words = _words(re.sub(r"\(knowledge retrieved\)", "", source, flags=re.I))
            ok = bool(retrieved) and bool(words) and (
                len(words & retrieved) / len(words) >= EVIDENCE_WORD_SUPPORT)
        if ok:
            supported += 1
        else:
            unsupported.append(source)
    return {"evidence_sources": len(sources), "evidence_supported": supported,
            "evidence_knowledge": sum(_evidence_kind(s) == "knowledge" for s in sources),
            "evidence_unsupported": " || ".join(unsupported)}


def check_action(action: dict[str, Any], agent: str, doc: dict[str, Any],
                 history: list[dict[str, Any]], rows: list[dict[str, Any]] = ()) -> dict[str, Any]:
    telemetry = doc.get("telemetry") or {}
    params = action.get("params") or {}
    tool = action.get("tool")
    cause = diagnosis(telemetry)
    share = _share(telemetry)
    direction = feasible = no_op = None
    if tool == "adjust_prb_allocation" and "prb_percentage" in params:
        requested = params["prb_percentage"]
        direction = requested > share
        available = 100 * (telemetry["slice_prb"] + (telemetry.get("cell_spare_prb") or 0)) \
            / telemetry["cell_prb"]
        feasible = requested <= available + 1e-9
        no_op = abs(requested - share) < 1
    elif tool == "update_slice_resource_policy" and "min_prb_ratio_pct" in params:
        direction = params["min_prb_ratio_pct"] >= share
    elif tool == "optimize_scheduler":
        no_op = params.get("scheduler") == telemetry.get("scheduler")

    reasoning = action.get("reasoning") or {}
    context = _agent_context(doc, history, agent, rows)
    numbers = number_check([reasoning.get("rationale"), reasoning.get("expected_effect")],
                           context)
    run = (doc.get("agents") or {}).get(agent) or {}
    return {
        "diagnosis": cause,
        "tool_cause": TOOL_CAUSE.get(tool, {}).get(cause, "neutral"),
        "direction_ok": direction,
        "feasible": feasible,
        "no_op": no_op,
        "numbers_quoted": numbers["quoted"],
        "numbers_untraced": numbers["untraced"],
        "untraced_values": " ".join(numbers["untraced_values"]),
        "retrieved_knowledge": len(run.get("knowledge", [])),
        **evidence_check(reasoning.get("evidence_sources"), _knowledge(run), context),
    }


def annotate(
    records: list[dict[str, Any]],
    rows_by_run: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[dict[tuple, dict], dict[tuple, dict]]:
    """Checks of every record and action, keyed (run_id, seq) and
    (run_id, seq, agent, i). History = the run's earlier records of the cell;
    rows_by_run = the dataset rows each run processed (telemetry dicts), for
    the healthy records MongoDB does not store."""
    record_checks, action_checks = {}, {}
    ordered = sorted(records, key=lambda d: (d["run_id"], d["seq"]))
    for index, doc in enumerate(ordered):
        telemetry = doc.get("telemetry") or {}
        cell = telemetry.get("cell_id")
        history = [d for d in ordered[:index]
                   if d["run_id"] == doc["run_id"] and d.get("telemetry", {}).get("cell_id") == cell]
        seen = [row for row in (rows_by_run or {}).get(doc["run_id"], [])
                if row.get("cell_id") == cell and row.get("run") == telemetry.get("run")
                and row.get("window_start_s", 0) <= telemetry.get("window_start_s", 0)]
        rows = [row for slice_id in {r.get("slice_id") for r in seen}
                for row in sorted((r for r in seen if r.get("slice_id") == slice_id),
                                  key=lambda r: r.get("window_start_s", 0))[-MEMORY_RECORDS:]]
        record_checks[(doc["run_id"], doc["seq"])] = check_record(doc, history, rows)
        for agent, run in (doc.get("agents") or {}).items():
            for position, action in enumerate(run.get("actions", [])):
                key = (doc["run_id"], doc["seq"], agent, action.get("i", position))
                action_checks[key] = check_action(action, agent, doc, history, rows)
    return record_checks, action_checks
