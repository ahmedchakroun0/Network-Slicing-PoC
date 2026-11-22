"""Deterministic verification of executed actions (closed loop).

An action taken on record k (emulator step k) applies from step k+1. The
first record of the same slice at or after that step shows whether the
violation the action targeted is gone:

    resolved   none of the KPIs violated at action time is violated any more
    improved   still violated, but every one of them is closer to its budget
    worse      at least one of them is further from its budget
    unchanged  otherwise (within TOLERANCE)
    unknown    one of them cannot be measured in the new record

Distance to the budget is budget_ratio (src/schema/sla.py), the measure the
Monitoring Agent's severity uses.
"""

from src.schema.monitoring import SLAAnalysis, TelemetryRecord
from src.schema.sla import budget_ratio
from src.schema.working_memory import ActionOutcome, Outcome, PendingCheck

# A relative change smaller than this is noise, not an effect.
TOLERANCE = 0.02


def _classify(before: dict[str, float], after: dict[str, float],
              still_violated: bool) -> Outcome:
    if not still_violated:
        return "resolved"
    changes = []
    for kpi, b in before.items():
        a = after[kpi]
        if b == a:
            changes.append(0.0)
        elif b == float("inf"):
            changes.append(-1.0)
        elif a == float("inf"):
            changes.append(1.0)
        else:
            changes.append((a - b) / b)
    if all(c <= -TOLERANCE for c in changes):
        return "improved"
    if any(c >= TOLERANCE for c in changes):
        return "worse"
    return "unchanged"


def verify(check: PendingCheck, record: TelemetryRecord, analysis: SLAAnalysis) -> ActionOutcome:
    """The outcome of one executed action, from the slice's next record."""
    kpis: dict[str, dict[str, float | None]] = {}
    before: dict[str, float] = {}
    after: dict[str, float] = {}
    for name, then in check.kpis.items():
        now = analysis.kpis.get(name)
        measured = now.measured if now is not None else None
        kpis[name] = {"before": then.measured, "after": measured, "budget": then.budget}
        before[name] = budget_ratio(then.measured, then.budget, then.bound)
        if measured is not None:
            after[name] = budget_ratio(measured, then.budget, then.bound)

    if len(after) < len(before):
        outcome: Outcome = "unknown"
    else:
        outcome = _classify(before, after,
                            still_violated=any(k in analysis.violations for k in check.kpis))
    return ActionOutcome(
        rec=check.rec, t=check.t, slice=check.slice, agent=check.agent, i=check.i,
        tool=check.tool, outcome=outcome, checked_rec=record.record_id,
        checked_t=record.window_start_s, kpis=kpis)
