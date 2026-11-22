"""SLA profiles per slice type, evaluated at runtime by the Monitoring Agent.

The telemetry carries measurements only; what counts as a violation lives
here, so budgets can change without re-preparing the dataset.

Budgets come from the ColO-RAN traffic classes (dataset README), not from
the measured KPIs:
  eMBB   CBR 4 Mbps per UE      -> at least 90 % of it delivered (3.6 Mbps),
                                   queueing delay <= 100 ms (video)
  MTC    Poisson 30 x 125 B/s   -> at least 50 % of the nominal 0.03 Mbps,
                                   queueing delay <= 100 ms
  URLLC  Poisson 10 x 125 B/s   -> at least 50 % of the nominal 0.01 Mbps,
                                   queueing delay <= 10 ms
These are PoC values, not standards. Tune them freely.

PRB utilisation and spare PRBs are not SLA KPIs: they are context that tells
the agents whether a slice is resource-limited and whether PRBs are free.
"""

from typing import Literal

from pydantic import BaseModel

SliceType = Literal["eMBB", "MTC", "URLLC"]


class KPIBudget(BaseModel):
    """One SLA term: `field` of TelemetryRecord compared with `budget`."""

    field: str
    budget: float
    unit: str
    # "min": violated when measured < budget; "max": when measured > budget.
    bound: Literal["min", "max"]


SLA_PROFILES: dict[str, dict[str, KPIBudget]] = {
    "eMBB": {
        "throughput": KPIBudget(field="dl_throughput_per_ue_mbps", budget=3.6, unit="Mbps", bound="min"),
        "queueing_delay": KPIBudget(field="queueing_delay_ms", budget=100.0, unit="ms", bound="max"),
    },
    "MTC": {
        "throughput": KPIBudget(field="dl_throughput_per_ue_mbps", budget=0.015, unit="Mbps", bound="min"),
        "queueing_delay": KPIBudget(field="queueing_delay_ms", budget=100.0, unit="ms", bound="max"),
    },
    "URLLC": {
        "throughput": KPIBudget(field="dl_throughput_per_ue_mbps", budget=0.005, unit="Mbps", bound="min"),
        "queueing_delay": KPIBudget(field="queueing_delay_ms", budget=10.0, unit="ms", bound="max"),
    },
}

# Every KPI name used by any profile (persistence counters, prompts).
KPI_NAMES: tuple[str, ...] = tuple(dict.fromkeys(k for p in SLA_PROFILES.values() for k in p))


def budget_ratio(measured: float, budget: float, bound: Literal["min", "max"]) -> float:
    """How far a KPI is beyond its budget, in multiples of the budget: above 1
    is a violation, and lower is better. A zero measurement of a `min` KPI is
    infinitely far."""
    if bound == "min":
        return budget / measured if measured else float("inf")
    return measured / budget if budget else float("inf")
