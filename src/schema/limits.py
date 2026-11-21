"""Safety envelope of the RAN controls.

Shared by the Governance agent (stage 4, resource/safety) and the RAN
emulator (last line of defense: a control outside these ranges is refused
with HTTP 422). PoC values, not standards.
"""

from typing import get_args

from src.schema.monitoring import Scheduler

PRB_PERCENTAGE_RANGE = (5, 90)
HANDOVER_MARGIN_RANGE_DB = (0.0, 10.0)
POWER_OFFSET_RANGE_DB = (-6.0, 6.0)
# The schedulers of the ColO-RAN testbed (same list as the telemetry).
ALLOWED_SCHEDULERS = frozenset(get_args(Scheduler))
QOS_PRIORITY_RANGE = (1, 10)                 # 1 = highest priority
QOS_DELAY_BUDGET_RANGE_MS = (5.0, 300.0)     # 3GPP 5QI packet delay budgets
QOS_GBR_RANGE_MBPS = (0.0, 10.0)             # per UE, 10 MHz carrier
MCS_OFFSET_RANGE = (-5, 5)
TARGET_BLER_RANGE = (0.001, 0.2)
SLICE_POLICY_MIN_RANGE_PCT = (5, 80)
SLICE_POLICY_MAX_RANGE_PCT = (20, 100)


def outside(value: float, bounds: tuple[float, float], name: str) -> str | None:
    """Error message when value is outside [low, high], else None."""
    low, high = bounds
    return None if low <= value <= high else f"{name} must be within [{low}, {high}]."
