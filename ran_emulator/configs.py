"""The PRB splits recorded in ColO-RAN and how a requested PRB share is snapped to one.

ColO-RAN measured 28 splits of the 50 PRBs of a cell (scripts/explore_coloran.py,
data/exploration/07_config_slice_prb.csv): eMBB in {6, 12, ..., 42}, URLLC in
{5, 11, ..., 41}, and MTC takes the rest (at least 3 PRBs). The index of a split
in that table is the `tr` folder of the raw dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.schema.sla import SliceType

CELL_PRB = 50
SLICES: tuple[SliceType, ...] = ("eMBB", "MTC", "URLLC")
SCHEDULER_BY_ID = {0: "round_robin", 1: "waterfilling", 2: "proportional_fair"}
SCHEDULER_ID = {name: i for i, name in SCHEDULER_BY_ID.items()}


@dataclass(frozen=True)
class Split:
    """PRBs per slice; they always sum to CELL_PRB."""

    embb: int
    mtc: int
    urllc: int

    def of(self, slice_id: SliceType) -> int:
        return {"eMBB": self.embb, "MTC": self.mtc, "URLLC": self.urllc}[slice_id]

    @property
    def tr(self) -> int:
        return SPLITS.index(self)

    def as_dict(self) -> dict[str, int]:
        return {"eMBB": self.embb, "MTC": self.mtc, "URLLC": self.urllc}


def _grid() -> list[Split]:
    out = []
    for urllc in range(5, 42, 6):
        for embb in range(6, 43, 6):
            if CELL_PRB - embb - urllc >= 3:
                out.append(Split(embb=embb, mtc=CELL_PRB - embb - urllc, urllc=urllc))
    return out


SPLITS: list[Split] = _grid()

Policy = tuple[float, float]  # (min_prb_ratio_pct, max_prb_ratio_pct)
# Cost of moving PRBs of a slice that was not the target of the request.
ABSORB_WEIGHT: dict[SliceType, float] = {"MTC": 1.0, "eMBB": 2.0, "URLLC": 3.0}


def within_policies(split: Split, policies: dict[SliceType, Policy]) -> bool:
    return all(
        lo * CELL_PRB / 100 <= split.of(s) <= hi * CELL_PRB / 100
        for s, (lo, hi) in policies.items()
    )


def snap(
    current: Split,
    slice_id: SliceType | None,
    target_prb: float | None,
    policies: dict[SliceType, Policy],
) -> Split:
    """The recorded split closest to the request that respects the slice policies.

    Ranked, in order, by: distance of the target slice to its request; number
    of other slices changed; weighted change of the other slices, where MTC
    (delay tolerant) absorbs a change first and URLLC last; tr. With no target
    (slice_id None) only the other criteria apply: used to re-snap after a
    policy change.
    """
    candidates = [s for s in SPLITS if within_policies(s, policies)]
    if not candidates:
        raise ValueError("no recorded PRB split satisfies the slice policies in force")
    others = [o for o in SLICES if o != slice_id]

    def cost(s: Split) -> tuple[float, int, float, int]:
        target = 0.0 if slice_id is None or target_prb is None else abs(s.of(slice_id) - target_prb)
        moved = [o for o in others if s.of(o) != current.of(o)]
        weighted = sum(ABSORB_WEIGHT[o] * abs(s.of(o) - current.of(o)) for o in moved)
        return target, len(moved), weighted, s.tr

    return min(candidates, key=cost)
