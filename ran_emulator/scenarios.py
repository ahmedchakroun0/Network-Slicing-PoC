"""Scenario files: initial cell configuration, slice policies and timed events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.schema.monitoring import Scheduler
from src.schema.sla import SliceType

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"

EventType = Literal["demand_surge", "interference", "link_degradation", "cell_outage"]


class EventSpec(BaseModel):
    """A timed perturbation of one cell (or one slice of it)."""

    model_config = ConfigDict(extra="forbid")

    type: EventType
    cell: str
    slice: SliceType | None = Field(
        default=None, description="Required for demand_surge and link_degradation.")
    start_step: int = Field(default=0, ge=0)
    duration: int = Field(default=60, ge=1)
    factor: float = Field(default=1.0, gt=0, description="demand_surge: demand multiplier.")
    db: float = Field(default=0.0, ge=0, description="interference / link_degradation: SINR loss.")

    @model_validator(mode="after")
    def _slice_when_needed(self) -> EventSpec:
        if self.type in ("demand_surge", "link_degradation") and self.slice is None:
            raise ValueError(f"{self.type} needs a slice")
        return self

    def active(self, step: int) -> bool:
        return self.start_step <= step < self.start_step + self.duration


class CellSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scheduler: Scheduler = "round_robin"
    tr: int = Field(default=9, ge=0, le=27,
                    description="Initial PRB split, index in configs.SPLITS (ColO-RAN tr).")
    start_offset: int | None = Field(
        default=None, ge=0, description="Initial window in the trace; None = drawn from the seed.")
    policies: dict[SliceType, tuple[float, float]] = Field(
        default_factory=dict, description="Slice PRB policies (min %, max %).")


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    seed: int = 0
    steps: int = Field(default=300, ge=1, description="Suggested run length.")
    cells: dict[str, CellSpec]
    events: list[EventSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _events_target_known_cells(self) -> Scenario:
        unknown = {e.cell for e in self.events} - set(self.cells)
        if unknown:
            raise ValueError(f"events target cells not in the scenario: {sorted(unknown)}")
        return self


def load_scenario(name_or_path: str | Path) -> Scenario:
    """Load a scenario by file path, or by name from ran_emulator/scenarios/."""
    path = Path(name_or_path)
    if not path.exists():
        path = SCENARIO_DIR / f"{Path(name_or_path).stem}.json"
    if not path.exists():
        known = sorted(p.stem for p in SCENARIO_DIR.glob("*.json"))
        raise FileNotFoundError(f"scenario {name_or_path!r} not found; known: {known}")
    return Scenario.model_validate(json.loads(path.read_text()))
