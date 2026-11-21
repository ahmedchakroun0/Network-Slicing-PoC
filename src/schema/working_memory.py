"""Shapes of the Redis working memory (src/memory/working_memory.py).

Only structured facts are stored: no LLM rationale or reason text, so the
model's own words are never fed back into later prompts.
"""

from typing import Any

from pydantic import BaseModel, Field

from src.schema.monitoring import SLAAnalysis, TelemetryRecord


class RecordEvent(BaseModel):
    """One processed telemetry record, as the next decisions need it."""

    rec: str
    t: float = Field(description="Window start (s) of the record.")
    slice: str
    kpis: dict[str, float | None] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)
    severity: str
    agents: list[str] = Field(default_factory=list)

    @classmethod
    def of(
        cls,
        record: TelemetryRecord,
        analysis: SLAAnalysis,
        severity: str = "none",
        agents: list[str] | None = None,
    ) -> "RecordEvent":
        return cls(
            rec=record.record_id,
            t=record.window_start_s,
            slice=record.slice_id,
            kpis={**{name: kpi.measured for name, kpi in analysis.kpis.items()},
                  "prb_utilisation": record.prb_utilisation},
            violations=analysis.violations,
            severity=severity,
            agents=agents or [],
        )


class ActionEvent(BaseModel):
    """One protected tool call of a RIC agent and its governance outcome."""

    rec: str
    t: float
    slice: str
    agent: str
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    approved: bool
    executed: bool
    stage: str | None = Field(default=None, description="Governance stage that rejected it.")
    why: str | None = Field(default=None, description="Deterministic rejection reason only.")
    review: dict[str, bool] | None = Field(
        default=None,
        description="LLM review verdicts (goal_aligned, reasoning_sound, "
        "behavior_acceptable), without the reviewer's text.",
    )


def _compact_events(events: list[BaseModel], drop: set[str]) -> list[dict[str, Any]]:
    """Events without empty fields and without fields the list already implies."""
    return [{key: value for key, value in event.model_dump(mode="json", exclude_none=True).items()
             if key not in drop and value not in ([], {})}
            for event in events]


class MemoryView(BaseModel):
    """What an agent sees of the working memory for one cell and slice."""

    cell: dict[str, str] = Field(
        default_factory=dict,
        description="Current configuration of the base station: run, scheduler, "
        "prb:<slice>, util:<slice>.",
    )
    slice_records: list[RecordEvent] = Field(default_factory=list)
    cell_records: list[RecordEvent] = Field(default_factory=list)
    slice_actions: list[ActionEvent] = Field(default_factory=list)
    cell_actions: list[ActionEvent] = Field(default_factory=list)
    persistence: dict[str, int] = Field(default_factory=dict)

    def compact(self, persistence: bool = True) -> dict[str, Any]:
        """The view as it goes into a prompt: no empty fields or lists, no
        record ids (`t` places an event in time), no slice name in the lists
        of this slice, no bookkeeping field of the cell."""
        view = {
            "cell": {k: v for k, v in self.cell.items() if k != "updated"},
            "slice_records": _compact_events(self.slice_records, {"rec", "slice"}),
            "cell_records": _compact_events(self.cell_records, {"rec"}),
            "slice_actions": _compact_events(self.slice_actions, {"rec", "slice"}),
            "cell_actions": _compact_events(self.cell_actions, {"rec"}),
            "persistence": self.persistence if persistence else {},
        }
        return {key: value for key, value in view.items() if value}
