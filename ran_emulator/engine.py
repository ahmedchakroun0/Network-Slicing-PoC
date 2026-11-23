"""Time-stepped RAN emulator driven by recorded ColO-RAN traces.

Each step is one window (1 s) and emits one TelemetryRecord per cell and slice.

- PRB split and scheduler: trace switching. The cell plays the recorded trace
  of its current (scheduler, split); a change switches to the trace of the new
  configuration at the same window offset, so the next KPIs are measured ones.
- Power, handover, QoS, link adaptation and events: modelled overlay
  (ran_emulator/overlay.py) on top of the trace row.

A control takes effect at the next step. A control outside the safety
envelope (src/schema/limits.py) raises ControlError and is logged as a
violation: Governance should have stopped it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ran_emulator import overlay
from ran_emulator.configs import CELL_PRB, SLICES, SPLITS, Policy, Split, snap, within_policies
from ran_emulator.scenarios import EventSpec, Scenario
from ran_emulator.traces import Trace, TraceBank
from src.schema import limits
from src.schema.monitoring import TelemetryRecord
from src.schema.sla import SliceType


class ControlError(ValueError):
    """A control request outside the safety envelope or otherwise invalid."""


@dataclass
class SliceControls:
    mcs_offset: int = 0
    target_bler: float = overlay.BASE_BLER
    priority_level: int = overlay.NEUTRAL_PRIORITY
    policy: Policy | None = None
    backlog_bytes: float = 0.0  # extra backlog from overlay effects, carried over


@dataclass
class CellState:
    cell_id: str
    scheduler: str
    split: Split
    trace: Trace
    offset: int
    power_offset_db: float = 0.0
    handover_margin_db: float = overlay.NEUTRAL_HANDOVER_DB
    slices: dict[SliceType, SliceControls] = field(
        default_factory=lambda: {s: SliceControls() for s in SLICES}
    )
    note: str | None = None

    def policies(self) -> dict[SliceType, Policy]:
        return {s: c.policy for s, c in self.slices.items() if c.policy is not None}

    def controls(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler,
            "prb": self.split.as_dict(),
            "prb_percentage": {s: round(self.split.of(s) * 100 / CELL_PRB, 1) for s in SLICES},
            "power_offset_db": self.power_offset_db,
            "handover_margin_db": self.handover_margin_db,
            "slices": {
                s: {k: v for k, v in asdict(c).items() if k != "backlog_bytes"}
                for s, c in self.slices.items()
            },
        }


@dataclass
class StepResult:
    step: int
    records: list[TelemetryRecord]
    meta: dict[str, dict[str, Any]]


class Emulator:
    def __init__(self, bank: TraceBank, scenario: Scenario):
        self.bank = bank
        self.scenario = scenario
        self.step_index = 0
        self.events: list[EventSpec] = list(scenario.events)
        self.violations: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.history: dict[str, list[dict[str, Any]]] = {}
        # Demand of a surging slice, anchored at the event's first window: a
        # saturated slice's trace rate is its capacity, not its demand.
        self._surge_base: dict[tuple[int, SliceType], float] = {}
        self.cells: dict[str, CellState] = {}
        rng = np.random.default_rng(scenario.seed)
        for cell_id, spec in scenario.cells.items():
            split = SPLITS[spec.tr]
            if spec.policies and not within_policies(split, spec.policies):
                split = snap(split, None, None, spec.policies)
            trace, note = bank.get(cell_id, spec.scheduler, split)
            offset = (
                spec.start_offset
                if spec.start_offset is not None
                else int(rng.integers(trace.length))
            )
            state = CellState(
                cell_id=cell_id,
                scheduler=trace.scheduler,
                split=trace.split,
                trace=trace,
                offset=offset % trace.length,
                note=note,
            )
            for s, pol in spec.policies.items():
                state.slices[s].policy = pol
            self.cells[cell_id] = state
            self.history[cell_id] = []

    # ------------------------------------------------------------------ step
    def step(self) -> StepResult:
        step = self.step_index
        records: list[TelemetryRecord] = []
        meta: dict[str, dict[str, Any]] = {}
        for cell in self.cells.values():
            events = [e for e in self.events if e.cell == cell.cell_id and e.active(step)]
            rows = {s: self._slice_kpis(cell, s, events) for s in SLICES}
            if any(modified for _, modified in rows.values()):
                self._recompute_spare(cell, {s: r for s, (r, _) in rows.items()})
            for s in SLICES:
                rec = TelemetryRecord(
                    record_id=f"emu-{self.scenario.name}-{cell.cell_id}-s{step:05d}-{s}",
                    cell_id=cell.cell_id,
                    slice_id=s,
                    # One run per episode: a control switches the source trace
                    # (meta.source_run), not the cell's configuration run, so
                    # the agents' working memory of the cell is kept.
                    run=f"emu/{self.scenario.name}/seed{self.scenario.seed}",
                    window_start_s=step * cell.trace.window_s,
                    window_s=cell.trace.window_s,
                    slice_prb=cell.split.of(s),
                    cell_prb=CELL_PRB,
                    scheduler=cell.scheduler,
                    **rows[s][0],
                )
                records.append(rec)
                self.history[cell.cell_id].append({"step": step, **rec.model_dump()})
            wrapped = cell.offset + 1 >= cell.trace.length
            meta[cell.cell_id] = {
                "source_run": cell.trace.run,
                "trace_offset": cell.offset,
                "wrapped": wrapped,
                "events": [e.model_dump() for e in events],
                "note": cell.note,
                "controls": cell.controls(),
            }
            cell.offset = 0 if wrapped else cell.offset + 1
        self.step_index += 1
        return StepResult(step=step, records=records, meta=meta)

    def _slice_kpis(
        self, cell: CellState, s: SliceType, events: list[EventSpec]
    ) -> tuple[dict[str, Any], bool]:
        """KPIs of one slice for this window, and whether the overlay changed them."""
        row = cell.trace.row(s, cell.offset)
        ctl = cell.slices[s]
        interference = sum(e.db for e in events if e.type == "interference")
        degradation = sum(e.db for e in events if e.type == "link_degradation" and e.slice == s)
        surges = [e for e in events if e.type == "demand_surge" and e.slice == s]
        surge = math.prod(e.factor for e in surges)
        demand = None
        if surges:
            base = self._surge_base.setdefault((id(surges[0]), s), row["dl_throughput_mbps"])
            demand = base * surge
        outage = any(e.type == "cell_outage" for e in events)
        sinr_shift = (
            cell.power_offset_db
            + overlay.handover_sinr_db(cell.handover_margin_db)
            - interference
            - degradation
        )
        borrow = 0.0
        if ctl.priority_level < overlay.NEUTRAL_PRIORITY:
            share = (overlay.NEUTRAL_PRIORITY - ctl.priority_level) / (overlay.NEUTRAL_PRIORITY - 1)
            borrow = (row["cell_spare_prb"] or 0.0) * share

        neutral = (
            sinr_shift == 0
            and degradation == 0
            and ctl.mcs_offset == 0
            and ctl.target_bler == overlay.BASE_BLER
            and surge == 1
            and borrow == 0
            and ctl.backlog_bytes == 0
            and not outage
        )
        if neutral:
            return row, False

        ues = max(row["active_ues"], 1)
        window = cell.trace.window_s
        if outage:
            arrived = (
                (demand if demand is not None else row["dl_throughput_mbps"]) * window * 1e6 / 8
            )
            ctl.backlog_bytes = (ctl.backlog_bytes + arrived) * overlay.BACKLOG_KEEP
            row.update(
                dl_throughput_mbps=0.0,
                dl_throughput_per_ue_mbps=0.0,
                dl_buffer_bytes=row["dl_buffer_bytes"] + ctl.backlog_bytes / ues,
                queueing_delay_ms=window * 1e3,
                prb_utilisation=0.0,
                dl_cqi=0.0,
                dl_mcs=0.0,
            )
            return row, True

        radio = overlay.radio_effect(
            row["dl_cqi"], row["dl_mcs"], sinr_shift, ctl.mcs_offset, ctl.target_bler, degradation
        )
        q = overlay.queue_step(
            rate_mbps=row["dl_throughput_mbps"],
            utilisation=row["prb_utilisation"],
            delay_ms=row["queueing_delay_ms"],
            buffer_bytes=row["dl_buffer_bytes"],
            ues=ues,
            slice_prb=cell.split.of(s),
            window_s=window,
            rate_factor=radio.rate_factor,
            surge=surge,
            extra_prb=borrow,
            backlog_bytes=ctl.backlog_bytes,
            harq_extra_ms=overlay.harq_extra_delay_ms(radio.bler),
            demand_mbps=demand,
        )
        ctl.backlog_bytes = q.backlog_bytes if q.backlog_bytes > 1.0 else 0.0
        row.update(
            dl_throughput_mbps=q.throughput_mbps,
            dl_throughput_per_ue_mbps=q.throughput_mbps / ues,
            dl_buffer_bytes=q.buffer_bytes,
            queueing_delay_ms=q.delay_ms,
            prb_utilisation=q.utilisation,
            dl_cqi=radio.cqi,
            dl_mcs=radio.mcs,
        )
        return row, True

    @staticmethod
    def _recompute_spare(cell: CellState, rows: dict[SliceType, dict[str, Any]]) -> None:
        idle = {s: cell.split.of(s) * (1 - rows[s]["prb_utilisation"]) for s in SLICES}
        for s in SLICES:
            rows[s]["cell_spare_prb"] = sum(v for o, v in idle.items() if o != s)

    # -------------------------------------------------------------- controls
    def _cell(self, cell_id: str) -> CellState:
        if cell_id not in self.cells:
            raise KeyError(f"unknown cell {cell_id!r}; emulated: {sorted(self.cells)}")
        return self.cells[cell_id]

    def _check(self, control: str, cell_id: str, args: dict[str, Any], error: str | None) -> None:
        if error:
            self.violations.append(
                {
                    "step": self.step_index,
                    "control": control,
                    "cell": cell_id,
                    "args": args,
                    "reason": error,
                }
            )
            raise ControlError(error)

    def _result(
        self,
        control: str,
        cell: CellState,
        slice_id: SliceType | None,
        previous: Any,
        current: Any,
        note: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        out = {
            "control": control,
            "cell": cell.cell_id,
            "slice": slice_id,
            "previous": previous,
            "current": current,
            "effective_step": self.step_index,
            "note": note,
            **extra,
        }
        self.actions.append(out)
        return out

    def _switch(self, cell: CellState, scheduler: str, split: Split) -> str | None:
        trace, note = self.bank.get(cell.cell_id, scheduler, split)
        cell.trace, cell.scheduler, cell.split = trace, trace.scheduler, trace.split
        cell.offset %= trace.length
        cell.note = note
        return note

    def set_prb(self, cell_id: str, slice_id: SliceType, prb_percentage: float) -> dict[str, Any]:
        cell = self._cell(cell_id)
        args = {"slice": slice_id, "prb_percentage": prb_percentage}
        self._check(
            "prb",
            cell_id,
            args,
            limits.outside(prb_percentage, limits.PRB_PERCENTAGE_RANGE, "prb_percentage"),
        )
        try:
            target = snap(cell.split, slice_id, prb_percentage * CELL_PRB / 100, cell.policies())
        except ValueError as exc:
            self._check("prb", cell_id, args, str(exc))
            raise
        previous = cell.split.as_dict()
        note = self._switch(cell, cell.scheduler, target)
        applied_pct = round(cell.split.of(slice_id) * 100 / CELL_PRB, 1)
        notes = [n for n in (note,) if n]
        if applied_pct != prb_percentage:
            notes.insert(
                0,
                f"snapped to the nearest recorded split: {slice_id} gets "
                f"{applied_pct}% ({cell.split.of(slice_id)} PRBs)",
            )
        return self._result(
            "prb",
            cell,
            slice_id,
            previous,
            cell.split.as_dict(),
            "; ".join(notes) or None,
            requested_pct=prb_percentage,
            applied_pct=applied_pct,
        )

    def set_scheduler(self, cell_id: str, scheduler: str) -> dict[str, Any]:
        cell = self._cell(cell_id)
        args = {"scheduler": scheduler}
        self._check(
            "scheduler",
            cell_id,
            args,
            None
            if scheduler in limits.ALLOWED_SCHEDULERS
            else f"scheduler {scheduler!r} is not allowed",
        )
        previous = cell.scheduler
        note = self._switch(cell, scheduler, cell.split)
        msg = "the ColO-RAN traces use one scheduler per cell: applied to all slices"
        return self._result(
            "scheduler",
            cell,
            None,
            previous,
            cell.scheduler,
            "; ".join(n for n in (msg, note) if n),
        )

    def set_handover(self, cell_id: str, handover_margin_db: float) -> dict[str, Any]:
        cell = self._cell(cell_id)
        self._check(
            "handover",
            cell_id,
            {"handover_margin_db": handover_margin_db},
            limits.outside(
                handover_margin_db, limits.HANDOVER_MARGIN_RANGE_DB, "handover_margin_db"
            ),
        )
        previous, cell.handover_margin_db = cell.handover_margin_db, float(handover_margin_db)
        return self._result(
            "handover",
            cell,
            None,
            {"handover_margin_db": previous},
            {"handover_margin_db": cell.handover_margin_db},
            "modelled effect",
        )

    def set_power(self, cell_id: str, power_offset_db: float) -> dict[str, Any]:
        cell = self._cell(cell_id)
        self._check(
            "power",
            cell_id,
            {"power_offset_db": power_offset_db},
            limits.outside(power_offset_db, limits.POWER_OFFSET_RANGE_DB, "power_offset_db"),
        )
        previous, cell.power_offset_db = cell.power_offset_db, float(power_offset_db)
        return self._result(
            "power",
            cell,
            None,
            {"power_offset_db": previous},
            {"power_offset_db": cell.power_offset_db},
            "modelled effect",
        )

    def set_qos(self, cell_id: str, slice_id: SliceType, priority_level: int) -> dict[str, Any]:
        cell = self._cell(cell_id)
        self._check(
            "qos",
            cell_id,
            {"slice": slice_id, "priority_level": priority_level},
            limits.outside(priority_level, limits.QOS_PRIORITY_RANGE, "priority_level"),
        )
        ctl = cell.slices[slice_id]
        previous, ctl.priority_level = ctl.priority_level, int(priority_level)
        return self._result(
            "qos",
            cell,
            slice_id,
            {"priority_level": previous},
            {"priority_level": ctl.priority_level},
            "modelled effect: priority below 5 borrows the cell's idle PRBs",
        )

    def set_link_adaptation(
        self, cell_id: str, slice_id: SliceType, mcs_offset: int, target_bler: float
    ) -> dict[str, Any]:
        cell = self._cell(cell_id)
        args = {"slice": slice_id, "mcs_offset": mcs_offset, "target_bler": target_bler}
        self._check(
            "link_adaptation",
            cell_id,
            args,
            (
                limits.outside(mcs_offset, limits.MCS_OFFSET_RANGE, "mcs_offset")
                or limits.outside(target_bler, limits.TARGET_BLER_RANGE, "target_bler")
            ),
        )
        ctl = cell.slices[slice_id]
        previous = {"mcs_offset": ctl.mcs_offset, "target_bler": ctl.target_bler}
        ctl.mcs_offset, ctl.target_bler = int(mcs_offset), float(target_bler)
        return self._result(
            "link_adaptation",
            cell,
            slice_id,
            previous,
            {"mcs_offset": ctl.mcs_offset, "target_bler": ctl.target_bler},
            "modelled effect",
        )

    def set_policy(
        self, cell_id: str, slice_id: SliceType, min_prb_ratio_pct: float, max_prb_ratio_pct: float
    ) -> dict[str, Any]:
        cell = self._cell(cell_id)
        args = {
            "slice": slice_id,
            "min_prb_ratio_pct": min_prb_ratio_pct,
            "max_prb_ratio_pct": max_prb_ratio_pct,
        }
        self._check(
            "policy",
            cell_id,
            args,
            (
                limits.outside(
                    min_prb_ratio_pct, limits.SLICE_POLICY_MIN_RANGE_PCT, "min_prb_ratio_pct"
                )
                or limits.outside(
                    max_prb_ratio_pct, limits.SLICE_POLICY_MAX_RANGE_PCT, "max_prb_ratio_pct"
                )
                or (
                    None
                    if min_prb_ratio_pct <= max_prb_ratio_pct
                    else "min_prb_ratio_pct must not exceed max_prb_ratio_pct"
                )
            ),
        )
        ctl = cell.slices[slice_id]
        policies = {**cell.policies(), slice_id: (min_prb_ratio_pct, max_prb_ratio_pct)}
        note = None
        if not within_policies(cell.split, policies):
            try:
                target = snap(cell.split, None, None, policies)
            except ValueError as exc:
                self._check("policy", cell_id, args, str(exc))
                raise
            before = cell.split.as_dict()
            switch_note = self._switch(cell, cell.scheduler, target)
            note = "; ".join(
                n
                for n in (
                    f"PRB split {before} violated the policy; moved to {cell.split.as_dict()}",
                    switch_note,
                )
                if n
            )
        previous = ctl.policy
        ctl.policy = (float(min_prb_ratio_pct), float(max_prb_ratio_pct))
        return self._result(
            "policy",
            cell,
            slice_id,
            None
            if previous is None
            else {"min_prb_ratio_pct": previous[0], "max_prb_ratio_pct": previous[1]},
            {"min_prb_ratio_pct": ctl.policy[0], "max_prb_ratio_pct": ctl.policy[1]},
            note,
            prb=cell.split.as_dict(),
        )

    # ---------------------------------------------------------------- events
    def add_event(self, event: EventSpec) -> EventSpec:
        self._cell(event.cell)
        self.events.append(event)
        return event

    # ---------------------------------------------------------- observation
    def state(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.name,
            "seed": self.scenario.seed,
            "next_step": self.step_index,
            "cells": {
                c.cell_id: {
                    **c.controls(),
                    "source_run": c.trace.run,
                    "trace_offset": c.offset,
                    "trace_length": c.trace.length,
                    "note": c.note,
                }
                for c in self.cells.values()
            },
            "events": [e.model_dump() for e in self.events],
        }

    def kpis(
        self, cell_id: str, from_step: int = 0, to_step: int | None = None
    ) -> list[dict[str, Any]]:
        self._cell(cell_id)
        return [
            r
            for r in self.history[cell_id]
            if r["step"] >= from_step and (to_step is None or r["step"] <= to_step)
        ]
