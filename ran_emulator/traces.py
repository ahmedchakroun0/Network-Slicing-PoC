"""Bank of recorded ColO-RAN cell traces, indexed by configuration.

A trace is one (cell, scheduler, PRB split) run of the prepared dataset
(scripts/prepare_coloran.py): per slice, one row per 1 s window, aligned on the
window offset. The emulator plays a trace and switches to another one when an
agent changes the PRB split or the scheduler.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ran_emulator.configs import SLICES, Split
from src.schema.sla import SliceType

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = (
    ROOT / "data/coloran_prepared_dataset.csv",   # full dataset (gitignored)
    ROOT / "data/coloran_prepared_sample.csv",    # tracked sample, fewer traces
)

# Per-window values copied from the trace (the rest of a record is set by the emulator).
KPI_FIELDS = (
    "active_ues", "dl_throughput_mbps", "dl_throughput_per_ue_mbps", "dl_buffer_bytes",
    "queueing_delay_ms", "prb_utilisation", "cell_spare_prb", "dl_cqi", "dl_mcs", "ul_sinr_db",
)
RUN_RE = re.compile(r"sched(\d)/tr(\d+)/(exp\d+)")


@dataclass
class Trace:
    run: str                 # e.g. "sched0/tr6/exp1"
    cell_id: str
    scheduler: str
    split: Split
    window_s: float
    kpis: dict[SliceType, dict[str, np.ndarray]]

    @property
    def length(self) -> int:
        return len(self.kpis["eMBB"]["dl_throughput_mbps"])

    def row(self, slice_id: SliceType, offset: int) -> dict[str, float | int | None]:
        out: dict[str, float | int | None] = {}
        for name, values in self.kpis[slice_id].items():
            v = float(values[offset])
            if math.isnan(v):
                out[name] = None
            else:
                out[name] = int(v) if name == "active_ues" else v
        return out


class TraceBank:
    def __init__(self, traces: list[Trace]):
        if not traces:
            raise ValueError("empty trace bank")
        self._by_key: dict[tuple[str, str, Split], Trace] = {}
        for t in traces:
            self._by_key.setdefault((t.cell_id, t.scheduler, t.split), t)

    # ---- construction --------------------------------------------------
    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> TraceBank:
        traces = []
        df = df.sort_values(["cell_id", "run", "slice_id", "window_start_s"])
        for (cell, run), g in df.groupby(["cell_id", "run"], sort=True):
            if not RUN_RE.search(str(run)) or set(g.slice_id) != set(SLICES):
                continue
            # Keep only the windows present for all three slices.
            common = set.intersection(*(set(g[g.slice_id == s].window_start_s) for s in SLICES))
            g = g[g.window_start_s.isin(common)]
            if g.empty:
                continue
            prb = g.groupby("slice_id").slice_prb.first()
            split = Split(embb=int(prb["eMBB"]), mtc=int(prb["MTC"]), urllc=int(prb["URLLC"]))
            kpis = {
                s: {f: g[g.slice_id == s][f].to_numpy(dtype=float) for f in KPI_FIELDS}
                for s in SLICES
            }
            traces.append(Trace(run=str(run), cell_id=str(cell),
                                scheduler=str(g.scheduler.iloc[0]), split=split,
                                window_s=float(g.window_s.iloc[0]), kpis=kpis))
        return cls(traces)

    @classmethod
    def load(cls, path: str | Path | None = None) -> TraceBank:
        if path is None:
            path = next((p for p in DEFAULT_DATASETS if p.exists()), None)
            if path is None:
                raise FileNotFoundError("no prepared ColO-RAN dataset found; run "
                                        "scripts/prepare_coloran.py first")
        return cls.from_frame(pd.read_csv(path))

    # ---- lookup ---------------------------------------------------------
    @property
    def cells(self) -> list[str]:
        return sorted({k[0] for k in self._by_key})

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self) -> Iterator[Trace]:
        return iter(self._by_key.values())

    def get(self, cell_id: str, scheduler: str, split: Split) -> tuple[Trace, str | None]:
        """The trace of this exact configuration, or the nearest recorded one.

        Returns (trace, note); note explains a fallback and is None on an exact match.
        """
        exact = self._by_key.get((cell_id, scheduler, split))
        if exact is not None:
            return exact, None
        same_cell = [t for (c, _, _), t in self._by_key.items() if c == cell_id]
        if not same_cell:
            raise KeyError(f"unknown cell {cell_id!r}; known: {self.cells}")

        def distance(t: Trace) -> tuple[int, int, int]:
            # Same scheduler first, then the closest split; tr breaks ties.
            prb = sum(abs(t.split.of(s) - split.of(s)) for s in SLICES)
            return (int(t.scheduler != scheduler), prb, t.split.tr)

        best = min(same_cell, key=distance)
        return best, (f"no trace recorded for {cell_id} {scheduler} {split.as_dict()}; "
                      f"using nearest {best.run} {best.scheduler} {best.split.as_dict()}")
