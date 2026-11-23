"""Redis working memory: network context the agents use for their next decision.

Keys are named after network entities (base station = cell, slice):

  wm:cell:{cell}:meta                 HASH  configuration in force: run, scheduler,
                                            prb:<slice>, util:<slice>, updated
  wm:cell:{cell}:records              LIST  recent records of every slice of the cell
  wm:cell:{cell}:actions              LIST  recent tool calls on any slice of the cell
  wm:slice:{cell}:{slice}:records     LIST  recent records of this slice (KPI trend)
  wm:slice:{cell}:{slice}:actions     LIST  recent tool calls on this slice
  wm:slice:{cell}:{slice}:persistence HASH  consecutive violated records per SLA KPI
  wm:slice:{cell}:{slice}:controls    HASH  PRB controls in force, as the RAN applied
                                            them: prb_percentage, policy_min, policy_max
  wm:slice:{cell}:{slice}:pending     LIST  executed actions awaiting verification
  wm:slice:{cell}:{slice}:outcomes    LIST  verified outcomes of those actions
  wm:cells                            SET   every cell seen (for reset)

Lists hold compact JSON events, newest first, capped at MEMORY_MAX_RECORDS
(default 30) with LPUSH + LTRIM; every write also refreshes the key TTL
(MEMORY_TTL_SECONDS, unset = no expiry). Each read or write is one pipeline,
so its cost does not depend on how much history exists.

A record from a new configuration of a cell (ColO-RAN `run`) means the cell
was reconfigured: that cell's memory is cleared before the record is stored.
"""

import os
from typing import Any

from redis.asyncio import Redis

from src.schema.monitoring import TelemetryRecord
from src.schema.sla import KPI_NAMES, SLA_PROFILES
from src.schema.working_memory import (
    ActionEvent,
    ActionOutcome,
    MemoryView,
    PendingCheck,
    RecordEvent,
)

DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_MAX_EVENTS = 30
SLICES = tuple(SLA_PROFILES)
CELLS_KEY = "wm:cells"


def _env_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def cell_key(cell: str, kind: str) -> str:
    return f"wm:cell:{cell}:{kind}"


def slice_key(cell: str, slice_id: str, kind: str) -> str:
    return f"wm:slice:{cell}:{slice_id}:{kind}"


def _cell_keys(cell: str) -> list[str]:
    """Every key that belongs to one cell."""
    keys = [cell_key(cell, k) for k in ("meta", "records", "actions")]
    keys += [
        slice_key(cell, s, k)
        for s in SLICES
        for k in ("records", "actions", "persistence", "controls", "pending", "outcomes")
    ]
    return keys


def _shares(split: dict[str, Any]) -> dict[str, dict[str, float]]:
    """PRB share (%) of every slice, from a cell's PRB split {slice: PRBs}."""
    total = sum(split.values())
    return (
        {s: {"prb_percentage": round(100 * n / total, 1)} for s, n in split.items()}
        if total
        else {}
    )


def _controls_of(event: ActionEvent) -> dict[str, dict[str, float]]:
    """The PRB controls an executed tool call put in force, per slice, as the
    RAN applied them: a PRB request is snapped to a recorded split, and both a
    PRB change and a policy can move the other slices' shares too. Without a
    RAN report, the requested values are used."""
    if not event.executed:
        return {}
    applied = event.applied or {}
    if event.tool == "adjust_prb_allocation":
        if isinstance(applied.get("current"), dict):
            return _shares(applied["current"])
        return {event.slice: {"prb_percentage": event.params["prb_percentage"]}}
    if event.tool == "update_slice_resource_policy":
        policy = applied.get("current") or {
            "min_prb_ratio_pct": event.params["min_prb_ratio_pct"],
            "max_prb_ratio_pct": event.params["max_prb_ratio_pct"],
        }
        controls = _shares(applied["prb"]) if isinstance(applied.get("prb"), dict) else {}
        controls.setdefault(event.slice, {}).update(
            policy_min=policy["min_prb_ratio_pct"], policy_max=policy["max_prb_ratio_pct"]
        )
        return controls
    return {}


class RedisWorkingMemory:
    """Per-cell and per-slice working memory of the agents."""

    def __init__(
        self,
        redis: Redis,
        max_events: int | None = None,
        ttl_seconds: int | None = None,
    ):
        self.redis = redis
        self.max_events = (
            max_events
            if max_events is not None
            else _env_int("MEMORY_MAX_RECORDS", DEFAULT_MAX_EVENTS)
        )
        self.ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else _env_int("MEMORY_TTL_SECONDS", None)
        )

    @classmethod
    def from_url(cls, url: str | None = None, **kwargs) -> "RedisWorkingMemory":
        redis = Redis.from_url(
            url or os.environ.get("REDIS_URL", DEFAULT_REDIS_URL),
            decode_responses=True,
        )
        return cls(redis, **kwargs)

    def _expire(self, pipe, *keys: str) -> None:
        if self.ttl_seconds:
            for key in keys:
                pipe.expire(key, self.ttl_seconds)

    def _push(self, pipe, key: str, values: list[str]) -> None:
        pipe.lpush(key, *values)
        pipe.ltrim(key, 0, self.max_events - 1)
        self._expire(pipe, key)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def observe(self, record: TelemetryRecord) -> None:
        """Store the cell configuration seen in this record. A new
        configuration (run) of the cell clears the cell's memory first."""
        meta = cell_key(record.cell_id, "meta")
        previous_run = await self.redis.hget(meta, "run")
        pipe = self.redis.pipeline(transaction=False)
        if previous_run is not None and previous_run != record.run:
            pipe.delete(*_cell_keys(record.cell_id))
        pipe.hset(
            meta,
            mapping={
                "run": record.run,
                "scheduler": record.scheduler,
                f"prb:{record.slice_id}": record.slice_prb,
                f"util:{record.slice_id}": round(record.prb_utilisation, 3),
                "updated": record.record_id,
            },
        )
        pipe.sadd(CELLS_KEY, record.cell_id)
        self._expire(pipe, meta)
        await pipe.execute()

    async def update_persistence(
        self, record: TelemetryRecord, violations: list[str]
    ) -> dict[str, int]:
        """Count consecutive violated records per SLA KPI for this slice."""
        key = slice_key(record.cell_id, record.slice_id, "persistence")
        pipe = self.redis.pipeline(transaction=False)
        for kpi in KPI_NAMES:
            if kpi in violations:
                pipe.hincrby(key, kpi, 1)
            else:
                pipe.hset(key, kpi, 0)
        self._expire(pipe, key)
        pipe.hgetall(key)
        counters = (await pipe.execute())[-1]
        return {kpi: int(counters.get(kpi, 0)) for kpi in KPI_NAMES}

    async def record_decision(self, cell: str, event: RecordEvent, cell_level: bool = True) -> None:
        """Add a processed record to its slice history and, unless cell_level
        is False (healthy records), to the cell history."""
        value = [event.model_dump_json()]
        pipe = self.redis.pipeline(transaction=False)
        self._push(pipe, slice_key(cell, event.slice, "records"), value)
        if cell_level:
            self._push(pipe, cell_key(cell, "records"), value)
        await pipe.execute()

    async def record_actions(self, cell: str, events: list[ActionEvent]) -> None:
        """Add the tool calls of one RIC invocation (in call order)."""
        if not events:
            return
        pipe = self.redis.pipeline(transaction=False)
        for slice_id in dict.fromkeys(e.slice for e in events):
            values = [e.model_dump_json() for e in events if e.slice == slice_id]
            self._push(pipe, slice_key(cell, slice_id, "actions"), values)
        self._push(pipe, cell_key(cell, "actions"), [e.model_dump_json() for e in events])
        for event in events:
            for slice_id, controls in _controls_of(event).items():
                key = slice_key(cell, slice_id, "controls")
                pipe.hset(key, mapping=controls)
                self._expire(pipe, key)
        await pipe.execute()

    async def controls(self, cell: str, slice_id: str) -> dict[str, float]:
        """PRB controls in force on this slice (last applied values). A
        hash, not the action lists: those are capped and would lose an old
        policy that still applies."""
        stored = await self.redis.hgetall(slice_key(cell, slice_id, "controls"))
        return {name: float(value) for name, value in stored.items()}

    # ------------------------------------------------------------------
    # Action verification (closed loop)
    # ------------------------------------------------------------------

    async def add_pending(self, cell: str, checks: list[PendingCheck]) -> None:
        """Executed actions to verify on a later record of their slice."""
        if not checks:
            return
        pipe = self.redis.pipeline(transaction=False)
        for check in checks:
            key = slice_key(cell, check.slice, "pending")
            pipe.rpush(key, check.model_dump_json())
            self._expire(pipe, key)
        await pipe.execute()

    async def pop_due(self, cell: str, slice_id: str, step: int) -> list[PendingCheck]:
        """Remove and return the slice's checks due at `step` or before."""
        key = slice_key(cell, slice_id, "pending")
        raw = await self.redis.lrange(key, 0, -1)
        if not raw:
            return []
        checks = [PendingCheck.model_validate_json(v) for v in raw]
        due = [c for c in checks if c.due_step <= step]
        if due:
            later = [c.model_dump_json() for c in checks if c.due_step > step]
            pipe = self.redis.pipeline(transaction=True)
            pipe.delete(key)
            if later:
                pipe.rpush(key, *later)
                self._expire(pipe, key)
            await pipe.execute()
        return due

    async def record_outcomes(self, cell: str, outcomes: list[ActionOutcome]) -> None:
        if not outcomes:
            return
        pipe = self.redis.pipeline(transaction=False)
        for slice_id in dict.fromkeys(o.slice for o in outcomes):
            self._push(
                pipe,
                slice_key(cell, slice_id, "outcomes"),
                [o.model_dump_json() for o in outcomes if o.slice == slice_id],
            )
        await pipe.execute()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def view(self, cell: str, slice_id: str, k: int = 5) -> MemoryView:
        """The memory an agent gets for one cell and slice: the last k events
        of the slice, and of the cell's OTHER slices."""
        pipe = self.redis.pipeline(transaction=False)
        pipe.hgetall(cell_key(cell, "meta"))
        pipe.lrange(slice_key(cell, slice_id, "records"), 0, k - 1)
        pipe.lrange(cell_key(cell, "records"), 0, 3 * k - 1)
        pipe.lrange(slice_key(cell, slice_id, "actions"), 0, k - 1)
        pipe.lrange(cell_key(cell, "actions"), 0, 3 * k - 1)
        pipe.hgetall(slice_key(cell, slice_id, "persistence"))
        pipe.lrange(slice_key(cell, slice_id, "outcomes"), 0, k - 1)
        meta, s_rec, c_rec, s_act, c_act, persistence, s_out = await pipe.execute()

        def others(raw: list[str], model):
            events = [model.model_validate_json(v) for v in raw]
            return [e for e in events if e.slice != slice_id][:k]

        return MemoryView(
            cell=meta,
            slice_records=[RecordEvent.model_validate_json(v) for v in s_rec],
            cell_records=others(c_rec, RecordEvent),
            slice_actions=[ActionEvent.model_validate_json(v) for v in s_act],
            cell_actions=others(c_act, ActionEvent),
            slice_outcomes=[ActionOutcome.model_validate_json(v) for v in s_out],
            persistence={kpi: int(v) for kpi, v in persistence.items()},
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def reset(self) -> None:
        """Delete the whole working memory (every cell seen)."""
        cells = await self.redis.smembers(CELLS_KEY)
        keys = [key for cell in cells for key in _cell_keys(cell)] + [CELLS_KEY]
        await self.redis.delete(*keys)

    async def close(self) -> None:
        await self.redis.aclose()
