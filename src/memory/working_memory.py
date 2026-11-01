"""Redis working memory: network context the agents use for their next decision.

Keys are named after network entities (base station = cell, slice):

  wm:cell:{cell}:meta                 HASH  configuration in force: run, scheduler,
                                            prb:<slice>, util:<slice>, updated
  wm:cell:{cell}:records              LIST  recent records of every slice of the cell
  wm:cell:{cell}:actions              LIST  recent tool calls on any slice of the cell
  wm:slice:{cell}:{slice}:records     LIST  recent records of this slice (KPI trend)
  wm:slice:{cell}:{slice}:actions     LIST  recent tool calls on this slice
  wm:slice:{cell}:{slice}:persistence HASH  consecutive violated records per SLA KPI
  wm:cells                            SET   every cell seen (for reset)

Lists hold compact JSON events, newest first, capped at MEMORY_MAX_RECORDS
(default 30) with LPUSH + LTRIM; every write also refreshes the key TTL
(MEMORY_TTL_SECONDS, unset = no expiry). Each read or write is one pipeline,
so its cost does not depend on how much history exists.

A record from a new configuration of a cell (ColO-RAN `run`) means the cell
was reconfigured: that cell's memory is cleared before the record is stored.
"""

import os

from redis.asyncio import Redis

from src.schema.monitoring import TelemetryRecord
from src.schema.sla import KPI_NAMES, SLA_PROFILES
from src.schema.working_memory import ActionEvent, MemoryView, RecordEvent

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
    keys += [slice_key(cell, s, k) for s in SLICES for k in ("records", "actions", "persistence")]
    return keys


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
            max_events if max_events is not None
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
        pipe.hset(meta, mapping={
            "run": record.run,
            "scheduler": record.scheduler,
            f"prb:{record.slice_id}": record.slice_prb,
            f"util:{record.slice_id}": round(record.prb_utilisation, 3),
            "updated": record.record_id,
        })
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
        await pipe.execute()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def view(self, cell: str, slice_id: str, k: int = 10) -> MemoryView:
        """The memory an agent gets for one cell and slice: the last k events
        of the slice, and of the cell's OTHER slices."""
        pipe = self.redis.pipeline(transaction=False)
        pipe.hgetall(cell_key(cell, "meta"))
        pipe.lrange(slice_key(cell, slice_id, "records"), 0, k - 1)
        pipe.lrange(cell_key(cell, "records"), 0, 3 * k - 1)
        pipe.lrange(slice_key(cell, slice_id, "actions"), 0, k - 1)
        pipe.lrange(cell_key(cell, "actions"), 0, 3 * k - 1)
        pipe.hgetall(slice_key(cell, slice_id, "persistence"))
        meta, s_rec, c_rec, s_act, c_act, persistence = await pipe.execute()

        def others(raw: list[str], model):
            events = [model.model_validate_json(v) for v in raw]
            return [e for e in events if e.slice != slice_id][:k]

        return MemoryView(
            cell=meta,
            slice_records=[RecordEvent.model_validate_json(v) for v in s_rec],
            cell_records=others(c_rec, RecordEvent),
            slice_actions=[ActionEvent.model_validate_json(v) for v in s_act],
            cell_actions=others(c_act, ActionEvent),
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
