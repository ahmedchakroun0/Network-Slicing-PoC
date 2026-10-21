import os
from datetime import datetime, timezone

from redis.asyncio import Redis

from src.schema.working_memory import WorkingMemory, ToolExecution

DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_MAX_RECORDS = 30


def _env_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


class RedisWorkingMemory:
    """Redis storage for the shared network working memory.

    The memory is bounded: only the most recent `max_tool_executions`
    entries are kept (env MEMORY_MAX_RECORDS, default 30). Without a bound
    the document grows with every record, is rewritten on every update and
    is injected verbatim into every RIC prompt, so cost grows quadratically
    and the prompt overflows the model context after a few hundred records.
    An optional TTL (env MEMORY_TTL_SECONDS) expires an idle memory.
    """

    KEY = "network:working_memory"

    def __init__(
        self,
        redis: Redis,
        max_tool_executions: int | None = None,
        ttl_seconds: int | None = None,
    ):
        self.redis = redis
        self.max_tool_executions = (
            max_tool_executions
            if max_tool_executions is not None
            else _env_int("MEMORY_MAX_RECORDS", DEFAULT_MAX_RECORDS)
        )
        self.ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else _env_int("MEMORY_TTL_SECONDS", None)
        )

    @classmethod
    def from_url(
        cls,
        url: str | None = None,
        **kwargs,
    ) -> "RedisWorkingMemory":
        redis = Redis.from_url(
            url or os.environ.get("REDIS_URL", DEFAULT_REDIS_URL),
            decode_responses=True,
        )
        return cls(redis, **kwargs)

    async def get(self) -> WorkingMemory:
        raw = await self.redis.get(self.KEY)

        if raw is None:
            return WorkingMemory()

        return WorkingMemory.model_validate_json(raw)

    async def save(self, memory: WorkingMemory) -> None:
        memory.updated_at = datetime.now(timezone.utc)
        if self.max_tool_executions is not None and self.max_tool_executions > 0:
            memory.tools_executed = memory.tools_executed[-self.max_tool_executions:]

        await self.redis.set(
            self.KEY,
            memory.model_dump_json(),
            ex=self.ttl_seconds if self.ttl_seconds else None,
        )

    async def update_network_status(self, status: str) -> None:
        memory = await self.get()
        memory.network_status = status
        await self.save(memory)

    async def add_tool_execution(
        self,
        execution: ToolExecution,
    ) -> None:
        memory = await self.get()
        memory.tools_executed.append(execution)
        await self.save(memory)

    async def close(self) -> None:
        await self.redis.aclose()

    async def reset(self) -> None:
        """Reset the working memory to an empty state."""
        await self.redis.delete(self.KEY)