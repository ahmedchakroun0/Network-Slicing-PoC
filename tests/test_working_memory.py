import pytest
from redis.asyncio import Redis

from src.memory.working_memory import RedisWorkingMemory
from src.schema.working_memory import ToolExecution, WorkingMemory


@pytest.fixture
async def redis_memory():
    redis = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
    await redis.flushdb()
    memory = RedisWorkingMemory(redis)
    yield memory
    await redis.flushdb()
    await redis.aclose()


@pytest.mark.asyncio
async def test_redis_working_memory_persists_status_and_exec_history(redis_memory):
    memory = redis_memory

    initial = await memory.get()
    assert isinstance(initial, WorkingMemory)
    assert initial.network_status == "Network status not yet available"
    assert initial.tools_executed == []

    await memory.update_network_status("Persistent throughput degradation in slice 7")
    await memory.add_tool_execution(
        ToolExecution(
            agent="non_rt_ric",
            tool_name="update_traffic_steering_policy",
            result={"status": "executed", "channel": "A1"},
        )
    )

    stored = await memory.get()
    assert stored.network_status == "Persistent throughput degradation in slice 7"
    assert len(stored.tools_executed) == 1
    assert stored.tools_executed[0].agent == "non_rt_ric"
    assert stored.tools_executed[0].tool_name == "update_traffic_steering_policy"
    assert stored.tools_executed[0].result["channel"] == "A1"
