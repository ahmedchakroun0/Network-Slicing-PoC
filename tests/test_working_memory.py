"""Redis working memory (src/memory/working_memory.py) on fakeredis."""

import pytest

from src.memory.working_memory import CELLS_KEY, RedisWorkingMemory, cell_key, slice_key
from src.schema.working_memory import ActionEvent, RecordEvent


def _event(rec, slice_id="eMBB", violations=("throughput",)):
    return RecordEvent(
        rec=rec,
        t=0.0,
        slice=slice_id,
        kpis={"throughput": 1.0},
        violations=list(violations),
        severity="high",
        agents=["near_rt_ric"],
    )


def _action(rec, slice_id="eMBB", approved=True):
    return ActionEvent(
        rec=rec,
        t=0.0,
        slice=slice_id,
        agent="near_rt_ric",
        tool="adjust_prb_allocation",
        params={"prb_percentage": 60},
        approved=approved,
        executed=approved,
        stage=None if approved else "resource_safety",
        why=None if approved else "prb_percentage must be within [5, 90].",
    )


@pytest.mark.asyncio
async def test_observe_stores_the_cell_configuration(memory, redis, make_record):
    await memory.observe(make_record(slice_id="eMBB", slice_prb=30, prb_utilisation=0.99))
    await memory.observe(make_record(slice_id="MTC", slice_prb=15, prb_utilisation=0.08))

    meta = await redis.hgetall(cell_key("bs1", "meta"))
    assert meta["run"] == "sched0/tr6/exp1" and meta["scheduler"] == "round_robin"
    assert meta["prb:eMBB"] == "30" and meta["prb:MTC"] == "15"
    assert meta["util:MTC"] == "0.08"
    assert await redis.smembers(CELLS_KEY) == {"bs1"}


@pytest.mark.asyncio
async def test_new_configuration_of_a_cell_clears_its_memory(memory, redis, make_record):
    await memory.observe(make_record())
    await memory.record_decision("bs1", _event("r1"))
    await memory.update_persistence(make_record(), ["throughput"])
    await memory.observe(make_record(record_id="other-cell", cell_id="bs2"))
    await memory.record_decision("bs2", _event("r2"))

    await memory.observe(make_record(run="sched2/tr0/exp1"))  # bs1 reconfigured

    view = await memory.view("bs1", "eMBB")
    assert view.slice_records == [] and view.persistence == {}
    assert view.cell["run"] == "sched2/tr0/exp1"
    assert len((await memory.view("bs2", "eMBB")).slice_records) == 1  # other cell kept


@pytest.mark.asyncio
async def test_persistence_counts_consecutive_violations_per_kpi(memory, make_record):
    rec = make_record()
    assert await memory.update_persistence(rec, ["throughput"]) == {
        "throughput": 1,
        "queueing_delay": 0,
    }
    assert await memory.update_persistence(rec, ["throughput", "queueing_delay"]) == {
        "throughput": 2,
        "queueing_delay": 1,
    }
    assert await memory.update_persistence(rec, ["queueing_delay"]) == {
        "throughput": 0,
        "queueing_delay": 2,
    }
    # another slice of the same cell has its own counters
    assert await memory.update_persistence(make_record(slice_id="URLLC"), []) == {
        "throughput": 0,
        "queueing_delay": 0,
    }


@pytest.mark.asyncio
async def test_view_separates_this_slice_from_the_other_slices(memory):
    await memory.record_decision("bs1", _event("e1", "eMBB"))
    await memory.record_decision("bs1", _event("u1", "URLLC", ["queueing_delay"]))
    await memory.record_actions(
        "bs1", [_action("e1", "eMBB"), _action("u1", "URLLC", approved=False)]
    )

    view = await memory.view("bs1", "eMBB")
    assert [e.rec for e in view.slice_records] == ["e1"]
    assert [e.rec for e in view.cell_records] == ["u1"]  # other slices only
    assert [a.rec for a in view.slice_actions] == ["e1"]
    rejected = view.cell_actions[0]
    assert rejected.slice == "URLLC" and rejected.approved is False
    assert rejected.stage == "resource_safety" and "prb_percentage" in rejected.why


@pytest.mark.asyncio
async def test_healthy_records_go_to_the_slice_history_only(memory):
    await memory.record_decision("bs1", _event("m1", "MTC", violations=()), cell_level=False)
    assert [e.rec for e in (await memory.view("bs1", "MTC")).slice_records] == ["m1"]
    assert (await memory.view("bs1", "eMBB")).cell_records == []


@pytest.mark.asyncio
async def test_lists_are_capped_newest_first(redis):
    memory = RedisWorkingMemory(redis, max_events=5)
    for i in range(12):
        await memory.record_decision("bs1", _event(f"r{i}"))
    assert await redis.llen(slice_key("bs1", "eMBB", "records")) == 5
    assert await redis.llen(cell_key("bs1", "records")) == 5
    view = await memory.view("bs1", "eMBB", k=3)
    assert [e.rec for e in view.slice_records] == ["r11", "r10", "r9"]


@pytest.mark.asyncio
async def test_ttl_is_applied_on_write(redis, make_record):
    memory = RedisWorkingMemory(redis, ttl_seconds=60)
    await memory.observe(make_record())
    await memory.record_decision("bs1", _event("r1"))
    assert 0 < await redis.ttl(cell_key("bs1", "meta")) <= 60
    assert 0 < await redis.ttl(slice_key("bs1", "eMBB", "records")) <= 60


@pytest.mark.asyncio
async def test_reset_deletes_every_working_memory_key(memory, redis, make_record):
    await redis.set("unrelated", "kept")
    await memory.observe(make_record())
    await memory.record_decision("bs1", _event("r1"))
    await memory.record_actions("bs1", [_action("r1")])
    await memory.update_persistence(make_record(), ["throughput"])

    await memory.reset()

    assert await redis.keys("wm:*") == []
    assert await redis.get("unrelated") == "kept"


def test_limits_read_env(monkeypatch, redis):
    monkeypatch.setenv("MEMORY_MAX_RECORDS", "7")
    monkeypatch.setenv("MEMORY_TTL_SECONDS", "120")
    memory = RedisWorkingMemory(redis)
    assert memory.max_events == 7 and memory.ttl_seconds == 120


def _control_action(tool, params, executed=True):
    return ActionEvent(
        rec="r1",
        t=1.0,
        slice="eMBB",
        agent="near_rt_ric",
        tool=tool,
        params=params,
        approved=executed,
        executed=executed,
    )


@pytest.mark.asyncio
async def test_controls_keep_the_last_executed_prb_values(memory, make_record):
    await memory.observe(make_record())
    await memory.record_actions(
        "bs1",
        [
            _control_action("adjust_prb_allocation", {"prb_percentage": 50}),
            _control_action("adjust_prb_allocation", {"prb_percentage": 90}, executed=False),
            _control_action(
                "update_slice_resource_policy", {"min_prb_ratio_pct": 40, "max_prb_ratio_pct": 70}
            ),
            _control_action("optimize_scheduler", {"scheduler": "waterfilling"}),
        ],
    )

    assert await memory.controls("bs1", "eMBB") == {
        "prb_percentage": 50,
        "policy_min": 40,
        "policy_max": 70,
    }
    assert await memory.controls("bs1", "URLLC") == {}


@pytest.mark.asyncio
async def test_controls_are_cleared_with_a_new_configuration(memory, make_record):
    await memory.observe(make_record())
    await memory.record_actions(
        "bs1", [_control_action("adjust_prb_allocation", {"prb_percentage": 50})]
    )

    await memory.observe(make_record(run="sched1/tr0/exp1"))

    assert await memory.controls("bs1", "eMBB") == {}
