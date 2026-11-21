"""End-to-end test of scripts/run_dataset.py without external services.

Everything is the real code path (MonitoringAgent, LangChain RIC agents built
by create_agent, RANAuthorizationMiddleware, RANTrustGovernanceAgent with its
LLM review stage, MongoRepository, RedisWorkingMemory) except the three
external services, replaced by in-memory fakes:
  * the LLM  -> ScriptedChatModel (tool-calling chat model, scripted)
  * Redis    -> fakeredis
  * MongoDB  -> mongomock

Scenario: two slices of the same cell are interleaved. The URLLC slice
violates its queueing-delay budget on every record, the MTC slice never
violates. MTC records are skipped before the Monitoring Agent (only counted).
Per-(cell, slice) persistence, kept in Redis, must reach 3 on the 3rd URLLC
record (row 5): the Non-RT RIC is not invoked before (orchestration note) and
its policy action is approved then, while Near-RT actions are approved
immediately. With a single
global counter, the healthy MTC records would reset it and no policy action
could ever pass.
"""

from __future__ import annotations

import json

import fakeredis.aioredis
import mongomock
import pandas as pd
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import agents.base_ric as base_ric
import scripts.run_dataset as run_dataset
import src.core.llm as core_llm
import src.repositories.mongo_repository as mongo_repo
from agents.ran_trust_governance import _LLMReview
from src.memory.working_memory import RedisWorkingMemory
from src.schema.monitoring import AgentInvocation, MonitoringDecision


class _Structured:
    def __init__(self, schema):
        self.schema = schema

    async def ainvoke(self, _input, *args, **kwargs):
        if self.schema is MonitoringDecision:
            return MonitoringDecision(
                situation="critical", severity="high", global_situation="queueing delay degradation",
                action_required=True,
                selected_agents=[
                    AgentInvocation(agent="near_rt_ric", responsibility="fast mitigation",
                                    objective="Reduce latency", priority="high", reason="latency violation"),
                    AgentInvocation(agent="non_rt_ric", responsibility="policy",
                                    objective="Prevent persistent latency", priority="high", reason="persistence"),
                ],
                reason="latency violation",
            )
        if self.schema is _LLMReview:
            return _LLMReview(goal_aligned=True, reasoning_sound=True, behavior_acceptable=True,
                              reason="Scripted approval.")
        raise AssertionError(f"unexpected schema {self.schema}")


_CALLS = {"n": 0}


class ScriptedChatModel(BaseChatModel):
    """Tool-calling chat model: first turn emits one protected tool call built
    from the invocation context; after the tool result it stops."""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return _Structured(schema)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if any(isinstance(m, ToolMessage) for m in messages):
            msg = AIMessage(content="Done.")
        else:
            system = next((m.content for m in messages if m.type == "system"), "")
            # The invocation is sent once, as one JSON line of the system prompt.
            inv = json.loads(system.split(" invocation:\n", 1)[1].split("\n", 1)[0])
            ctx = inv["context"]
            base = {"target": ctx["target"], "rationale": "Latency exceeds its budget on this slice.",
                    "expected_effect": "Lower latency.", "confidence": 0.8}
            if "Non-RT RIC Agent" in system:
                name, args = "update_slice_resource_policy", {**base, "slice_id": ctx["slice_id"],
                                                               "min_prb_ratio_pct": 30, "max_prb_ratio_pct": 80}
            else:
                name, args = "adjust_prb_allocation", {**base, "slice_id": ctx["slice_id"], "prb_percentage": 60}
            _CALLS["n"] += 1
            msg = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{_CALLS['n']}"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _row(record_id, slice_id, t, delay_ms):
    return {
        "record_id": record_id, "cell_id": "bs7", "slice_id": slice_id, "run": "sched0/tr0/exp1",
        "window_start_s": t, "window_s": 1.0, "slice_prb": 5, "cell_prb": 50,
        "scheduler": "round_robin", "active_ues": 2, "dl_throughput_mbps": 0.24,
        "dl_throughput_per_ue_mbps": 0.12, "dl_buffer_bytes": 300.0, "queueing_delay_ms": delay_ms,
        "prb_utilisation": 0.3, "cell_spare_prb": 30.0, "dl_cqi": 8.0, "dl_mcs": 12.0,
        "ul_sinr_db": None,
    }


@pytest.mark.asyncio
async def test_run_dataset_end_to_end(tmp_path, monkeypatch):
    rows = []
    for i in range(3):
        rows.append(_row(f"r{2 * i}", "URLLC", 50.0 + i, 40.0))   # > 10 ms URLLC budget
        rows.append(_row(f"r{2 * i + 1}", "MTC", 50.0 + i, 1.0))  # healthy
    csv = tmp_path / "telemetry.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    from src.core.llm import PROVIDER_GUARD

    fake_model = ScriptedChatModel(callbacks=[PROVIDER_GUARD])
    monkeypatch.setattr(run_dataset, "create_llm", lambda: fake_model)
    monkeypatch.setattr(base_ric, "create_llm", lambda: fake_model)
    monkeypatch.setattr(core_llm, "create_llm", lambda: fake_model)   # governance LLM review (lazy import)
    monkeypatch.setattr(mongo_repo, "MongoClient", mongomock.MongoClient)
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(RedisWorkingMemory, "from_url", classmethod(lambda cls, *a, **k: cls(redis)))

    config = run_dataset.RunConfig(dataset_path=csv, run_id="e2e", start_row=1, max_rows=None,
                                   knowledge_file=run_dataset.DEFAULT_KNOWLEDGE_FILE,
                                   results_dir=tmp_path / "results")
    store = {}
    original_init = mongo_repo.MongoRepository.__init__

    def _capture(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        store["repo"] = self

    monkeypatch.setattr(mongo_repo.MongoRepository, "__init__", _capture)

    await run_dataset.run_dataset(config)

    repo = store["repo"]
    summary = repo.runs.find_one({"_id": "e2e"})["summary"]
    assert summary["rows_processed"] == 3
    assert summary["rows_without_violation"] == 3
    assert summary.get("rows_failed", 0) == 0
    assert summary["agent_invocations"] == {"near_rt_ric": 3, "non_rt_ric": 1}

    recs = {d["_id"].split(":", 1)[1]: d for d in repo.records.find({"run_id": "e2e"})}
    # healthy slice: skipped before the Monitoring Agent, no record document
    assert sorted(recs) == ["r0", "r2", "r4"]
    assert all(d["status"] == "done" for d in recs.values())

    def outcome(rid, agent):
        actions = recs[rid]["agents"][agent]["actions"]
        assert len(actions) == 1
        return actions[0]

    # Near-RT: approved on every violated record (LLM review included)
    for rid in ("r0", "r2", "r4"):
        action = outcome(rid, "near_rt_ric")
        assert action["governance"]["approved"] is True
        assert "llm_review" in action["governance"]["checks_passed"]
        assert action["governance"]["llm_review"]["reason"] == "Scripted approval."
        assert action["target"] == "bs7" and action["params"]["slice_id"] == "URLLC"
        assert action["reasoning"]["rationale"]

    # Non-RT: per-(cell, slice) persistence 1, 2 -> not invoked; 3 -> approved
    for rid, observed in (("r0", 1), ("r2", 2)):
        assert "non_rt_ric" not in recs[rid].get("agents", {})
        note = recs[rid]["monitoring"]["orchestration_notes"][0]
        assert f"persisted {observed} consecutive" in note
    assert outcome("r4", "non_rt_ric")["governance"]["approved"] is True
    assert recs["r4"]["persistence"]["queueing_delay"] == 3
    assert recs["r4"]["sla"]["violations"] == ["queueing_delay"]

    # model usage per record: the RIC agents' tool-calling turns (the scripted
    # structured outputs are not chat-model calls); healthy records make none
    assert recs["r4"]["llm"]["by_caller"]["near_rt_ric"]["calls"] >= 1
    assert recs["r4"]["llm"]["by_caller"]["non_rt_ric"]["calls"] >= 1

    # the run's result files, in a directory named after the run
    out = tmp_path / "results" / "e2e"
    assert sorted(f.name for f in out.iterdir()) == [
        "actions.csv", "e4_table.tex", "expert_rating.csv", "metrics.json", "records.csv"]
    assert json.loads((out / "metrics.json").read_text())["governance"]["proposals"] == 4

    # working memory: bounded lists; the healthy MTC records are in their slice history
    from src.memory.working_memory import cell_key, slice_key
    limit = RedisWorkingMemory(redis).max_events
    for key in (cell_key("bs7", "records"), slice_key("bs7", "URLLC", "actions")):
        assert 0 < await redis.llen(key) <= limit
    assert await redis.llen(slice_key("bs7", "MTC", "records")) == 3


@pytest.mark.asyncio
async def test_run_stops_when_every_model_is_rate_limited(tmp_path, monkeypatch):
    """The 2nd violated record exhausts the fallback chain: it is failed (its
    calls were absorbed by fail-closed agents), the run stops and the 3rd
    record is never processed."""
    from src.core.llm import PROVIDER_GUARD
    from src.schema.monitoring import MonitoringDecision

    seen = []

    class _Workflow:
        def __init__(self, **kwargs):
            pass

        async def run(self, record, seq=0):
            seen.append(record.record_id)
            if len(seen) == 2:
                PROVIDER_GUARD.stopped = "RateLimitError: all deployments in cooldown"
            decision = MonitoringDecision(situation="critical", severity="high", global_situation="x",
                                          action_required=False, reason="x")
            return {"outcome": "done", "decision": decision, "agent_results": {}}

    pd.DataFrame([_row(f"r{i}", "URLLC", 50.0 + i, 40.0) for i in range(3)]).to_csv(
        tmp_path / "t.csv", index=False)
    monkeypatch.setattr(run_dataset, "RecordWorkflow", _Workflow)
    monkeypatch.setattr(run_dataset, "create_llm", lambda: ScriptedChatModel())
    monkeypatch.setattr(mongo_repo, "MongoClient", mongomock.MongoClient)
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(RedisWorkingMemory, "from_url", classmethod(lambda cls, *a, **k: cls(redis)))
    store = {}
    original_init = mongo_repo.MongoRepository.__init__

    def _capture(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        store["repo"] = self

    monkeypatch.setattr(mongo_repo.MongoRepository, "__init__", _capture)

    try:
        await run_dataset.run_dataset(run_dataset.RunConfig(
            dataset_path=tmp_path / "t.csv", run_id="stop", start_row=1, max_rows=None,
            knowledge_file=run_dataset.DEFAULT_KNOWLEDGE_FILE, results_dir=tmp_path / "results"))
    finally:
        PROVIDER_GUARD.reset()

    repo = store["repo"]
    run = repo.runs.find_one({"_id": "stop"})
    assert seen == ["r0", "r1"]
    assert run["status"] == "stopped" and "row 2" in run["stop_reason"]
    failed = repo.records.find_one({"_id": "stop:r1"})
    assert failed["status"] == "failed" and failed["error"]["type"] == "ModelProviderStopped"
    metrics = json.loads((tmp_path / "results" / "stop" / "metrics.json").read_text())
    assert metrics["stopped"] == {"stop": run["stop_reason"]}
    assert metrics["dataset"]["records_not_evaluated"] == 1

