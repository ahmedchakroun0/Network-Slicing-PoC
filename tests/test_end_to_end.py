"""End-to-end test of scripts/run_dataset.py without external services.

Everything is the real code path (MonitoringAgent, LangChain RIC agents built
by create_agent, RANAuthorizationMiddleware, RANTrustGovernanceAgent with its
LLM review stage, MongoRepository, RedisWorkingMemory) except the three
external services, replaced by in-memory fakes:
  * the LLM  -> ScriptedChatModel (tool-calling chat model, scripted)
  * Redis    -> fakeredis
  * MongoDB  -> mongomock

Scenario: two slices of the same cell are interleaved. Slice A violates
latency on every record, slice B never violates. Per-(cell, slice)
persistence must reach 3 on the 3rd A record (row 5), so the Non-RT policy
action is rejected before and approved from then on, while Near-RT actions
are approved immediately. With the former single global counter, B's
healthy records reset A's persistence and no policy action could ever pass.
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
                situation="critical", severity="high", global_situation="latency degradation",
                action_required=True,
                sla_analysis={k: {"status": "ok"} for k in ("latency", "jitter", "packet_loss", "throughput")},
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
            human = [m.content for m in messages if m.type == "human"][-1]
            inv = json.loads(human[human.index("{"):])
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


def _row(record_id, slice_type, latency_ns):
    return {
        "record_id": record_id, "Use Case Type": "test", "Packet Loss Budget": 0.01,
        "Latency Budget (ns)": 1000.0, "Jitter Budget (ns)": 1000.0, "Data Rate Budget (Gbps)": 1.0,
        "Required Mobility": "no", "Required Connectivity": "yes",
        "Slice Available Transfer Rate (Gbps)": 5.0, "Slice Latency (ns)": latency_ns,
        "Slice Packet Loss": 0.0, "Slice Jitter (ns)": 10.0, "Slice Type": slice_type,
        "Slice Handover": "no", "Cell ID": "bs7",
    }


@pytest.mark.asyncio
async def test_run_dataset_end_to_end(tmp_path, monkeypatch):
    rows = []
    for i in range(3):
        rows.append(_row(f"r{2 * i}", "sliceA", 5000.0))   # violates latency
        rows.append(_row(f"r{2 * i + 1}", "sliceB", 10.0))  # healthy
    csv = tmp_path / "telemetry.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    fake_model = ScriptedChatModel()
    monkeypatch.setattr(run_dataset, "create_llm", lambda: fake_model)
    monkeypatch.setattr(base_ric, "create_llm", lambda: fake_model)
    monkeypatch.setattr(core_llm, "create_llm", lambda: fake_model)   # governance LLM review (lazy import)
    monkeypatch.setattr(mongo_repo, "MongoClient", mongomock.MongoClient)
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(RedisWorkingMemory, "from_url", classmethod(lambda cls, *a, **k: cls(redis)))

    config = run_dataset.RunConfig(dataset_path=csv, run_id="e2e", start_row=1, max_rows=None,
                                   knowledge_file=run_dataset.DEFAULT_KNOWLEDGE_FILE)
    store = {}
    original_init = mongo_repo.MongoRepository.__post_init__

    def _capture(self):
        original_init(self)
        store["collection"] = self._collection

    monkeypatch.setattr(mongo_repo.MongoRepository, "__post_init__", _capture)

    await run_dataset.run_dataset(config)

    doc = store["collection"].find_one({"_id": "e2e"})
    assert doc["summary"]["rows_processed"] == 6
    assert doc["summary"]["rows_failed"] == 0

    recs = doc["records"]
    # healthy slice: deterministic override -> no agent invoked
    for rid in ("r1", "r3", "r5"):
        assert not recs[rid]["monitoring_decision"]["selected_agents"]

    def outcome(rid, agent):
        audit = recs[rid]["agent_executions"][agent]["middleware_audit"]
        assert len(audit) == 1
        return audit[0]["governance"]

    # Near-RT: approved on every violated record (LLM review included)
    for rid in ("r0", "r2", "r4"):
        g = outcome(rid, "near_rt_ric")
        assert g["approved"] is True and "llm_review" in g["checks_passed"]
        ctx = recs[rid]["agent_executions"]["near_rt_ric"]["invocation"]["context"]
        assert ctx["target"] == "bs7" and ctx["slice_id"] == "sliceA"

    # Non-RT: per-(cell, slice) persistence 1, 2 -> rejected; 3 -> approved
    assert outcome("r0", "non_rt_ric")["failed_stage"] == "sla"
    assert outcome("r2", "non_rt_ric")["failed_stage"] == "sla"
    assert outcome("r4", "non_rt_ric")["approved"] is True
    assert recs["r4"]["agent_executions"]["non_rt_ric"]["invocation"]["context"]["persistence"]["latency"] == 3

    # working memory stayed bounded
    from src.schema.working_memory import WorkingMemory
    wm = WorkingMemory.model_validate_json(await redis.get(RedisWorkingMemory.KEY))
    assert len(wm.tools_executed) <= RedisWorkingMemory(redis).max_tool_executions
