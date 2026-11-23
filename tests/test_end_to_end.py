"""End-to-end test of the closed loop (scripts/run_loop.py) without external services.

Everything is the real code path (MonitoringAgent, LangChain RIC agents built
by create_agent, RANAuthorizationMiddleware, RANTrustGovernanceAgent with its
LLM review stage, the MCP servers and client adapter, the RAN emulator,
MongoRepository, RedisWorkingMemory, action verification) except the
external services, replaced by in-process stand-ins:
  * the LLM      -> ScriptedChatModel (tool-calling chat model, scripted)
  * Redis        -> fakeredis
  * MongoDB      -> mongomock
  * emulator     -> the FastAPI app over httpx.ASGITransport (synthetic traces)
  * MCP/LiteLLM  -> in-process MCP clients connected straight to the servers

The scripted Near-RT RIC always raises the eMBB PRB share to 72 % (snapped to
36 of 50 PRBs). On a cell where that fixes eMBB, the next step verifies the
action as "resolved" and the agents stay quiet. On a cell where it cannot
(throughput does not scale enough), the violation persists: the action is
verified "improved" then "unchanged", and at the 3rd consecutive violated
record the Non-RT RIC's policy is approved.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import fakeredis.aioredis
import httpx
import mongomock
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from mcp import Client

import agents.base_ric as base_ric
import scripts.run_loop as run_loop
import src.core.llm as core_llm
import src.repositories.mongo_repository as mongo_repo
from agents.ran_trust_governance import _LLMReview
from ran_emulator.api import create_app
from ran_emulator.client import EmulatorClient
from ran_emulator.traces import TraceBank
from ric_tools.server import build_server
from src.memory.working_memory import RedisWorkingMemory, slice_key
from src.schema.monitoring import AgentInvocation, MonitoringDecision


class _Structured:
    def __init__(self, schema):
        self.schema = schema

    async def ainvoke(self, _input, *args, **kwargs):
        if self.schema is MonitoringDecision:
            return MonitoringDecision(
                situation="critical",
                severity="high",
                global_situation="queueing delay degradation",
                action_required=True,
                selected_agents=[
                    AgentInvocation(
                        agent="near_rt_ric",
                        responsibility="fast mitigation",
                        objective="Reduce latency",
                        priority="high",
                        reason="latency violation",
                    ),
                    AgentInvocation(
                        agent="non_rt_ric",
                        responsibility="policy",
                        objective="Prevent persistent latency",
                        priority="high",
                        reason="persistence",
                    ),
                ],
                reason="latency violation",
            )
        if self.schema is _LLMReview:
            return _LLMReview(
                goal_aligned=True,
                reasoning_sound=True,
                behavior_acceptable=True,
                reason="Scripted approval.",
            )
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
            base = {
                "target": ctx["target"],
                "rationale": "eMBB throughput and delay exceed their budgets on this slice.",
                "expected_effect": "Higher throughput, lower delay.",
                "confidence": 0.8,
            }
            if "Non-RT RIC Agent" in system:
                name, args = (
                    "update_slice_resource_policy",
                    {
                        **base,
                        "slice_id": ctx["slice_id"],
                        "min_prb_ratio_pct": 60,
                        "max_prb_ratio_pct": 90,
                    },
                )
            else:
                name, args = (
                    "adjust_prb_allocation",
                    {**base, "slice_id": ctx["slice_id"], "prb_percentage": 72},
                )
            _CALLS["n"] += 1
            msg = AIMessage(
                content="", tool_calls=[{"name": name, "args": args, "id": f"call_{_CALLS['n']}"}]
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])


SCENARIO = {
    "name": "e2e",
    "seed": 0,
    # tr 9: eMBB 18 PRBs, MTC 21, URLLC 11 -> eMBB violates both KPIs.
    "cells": {"bs2": {"scheduler": "round_robin", "tr": 9, "start_offset": 0}},
}


@pytest.fixture
def services(monkeypatch, tmp_path):
    """Scripted model, fakeredis, mongomock; returns the captured repository."""
    from src.core.llm import PROVIDER_GUARD

    # The usage callback counts the model calls per caller, as create_llm does.
    fake_model = ScriptedChatModel(callbacks=[PROVIDER_GUARD])
    monkeypatch.setattr(run_loop, "create_llm", lambda: fake_model)
    monkeypatch.setattr(base_ric, "create_llm", lambda: fake_model)
    monkeypatch.setattr(core_llm, "create_llm", lambda: fake_model)  # governance review
    monkeypatch.setattr(mongo_repo, "MongoClient", mongomock.MongoClient)
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(
        RedisWorkingMemory, "from_url", classmethod(lambda cls, *a, **k: cls(redis))
    )
    store = {"redis": redis}
    original_init = mongo_repo.MongoRepository.__init__

    def _capture(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        store["repo"] = self

    monkeypatch.setattr(mongo_repo.MongoRepository, "__init__", _capture)
    (tmp_path / "e2e.json").write_text(json.dumps(SCENARIO))
    return store


@asynccontextmanager
async def closed_loop(bank):
    """The emulator and both MCP servers, in process."""
    emulator = EmulatorClient(
        base_url="http://emu", transport=httpx.ASGITransport(app=create_app(bank, _scenario()))
    )
    async with (
        Client(build_server("near_rt_ric", emulator)) as near,
        Client(build_server("non_rt_ric", emulator)) as non,
    ):
        yield emulator, {"near_rt_ric": near, "non_rt_ric": non}


def _scenario():
    from ran_emulator.scenarios import Scenario

    return Scenario.model_validate(SCENARIO)


def _actions(doc, agent):
    return doc["agents"][agent]["actions"]


async def test_action_fixes_the_violation_and_is_verified_resolved(frame, services, tmp_path):
    # eMBB rate that meets its 3.6 Mbps/UE budget with 36 PRBs (not with 18).
    frame = frame.copy()
    embb = frame.slice_id == "eMBB"
    frame.loc[embb, "dl_throughput_per_ue_mbps"] = 0.12 * frame.loc[embb, "slice_prb"]
    bank = TraceBank.from_frame(frame)

    async with closed_loop(bank) as (emulator, mcp_clients):
        await run_loop.run_loop(
            run_loop.RunConfig(
                run_id="loop",
                scenario=str(tmp_path / "e2e.json"),
                steps=3,
                results_dir=tmp_path / "results",
            ),
            emulator=emulator,
            mcp_clients=mcp_clients,
        )
        sim_actions = await emulator._call("GET", "/sim/actions")
        violations = await emulator.violations()

    repo = services["repo"]
    run = repo.runs.find_one({"_id": "loop"})
    assert run["status"] == "finished" and run["config"]["source"] == "emulator"
    summary = run["summary"]
    # 3 steps x 3 slices; only step 0's eMBB record violates.
    assert summary["steps"] == 3 and summary["rows_selected"] == 9
    assert summary["rows_processed"] == 1 and summary["rows_without_violation"] == 8
    assert summary["action_outcomes"] == {"resolved": 1}

    # The approved call really changed the RAN, inside its envelope.
    assert [(a["control"], a["requested_pct"], a["applied_pct"]) for a in sim_actions] == [
        ("prb", 72, 72.0)
    ]
    assert violations == []

    (doc,) = repo.records.find({"run_id": "loop"})
    (action,) = _actions(doc, "near_rt_ric")
    assert action["governance"]["approved"] and action["execution"]["status"] == "executed"
    assert action["execution"]["result"]["current"]["eMBB"] == 36
    outcome = action["outcome"]
    assert outcome["outcome"] == "resolved"
    assert outcome["checked_rec"] == "emu-e2e-bs2-s00001-eMBB"
    assert outcome["kpis"]["throughput"]["before"] < 3.6 <= outcome["kpis"]["throughput"]["after"]

    # Working memory keeps the action (stable run id) and shows the outcome.
    memory = RedisWorkingMemory(services["redis"])
    view = (await memory.view("bs2", "eMBB")).compact()
    assert view["slice_actions"][0]["applied"]["applied_pct"] == 72.0
    assert view["slice_outcomes"][0]["outcome"] == "resolved"
    assert await memory.controls("bs2", "eMBB") == {"prb_percentage": 72.0}
    assert await memory.controls("bs2", "MTC") == {"prb_percentage": 6.0}

    metrics = json.loads((tmp_path / "results" / "loop" / "metrics.json").read_text())
    assert metrics["closed_loop"]["outcomes"] == {"resolved": 1}


async def test_unfixable_violation_escalates_to_the_non_rt_policy(bank, services, tmp_path):
    """Synthetic eMBB throughput cannot reach its budget with any split: the
    same PRB action is verified improved, then unchanged, and at the 3rd
    violated record the Non-RT RIC's policy is approved."""
    async with closed_loop(bank) as (emulator, mcp_clients):
        await run_loop.run_loop(
            run_loop.RunConfig(run_id="esc", scenario=str(tmp_path / "e2e.json"), steps=3),
            emulator=emulator,
            mcp_clients=mcp_clients,
        )

    repo = services["repo"]
    docs = sorted(repo.records.find({"run_id": "esc"}), key=lambda d: d["seq"])
    assert [d["seq"] for d in docs] == [0, 1, 2]
    assert [d["persistence"]["throughput"] for d in docs] == [1, 2, 3]

    outcomes = [_actions(d, "near_rt_ric")[0].get("outcome", {}).get("outcome") for d in docs]
    # step 0's action (18 -> 36 PRBs) halves the gap; step 1 repeats it: no change.
    assert outcomes[:2] == ["improved", "unchanged"]

    for doc, observed in zip(docs[:2], (1, 2), strict=True):
        assert "non_rt_ric" not in doc["agents"]
        assert f"persisted {observed} consecutive" in doc["monitoring"]["orchestration_notes"][0]
    (policy,) = _actions(docs[2], "non_rt_ric")
    assert policy["governance"]["approved"] and policy["execution"]["status"] == "executed"
    assert policy["execution"]["result"]["current"] == {
        "min_prb_ratio_pct": 60.0,
        "max_prb_ratio_pct": 90.0,
    }
    assert await RedisWorkingMemory(services["redis"]).controls("bs2", "eMBB") == {
        "prb_percentage": 72.0,
        "policy_min": 60.0,
        "policy_max": 90.0,
    }

    # The model usage of a record is split per caller.
    assert docs[2]["llm"]["by_caller"]["non_rt_ric"]["calls"] >= 1
    assert await services["redis"].llen(slice_key("bs2", "eMBB", "outcomes")) == 2


async def test_run_stops_when_every_model_is_rate_limited(bank, services, tmp_path, monkeypatch):
    """The 2nd violated record exhausts the fallback chain: it is failed (its
    calls were absorbed by fail-closed agents), the run stops and no later
    record is processed."""
    from src.core.llm import PROVIDER_GUARD

    seen = []

    class _Workflow:
        def __init__(self, **kwargs):
            pass

        async def run(self, record, seq=0):
            seen.append(record.record_id)
            if len(seen) == 2:
                PROVIDER_GUARD.stopped = "RateLimitError: all deployments in cooldown"
            decision = MonitoringDecision(
                situation="critical",
                severity="high",
                global_situation="x",
                action_required=False,
                reason="x",
            )
            return {"outcome": "done", "decision": decision, "agent_results": {}}

    monkeypatch.setattr(run_loop, "RecordWorkflow", _Workflow)
    try:
        async with closed_loop(bank) as (emulator, mcp_clients):
            await run_loop.run_loop(
                run_loop.RunConfig(
                    run_id="stop",
                    scenario=str(tmp_path / "e2e.json"),
                    steps=3,
                    results_dir=tmp_path / "results",
                ),
                emulator=emulator,
                mcp_clients=mcp_clients,
            )
    finally:
        PROVIDER_GUARD.reset()

    repo = services["repo"]
    run = repo.runs.find_one({"_id": "stop"})
    assert seen == ["emu-e2e-bs2-s00000-eMBB", "emu-e2e-bs2-s00000-MTC"]
    assert (
        run["status"] == "stopped" and "step 0, record emu-e2e-bs2-s00000-MTC" in run["stop_reason"]
    )
    failed = repo.records.find_one({"_id": "stop:emu-e2e-bs2-s00000-MTC"})
    assert failed["status"] == "failed" and failed["error"]["type"] == "ModelProviderStopped"
    metrics = json.loads((tmp_path / "results" / "stop" / "metrics.json").read_text())
    assert metrics["stopped"] == {"stop": run["stop_reason"]}
