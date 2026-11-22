# tests/conftest.py
"""Fixtures shared by every test module.

Nothing here touches the network, a real Redis/MongoDB or a real model:

- FakeLLM            replaces the chat model. It mimics `with_structured_output`
                     and returns whatever the test queued (or raises it).
- memory             the real RedisWorkingMemory on fakeredis.
- mongo              the real MongoRepository on mongomock.
- make_record / make_invocation / make_decision
                     factories with sane defaults; override only what a test
                     is about.
- make_sub_agent     async callable that records how it was invoked.
- frame / bank / make_scenario
                     a small synthetic ColO-RAN trace bank and scenario factory
                     for the RAN emulator (tests/emulator_fixtures.py).
"""

import os
from collections import deque
from typing import Any

# Tests never send traces, whatever .env says (load_dotenv keeps this value).
os.environ["LANGSMITH_TRACING"] = "false"

import fakeredis.aioredis
import mongomock
import pytest

import src.repositories.mongo_repository as mongo_repository_module
from src.memory.working_memory import RedisWorkingMemory
from src.repositories.mongo_repository import MongoRepository
from src.schema.monitoring import (
    AgentInvocation,
    KPIStatus,
    MonitoringDecision,
    SLAAnalysis,
    TelemetryRecord,
)
from tests.emulator_fixtures import bank, frame, make_scenario  # noqa: F401 (fixtures)

# --------------------------------------------------------------------------
# Fake LLM
# --------------------------------------------------------------------------


class _FakeStructuredLLM:
    """What `llm.with_structured_output(Schema)` returns."""

    def __init__(self, parent: "FakeLLM", schema: Any) -> None:
        self._parent = parent
        self.schema = schema

    async def ainvoke(self, prompt_value: Any, *args: Any, **kwargs: Any) -> Any:
        self._parent.calls.append({"schema": self.schema, "input": prompt_value})
        if not self._parent.responses:
            raise AssertionError("FakeLLM was called but no response was queued.")
        response = self._parent.responses.popleft()
        if isinstance(response, BaseException):
            raise response
        return response


class FakeLLM:
    """Deterministic chat model.

    llm.queue(obj_1, obj_2, ...)   responses returned in order; an exception
                                   instance in the queue is raised instead.
    llm.calls                      every call: {"schema", "input"}.
    llm.last_prompt                text of the last prompt it received.
    """

    def __init__(self) -> None:
        self.responses: deque[Any] = deque()
        self.calls: list[dict[str, Any]] = []

    def queue(self, *items: Any) -> "FakeLLM":
        self.responses.extend(items)
        return self

    def with_structured_output(self, schema: Any) -> _FakeStructuredLLM:
        return _FakeStructuredLLM(self, schema)

    @property
    def last_prompt(self) -> str:
        prompt = self.calls[-1]["input"]
        return prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


# --------------------------------------------------------------------------
# Stores: real code on in-memory backends
# --------------------------------------------------------------------------


@pytest.fixture
def redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def memory(redis) -> RedisWorkingMemory:
    return RedisWorkingMemory(redis, max_events=30)


@pytest.fixture
def mongo(monkeypatch) -> MongoRepository:
    monkeypatch.setattr(mongo_repository_module, "MongoClient", mongomock.MongoClient)
    return MongoRepository()


# --------------------------------------------------------------------------
# Data factories
# --------------------------------------------------------------------------

# An eMBB record where every KPI of the eMBB SLA profile is within budget
# (throughput >= 3.6 Mbps per UE, queueing delay <= 100 ms).
_OK_RECORD: dict[str, Any] = {
    "record_id": "rec-1",
    "cell_id": "bs1",
    "slice_id": "eMBB",
    "run": "sched0/tr6/exp1",
    "window_start_s": 60.0,
    "window_s": 1.0,
    "slice_prb": 42,
    "cell_prb": 50,
    "scheduler": "round_robin",
    "active_ues": 2,
    "dl_throughput_mbps": 8.0,
    "dl_throughput_per_ue_mbps": 4.0,
    "dl_buffer_bytes": 10_000.0,
    "queueing_delay_ms": 20.0,
    "prb_utilisation": 0.9,
    "cell_spare_prb": 6.0,
    "dl_cqi": 8.5,
    "dl_mcs": 15.0,
    "ul_sinr_db": 10.0,
}


@pytest.fixture
def make_record():
    """make_record(queueing_delay_ms=400.0, slice_id="URLLC", ...)"""

    def _make(**overrides: Any) -> TelemetryRecord:
        return TelemetryRecord(**{**_OK_RECORD, **overrides})

    return _make


@pytest.fixture
def make_invocation():
    def _make(agent: str = "near_rt_ric", **overrides: Any) -> AgentInvocation:
        data: dict[str, Any] = {
            "agent": agent,
            "responsibility": "Mitigate the SLA violation.",
            "objective": "Restore the violated KPIs.",
            "priority": "high",
            "reason": "SLA violation detected.",
            "context": {"target": "cell-1", "slice_id": "slice-1"},
        }
        data.update(overrides)
        return AgentInvocation(**data)

    return _make


def _unavailable_sla() -> SLAAnalysis:
    kpi = KPIStatus(status="unavailable")
    return SLAAnalysis(kpis={"throughput": kpi, "queueing_delay": kpi})


@pytest.fixture
def make_decision(make_invocation):
    """A MonitoringDecision as the LLM would return it. Its sla_analysis is
    deliberately a placeholder: the agent must overwrite it."""

    def _make(**overrides: Any) -> MonitoringDecision:
        data: dict[str, Any] = {
            "situation": "critical",
            "severity": "high",
            "global_situation": "Slice degraded.",
            "action_required": True,
            "sla_analysis": _unavailable_sla(),
            "selected_agents": [make_invocation()],
            "reason": "LLM reason.",
        }
        data.update(overrides)
        return MonitoringDecision(**data)

    return _make


class RecordingSubAgent:
    """Async sub-agent double: records invocations, returns or raises."""

    def __init__(self, result: Any = None, error: BaseException | None = None):
        self.result = {"status": "ok"} if result is None else result
        self.error = error
        self.calls: list[AgentInvocation] = []

    async def __call__(self, invocation: AgentInvocation) -> Any:
        self.calls.append(invocation)
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def make_sub_agent():
    def _make(result: Any = None, error: BaseException | None = None):
        return RecordingSubAgent(result=result, error=error)

    return _make


# --------------------------------------------------------------------------
# Monitoring agent wiring
# --------------------------------------------------------------------------


@pytest.fixture
def MonitoringAgent():
    # The only place that knows where the class lives: adjust this import
    # to your project layout.
    from agents.monitoring_agent import MonitoringAgent as cls

    return cls


@pytest.fixture
def monitoring_prompt_dir(tmp_path, monkeypatch):
    """MonitoringAgent reads Path("prompts/monitoring.txt") relative to the
    working directory, so tests run in a temp dir holding a stub prompt.
    (Keep braces out of the stub: it is used as a prompt template.)"""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "monitoring.txt").write_text(
        "You are the Monitoring Agent (test prompt).", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def make_monitoring_agent(MonitoringAgent, fake_llm, memory, monitoring_prompt_dir):
    def _make(sub_agents: dict[str, Any] | None = None, mongo: MongoRepository | None = None,
              run_id: str | None = None):
        return MonitoringAgent(llm=fake_llm, memory=memory, sub_agents=sub_agents or {},
                               mongo_repository=mongo, run_id=run_id)

    return _make
