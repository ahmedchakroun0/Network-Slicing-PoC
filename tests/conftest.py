# tests/conftest.py
"""Fixtures shared by every test module.

Nothing here touches the network, Redis or a real model:

- FakeLLM            replaces the chat model. It mimics `with_structured_output`
                     and returns whatever the test queued (or raises it).
- FakeWorkingMemory  in-memory stand-in for RedisWorkingMemory.
- make_record / make_invocation / make_decision
                     factories with sane defaults; override only what a test
                     is about.
- make_sub_agent     async callable that records how it was invoked.
"""

from collections import deque
from typing import Any

import pytest

from src.schema.monitoring import (
    AgentInvocation,
    KPIStatus,
    MonitoringDecision,
    SLAAnalysis,
    TelemetryRecord,
)
from src.schema.working_memory import ToolExecution, WorkingMemory

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
# Fake working memory
# --------------------------------------------------------------------------


class FakeWorkingMemory:
    """Same async API as RedisWorkingMemory, backed by a plain object."""

    def __init__(self) -> None:
        self.state = WorkingMemory()

    async def get(self) -> WorkingMemory:
        return self.state.model_copy(deep=True)

    async def update_network_status(self, status: str) -> None:
        self.state.network_status = status

    async def add_tool_execution(self, execution: ToolExecution) -> None:
        self.state.tools_executed.append(execution)

    # test helper
    def executions_of(self, agent: str) -> list[ToolExecution]:
        return [e for e in self.state.tools_executed if e.agent == agent]


@pytest.fixture
def memory() -> FakeWorkingMemory:
    return FakeWorkingMemory()


# --------------------------------------------------------------------------
# Data factories
# --------------------------------------------------------------------------

# A record where every KPI is within budget.
_OK_RECORD: dict[str, Any] = {
    "use_case_type": "eMBB",
    "packet_loss_budget": 0.01,
    "latency_budget_ns": 1000.0,
    "jitter_budget_ns": 100.0,
    "data_rate_budget_gbps": 10.0,
    "slice_available_transfer_rate_gbps": 20.0,
    "slice_latency_ns": 500.0,
    "slice_packet_loss": 0.001,
    "slice_jitter_ns": 50.0,
    "slice_type": "eMBB",
    "record_id": "rec-1",
}


@pytest.fixture
def make_record():
    """make_record(slice_latency_ns=2000.0, jitter_budget_ns=None, ...)"""

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
    return SLAAnalysis(latency=kpi, jitter=kpi, packet_loss=kpi, throughput=kpi)


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
    def _make(sub_agents: dict[str, Any] | None = None):
        return MonitoringAgent(llm=fake_llm, memory=memory, sub_agents=sub_agents or {})

    return _make
