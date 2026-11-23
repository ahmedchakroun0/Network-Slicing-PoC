"""The Monitoring agent as an A2A service.

Skill ran.monitoring.observe: {"run_id", "seq", "records": [TelemetryRecord]}
-> {"results": [RecordOutcome], "stopped"}. Each record goes through the
record workflow (agents/record_workflow.py) unchanged; the RIC agents it
selects are RemoteRICAgent clients found by their Agent Cards, so every
RIC invocation is an A2A task.

The experiment driver (scripts/run_loop.py --agents a2a) steps the emulator
and sends each step's records here; it is the RAN clock, not an agent.
"""

from __future__ import annotations

import asyncio
from typing import Any

import src.core.llm as core_llm
from a2a_agents.client import A2AJsonClient
from agents.monitoring_agent import MonitoringAgent
from agents.record_runner import process_record
from agents.record_workflow import RecordWorkflow
from src.core.llm import PROVIDER_GUARD
from src.memory.working_memory import RedisWorkingMemory
from src.repositories.mongo_repository import MongoRepository
from src.schema.monitoring import TelemetryRecord


class MonitoringService:
    """One Monitoring agent and record workflow per run (both carry the run id)."""

    def __init__(self, *, sub_agents: dict[str, Any], memory: RedisWorkingMemory,
                 mongo_repository: MongoRepository):
        self.sub_agents = sub_agents
        self.memory = memory
        self.mongo_repository = mongo_repository
        self._run_id: str | None = None
        self._workflow: RecordWorkflow | None = None
        self._lock = asyncio.Lock()         # one step at a time: the loop is lockstep

    def workflow(self, run_id: str) -> RecordWorkflow:
        if self._workflow is None or run_id != self._run_id:
            PROVIDER_GUARD.reset()
            monitoring = MonitoringAgent(
                llm=core_llm.create_llm(), memory=self.memory,
                mongo_repository=self.mongo_repository, run_id=run_id,
                sub_agents=self.sub_agents)
            self._workflow = RecordWorkflow(monitoring=monitoring, memory=self.memory,
                                            mongo_repository=self.mongo_repository,
                                            run_id=run_id)
            self._run_id = run_id
        return self._workflow

    async def observe(self, payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        run_id, seq = str(payload["run_id"]), int(payload["seq"])
        records = [TelemetryRecord.model_validate(r) for r in payload["records"]]
        async with self._lock:
            workflow = self.workflow(run_id)
            results, stopped = [], None
            for record in records:
                outcome = await process_record(workflow, self.mongo_repository, run_id,
                                               record, seq)
                results.append(outcome.as_dict())
                if outcome.stopped is not None:
                    stopped = outcome.stopped
                    break
        return {"results": results, "stopped": stopped}


class RemoteMonitoring:
    """The driver's client of the Monitoring service."""

    def __init__(self, client: A2AJsonClient):
        self.client = client

    async def observe(self, run_id: str, seq: int,
                      records: list[TelemetryRecord]) -> dict[str, Any]:
        return await self.client.call({
            "run_id": run_id, "seq": seq,
            "records": [r.model_dump(mode="json") for r in records]})
