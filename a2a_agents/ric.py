"""The RIC agents as A2A services, and the Monitoring agent's client of them.

Server side: `ric_handler` wraps BaseRICAgent.invoke unchanged (working
memory, governance middleware, MCP tools, audit). Client side: RemoteRICAgent
has the `invocation -> RICResult` signature of MonitoringAgent.sub_agents, so
the Monitoring agent invokes a remote RIC agent exactly like a local one.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from a2a_agents.client import A2ACallError, A2AJsonClient
from a2a_agents.protocol import Handler
from a2a_agents.scope import RunScope
from agents.base_ric import BaseRICAgent
from agents.near_rt_ric_agent import NearRTRICAgent
from agents.non_rt_ric_agent import NonRTRICAgent
from src.core.llm import PROVIDER_GUARD
from src.memory.working_memory import RedisWorkingMemory
from src.repositories.mongo_repository import MongoRepository
from src.schema.monitoring import AgentInvocation, AgentName
from src.schema.near_rt_ric import NearRTRICResult as RICResult

AGENT_CLASSES: dict[AgentName, type[BaseRICAgent]] = {
    "near_rt_ric": NearRTRICAgent,
    "non_rt_ric": NonRTRICAgent,
}


def build_ric_agent(role: AgentName, *, governance: Any, memory: RedisWorkingMemory | None,
                    mongo_repository: MongoRepository | None, action_tools: list[BaseTool],
                    run_id: str | None = None) -> BaseRICAgent:
    """One RIC agent, as the in-process runner and the RIC service build it."""
    return AGENT_CLASSES[role](
        trust_governance_agent=governance, memory=memory, mongo_repository=mongo_repository,
        run_id=run_id, action_tools=action_tools)


def ric_handler(agent: BaseRICAgent) -> Handler:
    """The A2A skill of a RIC agent: {"invocation", "run_id"} -> {"result", "usage", "stopped"}."""
    scope = RunScope()

    async def handle(payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        invocation = AgentInvocation.model_validate(payload["invocation"])
        if invocation.agent != agent.agent_name:
            raise ValueError(f"{agent.agent_name} does not handle invocations for "
                             f"{invocation.agent!r}")
        scope.enter(payload.get("run_id") or invocation.context.get("run_id"))
        try:
            result = await agent.invoke(invocation)
        except BaseException:
            scope.discard()
            raise
        return {"result": result.model_dump(mode="json"), **scope.report()}

    return handle


class RemoteRICAgent:
    """A RIC agent reached over A2A; a drop-in for BaseRICAgent.invoke."""

    def __init__(self, role: AgentName, client: A2AJsonClient):
        self.role = role
        self.client = client

    async def invoke(self, invocation: AgentInvocation) -> RICResult:
        """Raises A2ACallError when the agent is unreachable or its task failed:
        MonitoringAgent.invoke_agent records the failure, as for a local crash."""
        response = await self.client.call(
            {"invocation": invocation.model_dump(mode="json"),
             "run_id": invocation.context.get("run_id")},
            metadata={"caller": "monitoring-agent"})
        try:
            result = RICResult.model_validate(response["result"])
        except (KeyError, ValueError) as exc:
            raise A2ACallError(f"{self.role}: malformed result: {exc}") from exc
        # The remote agent's model calls count on this record, under its role
        # (in its own process they are not nested in the record workflow).
        PROVIDER_GUARD.merge(response.get("usage"), response.get("stopped"),
                             rename={"other": self.role})
        return result
