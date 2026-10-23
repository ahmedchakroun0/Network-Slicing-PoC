# base_ric/agent.py
"""Shared base for the RIC agents (Near-RT and Non-RT).

The two RIC agents are siblings: neither inherits from the other. All the
common machinery lives here (prompt injection, governance middleware, audit
collection, typed result). A concrete agent only declares its identity:

    agent_name   - value of AgentInvocation.agent it accepts
    prompt_path  - its system prompt file
    tools        - tools given to the LLM (knowledge search + its action
                   tools). The agent picks the tool itself; every tool
                   listed in middleware PROTECTED_TOOLS is gated by the
                   Trust & Governance Agent.
"""

import logging
from abc import ABC
from pathlib import Path
from typing import Any, ClassVar

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt

from src.core.llm import create_llm
from src.memory.working_memory import RedisWorkingMemory
from src.schema.monitoring import AgentInvocation
from src.schema.near_rt_ric import (
    GovernanceResult,
    RANActionProposal,
    RANToolResult,
)
from src.schema.near_rt_ric import NearRTRICResult as RICResult  # shared result
from src.repositories.mongo_repository import MongoRepository

from middleware.ran_authorisation import (
    PROTECTED_TOOLS,
    RAN_AUDIT,
    RANAuthorizationMiddleware,
)

logger = logging.getLogger(__name__)


class RICState(AgentState):
    agent_invocation: dict[str, Any] | None
    working_memory: dict[str, Any] | None


class BaseRICAgent(ABC):
    """Common behaviour of the RIC agents. Subclasses only set the class
    attributes below."""

    agent_name: ClassVar[str]
    prompt_path: ClassVar[Path]
    tools: ClassVar[list[Any]]

    def __init__(
        self,
        agent=None,
        trust_governance_agent=None,
        memory: RedisWorkingMemory | None = None,
        mongo_repository: MongoRepository | None = None,
        run_id: str | None = None,
    ):
        self.agent = agent
        self.trust_governance_agent = trust_governance_agent
        self.memory = memory
        self.mongo_repository = mongo_repository
        self.run_id = run_id

    async def initialize(self) -> None:
        if self.agent is not None:
            return

        if self.trust_governance_agent is None:
            raise RuntimeError(
                f"{self.agent_name} requires a RAN Trust & Governance Agent."
            )

        if not self.prompt_path.is_file():
            raise FileNotFoundError(
                f"{self.agent_name} prompt not found: {self.prompt_path}"
            )

        self.agent = await self._build_agent()

    async def _build_agent(self):
        base_prompt = self.prompt_path.read_text(encoding="utf-8")
        agent_name = self.agent_name
        action_tool_names = ", ".join(
            f"`{t.name}`" for t in self.tools if t.name in PROTECTED_TOOLS
        )

        # The dynamic prompt returns the FULL system prompt (base prompt
        # file + per-invocation context), so the base prompt is never lost
        # whichever way the installed LangChain version applies it.
        @dynamic_prompt
        def inject_context(request: ModelRequest) -> str:
            state = request.state or {}

            invocation = state.get("agent_invocation") or {}
            working_memory = state.get("working_memory") or {}

            return f"""{base_prompt}

---
Current {agent_name} invocation:
{invocation}

Shared network working memory:
{working_memory}

Your action tools: {action_tool_names}.
Pass `target` (and `slice_id` only when the tool declares that parameter)
exactly as given in the invocation context.
Do not claim an action executed unless its tool result confirms it.
"""

        return create_agent(
            model=create_llm(),
            tools=self.tools,
            middleware=[
                inject_context,
                RANAuthorizationMiddleware(
                    trust_governance_agent=self.trust_governance_agent,
                ),
            ],
            state_schema=RICState,
        )

    @staticmethod
    def _build_typed_result(
        audit: list[dict[str, Any]],
        messages: list[Any],
    ) -> RICResult:
        """Convert middleware audit records into a typed result."""

        # Fail safely if a protected tool was called but no audit
        # record was collected.
        protected_calls = [
            call
            for message in messages
            for call in (getattr(message, "tool_calls", None) or [])
            if call.get("name") in PROTECTED_TOOLS
        ]

        if protected_calls and not audit:
            return RICResult(status="error")

        if not audit:
            # No protected call: the agent abstained (or forcing failed).
            return RICResult(status="no_action")

        # One audit entry per protected call. The typed result holds a single
        # proposal/governance/execution, so it shows the first executed
        # action (or the first entry if none executed). Every action is in
        # `middleware_audit`.
        def _entry_status(item: dict[str, Any]) -> str:
            governance_item = item.get("governance") or {}
            execution_item = item.get("execution") or {}
            if not governance_item.get("approved"):
                return "rejected"
            if execution_item.get("status") == "executed":
                return "completed"
            if execution_item.get("status") == "rejected":
                return "rejected"
            return "error"

        statuses = [_entry_status(item) for item in audit]
        if "completed" in statuses:
            overall_status = "completed"
            entry = audit[statuses.index("completed")]
        else:
            overall_status = "rejected" if "rejected" in statuses else "error"
            entry = audit[0]

        proposal_data = entry.get("proposal")
        governance_data = entry.get("governance")
        execution_data = entry.get("execution")

        proposal = (
            RANActionProposal.model_validate(proposal_data)
            if proposal_data is not None
            else None
        )

        governance = (
            GovernanceResult.model_validate(governance_data)
            if governance_data is not None
            else None
        )

        execution = (
            RANToolResult.model_validate(execution_data)
            if execution_data is not None
            else None
        )

        status = overall_status

        return RICResult(
            status=status,
            proposal=proposal,
            governance=governance,
            execution=execution,
        )

    def _log_outcome(self, typed_result: RICResult, messages: list[Any]) -> None:
        """Say in the logs why an invocation ended the way it did."""
        calls = [
            (call.get("name"), call.get("args"))
            for message in messages
            for call in (getattr(message, "tool_calls", None) or [])
        ]
        if typed_result.status == "no_action":
            last = messages[-1] if messages else None
            logger.warning(
                "%s: NO action tool was called. Tool calls: %s. "
                "Final model message: %r",
                self.agent_name,
                calls,
                getattr(last, "content", last),
            )
        else:
            logger.info(
                "%s: status=%s, tool calls: %s",
                self.agent_name,
                typed_result.status,
                calls,
            )

    async def invoke(self, invocation: AgentInvocation) -> RICResult:
        """Invoke this RIC agent for one Monitoring Agent instruction."""

        if invocation.agent != self.agent_name:
            raise ValueError(
                f"{type(self).__name__} cannot handle agent={invocation.agent!r}"
            )

        try:
            await self.initialize()

            memory = await self.memory.get() if self.memory is not None else None
            working_memory = (
                memory.model_dump(mode="json") if memory is not None else {}
            )

            context = invocation.context or {}
            record_id = context.get("record_id")
            run_id = context.get("run_id") or self.run_id

            if self.mongo_repository is not None and run_id and record_id:
                self.mongo_repository.start_agent_execution(
                    run_id=run_id,
                    record_id=record_id,
                    agent=self.agent_name,
                    invocation=invocation,
                )

            # Fresh audit collector for this invocation only.
            audit: list[dict[str, Any]] = []
            token = RAN_AUDIT.set(audit)
            result_payload: dict[str, Any]

            try:
                result = await self.agent.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "Handle this Monitoring Agent invocation. Decide which of your action "
                                    "tools address the KPI violations named in your objective and call them. "
                                    "The governance middleware validates each call.\n\n"
                                    f"{invocation.model_dump_json()}"
                                ),
                            }
                        ],
                        "agent_invocation": invocation.model_dump(mode="json"),
                        "working_memory": working_memory,
                    }
                )
            finally:
                RAN_AUDIT.reset(token)

            typed_result = self._build_typed_result(
                audit=audit,
                messages=result.get("messages", []),
            )

            self._log_outcome(typed_result, result.get("messages", []))

            result_payload = typed_result.model_dump(mode="json")

            if self.mongo_repository is not None and run_id and record_id:
                self.mongo_repository.finish_agent_execution(
                    run_id=run_id,
                    record_id=record_id,
                    agent=self.agent_name,
                    invocation=invocation,
                    result=result_payload,
                    middleware_audit=audit,
                )

            return typed_result

        except Exception:
            # Do not disguise an internal failure as successful execution,
            # but do not lose the cause either.
            logger.exception("%s invocation failed", self.agent_name)

            context = invocation.context or {}
            record_id = context.get("record_id")
            run_id = context.get("run_id") or self.run_id

            if self.mongo_repository is not None and run_id and record_id:
                self.mongo_repository.finish_agent_execution(
                    run_id=run_id,
                    record_id=record_id,
                    agent=self.agent_name,
                    invocation=invocation,
                    result={"status": "error"},
                    middleware_audit=[],
                )

            return RICResult(status="error")