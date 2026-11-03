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

import json
import logging
from abc import ABC
from pathlib import Path
from typing import Any, ClassVar

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt

from src.core.llm import create_llm, message_text
from src.memory.working_memory import RedisWorkingMemory
from src.schema.monitoring import AgentInvocation
from src.schema.near_rt_ric import (
    GovernanceResult,
    RANActionProposal,
    RANToolResult,
)
from src.schema.near_rt_ric import NearRTRICResult as RICResult  # shared result
from src.repositories.mongo_repository import MongoRepository

from langchain_core.messages import AIMessage

from src.schema.mongo_run_document import ActionEntry, AgentMessage, KnowledgeEntry
from src.schema.working_memory import ActionEvent

from middleware.ran_authorisation import (
    PROTECTED_TOOLS,
    RAN_AUDIT,
    AuditSink,
    RANAuthorizationMiddleware,
)

logger = logging.getLogger(__name__)

# Rejection reasons from these stages are deterministic and safe to show the
# agents again; the LLM review's reason is not kept in working memory.
DETERMINISTIC_STAGES = frozenset(
    {"schema", "policy_trust", "sla", "resource_safety", "invocation_limit", "invalid_proposal"}
)
# The LLM review's verdicts are kept in working memory as booleans only.
REVIEW_VERDICTS = ("goal_aligned", "reasoning_sound", "behavior_acceptable")


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

            invocation = json.dumps(state.get("agent_invocation") or {}, separators=(",", ":"))
            working_memory = json.dumps(state.get("working_memory") or {}, separators=(",", ":"))

            return f"""{base_prompt}

---
Current {agent_name} invocation:
{invocation}

Working memory of this base station and slice (facts, most recent first:
cell configuration, last records of this slice and of the other slices,
last tool calls executed or rejected on them):
{working_memory}
Do not repeat a call that was just rejected with the same parameters, and
prefer a different action when the same violation persists after executed
actions.

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
    def _select_entry(audit: list[dict[str, Any]]) -> tuple[str, int]:
        """Overall status of a non-empty audit and the entry the typed result
        reports: the first executed action, or the first entry if none ran."""

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
            return "completed", statuses.index("completed")
        return ("rejected" if "rejected" in statuses else "error"), 0

    @staticmethod
    def _final_message(messages: list[Any]) -> AgentMessage | None:
        """The agent's last model turn (its closing answer after the tools)."""
        last = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last is None:
            return None
        return AgentMessage(text=message_text(last))

    @staticmethod
    def _action_events(
        actions: list[ActionEntry], context: dict[str, Any], agent: str
    ) -> list[ActionEvent]:
        """Working-memory facts of the calls (no LLM text)."""
        events = []
        for action in actions:
            stage = action.governance.get("failed_stage")
            reason = action.governance.get("reason") or ""
            review = action.governance.get("llm_review") or {}
            events.append(ActionEvent(
                rec=context.get("record_id", ""),
                t=context.get("window_start_s", 0.0),
                slice=context.get("slice_id", ""),
                agent=agent,
                tool=action.tool,
                params=action.params,
                approved=bool(action.governance.get("approved")),
                executed=action.execution.get("status") == "executed",
                stage=stage,
                why=reason[:160] if stage in DETERMINISTIC_STAGES else None,
                review={k: review[k] for k in REVIEW_VERDICTS if k in review} or None,
            ))
        return events

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

        overall_status, index = BaseRICAgent._select_entry(audit)
        entry = audit[index]

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

        context = invocation.context or {}
        record_id = context.get("record_id")
        run_id = context.get("run_id") or self.run_id
        cell, slice_id = context.get("target"), context.get("slice_id")
        store = self.mongo_repository if (self.mongo_repository is not None
                                          and run_id and record_id) else None

        def push_action(action: ActionEntry) -> None:
            store.push_action(run_id=run_id, record_id=record_id,
                              agent=self.agent_name, entry=action)

        def push_knowledge(entry: KnowledgeEntry) -> None:
            store.push_knowledge(run_id=run_id, record_id=record_id,
                                 agent=self.agent_name, entry=entry)

        # Fresh audit collector for this invocation only; each call and each
        # knowledge lookup is written to the results store as it happens.
        audit = AuditSink(
            on_action=push_action if store is not None else None,
            on_knowledge=push_knowledge if store is not None else None,
        )

        try:
            await self.initialize()

            working_memory = (
                (await self.memory.view(cell, slice_id)).compact()
                if self.memory is not None and cell and slice_id
                else {}
            )

            if store is not None:
                store.start_agent(run_id=run_id, record_id=record_id, agent=self.agent_name)

            token = RAN_AUDIT.set(audit)
            try:
                result = await self.agent.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                # The invocation is in the system prompt
                                # (sent once, not repeated here).
                                "content": (
                                    "Handle the Monitoring Agent invocation given in your "
                                    "instructions. Decide which of your action tools address "
                                    "the KPI violations named in its objective and call them. "
                                    "The governance middleware validates each call."
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
                audit=audit.entries,
                messages=result.get("messages", []),
            )
            self._log_outcome(typed_result, result.get("messages", []))

            if store is not None:
                selected = self._select_entry(audit.entries)[1] if audit.entries else None
                store.finish_agent(run_id=run_id, record_id=record_id, agent=self.agent_name,
                                   status=typed_result.status, selected_action=selected,
                                   final_message=self._final_message(result.get("messages", [])))
            return typed_result

        except Exception as exc:
            # Do not disguise an internal failure as successful execution,
            # but do not lose the cause either.
            logger.exception("%s invocation failed", self.agent_name)
            if store is not None:
                store.finish_agent(run_id=run_id, record_id=record_id, agent=self.agent_name,
                                   status="error",
                                   error={"type": type(exc).__name__, "message": str(exc)})
            return RICResult(status="error")

        finally:
            # Working memory: what was executed or rejected on this slice.
            if self.memory is not None and cell and audit.actions:
                try:
                    await self.memory.record_actions(
                        cell, self._action_events(audit.actions, context, self.agent_name))
                except Exception:
                    logger.exception("%s: could not update working memory", self.agent_name)