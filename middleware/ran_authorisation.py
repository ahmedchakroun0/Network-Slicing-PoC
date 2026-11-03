# middleware/ran_authorisation.py

import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field, ValidationError, create_model

from src.core.llm import message_text
from src.schema.mongo_run_document import ActionEntry, AgentMessage, KnowledgeEntry
from src.schema.near_rt_ric import (
    GovernanceResult,
    RANActionProposal,
    RANToolResult,
)

logger = logging.getLogger(__name__)

# Invocation-level limits (checked before governance, fail closed):
# a bounded number of protected calls per invocation (stops retry loops),
# at most ONE executed call per tool per invocation (no duplicate or
# conflicting policies for the same record), and no resubmission of the same
# parameters after an LLM review rejection.
MAX_PROTECTED_CALLS = int(os.getenv("MAX_PROTECTED_CALLS_PER_INVOCATION", "20"))

# The RIC agents choose and call the action tools themselves. Each of these
# tools is protected: the call is turned into a RANActionProposal, validated
# by the Trust & Governance Agent here, and only then executed.
NEAR_RT_TOOLS = frozenset(
    [
        "adjust_prb_allocation",
        "optimize_scheduler",
        "optimize_handover",
        "adjust_power_control",
        "adjust_qos_parameters",
        "adjust_link_adaptation",
    ]
)
NON_RT_POLICY_TOOLS = frozenset(["update_slice_resource_policy"])
PROTECTED_TOOLS = NEAR_RT_TOOLS | NON_RT_POLICY_TOOLS

# The network resource each protected tool controls. At most one call per
# resource is executed per invocation, so two tools acting on the same
# resource cannot apply duplicate or conflicting changes for one record.
RESOURCE_OF = {
    "adjust_prb_allocation": "prb_allocation",
    "update_slice_resource_policy": "prb_policy",
    "optimize_scheduler": "scheduler",
    "optimize_handover": "handover",
    "adjust_power_control": "power",
    "adjust_qos_parameters": "qos",
    "adjust_link_adaptation": "link_adaptation",
}

# Arguments every protected tool takes besides its own action parameters.
REASONING_FIELDS = ("rationale", "expected_effect", "evidence_sources", "confidence")

# Set on the AIMessage of a tool call the middleware produced itself.
FORCED_FLAG = "forced_by_middleware"


class AuditSink:
    """Per-invocation audit collector.

    `entries` keeps the full entries (proposal, governance, execution) in
    memory for the RIC agent's typed result. Each call is also turned into a
    compact ActionEntry (no invocation context, no duplicated parameters) and
    handed to `on_action`, which the RIC agent wires to the results store, so
    the trail is written as the calls happen. Knowledge lookups are kept the
    same way (`on_knowledge`), so the evidence an agent cites can be checked.
    """

    def __init__(
        self,
        on_action: Callable[[ActionEntry], None] | None = None,
        on_knowledge: Callable[[KnowledgeEntry], None] | None = None,
    ) -> None:
        self.entries: list[dict[str, Any]] = []
        self.actions: list[ActionEntry] = []
        self.knowledge: list[KnowledgeEntry] = []
        self._on_action = on_action
        self._on_knowledge = on_knowledge

    @staticmethod
    def _emit(callback: Callable[[Any], None] | None, item: Any, tool_call_id: str) -> None:
        if callback is None:
            return
        try:
            callback(item)
        except Exception:
            # The results store must never decide whether an action runs.
            logger.exception("Could not persist audit item %s", tool_call_id)

    def add_knowledge(self, tool_call_id: str, tool: str, args: dict[str, Any], result: Any) -> None:
        entry = KnowledgeEntry(tool_call_id=tool_call_id, at=datetime.now(timezone.utc),
                               tool=tool, args=args, result=str(result))
        self.knowledge.append(entry)
        self._emit(self._on_knowledge, entry, tool_call_id)

    def __len__(self) -> int:
        return len(self.entries)

    def add(
        self,
        tool_call_id: str,
        submitted: dict[str, Any],
        proposal: RANActionProposal | None,
        decision: GovernanceResult,
        execution: RANToolResult,
        agent_message: AgentMessage | None = None,
    ) -> None:
        self.entries.append(
            {
                "tool_call_id": tool_call_id,
                "proposal": (
                    proposal.model_dump(mode="json") if proposal is not None else None
                ),
                "governance": decision.model_dump(mode="json"),
                "execution": execution.model_dump(mode="json"),
            }
        )
        action = ActionEntry(
            i=len(self.actions),
            tool_call_id=tool_call_id,
            at=datetime.now(timezone.utc),
            tool=execution.tool_name,
            target=execution.target,
            params={k: v for k, v in execution.parameters.items() if k != "target"},
            reasoning={k: submitted[k] for k in REASONING_FIELDS if k in submitted} or None,
            agent_message=agent_message,
            governance=decision.model_dump(mode="json"),
            execution={"status": execution.status, "result": execution.result,
                       "message": execution.message},
        )
        self.actions.append(action)
        self._emit(self._on_action, action, tool_call_id)


# The RIC agent sets this to a fresh AuditSink before calling the LangChain
# agent and reads the same sink afterwards. This avoids depending on in-place
# mutation of the agent state, which LangChain does not guarantee to persist.
RAN_AUDIT: ContextVar[AuditSink | None] = ContextVar("ran_audit", default=None)


def build_submission(tool_name, args):
    remaining = dict(args or {})
    target = remaining.pop("target", None)
    reasoning = {
        key: remaining.pop(key)
        for key in REASONING_FIELDS
        if key in remaining and remaining[key] is not None
    }
    for key in REASONING_FIELDS:
        remaining.pop(key, None)

    submitted = {
        "tool_name": tool_name,
        "target": target,
        "parameters": remaining,
        **reasoning,
    }
    return submitted, {**remaining, "target": target}



class RANAuthorizationMiddleware(AgentMiddleware):
    """Authorize and audit protected RAN actions before execution.

    An agent may call several protected tools in one invocation. Each call
    is reviewed on its own and gets its own audit entry.
    """

    def __init__(
        self,
        trust_governance_agent,
        protected_tools: frozenset[str] | set[str] = PROTECTED_TOOLS,
    ):
        self.trust_governance_agent = trust_governance_agent
        self.protected_tools = frozenset(protected_tools)

    @staticmethod
    def _tool_result(
        tool_name: str,
        arguments: dict[str, Any],
        status: str,
        result: Any = None,
        message: str | None = None,
    ) -> RANToolResult:
        target = arguments.get("target")

        if status not in {"executed", "rejected", "error"}:
            status = "error"

        if isinstance(result, dict):
            result_data = result
        elif result is None:
            result_data = {}
        else:
            result_data = {"content": str(result)}

        return RANToolResult(
            tool_name=tool_name,
            status=status,
            target=target,
            parameters=arguments,
            result=result_data,
            message=message,
        )

    @staticmethod
    def _calling_message(request, tool_call_id: str) -> AgentMessage | None:
        """The model turn that emitted this tool call: its visible text and
        whether the middleware forced it."""
        for message in reversed((request.state or {}).get("messages") or []):
            if not isinstance(message, AIMessage):
                continue
            if any(call.get("id") == tool_call_id for call in message.tool_calls or []):
                return AgentMessage(
                    text=message_text(message),
                    forced=bool((message.additional_kwargs or {}).get(FORCED_FLAG)),
                )
        return None

    async def awrap_tool_call(
        self,
        request,
        handler: Callable[..., Awaitable[Any]],
    ):
        tool_call = request.tool_call
        # The tool the agent actually chose (kept for the audit record).
        tool_name = tool_call["name"]

        # Read-only tools do not need action authorization; their results are
        # kept so the evidence the agent cites can be checked.
        if tool_name not in self.protected_tools:
            response = await handler(request)
            audit = RAN_AUDIT.get()
            if audit is not None:
                content = response.content if isinstance(response, ToolMessage) else response
                audit.add_knowledge(tool_call.get("id", ""), tool_name,
                                    tool_call.get("args") or {}, content)
            return response

        # One traced span per protected call, whatever its outcome: a rejected
        # tool never runs, so it would otherwise leave no trace of its own.
        async def gate(_tool_call: dict[str, Any]) -> Any:
            return await self._gate(request, handler)

        return await RunnableLambda(gate).ainvoke(
            tool_call,
            config={"run_name": f"governance {tool_name}", "tags": ["governance"]},
        )

    async def _gate(
        self,
        request,
        handler: Callable[..., Awaitable[Any]],
    ):
        """Authorize one protected tool call and execute it only if approved."""
        tool_call = request.tool_call
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]
        # Every argument the model sent is submitted: one the tool does not
        # declare is rejected by governance (schema stage), never dropped.
        submitted, arguments = build_submission(tool_name, tool_call.get("args") or {})
        agent_message = self._calling_message(request, tool_call_id)

        # Fail closed: without an audit collector the action could not be
        # traced, so it is not executed.
        audit = RAN_AUDIT.get()
        if audit is None:
            return ToolMessage(
                content=json.dumps(
                    {
                        "status": "error",
                        "tool_name": tool_name,
                        "message": (
                            "Audit collector not configured; protected action blocked."
                        ),
                    }
                ),
                tool_call_id=tool_call_id,
            )

        state = request.state
        invocation = state.get("agent_invocation") or {}

        limit_reason = None
        if len(audit) >= MAX_PROTECTED_CALLS:
            limit_reason = (
                f"[invocation_limit] at most {MAX_PROTECTED_CALLS} protected calls "
                "per invocation; stop and report."
            )
        elif any(
            RESOURCE_OF.get((entry.get("execution") or {}).get("tool_name"))
            == RESOURCE_OF.get(tool_name, tool_name)
            and (entry.get("execution") or {}).get("status") == "executed"
            for entry in audit.entries
        ):
            limit_reason = (
                f"[invocation_limit] the {RESOURCE_OF.get(tool_name, tool_name)} of this "
                "slice was already changed in this invocation; a second (duplicate or "
                "conflicting) change is not allowed."
            )
        elif any(
            (entry.get("governance") or {}).get("failed_stage") == "llm_review"
            and (entry.get("proposal") or {}).get("tool_name") == tool_name
            and (entry.get("proposal") or {}).get("parameters") == submitted["parameters"]
            for entry in audit.entries
        ):
            # The review judges the reasoning: rewording it must not be enough
            # to get the same action through.
            limit_reason = (
                f"[invocation_limit] {tool_name} with these parameters was already "
                "rejected by the review in this invocation; rewording the rationale "
                "does not change the action. Change the parameters or use another tool."
            )
        if limit_reason is not None:
            decision = GovernanceResult(
                approved=False, reason=limit_reason, failed_stage="invocation_limit"
            )
            execution = self._tool_result(
                tool_name, arguments, "rejected", message=limit_reason
            )
            audit.add(tool_call_id, submitted, None, decision, execution, agent_message)
            return ToolMessage(
                content=json.dumps(
                    {"status": "rejected", "tool_name": tool_name, "reason": limit_reason}
                ),
                tool_call_id=tool_call_id,
            )

        # A malformed proposal never reaches governance or the tool. It is
        # still written to the audit (so the reason is visible in Mongo and
        # the invocation is not reported as a bare "error"), and the RIC
        # agent is told what to fix so it can call the tool again.
        try:
            submitted_proposal = RANActionProposal.model_validate(submitted)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in exc.errors()
            )
            reason = f"[invalid_proposal] {details}"
            decision = GovernanceResult(
                approved=False,
                reason=reason,
                failed_stage="invalid_proposal",
            )
            execution = self._tool_result(
                tool_name, arguments, "rejected", message=reason
            )
            audit.add(tool_call_id, submitted, None, decision, execution, agent_message)

            return ToolMessage(
                content=json.dumps(
                    {
                        "status": "invalid_proposal",
                        "message": (
                            f"Proposal does not match the required format: {details}. "
                            "Fix these arguments and call the tool again."
                        ),
                    }
                ),
                tool_call_id=tool_call_id,
            )

        target = arguments.get("target")
        proposal: RANActionProposal | None = None

        try:
            if not isinstance(target, str) or not target.strip():
                raise ValueError("A non-empty action target is required.")

            if self.trust_governance_agent is None:
                raise RuntimeError("Trust & Governance Agent is not configured.")

            # The proposal is exactly what the RIC agent submitted, except
            # `invocation_context`, which is supplied by the system and never
            # trusted from the RIC agent. Governance gets the proposal and
            # that context only: no agent state, messages or raw telemetry.
            proposal = submitted_proposal.model_copy(
                update={"invocation_context": {"invocation": invocation}}
            )

            decision = await self.trust_governance_agent.authorize(
                proposal=proposal,
            )
            logger.info(
                "Governance decision: tool=%s approved=%s failed_stage=%s reason=%s checks_passed=%s",
                proposal.tool_name,
                decision.approved,
                decision.failed_stage,
                decision.reason,
                decision.checks_passed,
            )

            if not isinstance(decision, GovernanceResult):
                decision = GovernanceResult.model_validate(decision)

        except Exception as exc:
            decision = GovernanceResult(
                approved=False,
                reason=(f"Authorization failed: {type(exc).__name__}: {exc}"),
                failed_stage="authorization_error",
            )

        if not decision.approved:
            execution = self._tool_result(
                tool_name,
                arguments,
                "rejected",
                message=decision.reason,
            )
            audit.add(tool_call_id, submitted, proposal, decision, execution, agent_message)

            return ToolMessage(
                content=json.dumps(
                    {
                        "status": "rejected",
                        "tool_name": tool_name,
                        "reason": decision.reason,
                    }
                ),
                tool_call_id=tool_call_id,
            )

        # The protected tool is called only after approval.
        try:
            tool_response = await handler(request)

            raw_result = (
                tool_response.content
                if isinstance(tool_response, ToolMessage)
                else tool_response
            )

            if isinstance(raw_result, str):
                try:
                    raw_result = json.loads(raw_result)
                except (json.JSONDecodeError, TypeError):
                    pass

            execution = self._tool_result(
                tool_name,
                arguments,
                "executed",
                result=raw_result,
            )
            audit.add(tool_call_id, submitted, proposal, decision, execution, agent_message)

            return tool_response

        except Exception as exc:
            execution = self._tool_result(
                tool_name,
                arguments,
                "error",
                message=f"{type(exc).__name__}: {exc}",
            )
            audit.add(tool_call_id, submitted, proposal, decision, execution, agent_message)

            return ToolMessage(
                content=json.dumps(
                    {
                        "status": "error",
                        "tool_name": tool_name,
                        "message": str(exc),
                    }
                ),
                tool_call_id=tool_call_id,
            )
    @staticmethod
    def _ai_message(response) -> AIMessage | None:
        if isinstance(response, AIMessage):
            return response
        for m in getattr(response, "result", None) or []:
            if isinstance(m, AIMessage):
                return m
        return None

    async def awrap_model_call(self, request, handler):
        """Guarantee that a protected action tool call is produced."""
        called = [
            call.get("name")
            for message in request.messages
            for call in (getattr(message, "tool_calls", None) or [])
        ]
        if any(name in self.protected_tools for name in called):
            return await handler(request)

        # First step: let the model act freely (knowledge search or action).
        if not called:
            response = await handler(request.override(tool_choice="any"))
            ai = self._ai_message(response)
            if ai is not None and ai.tool_calls:
                return response
            logger.warning("Model returned no structured tool call; forcing action.")

        # No action yet (the model searched, or answered without a tool
        # call): one call that offers only the action tools and requires a
        # tool call. The model still chooses the tool and its arguments.
        actions = [t for t in request.tools if getattr(t, "name", None) in self.protected_tools]
        response = await handler(request.override(tools=actions, tool_choice="any"))
        ai = self._ai_message(response)
        if ai is not None and any(c.get("name") in self.protected_tools for c in ai.tool_calls):
            ai.additional_kwargs[FORCED_FLAG] = True
            return response
        logger.warning("No action tool call even when required; forcing it with structured output.")

        # Last resort for models that ignore tool calling: choose the tool and
        # fill its arguments with schema-constrained output (two calls).
        try:
            return await self._force_action(request)
        except Exception:
            logger.exception("Forced action failed")
            return await handler(request)

    async def _force_action(self, request) -> ModelResponse:
        tools = {
            t.name: t for t in request.tools
            if getattr(t, "name", None) in self.protected_tools
        }
        if not tools:
            raise RuntimeError("No protected tools available to force.")

        knowledge = "\n\n".join(
            str(m.content) for m in request.messages if isinstance(m, ToolMessage)
        )
        catalogue = "\n".join(f"- {n}: {t.description}" for n, t in tools.items())
        base = [
            SystemMessage(request.system_prompt or ""),
            HumanMessage(
                f"Knowledge retrieved so far:\n{knowledge or 'none'}\n\n"
                f"Available action tools:\n{catalogue}"
            ),
        ]
        model = request.model

        # 1) choose the tool
        Choice = create_model(
            "ToolChoice",
            tool_name=(Literal[tuple(tools)], Field(description="Action tool to use.")),  # type: ignore[valid-type]
        )
        choice = await model.with_structured_output(Choice).ainvoke(
            base + [HumanMessage(
                "Choose the single action tool that best addresses the violated "
                "KPIs named in your objective."
            )]
        )
        name = choice["tool_name"] if isinstance(choice, dict) else choice.tool_name
        tool = tools[name]

        # 2) fill its arguments using the tool's own schema
        filled = await model.with_structured_output(tool.args_schema).ainvoke(
            base + [HumanMessage(
                f"You chose `{name}`. Fill in its arguments.\n"
                f"Tool description:\n{tool.description}\n\n"
                "Copy target (and slice_id if present) exactly from the invocation "
                "context. Give a specific rationale citing the violated KPI values, "
                "a concrete expected_effect, and confidence between 0 and 1."
            )]
        )
        args = (
            filled.model_dump(exclude_none=True)
            if isinstance(filled, BaseModel)
            else {k: v for k, v in dict(filled).items() if v is not None}
        )

        ai = AIMessage(
            content="",
            additional_kwargs={FORCED_FLAG: True},
            tool_calls=[{
                "name": name,
                "args": args,
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "tool_call",
            }],
        )
        return ModelResponse(result=[ai])