# middleware/ran_authorisation.py

import json
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any
from pydantic import ValidationError
import logging
import uuid
from typing import Literal

from langchain.agents.middleware import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, ValidationError, create_model

logger = logging.getLogger(__name__)
from src.schema.near_rt_ric import (
    GovernanceResult,
    RANActionProposal,
    RANToolResult,
)

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
NON_RT_POLICY_TOOLS = frozenset(
    [
        "update_slice_resource_policy",
        "update_traffic_steering_policy",
        "update_admission_control_policy",
        "optimize_slice_capacity_policy",
    ]
)
PROTECTED_TOOLS = NEAR_RT_TOOLS | NON_RT_POLICY_TOOLS

# Arguments every protected tool takes besides its own action parameters.
REASONING_FIELDS = ("rationale", "expected_effect", "evidence_sources", "confidence")


# Per-invocation audit collector.
#
# The RIC agent sets this to a fresh list before calling the LangChain agent
# and reads the same list afterwards. This avoids depending on in-place
# mutation of the agent state, which LangChain does not guarantee to persist.
RAN_AUDIT: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "ran_audit", default=None
)


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
    def _record(
        audit: list[dict[str, Any]],
        tool_call_id: str,
        proposal: RANActionProposal | None,
        decision: GovernanceResult,
        execution: RANToolResult,
    ) -> None:
        audit.append(
            {
                "tool_call_id": tool_call_id,
                "proposal": (
                    proposal.model_dump(mode="json") if proposal is not None else None
                ),
                "governance": decision.model_dump(mode="json"),
                "execution": execution.model_dump(mode="json"),
            }
        )

    async def awrap_tool_call(
        self,
        request,
        handler: Callable[..., Awaitable[Any]],
    ):
        tool_call = request.tool_call
        # The tool the agent actually chose (kept for the audit record).
        tool_name = tool_call["name"]

        # Read-only tools do not need action authorization.
        if tool_name not in self.protected_tools:
            return await handler(request)

        tool_call_id = tool_call["id"]
        declared = set(request.tool.args) if getattr(request, "tool", None) else None
        args = {k: v for k, v in (tool_call.get("args") or {}).items()
                if declared is None or k in declared}
        submitted, arguments = build_submission(tool_name, args)

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
            self._record(audit, tool_call_id, None, decision, execution)

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
            self._record(audit, tool_call_id, proposal, decision, execution)

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
            self._record(audit, tool_call_id, proposal, decision, execution)

            return tool_response

        except Exception as exc:
            execution = self._tool_result(
                tool_name,
                arguments,
                "error",
                message=f"{type(exc).__name__}: {exc}",
            )
            self._record(audit, tool_call_id, proposal, decision, execution)

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

        # Model ignored tool calling (or already searched): force the action
        # with schema-constrained output, which this stack supports.
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
            tool_calls=[{
                "name": name,
                "args": args,
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "tool_call",
            }],
        )
        return ModelResponse(result=[ai])