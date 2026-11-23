"""The RAN Trust & Governance service: the policy decision point.

The RIC agents' RANAuthorizationMiddleware (the enforcement point) calls it
before every protected tool call through RemoteGovernance, which has the same
`authorize(proposal)` method as RANTrustGovernanceAgent, so the middleware does
not change. It is a plain request/response service, not an A2A agent: nobody
chooses to call it; every protected call must.

    POST /authorize  {"proposal": RANActionProposal, "run_id"}
                  -> {"result": GovernanceResult, "usage", "stopped"}

Fail closed: RemoteGovernance turns an unreachable or failing service into a
rejection (stage "governance_unavailable"); the tool is never executed.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from a2a_agents.protocol import require_bearer
from a2a_agents.scope import RunScope
from agents.ran_trust_governance import RANTrustGovernanceAgent
from src.core.llm import PROVIDER_GUARD
from src.schema.near_rt_ric import GovernanceResult, RANActionProposal

logger = logging.getLogger(__name__)

UNAVAILABLE_STAGE = "governance_unavailable"


class AuthorizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal: RANActionProposal
    run_id: str | None = None


def run_id_of(proposal: RANActionProposal) -> str | None:
    """The run id the middleware put in the proposal's invocation context."""
    invocation = (proposal.invocation_context or {}).get("invocation") or {}
    return (invocation.get("context") or {}).get("run_id")


def create_governance_app(governance: RANTrustGovernanceAgent, *,
                          token: str | None = None) -> FastAPI:
    app = FastAPI(title="ran-trust-governance")
    scope = RunScope()

    @app.post("/authorize")
    async def authorize(request: AuthorizeRequest) -> dict[str, Any]:
        scope.enter(request.run_id)
        try:
            result = await governance.authorize(proposal=request.proposal)
        except BaseException:
            scope.discard()
            raise
        return {"result": result.model_dump(mode="json"), **scope.report()}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "agent": "ran-trust-governance"}

    require_bearer(app, token)
    return app


class RemoteGovernance:
    """Drop-in for RANTrustGovernanceAgent in RANAuthorizationMiddleware."""

    def __init__(self, base_url: str, *, token: str | None = None, timeout_s: float = 200.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers,
                                       timeout=timeout_s, transport=transport)

    async def authorize(self, proposal: RANActionProposal,
                        state: dict | None = None) -> GovernanceResult:
        body = {"proposal": proposal.model_dump(mode="json"), "run_id": run_id_of(proposal)}
        try:
            response = await self._http.post("/authorize", json=body)
            response.raise_for_status()
            data = response.json()
            result = GovernanceResult.model_validate(data["result"])
        except Exception as exc:
            logger.error("governance service unavailable: %s: %s", type(exc).__name__, exc)
            return GovernanceResult(
                approved=False, failed_stage=UNAVAILABLE_STAGE,
                reason=f"[{UNAVAILABLE_STAGE}] the governance service could not decide "
                       f"({type(exc).__name__}); the action is not executed.")
        # The governance LLM review ran in the service: count it on this record.
        PROVIDER_GUARD.merge(data.get("usage"), data.get("stopped"))
        return result

    async def aclose(self) -> None:
        await self._http.aclose()
