"""Run one agent service.

    poetry run python -m a2a_agents governance             # :8310
    poetry run python -m a2a_agents near_rt_ric            # :8301, needs LiteLLM + governance
    poetry run python -m a2a_agents non_rt_ric             # :8302, needs LiteLLM + governance
    poetry run python -m a2a_agents monitoring             # :8300, needs both RIC agents

Shared infrastructure, reached over the network: Redis (REDIS_URL), MongoDB
(MONGO_URI), LiteLLM (LITELLM_API_BASE: models, MCP tools). See settings.py
for the service URLs and tokens.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

import uvicorn
from fastapi import FastAPI

from a2a_agents import settings
from a2a_agents.cards import MONITORING_SKILL, ROLE_OF_SKILL, monitoring_card, ric_card
from a2a_agents.client import discover
from a2a_agents.governance import RemoteGovernance, create_governance_app
from a2a_agents.monitoring import MonitoringService
from a2a_agents.protocol import build_a2a_app
from a2a_agents.ric import RemoteRICAgent, build_ric_agent, ric_handler
from agents.ran_trust_governance import RANTrustGovernanceAgent
from ric_tools.client import role_tools
from src.memory.working_memory import RedisWorkingMemory
from src.repositories.mongo_repository import MongoRepository

logger = logging.getLogger("a2a_agents")


class _Late:
    """A handler whose implementation is built in the app's lifespan."""

    def __init__(self) -> None:
        self.impl: Any = None

    async def __call__(self, payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        if self.impl is None:
            raise RuntimeError("the service is still starting")
        return await self.impl(payload, metadata)


def governance_app() -> FastAPI:
    return create_governance_app(RANTrustGovernanceAgent(), token=settings.token("governance"))


def ric_app(role: str, url: str) -> FastAPI:
    late = _Late()

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        governance = RemoteGovernance(settings.governance_url(),
                                      token=settings.token("governance"),
                                      timeout_s=settings.governance_timeout_s())
        memory = RedisWorkingMemory.from_url()
        async with role_tools(role) as tools:      # MCP session through LiteLLM
            agent = build_ric_agent(role, governance=governance, memory=memory,
                                    mongo_repository=MongoRepository(), action_tools=tools)
            await agent.initialize()
            late.impl = ric_handler(agent)
            logger.info("%s ready with tools %s", role, [t.name for t in tools])
            try:
                yield
            finally:
                await governance.aclose()
                await memory.close()

    return build_a2a_app(ric_card(role, url), late, artifact_name="ric_result",
                         token=settings.token(role), lifespan=lifespan)


def monitoring_app(url: str) -> FastAPI:
    late = _Late()

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        gateway_key = settings.gateway_key()
        tokens = {skill: gateway_key or settings.token(role)
                  for skill, role in ROLE_OF_SKILL.items()}
        clients = await discover(settings.ric_card_urls(), tokens=tokens,
                                 timeout_s=settings.ric_timeout_s())
        missing = set(ROLE_OF_SKILL) - set(clients)
        if missing:
            raise RuntimeError(f"no agent card declares the skills {sorted(missing)}")
        sub_agents = {ROLE_OF_SKILL[skill]: RemoteRICAgent(ROLE_OF_SKILL[skill], client).invoke
                      for skill, client in clients.items() if skill in ROLE_OF_SKILL}
        memory = RedisWorkingMemory.from_url()
        late.impl = MonitoringService(sub_agents=sub_agents, memory=memory,
                                      mongo_repository=MongoRepository()).observe
        logger.info("monitoring ready; RIC agents: %s", sorted(sub_agents))
        try:
            yield
        finally:
            for client in clients.values():
                await client.aclose()
            await memory.close()

    return build_a2a_app(monitoring_card(url), late, artifact_name="observation",
                         token=settings.token("monitoring"), lifespan=lifespan)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("service", choices=settings.SERVICES)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--public-url", default=None,
                   help="URL advertised in the agent card (default http://<host>:<port>)")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    port = a.port or settings.DEFAULT_PORT[a.service]
    url = a.public_url or f"http://{a.host}:{port}"
    if a.service == "governance":
        app = governance_app()
    elif a.service == "monitoring":
        app = monitoring_app(url)
    else:
        app = ric_app(a.service, url)
    logger.info("%s on %s (skill %s)", a.service, url,
                MONITORING_SKILL if a.service == "monitoring" else a.service)
    uvicorn.run(app, host=a.host, port=port)


if __name__ == "__main__":
    main()
