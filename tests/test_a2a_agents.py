"""The agents as services: A2A between agents, HTTP to the governance service.

Every service is its real ASGI app, reached through httpx (one in-process
transport per host, no sockets), so the requests go through the A2A JSON-RPC
binding, the agent cards, the bearer tokens and the task lifecycle exactly as
over the network. The closed-loop test reuses tests/test_end_to_end.py's
stand-ins (scripted model, fakeredis, mongomock, in-process emulator and MCP
servers) and checks that the A2A deployment gives the in-process results.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager

import httpx
import mongomock
import pytest

import scripts.run_loop as run_loop
import src.repositories.mongo_repository as mongo_repo
from a2a_agents.cards import MONITORING_SKILL, SKILL_OF_ROLE, monitoring_card, ric_card
from a2a_agents.client import A2ACallError, A2AJsonClient, discover
from a2a_agents.governance import UNAVAILABLE_STAGE, RemoteGovernance, create_governance_app
from a2a_agents.monitoring import MonitoringService, RemoteMonitoring
from a2a_agents.protocol import build_a2a_app, json_part, read_json
from a2a_agents.ric import RemoteRICAgent, build_ric_agent, ric_handler
from agents.monitoring_agent import MonitoringAgent
from agents.ran_trust_governance import RANTrustGovernanceAgent
from ran_emulator.traces import TraceBank
from ric_tools.client import role_tools
from src.core.llm import PROVIDER_GUARD
from src.memory.working_memory import RedisWorkingMemory
from src.repositories.mongo_repository import MongoRepository
from src.schema.monitoring import AgentInvocation
from src.schema.near_rt_ric import NearRTRICResult, RANActionProposal
from tests.test_end_to_end import ScriptedChatModel, closed_loop, services  # noqa: F401

ROLES = ("near_rt_ric", "non_rt_ric")


class HostRouter(httpx.AsyncBaseTransport):
    """One ASGI app per host name: http://<host>/... reaches that app."""

    def __init__(self, apps: dict[str, object]):
        self.transports = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}

    def add(self, host: str, app: object) -> None:
        self.transports[host] = httpx.ASGITransport(app=app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self.transports.get(request.url.host)
        if transport is None:
            raise httpx.ConnectError(f"no service at {request.url.host}", request=request)
        return await transport.handle_async_request(request)


def _invocation(agent: str = "near_rt_ric", **context) -> AgentInvocation:
    return AgentInvocation(agent=agent, responsibility="r", objective="o", priority="high",
                           reason="x", context={"target": "bs2", "slice_id": "eMBB",
                                                "run_id": "r1", **context})


class _FakeRIC:
    """Stands in for a BaseRICAgent behind the A2A server."""

    agent_name = "near_rt_ric"

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.seen: list[AgentInvocation] = []

    async def invoke(self, invocation: AgentInvocation) -> NearRTRICResult:
        if self.fail:
            raise RuntimeError("boom")
        self.seen.append(invocation)
        # One model call in this (remote) process: not nested in a record workflow.
        PROVIDER_GUARD.usage.setdefault("other", {"calls": 0})["calls"] += 1
        return NearRTRICResult(status="no_action")


def _ric_router(agent, token: str | None = "secret") -> HostRouter:
    app = build_a2a_app(ric_card("near_rt_ric", "http://near"), ric_handler(agent),
                        artifact_name="ric_result", token=token)
    return HostRouter({"near": app})


# ------------------------------------------------------------------ payloads

def test_json_part_keeps_integers_exact():
    payload = {"persistence": {"throughput": 3}, "x": 1.5, "s": "a", "n": None}
    value = read_json([json_part(payload)])
    assert value == payload and isinstance(value["persistence"]["throughput"], int)


# ---------------------------------------------------------- cards, discovery

async def test_cards_declare_one_skill_and_discovery_maps_them():
    router = HostRouter({
        "near": build_a2a_app(ric_card("near_rt_ric", "http://near"), ric_handler(_FakeRIC()),
                              artifact_name="ric_result"),
        "mon": build_a2a_app(monitoring_card("http://mon"), ric_handler(_FakeRIC()),
                             artifact_name="observation")})
    found = await discover(["http://near", "http://mon"], transport=router)
    assert set(found) == {SKILL_OF_ROLE["near_rt_ric"], MONITORING_SKILL}
    with pytest.raises(ValueError, match="two agents"):
        await discover(["http://near", "http://near"], transport=router)
    with pytest.raises(A2ACallError, match="no agent card"):
        await discover(["http://nowhere"], transport=router)


# -------------------------------------------------------------- RIC over A2A

async def test_remote_ric_round_trip_and_usage_under_its_role():
    agent = _FakeRIC()
    client = A2AJsonClient("http://near", token="secret", transport=_ric_router(agent))
    PROVIDER_GUARD.reset()
    result = await RemoteRICAgent("near_rt_ric", client).invoke(_invocation(persistence={"x": 2}))
    assert result == NearRTRICResult(status="no_action")
    # The invocation arrived intact (types included).
    assert agent.seen[0].context["persistence"] == {"x": 2}
    # The remote call was reported back and counted under the agent's role.
    by_caller = PROVIDER_GUARD.take_usage()["by_caller"]
    assert list(by_caller) == ["near_rt_ric"] and by_caller["near_rt_ric"]["calls"] == 1


async def test_card_is_public_but_tasks_need_the_token():
    router = _ric_router(_FakeRIC())
    async with httpx.AsyncClient(transport=router) as http:
        assert (await http.get("http://near/.well-known/agent-card.json")).status_code == 200
        assert (await http.post("http://near/", json={})).status_code == 401
    with pytest.raises(A2ACallError):
        await RemoteRICAgent("near_rt_ric", A2AJsonClient(
            "http://near", token="wrong", transport=router)).invoke(_invocation())


async def test_wrong_role_and_agent_crash_fail_the_task():
    remote = RemoteRICAgent("near_rt_ric", A2AJsonClient(
        "http://near", token="secret", transport=_ric_router(_FakeRIC())))
    with pytest.raises(A2ACallError, match="TASK_STATE_FAILED.*does not handle"):
        await remote.invoke(_invocation(agent="non_rt_ric"))
    crashing = RemoteRICAgent("near_rt_ric", A2AJsonClient(
        "http://near", token="secret", transport=_ric_router(_FakeRIC(fail=True))))
    with pytest.raises(A2ACallError, match="RuntimeError: boom"):
        await crashing.invoke(_invocation())


async def test_monitoring_records_an_unreachable_ric_agent(services):  # noqa: F811
    repo = MongoRepository()
    repo.start_run(run_id="r1", config={}, summary={})
    # The workflow's ingest step opens the record document of a violation.
    repo.insert_record(run_id="r1", record_id="rec-1", seq=0, telemetry={}, sla={},
                       persistence={})
    remote = RemoteRICAgent("near_rt_ric", A2AJsonClient(
        "http://gone", transport=HostRouter({})))
    monitoring = MonitoringAgent(llm=ScriptedChatModel(), memory=None, mongo_repository=repo,
                                 run_id="r1", sub_agents={"near_rt_ric": remote.invoke})
    assert await monitoring.invoke_agent(_invocation(), "rec-1") is None
    doc = repo.records.find_one({"_id": "r1:rec-1"})
    assert doc["agents"]["near_rt_ric"]["status"] == "error"
    assert doc["agents"]["near_rt_ric"]["error"]["type"] == "A2ACallError"


# -------------------------------------------------------- governance service

def _proposal(**params) -> RANActionProposal:
    return RANActionProposal(
        tool_name="adjust_power_control", target="bs2",
        parameters={"power_offset_db": 3.0, **params},
        rationale="r", expected_effect="e", confidence=0.5,
        invocation_context={"invocation": {"context": {"run_id": "g1", "target": "bs2"}}})


async def test_remote_governance_gives_the_local_decision():
    local = RANTrustGovernanceAgent(use_llm=False)
    app = create_governance_app(local, token="g")
    remote = RemoteGovernance("http://gov", token="g", transport=httpx.ASGITransport(app=app))
    for proposal in (_proposal(), _proposal(power_offset_db=40.0)):
        assert await remote.authorize(proposal) == await local.authorize(proposal)


async def test_governance_unreachable_or_unauthorized_fails_closed():
    app = create_governance_app(RANTrustGovernanceAgent(use_llm=False), token="g")
    for remote in (
        RemoteGovernance("http://gov", transport=httpx.ASGITransport(app=app)),   # no token
        RemoteGovernance("http://gov", token="g", transport=HostRouter({})),      # down
    ):
        decision = await remote.authorize(_proposal())
        assert not decision.approved and decision.failed_stage == UNAVAILABLE_STAGE


# ------------------------------------------------------ closed loop over A2A

@pytest.fixture
def one_mongo(services, monkeypatch):  # noqa: F811
    """Every service's repository on the same database, like one MongoDB
    server (each mongomock client is otherwise its own in-memory store)."""
    shared = mongomock.MongoClient()
    monkeypatch.setattr(mongo_repo, "MongoClient", lambda *a, **k: shared)
    return services


@asynccontextmanager
async def a2a_deployment(mcp_clients):
    """Governance, both RIC agents and Monitoring as separate apps, linked
    only through their URLs, cards and tokens."""
    router = HostRouter({})
    router.add("governance", create_governance_app(RANTrustGovernanceAgent(), token="tg"))
    memory = RedisWorkingMemory.from_url()
    async with AsyncExitStack() as stack:
        for role in ROLES:
            # Each RIC service: its own governance client, MCP session, repository.
            governance = RemoteGovernance("http://governance", token="tg", transport=router)
            tools = await stack.enter_async_context(role_tools(role, client=mcp_clients[role]))
            agent = build_ric_agent(role, governance=governance, memory=memory,
                                    mongo_repository=MongoRepository(), action_tools=tools)
            router.add(role, build_a2a_app(ric_card(role, f"http://{role}"), ric_handler(agent),
                                           artifact_name="ric_result", token=f"t-{role}"))
        clients = await discover([f"http://{r}" for r in ROLES], transport=router,
                                 tokens={SKILL_OF_ROLE[r]: f"t-{r}" for r in ROLES})
        sub_agents = {role: RemoteRICAgent(role, clients[SKILL_OF_ROLE[role]]).invoke
                      for role in ROLES}
        service = MonitoringService(sub_agents=sub_agents, memory=memory,
                                    mongo_repository=MongoRepository())
        router.add("monitoring", build_a2a_app(monitoring_card("http://monitoring"),
                                               service.observe, artifact_name="observation",
                                               token="tm"))
        yield RemoteMonitoring(A2AJsonClient("http://monitoring", token="tm", transport=router))


async def test_closed_loop_over_a2a_matches_the_in_process_run(frame, one_mongo, tmp_path):
    # Same set-up as test_end_to_end.test_action_fixes_the_violation_and_is_verified_resolved.
    frame = frame.copy()
    embb = frame.slice_id == "eMBB"
    frame.loc[embb, "dl_throughput_per_ue_mbps"] = 0.12 * frame.loc[embb, "slice_prb"]
    bank = TraceBank.from_frame(frame)

    async with closed_loop(bank) as (emulator, mcp_clients), \
            a2a_deployment(mcp_clients) as monitoring:
        await run_loop.run_loop(
            run_loop.RunConfig(run_id="a2a", scenario=str(tmp_path / "e2e.json"), steps=3,
                               agents="a2a"),
            emulator=emulator, monitoring=monitoring)
        sim_actions = await emulator._call("GET", "/sim/actions")

    repo = one_mongo["repo"]
    run = repo.runs.find_one({"_id": "a2a"})
    assert run["status"] == "finished" and run["config"]["agents"] == "a2a"
    summary = run["summary"]
    assert summary["steps"] == 3 and summary["rows_selected"] == 9
    assert summary["rows_processed"] == 1 and summary["rows_without_violation"] == 8
    assert summary["action_outcomes"] == {"resolved": 1}
    # The approved call, decided by the governance service, changed the RAN.
    assert [(a["control"], a["requested_pct"]) for a in sim_actions] == [("prb", 72)]

    (doc,) = repo.records.find({"run_id": "a2a"})
    (action,) = doc["agents"]["near_rt_ric"]["actions"]
    assert action["governance"]["approved"] and action["execution"]["status"] == "executed"
    assert action["governance"]["llm_review"]["goal_aligned"] is True
    assert action["outcome"]["outcome"] == "resolved"
    # The RIC agent's model calls, made in its own service, count on the
    # record under its role. (The scripted model's structured outputs, used by
    # Monitoring and the governance review, fire no callbacks, in process too.)
    assert doc["llm"]["by_caller"]["near_rt_ric"]["calls"] >= 1


async def test_run_stops_when_the_monitoring_agent_is_down(bank, services, tmp_path):  # noqa: F811
    down = RemoteMonitoring(A2AJsonClient("http://monitoring", transport=HostRouter({})))
    async with closed_loop(bank) as (emulator, _):
        await run_loop.run_loop(
            run_loop.RunConfig(run_id="down", scenario=str(tmp_path / "e2e.json"), steps=3,
                               agents="a2a"),
            emulator=emulator, monitoring=down)
    run = services["repo"].runs.find_one({"_id": "down"})
    assert run["status"] == "stopped"
    assert "Monitoring agent is unavailable (step 0)" in run["stop_reason"]
