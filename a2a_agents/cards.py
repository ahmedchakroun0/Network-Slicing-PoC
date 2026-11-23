"""Agent Cards: one skill per agent; callers find agents by skill id."""

from __future__ import annotations

from a2a.types import AgentCard, AgentSkill

from a2a_agents.protocol import agent_card
from src.schema.monitoring import AgentName

MONITORING_SKILL = "ran.monitoring.observe"
SKILL_OF_ROLE: dict[AgentName, str] = {
    "near_rt_ric": "ran.nearrt.mitigate",
    "non_rt_ric": "ran.nonrt.propose_policy",
}
ROLE_OF_SKILL: dict[str, AgentName] = {skill: role for role, skill in SKILL_OF_ROLE.items()}

_RIC = {
    "near_rt_ric": (
        "near-rt-ric-agent",
        "Near-RT RIC agent: fast, local RAN mitigation of one cell and slice (E2-style controls).",
        AgentSkill(
            id=SKILL_OF_ROLE["near_rt_ric"], name="Mitigate a RAN incident",
            description=("Input: {'invocation': AgentInvocation, 'run_id'}. Chooses and executes, "
                         "after governance approval, PRB, scheduler, handover, power, QoS or "
                         "link-adaptation controls. Output: {'result': RICResult, 'usage', "
                         "'stopped'}."),
            tags=["ran", "near-rt", "slicing", "e2"]),
    ),
    "non_rt_ric": (
        "non-rt-ric-agent",
        "Non-RT RIC agent: policy-level answer to persistent degradation (A1-style policies).",
        AgentSkill(
            id=SKILL_OF_ROLE["non_rt_ric"], name="Propose a slice resource policy",
            description=("Input: {'invocation': AgentInvocation, 'run_id'}. Sets, after governance "
                         "approval, the slice's PRB policy. Output: {'result': RICResult, "
                         "'usage', 'stopped'}."),
            tags=["ran", "non-rt", "slicing", "a1"]),
    ),
}


def ric_card(role: AgentName, url: str) -> AgentCard:
    name, description, skill = _RIC[role]
    return agent_card(name=name, description=description, url=url, skill=skill)


def monitoring_card(url: str) -> AgentCard:
    return agent_card(
        name="monitoring-agent",
        description=("Monitoring agent: analyses slice telemetry against the SLAs, decides which "
                     "RIC agent must act and invokes it over A2A."),
        url=url,
        skill=AgentSkill(
            id=MONITORING_SKILL, name="Observe a telemetry step",
            description=("Input: {'run_id', 'seq', 'records': [TelemetryRecord]}. Runs each record "
                         "through the record workflow. Output: {'results': [{record_id, status, "
                         "agents}], 'stopped'}."),
            tags=["ran", "monitoring", "sla"]),
    )
