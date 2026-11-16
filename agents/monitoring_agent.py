import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from agents.ran_trust_governance import NON_RT_MIN_PERSISTENCE
from src.memory.working_memory import RedisWorkingMemory
from src.repositories.mongo_repository import MongoRepository
from src.schema.monitoring import (
    AgentInvocation,
    AgentName,
    KPIStatus,
    MonitoringDecision,
    SLAAnalysis,
    TelemetryRecord,
)
from src.schema.sla import SLA_PROFILES, KPIBudget
from src.schema.working_memory import RecordEvent

logger = logging.getLogger(__name__)

SubAgent = Callable[[AgentInvocation], Awaitable[Any]]
# Not stored with the decision: the SLA analysis is on the record, and the
# agent context is injected by trusted code.
_STORED_EXCLUDE = {"sla_analysis": True, "record_id": True,
                   "selected_agents": {"__all__": {"context"}}}
_PROMPT_PATH = Path("prompts/monitoring.txt")


class MonitoringAgent:
    """Central orchestrator for telemetry analysis and agent invocation."""

    def __init__(
        self,
        llm: Any,
        memory: RedisWorkingMemory,
        sub_agents: dict[AgentName, SubAgent] | None = None,
        mongo_repository: MongoRepository | None = None,
        run_id: str | None = None,
    ) -> None:
        self.memory = memory
        self.sub_agents = sub_agents or {}
        self.mongo_repository = mongo_repository
        self.run_id = run_id
        if not _PROMPT_PATH.is_file():
            raise FileNotFoundError(f"Monitoring prompt not found: {_PROMPT_PATH}")

        system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        self.decision_llm = llm.with_structured_output(MonitoringDecision)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt,
                ),
                (
                    "human",
                    """
Incoming slice telemetry (one cell, one slice, one time window):
{telemetry}

Deterministic SLA analysis against the {slice_id} SLA profile:
{sla_analysis}

Working memory of this base station and slice (deterministic facts, most
recent first): the base-station configuration, the last records of this
slice and of the cell's other slices, and the last tool calls executed or
rejected on them:
{working_memory}

Consecutive-violation persistence for this cell and slice (per KPI,
computed deterministically; "not provided" if unknown):
{persistence}

Return a MonitoringDecision.
""",
                ),
            ]
        )

    @staticmethod
    def _compare(measured: float | None, term: KPIBudget) -> KPIStatus:
        """Compare a measured KPI with its SLA term."""
        status = KPIStatus(measured=measured, budget=term.budget, unit=term.unit,
                           bound=term.bound, status="unavailable")
        if measured is None:
            return status
        violation = measured < term.budget if term.bound == "min" else measured > term.budget
        status.status = "violation" if violation else "ok"
        return status

    @classmethod
    def analyze_sla(cls, record: TelemetryRecord) -> SLAAnalysis:
        """Evaluate the record against its slice's SLA profile deterministically."""
        profile = SLA_PROFILES.get(record.slice_id, {})
        kpis = {
            name: cls._compare(getattr(record, term.field), term)
            for name, term in profile.items()
        }
        return SLAAnalysis(
            slice_id=record.slice_id,
            kpis=kpis,
            violations=[n for n, s in kpis.items() if s.status == "violation"],
            unavailable_metrics=[n for n, s in kpis.items() if s.status == "unavailable"],
        )

    @staticmethod
    def _severity(analysis: SLAAnalysis) -> str:
        """How far the worst violated KPI is beyond its budget (x budget)."""
        worst = 0.0
        for name in analysis.violations:
            s = analysis.kpis[name]
            if not s.budget:
                continue
            if s.bound == "min":
                ratio = s.budget / s.measured if s.measured else float("inf")
            else:
                ratio = s.measured / s.budget
            worst = max(worst, ratio)
        return "high" if worst >= 3 else "medium" if worst >= 1.5 else "low"

    @staticmethod
    def _stored_decision(
        decision: MonitoringDecision, llm_output: dict[str, Any]
    ) -> dict[str, Any]:
        """The effective decision, and the model's own decision only when the
        deterministic override changed it."""
        stored = decision.model_dump(mode="json", exclude=_STORED_EXCLUDE)
        # orchestration_notes is set by code, never by the LLM: not compared.
        llm_output = {k: v for k, v in llm_output.items() if k != "orchestration_notes"}
        effective = {k: v for k, v in stored.items() if k != "orchestration_notes"}
        stored["overridden"] = effective != llm_output
        if stored["overridden"]:
            stored["llm_output"] = llm_output
        return stored

    @staticmethod
    def _apply_routing_rules(
        decision: MonitoringDecision,
        sla_analysis: SLAAnalysis,
        persistence: dict[str, int] | None,
    ) -> None:
        """Deterministic routing for a record with violations.

        1. The Non-RT RIC (policy level) is invoked only for a persistent
           violation: at least NON_RT_MIN_PERSISTENCE consecutive records.
        2. A violation is never left unhandled: with no agent left, the
           Near-RT RIC is invoked for the violated KPIs.
        Each change is noted on the decision (the LLM's own selection stays
        in the stored llm_output).
        """
        notes = decision.orchestration_notes
        observed = max(
            ((persistence or {}).get(kpi, 0) for kpi in sla_analysis.violations), default=0
        )
        if NON_RT_MIN_PERSISTENCE > 1 and observed < NON_RT_MIN_PERSISTENCE:
            if any(i.agent == "non_rt_ric" for i in decision.selected_agents):
                decision.selected_agents = [
                    i for i in decision.selected_agents if i.agent != "non_rt_ric"
                ]
                notes.append(
                    f"non_rt_ric not invoked: the violation persisted {observed} "
                    f"consecutive record(s), policy actions need {NON_RT_MIN_PERSISTENCE}."
                )

        if not decision.selected_agents:
            violated = ", ".join(
                f"{kpi} ({sla_analysis.kpis[kpi].measured} {sla_analysis.kpis[kpi].unit} vs "
                f"{sla_analysis.kpis[kpi].bound} {sla_analysis.kpis[kpi].budget})"
                for kpi in sla_analysis.violations
            )
            decision.selected_agents = [AgentInvocation(
                agent="near_rt_ric",
                responsibility="Mitigate the SLA violations of this slice.",
                objective=f"Restore the violated KPIs: {violated}.",
                priority={"high": "high", "medium": "medium"}.get(decision.severity, "low"),
                reason="Fallback: the record has SLA violations but no agent was selected.",
            )]
            notes.append("near_rt_ric invoked by fallback: SLA violations with no agent selected.")

    @staticmethod
    def _merge_duplicate_agents(selected: list) -> list:
        """One invocation per agent: the audit store keys executions by agent
        name, so a second invocation of the same agent would overwrite the
        first one's audit trail. Duplicates are merged (objectives joined)."""
        rank = {"low": 0, "medium": 1, "high": 2}
        merged: dict[str, Any] = {}
        for inv in selected:
            first = merged.get(inv.agent)
            if first is None:
                merged[inv.agent] = inv
                continue
            first.responsibility = f"{first.responsibility}, {inv.responsibility}"
            first.objective = f"{first.objective}; {inv.objective}"
            if rank.get(inv.priority, 0) > rank.get(first.priority, 0):
                first.priority = inv.priority
            first.reason = f"{first.reason} {inv.reason}".strip()
        return list(merged.values())

    async def decide(
        self,
        record: TelemetryRecord,
        persistence: dict[str, int] | None = None,
        sla_analysis: SLAAnalysis | None = None,
    ) -> MonitoringDecision:
        """Decide which agents act on one record (no agent is invoked here).

        The LLM decision is overridden by the deterministic SLA analysis and
        routing rules, then stored (MongoDB) and remembered (Redis)."""
        record_id = record.record_id
        store = self.mongo_repository if self.run_id is not None else None

        # 1. Calculate SLA violations without relying on the LLM.
        if sla_analysis is None:
            sla_analysis = self.analyze_sla(record)

        # 2. Read the working memory of this base station and slice.
        working_memory = (
            (await self.memory.view(record.cell_id, record.slice_id)).compact(persistence=False)
            if self.memory is not None
            else {}
        )

        # 3. Ask the LLM for a structured orchestration decision.
        prompt_value = self.prompt.invoke(
            {
                "telemetry": record.model_dump(mode="json"),
                "slice_id": record.slice_id,
                "sla_analysis": sla_analysis.model_dump(mode="json"),
                "working_memory": json.dumps(working_memory, separators=(",", ":")),
                "persistence": persistence if persistence is not None else "not provided",
            }
        )

        decision = await self.decision_llm.ainvoke(prompt_value)
        if not isinstance(decision, MonitoringDecision):
            decision = MonitoringDecision.model_validate(decision)
        # What the model decided, before the deterministic override (for evaluation).
        llm_output = decision.model_dump(mode="json", exclude=_STORED_EXCLUDE)

        # 4. Treat deterministic SLA results as authoritative.
        decision.sla_analysis = sla_analysis
        decision.record_id = record_id

        if not sla_analysis.violations:
            decision.action_required = False
            decision.selected_agents = []

            if sla_analysis.unavailable_metrics:
                # Some SLAs could not be evaluated, so do not claim
                # that the entire network is healthy.
                decision.situation = "warning"
                decision.severity = "none"
                decision.reason = (
                    "No SLA violations were detected among the evaluable KPIs, "
                    "but monitoring data is incomplete for: "
                    + ", ".join(sla_analysis.unavailable_metrics)
                    + ". No agents were invoked."
                )
            else:
                decision.situation = "normal"
                decision.severity = "none"
                decision.reason = (
                    "All monitored KPIs were evaluated and no SLA violations "
                    "were detected."
                )
        else:
            decision.severity = self._severity(sla_analysis)
            # A record with verified violations is never "normal" (run-24 had
            # situation="normal" with severity="high"); the LLM's choice between
            # warning and critical is kept.
            if decision.situation == "normal":
                decision.situation = "warning"
            decision.selected_agents = self._merge_duplicate_agents(decision.selected_agents)
            self._apply_routing_rules(decision, sla_analysis, persistence)
            decision.action_required = bool(decision.selected_agents)

        # 5. Store the EFFECTIVE decision (after the deterministic override), so
        # the results match what was executed. The SLA analysis is stored once
        # on the record, and the agent context is injected by trusted code.
        if store is not None:
            store.set_decision(
                run_id=self.run_id,
                record_id=record_id,
                decision=self._stored_decision(decision, llm_output),
            )

        # 6. Working memory: this record, for the next decisions on the cell.
        if self.memory is not None:
            await self.memory.record_decision(
                record.cell_id,
                RecordEvent.of(
                    record, sla_analysis, decision.severity,
                    [i.agent for i in decision.selected_agents],
                ),
            )

        return decision

    async def invoke_agent(
        self,
        invocation: AgentInvocation,
        record_id: str,
        trusted_context: dict[str, Any] | None = None,
    ) -> Any:
        """Invoke one selected agent. The trusted context (deterministic code)
        overrides anything the LLM put in the invocation context. A missing
        agent or a failure is recorded and returns None."""
        store = self.mongo_repository if self.run_id is not None else None
        agent = self.sub_agents.get(invocation.agent)
        invocation_context = {**invocation.context, **(trusted_context or {})}
        invocation_context.setdefault("record_id", record_id)
        if self.run_id is not None:
            invocation_context.setdefault("run_id", self.run_id)
        invocation = invocation.model_copy(update={"context": invocation_context})

        if agent is None:
            logger.error("No implementation registered for %s", invocation.agent)
            if store is not None:
                store.finish_agent(
                    run_id=self.run_id, record_id=record_id, agent=invocation.agent,
                    status="error",
                    error={"type": "MissingAgent",
                           "message": "No implementation registered for this agent."},
                )
            return None

        try:
            return await agent(invocation)
        except Exception as exc:
            # The RIC agents record their own failures; this catches a
            # failure around them so it is still visible in the results.
            logger.exception("%s invocation failed", invocation.agent)
            if store is not None:
                store.finish_agent(
                    run_id=self.run_id, record_id=record_id, agent=invocation.agent,
                    status="error",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            return None

    async def run(
        self,
        record: TelemetryRecord,
        persistence: dict[str, int] | None = None,
    ) -> MonitoringDecision:
        """Decide on one record and invoke the selected agents in order
        (without the record workflow graph; see agents/record_workflow.py)."""
        decision = await self.decide(record, persistence)
        if decision.action_required:
            for invocation in decision.selected_agents:
                await self.invoke_agent(invocation, record.record_id)
        return decision
