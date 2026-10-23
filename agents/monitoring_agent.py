from collections.abc import Awaitable, Callable
from typing import Any
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

from src.memory.working_memory import RedisWorkingMemory
from src.schema.monitoring import TelemetryRecord
from src.schema.monitoring import (
    AgentInvocation,
    AgentName,
    KPIStatus,
    MonitoringDecision,
    SLAAnalysis,
)
from src.schema.working_memory import ToolExecution
from src.repositories.mongo_repository import MongoRepository


SubAgent = Callable[[AgentInvocation], Awaitable[Any]]
_PROMPT_PATH = Path("prompts/monitoring.txt")


class MonitoringAgent:
    """Central orchestrator for telemetry analysis and agent invocation."""

    def __init__(
        self,
        llm: Any,
        memory: RedisWorkingMemory,
        sub_agents: dict[AgentName, SubAgent],
        mongo_repository: MongoRepository | None = None,
        run_id: str | None = None,
    ) -> None:
        self.memory = memory
        self.sub_agents = sub_agents
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
Incoming telemetry:
{telemetry}

Deterministic SLA analysis:
{sla_analysis}

Current shared network status:
{network_status}

Recent agent tool executions:
{recent_tool_executions}

Consecutive-violation persistence for this cell and slice (per KPI,
computed deterministically; "not provided" if unknown):
{persistence}

Return a MonitoringDecision.
""",
                ),
            ]
        )

    @staticmethod
    def _compare(
        measured: float | None,
        budget: float | None,
        *,
        lower_is_better: bool,
    ) -> KPIStatus:
        """Compare a measured KPI with its corresponding budget."""
        if measured is None or budget is None:
            return KPIStatus(
                measured=measured,
                budget=budget,
                status="unavailable",
            )

        violation = measured > budget if lower_is_better else measured < budget

        return KPIStatus(
            measured=measured,
            budget=budget,
            status="violation" if violation else "ok",
        )

    @classmethod
    def analyze_sla(cls, record: TelemetryRecord) -> SLAAnalysis:
        """Calculate SLA violations deterministically."""
        analysis = SLAAnalysis(
            latency=cls._compare(
                record.slice_latency_ns,
                record.latency_budget_ns,
                lower_is_better=True,
            ),
            jitter=cls._compare(
                record.slice_jitter_ns,
                record.jitter_budget_ns,
                lower_is_better=True,
            ),
            packet_loss=cls._compare(
                record.slice_packet_loss,
                record.packet_loss_budget,
                lower_is_better=True,
            ),
            throughput=cls._compare(
                record.slice_available_transfer_rate_gbps,
                record.data_rate_budget_gbps,
                lower_is_better=False,
            ),
        )
        kpis = {
            "latency": analysis.latency,
            "jitter": analysis.jitter,
            "packet_loss": analysis.packet_loss,
            "throughput": analysis.throughput,
        }

        analysis.violations = [
            name for name, status in kpis.items() if status.status == "violation"
        ]

        analysis.unavailable_metrics = [
            name for name, status in kpis.items() if status.status == "unavailable"
        ]

        return analysis
    @staticmethod
    def _severity(analysis: SLAAnalysis) -> str:
        worst = 0.0
        for name in analysis.violations:
            s = getattr(analysis, name)
            if not s.budget:
                continue
            if name == "throughput":
                ratio = s.budget / s.measured if s.measured else float("inf")
            else:
                ratio = s.measured / s.budget
            worst = max(worst, ratio)
        return "high" if worst >= 3 else "medium" if worst >= 1.5 else "low"

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

    async def run(
        self,
        record: TelemetryRecord,
        persistence: dict[str, int] | None = None,
    ) -> MonitoringDecision:
        """Analyze one telemetry record and invoke selected agents."""
        # 1. Read the latest shared network context.
        memory = await self.memory.get()

        # 2. Calculate SLA violations without relying on the LLM.
        sla_analysis = self.analyze_sla(record)

        record_id = record.record_id or f"record-{id(record)}"

        if self.mongo_repository is not None and self.run_id is not None:
            self.mongo_repository.start_record(
                run_id=self.run_id,
                record_document=self.mongo_repository.new_record_document(
                    record_id=record_id,
                    telemetry_record=record.model_dump(mode="json"),
                    ground_truth={},
                    comparison={},
                    computed_analysis=sla_analysis.model_dump(mode="json"),
                ),
            )

        # 3. Ask the LLM for a structured orchestration decision.
        prompt_value = self.prompt.invoke(
            {
                "telemetry": record.model_dump(
                    by_alias=True,
                    mode="json",
                ),
                "sla_analysis": sla_analysis.model_dump(mode="json"),
                "network_status": memory.network_status,
                "recent_tool_executions": [
                    execution.model_dump(mode="json")
                    for execution in memory.tools_executed[-10:]
                ],
                "persistence": persistence if persistence is not None else "not provided",
            }
        )

        decision: MonitoringDecision = await self.decision_llm.ainvoke(prompt_value)

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
            decision.action_required = bool(decision.selected_agents)
        else:
            decision.severity = self._severity(sla_analysis)
            # A record with verified violations is never "normal" (run-24 had
            # situation="normal" with severity="high"); the LLM's choice between
            # warning and critical is kept.
            if decision.situation == "normal":
                decision.situation = "warning"
            decision.selected_agents = self._merge_duplicate_agents(decision.selected_agents)
        # Persist the EFFECTIVE decision (after the deterministic override),
        # so the audit trail matches what was actually executed.
        if self.mongo_repository is not None and self.run_id is not None:
            self.mongo_repository.set_monitoring_decision(
                run_id=self.run_id,
                record_id=record_id,
                monitoring_decision=decision.model_dump(mode="json"),
            )

        # 5. Update the shared network status.
        await self.memory.update_network_status(decision.global_situation)

        # 6. Record the decision before invoking subagents.
        await self.memory.add_tool_execution(
            ToolExecution(
                agent="monitoring",
                tool_name="analyze_telemetry_and_decide",
                result={
                    "record_id": record_id,
                    "situation": decision.situation,
                    "severity": decision.severity,
                    "action_required": decision.action_required,
                    "sla_analysis": sla_analysis.model_dump(mode="json"),
                    "selected_agents": [
                        invocation.model_dump(mode="json")
                        for invocation in decision.selected_agents
                    ],
                    "reason": decision.reason,
                },
            )
        )

        # 7. Invoke only the agents selected by the Monitoring Agent.
        if decision.action_required:
            for invocation in decision.selected_agents:
                agent = self.sub_agents.get(invocation.agent)
                invocation_context = dict(invocation.context)
                invocation_context.setdefault("record_id", record_id)
                if self.run_id is not None:
                    invocation_context.setdefault("run_id", self.run_id)
                invocation = invocation.model_copy(
                    update={"context": invocation_context}
                )

                if agent is None:
                    await self.memory.add_tool_execution(
                        ToolExecution(
                            agent=invocation.agent,
                            tool_name="agent_invocation",
                            result={
                                "status": "error",
                                "message": "No implementation registered "
                                "for this agent.",
                            },
                        )
                    )

                    if self.mongo_repository is not None and self.run_id is not None:
                        self.mongo_repository.finish_agent_execution(
                            run_id=self.run_id,
                            record_id=record_id,
                            agent=invocation.agent,
                            invocation=invocation,
                            result={
                                "status": "error",
                                "message": "No implementation registered for this agent.",
                            },
                            middleware_audit=[],
                        )
                    continue

                try:
                    result = await agent(invocation)

                    await self.memory.add_tool_execution(
                        ToolExecution(
                            agent=invocation.agent,
                            tool_name="agent_invocation",
                            result={
                                "status": "completed",
                                "instruction": invocation.model_dump(mode="json"),
                                "result": result,
                            },
                        )
                    )

                except Exception as exc:
                    # Record the failure so it is visible in the PoC trace.
                    await self.memory.add_tool_execution(
                        ToolExecution(
                            agent=invocation.agent,
                            tool_name="agent_invocation",
                            result={
                                "status": "error",
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            },
                        )
                    )

        return decision
