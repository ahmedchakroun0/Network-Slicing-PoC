from pathlib import Path

from langchain_core.tools import tool

from src.schema.monitoring import Scheduler

# Resolved from the project root so the tool works from any working directory.
KNOWLEDGE_FILE = Path(__file__).resolve().parent.parent / "data" / "ran_knowledge.txt"


@tool
async def search_ran_knowledge(query: str) -> str:
    """
    Search the 6G RAN knowledge base for information relevant
    to the current Near-RT RIC problem.
    """

    text = KNOWLEDGE_FILE.read_text(encoding="utf-8")

    query_words = {word.lower() for word in query.split() if len(word) > 2}

    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    results = []

    for chunk in chunks:
        chunk_words = {word.lower().strip(".,:;()[]{}") for word in chunk.split()}

        score = len(query_words & chunk_words)

        if score > 0:
            results.append((score, chunk))

    results.sort(reverse=True)

    if not results:
        return "No relevant RAN knowledge was found."

    return "\n\n".join(chunk for _, chunk in results[:3])


# Every action tool below also takes the same reasoning arguments
# (rationale, expected_effect, confidence, evidence_sources). The tool body
# ignores them: RANAuthorizationMiddleware reads them from the call, builds
# the proposal and has the Trust & Governance Agent review it BEFORE the
# body runs.


@tool
async def adjust_prb_allocation(
    target: str,
    slice_id: str,
    prb_percentage: int,
    rationale: str,
    expected_effect: str,
    confidence: float,
    evidence_sources: list[str] | None = None,
) -> dict:
    """
    Adjust PRB allocation for a network slice.
    prb_percentage is the new PRB share of the slice (5-90).
    target and slice_id must be copied from the invocation context.
    rationale: why this tool and this exact value, citing the violated KPIs.
    expected_effect: which KPI should improve and how.
    confidence: float between 0 and 1.
    This is a dummy network operation for the PoC.
    """

    return {
        "status": "success",
        "tool": "adjust_prb_allocation",
        "target": target,
        "slice_id": slice_id,
        "prb_percentage": prb_percentage,
        "message": (
            f"PRB allocation changed to {prb_percentage}% for {slice_id} on {target}."
        ),
    }


@tool
async def optimize_scheduler(
    target: str,
    slice_id: str,
    scheduler: Scheduler,
    rationale: str,
    expected_effect: str,
    confidence: float,
    evidence_sources: list[str] | None = None,
) -> dict:
    """
    Change the MAC scheduler used for a slice.
    scheduler: round_robin | waterfilling | proportional_fair (the schedulers
    of the ColO-RAN testbed).
    target and slice_id must be copied from the invocation context.
    rationale: why this tool and this exact value, citing the violated KPIs.
    expected_effect: which KPI should improve and how.
    confidence: float between 0 and 1.
    This is a dummy network operation for the PoC.
    """

    return {
        "status": "success",
        "tool": "optimize_scheduler",
        "target": target,
        "slice_id": slice_id,
        "scheduler": scheduler,
        "message": (f"Scheduler '{scheduler}' configured for {slice_id} on {target}."),
    }


@tool
async def optimize_handover(
    target: str,
    handover_margin_db: float,
    rationale: str,
    expected_effect: str,
    confidence: float,
    evidence_sources: list[str] | None = None,
) -> dict:
    """
    Adjust handover parameters.
    handover_margin_db: 0.0-10.0.
    target must be copied from the invocation context.
    rationale: why this tool and this exact value, citing the violated KPIs.
    expected_effect: which KPI should improve and how.
    confidence: float between 0 and 1.
    This is a dummy network operation for the PoC.
    """

    return {
        "status": "success",
        "tool": "optimize_handover",
        "target": target,
        "handover_margin_db": handover_margin_db,
        "message": (f"Handover margin changed to {handover_margin_db} dB on {target}."),
    }


@tool
async def adjust_power_control(
    target: str,
    power_offset_db: float,
    rationale: str,
    expected_effect: str,
    confidence: float,
    evidence_sources: list[str] | None = None,
) -> dict:
    """
    Adjust RAN power-control parameters.
    power_offset_db: -6.0 to 6.0.
    target must be copied from the invocation context.
    rationale: why this tool and this exact value, citing the violated KPIs.
    expected_effect: which KPI should improve and how.
    confidence: float between 0 and 1.
    This is a dummy network operation for the PoC.
    """

    return {
        "status": "success",
        "tool": "adjust_power_control",
        "target": target,
        "power_offset_db": power_offset_db,
        "message": (
            f"Power control offset changed to {power_offset_db} dB on {target}."
        ),
    }
@tool
async def adjust_qos_parameters(
    target: str,
    slice_id: str,
    priority_level: int,
    packet_delay_budget_ms: float,
    guaranteed_bitrate_mbps: float,
    rationale: str,
    expected_effect: str,
    confidence: float,
    evidence_sources: list[str] | None = None,
) -> dict:
    """
    Adjust the QoS profile of a network slice (5QI-style characteristics).
    priority_level: 1-10, 1 is the highest scheduling priority.
    packet_delay_budget_ms: 5-300 ms, the delay the slice's packets may take.
    guaranteed_bitrate_mbps: 0-10 Mbps per UE (0 = non-GBR).
    target and slice_id must be copied from the invocation context.
    rationale: why this tool and these exact values, citing the violated KPIs.
    expected_effect: which KPI should improve and how.
    confidence: float between 0 and 1.
    This is a dummy network operation for the PoC.
    """

    return {
        "status": "success",
        "tool": "adjust_qos_parameters",
        "target": target,
        "slice_id": slice_id,
        "priority_level": priority_level,
        "packet_delay_budget_ms": packet_delay_budget_ms,
        "guaranteed_bitrate_mbps": guaranteed_bitrate_mbps,
        "message": (
            f"QoS of {slice_id} on {target}: priority {priority_level}, "
            f"PDB {packet_delay_budget_ms} ms, GBR {guaranteed_bitrate_mbps} Mbps/UE."
        ),
    }


@tool
async def adjust_link_adaptation(
    target: str,
    slice_id: str,
    mcs_offset: int,
    target_bler: float,
    rationale: str,
    expected_effect: str,
    confidence: float,
    evidence_sources: list[str] | None = None,
) -> dict:
    """
    Adjust downlink link adaptation for a network slice.
    mcs_offset: -5 to 5, added to the MCS chosen from the reported CQI
    (negative = more robust, positive = more aggressive).
    target_bler: 0.001-0.2, the block error rate link adaptation aims for.
    target and slice_id must be copied from the invocation context.
    rationale: why this tool and these exact values, citing the violated KPIs.
    expected_effect: which KPI should improve and how.
    confidence: float between 0 and 1.
    This is a dummy network operation for the PoC.
    """

    return {
        "status": "success",
        "tool": "adjust_link_adaptation",
        "target": target,
        "slice_id": slice_id,
        "mcs_offset": mcs_offset,
        "target_bler": target_bler,
        "message": (
            f"Link adaptation of {slice_id} on {target}: MCS offset {mcs_offset}, "
            f"target BLER {target_bler}."
        ),
    }
