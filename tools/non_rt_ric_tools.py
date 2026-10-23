from langchain_core.tools import tool

# Policy-level action tools of the Non-RT RIC. Same pattern as the Near-RT
# tools: the agent picks one, RANAuthorizationMiddleware has the Trust &
# Governance Agent review the call first, and the body only runs when the
# proposal is approved. In a real O-RAN deployment these would be A1 policy
# pushes towards the Near-RT RIC.
#
# rationale / expected_effect / confidence / evidence_sources are read by the
# middleware and governance; the bodies ignore them.


@tool
async def update_slice_resource_policy(
    target: str,
    slice_id: str,
    min_prb_ratio_pct: int,
    max_prb_ratio_pct: int,
    rationale: str,
    expected_effect: str,
    confidence: float,
    evidence_sources: list[str] | None = None,
) -> dict:
    """
    Update the resource policy of a slice: guaranteed (min) and capped (max)
    PRB share.
    min_prb_ratio_pct: 5-80. max_prb_ratio_pct: 20-100. min must be <= max.
    target and slice_id must be copied from the invocation context.
    rationale: why this policy and these values, citing the persistent KPI
    violations. expected_effect: which KPI should improve and how.
    confidence: float between 0 and 1.
    This is a dummy network operation for the PoC.
    """

    return {
        "status": "success",
        "tool": "update_slice_resource_policy",
        "channel": "A1",
        "target": target,
        "slice_id": slice_id,
        "min_prb_ratio_pct": min_prb_ratio_pct,
        "max_prb_ratio_pct": max_prb_ratio_pct,
        "message": (
            f"Resource policy for {slice_id} on {target}: "
            f"{min_prb_ratio_pct}%-{max_prb_ratio_pct}% PRB."
        ),
    }


@tool
async def update_traffic_steering_policy(
    target: str,
    steering_mode: str,
    load_balance_threshold_pct: int,
    rationale: str,
    expected_effect: str,
    confidence: float,
    evidence_sources: list[str] | None = None,
) -> dict:
    """
    Update the traffic steering policy.
    steering_mode: load_balancing | latency_aware | throughput_aware.
    load_balance_threshold_pct: 50-95.
    target must be copied from the invocation context.
    rationale: why this policy and these values, citing the persistent KPI
    violations. expected_effect: which KPI should improve and how.
    confidence: float between 0 and 1.
    This is a dummy network operation for the PoC.
    """

    return {
        "status": "success",
        "tool": "update_traffic_steering_policy",
        "channel": "A1",
        "target": target,
        "steering_mode": steering_mode,
        "load_balance_threshold_pct": load_balance_threshold_pct,
        "message": (
            f"Traffic steering on {target} set to '{steering_mode}' "
            f"(threshold {load_balance_threshold_pct}%)."
        ),
    }


@tool
async def update_admission_control_policy(
    target: str,
    slice_id: str,
    max_utilization_pct: int,
    rationale: str,
    expected_effect: str,
    confidence: float,
    evidence_sources: list[str] | None = None,
) -> dict:
    """
    Update the admission control policy of a slice: the maximum utilization
    before new sessions are refused.
    max_utilization_pct: 60-100.
    target and slice_id must be copied from the invocation context.
    rationale: why this policy and this value, citing the persistent KPI
    violations. expected_effect: which KPI should improve and how.
    confidence: float between 0 and 1.
    This is a dummy network operation for the PoC.
    """

    return {
        "status": "success",
        "tool": "update_admission_control_policy",
        "channel": "A1",
        "target": target,
        "slice_id": slice_id,
        "max_utilization_pct": max_utilization_pct,
        "message": (
            f"Admission control for {slice_id} on {target}: "
            f"max utilization {max_utilization_pct}%."
        ),
    }

@tool
async def optimize_slice_capacity_policy(
    target: str,
    slice_id: str,
    min_prb_ratio_pct: int,
    max_prb_ratio_pct: int,
    rationale: str,
    expected_effect: str,
    confidence: float,
    evidence_sources: list[str] | None = None,
) -> dict:
    """
    Optimize the capacity policy of a slice based on current network conditions.
    min_prb_ratio_pct: 5-80. max_prb_ratio_pct: 20-100. min must be <= max.
    target and slice_id must be copied from the invocation context.
    rationale: why this optimization and these values, citing the persistent KPI
    violations. expected_effect: which KPI should improve and how.
    confidence: float between 0 and 1.
    This is a dummy network operation for the PoC.
    """

    return {
        "status": "success",
        "tool": "optimize_slice_capacity_policy",
        "channel": "A1",
        "target": target,
        "slice_id": slice_id,
        "min_prb_ratio_pct": min_prb_ratio_pct,
        "max_prb_ratio_pct": max_prb_ratio_pct,
        "message": (
            f"Optimized capacity policy for {slice_id} on {target}: "
            f"{min_prb_ratio_pct}%-{max_prb_ratio_pct}% PRB."
        ),
    }