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
