from pathlib import Path

from agents.base_ric import BaseRICAgent

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


class NonRTRICAgent(BaseRICAgent):
    """Specialized agent for persistent-degradation, policy-level changes (A1-style)."""

    agent_name = "non_rt_ric"
    prompt_path = PROMPTS / "non_rt_ric.txt"
