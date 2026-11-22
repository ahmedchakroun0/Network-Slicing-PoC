from pathlib import Path

from agents.base_ric import BaseRICAgent

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


class NearRTRICAgent(BaseRICAgent):
    """Specialized agent for fast, local RAN optimization (E2-style controls)."""

    agent_name = "near_rt_ric"
    prompt_path = PROMPTS / "near_rt_ric.txt"
