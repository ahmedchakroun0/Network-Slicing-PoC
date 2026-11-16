from pathlib import Path

from agents.base_ric import BaseRICAgent
from tools.near_rt_ric_tools import search_ran_knowledge
from tools.registry import tools_of


class NonRTRICAgent(BaseRICAgent):
    """Specialized agent for persistent-degradation, policy-level changes."""

    agent_name = "non_rt_ric"
    prompt_path = Path("prompts/non_rt_ric.txt")
    # Knowledge search plus every action tool the registry gives this role.
    tools = [search_ran_knowledge, *tools_of("non_rt_ric")]
