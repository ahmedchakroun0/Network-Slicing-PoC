# non_rt_ric/agent.py
from pathlib import Path

from agents.base_ric import BaseRICAgent
from tools.near_rt_ric_tools import search_ran_knowledge
from tools.non_rt_ric_tools import update_slice_resource_policy


class NonRTRICAgent(BaseRICAgent):
    """Specialized agent for persistent-degradation, policy-level changes."""

    agent_name = "non_rt_ric"
    prompt_path = Path("prompts/non_rt_ric.txt")
    tools = [
        search_ran_knowledge,
        update_slice_resource_policy,
    ]
