# near_rt_ric/agent.py
from pathlib import Path

from agents.base_ric import BaseRICAgent
from tools.near_rt_ric_tools import (
    search_ran_knowledge,
    adjust_prb_allocation,
    optimize_scheduler,
    optimize_handover,
    adjust_power_control,
    adjust_qos_parameters,
    adjust_link_adaptation,
)


class NearRTRICAgent(BaseRICAgent):
    """Specialized agent for fast, local RAN optimization."""

    agent_name = "near_rt_ric"
    prompt_path = Path("prompts/near_rt_ric.txt")
    tools = [
        search_ran_knowledge,
        adjust_prb_allocation,
        optimize_scheduler,
        optimize_handover,
        adjust_power_control,
        adjust_qos_parameters,
        adjust_link_adaptation,

    ]