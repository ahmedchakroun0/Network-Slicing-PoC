"""The RAN action tools, each defined once.

One method per tool: its signature is the tool's arguments and its docstring
is the tool's description. Everything else is derived from it: the MCP
tool schema (server.py), what the model sees (client.py) and the argument
schema governance validates (registry.ARG_MODELS).

Each tool takes only the arguments the RAN uses. Value ranges are stated in
the descriptions, not enforced here: the safety envelope is governance's
resource/safety stage (src/schema/limits.py), and the emulator refuses a
value outside it (HTTP 422).

The model's justification of a call (rationale, expected_effect, ...) is
not a tool argument: the client adds it to what the model sees and strips it
before the MCP call (client.py), and governance reviews it.
"""

from typing import Annotated, Any

from pydantic import Field

from ran_emulator.client import EmulatorClient
from src.schema.monitoring import Scheduler
from src.schema.sla import SliceType

Cell = Annotated[
    str, Field(min_length=1, description="The base station to act on: the invocation's target.")
]
Slice = Annotated[SliceType, Field(description="The slice to act on: the invocation's slice_id.")]


class RANControls:
    """The action tools of the RIC agents, acting on the RAN emulator. A change
    applies from the next emulator step; each result reports the previous and
    the current value."""

    def __init__(self, emulator: EmulatorClient) -> None:
        self._emulator = emulator

    # ------------------------------------------------------------ Near-RT RIC

    async def adjust_prb_allocation(
        self,
        target: Cell,
        slice_id: Slice,
        prb_percentage: Annotated[int, Field(description="New PRB share of the slice, 5-90 %.")],
    ) -> dict[str, Any]:
        """Change the PRB share of a slice. The request is snapped to the closest
        recorded PRB split (the other slices give up or get PRBs); the result
        reports the requested and the applied share."""
        return await self._emulator.adjust_prb_allocation(target, slice_id, prb_percentage)

    async def optimize_scheduler(
        self,
        target: Cell,
        scheduler: Annotated[Scheduler, Field(description="The MAC scheduler to use.")],
    ) -> dict[str, Any]:
        """Change the MAC scheduler of the cell. The ColO-RAN testbed runs one
        scheduler per cell, so the change applies to every slice."""
        return await self._emulator.optimize_scheduler(target, scheduler)

    async def optimize_handover(
        self,
        target: Cell,
        handover_margin_db: Annotated[float, Field(description="Handover margin, 0-10 dB.")],
    ) -> dict[str, Any]:
        """Change the handover margin of the cell (affects the SINR of every slice)."""
        return await self._emulator.optimize_handover(target, handover_margin_db)

    async def adjust_power_control(
        self,
        target: Cell,
        power_offset_db: Annotated[float, Field(description="Transmit power offset, -6 to 6 dB.")],
    ) -> dict[str, Any]:
        """Change the transmit power offset of the cell (raises or lowers the
        SINR of every slice)."""
        return await self._emulator.adjust_power_control(target, power_offset_db)

    async def adjust_qos_parameters(
        self,
        target: Cell,
        slice_id: Slice,
        priority_level: Annotated[
            int, Field(description="Scheduling priority, 1-10; 1 is the highest.")
        ],
    ) -> dict[str, Any]:
        """Change the scheduling priority of a slice. Below 5, the slice may
        borrow the PRBs the cell's other slices leave idle."""
        return await self._emulator.adjust_qos_parameters(target, slice_id, priority_level)

    async def adjust_link_adaptation(
        self,
        target: Cell,
        slice_id: Slice,
        mcs_offset: Annotated[
            int,
            Field(
                description="Added to the MCS chosen from the CQI, -5 to 5 "
                "(negative = more robust, positive = more aggressive)."
            ),
        ],
        target_bler: Annotated[
            float, Field(description="Block error rate link adaptation aims for, 0.001-0.2.")
        ],
    ) -> dict[str, Any]:
        """Change the downlink link adaptation of a slice: a more robust setting
        trades throughput for fewer retransmissions (lower delay)."""
        return await self._emulator.adjust_link_adaptation(
            target, slice_id, mcs_offset, target_bler
        )

    # ------------------------------------------------------------- Non-RT RIC

    async def update_slice_resource_policy(
        self,
        target: Cell,
        slice_id: Slice,
        min_prb_ratio_pct: Annotated[
            int, Field(description="Guaranteed PRB share of the slice, 5-80 %.")
        ],
        max_prb_ratio_pct: Annotated[
            int,
            Field(description="Maximum PRB share of the slice, 20-100 %; not below the minimum."),
        ],
    ) -> dict[str, Any]:
        """Set the resource policy of a slice (an A1 policy): its guaranteed and
        maximum PRB share. If the current PRB split violates the policy, the
        cell moves to the closest split that respects it."""
        return await self._emulator.update_slice_resource_policy(
            target, slice_id, min_prb_ratio_pct, max_prb_ratio_pct
        )
