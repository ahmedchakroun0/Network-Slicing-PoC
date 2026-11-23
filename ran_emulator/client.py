"""Async client of the RAN emulator API, for the MCP tool servers and runners."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from src.schema.monitoring import TelemetryRecord


class EmulatorError(RuntimeError):
    def __init__(self, status_code: int, detail: Any):
        super().__init__(f"emulator returned {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class EmulatorClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8100",
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout_s, transport=transport)

    async def __aenter__(self) -> EmulatorClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        r = await self._http.request(method, path, **kwargs)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text
            raise EmulatorError(r.status_code, detail)
        return r.json()

    # ---- controls (one per action tool) ----------------------------------
    async def adjust_prb_allocation(
        self, cell: str, slice_id: str, prb_percentage: float
    ) -> dict[str, Any]:
        return await self._call(
            "PUT", f"/cells/{cell}/slices/{slice_id}/prb", json={"prb_percentage": prb_percentage}
        )

    async def optimize_scheduler(self, cell: str, scheduler: str) -> dict[str, Any]:
        return await self._call("PUT", f"/cells/{cell}/scheduler", json={"scheduler": scheduler})

    async def optimize_handover(self, cell: str, handover_margin_db: float) -> dict[str, Any]:
        return await self._call(
            "PUT", f"/cells/{cell}/handover", json={"handover_margin_db": handover_margin_db}
        )

    async def adjust_power_control(self, cell: str, power_offset_db: float) -> dict[str, Any]:
        return await self._call(
            "PUT", f"/cells/{cell}/power", json={"power_offset_db": power_offset_db}
        )

    async def adjust_qos_parameters(
        self, cell: str, slice_id: str, priority_level: int
    ) -> dict[str, Any]:
        return await self._call(
            "PUT", f"/cells/{cell}/slices/{slice_id}/qos", json={"priority_level": priority_level}
        )

    async def adjust_link_adaptation(
        self, cell: str, slice_id: str, mcs_offset: int, target_bler: float
    ) -> dict[str, Any]:
        return await self._call(
            "PUT",
            f"/cells/{cell}/slices/{slice_id}/link-adaptation",
            json={"mcs_offset": mcs_offset, "target_bler": target_bler},
        )

    async def update_slice_resource_policy(
        self, cell: str, slice_id: str, min_prb_ratio_pct: float, max_prb_ratio_pct: float
    ) -> dict[str, Any]:
        return await self._call(
            "PUT",
            f"/cells/{cell}/slices/{slice_id}/policy",
            json={"min_prb_ratio_pct": min_prb_ratio_pct, "max_prb_ratio_pct": max_prb_ratio_pct},
        )

    # ---- simulation ------------------------------------------------------
    async def reset(self, scenario: str | None = None, seed: int | None = None) -> dict[str, Any]:
        return await self._call("POST", "/sim/reset", json={"scenario": scenario, "seed": seed})

    async def run(self, steps: int = 1) -> list[dict[str, Any]]:
        return (await self._call("POST", "/sim/run", json={"steps": steps}))["steps"]

    async def step_records(self) -> list[TelemetryRecord]:
        """Advance one step (lockstep) and return its telemetry records."""
        (step,) = await self.run(1)
        return [TelemetryRecord.model_validate(r) for r in step["records"]]

    async def add_event(self, **event: Any) -> dict[str, Any]:
        return await self._call("POST", "/sim/events", json=event)

    async def state(self) -> dict[str, Any]:
        return await self._call("GET", "/state")

    async def kpis(
        self, cell: str, from_step: int = 0, to_step: int | None = None
    ) -> list[dict[str, Any]]:
        params: dict[str, int] = {"from_step": from_step}
        if to_step is not None:
            params["to_step"] = to_step
        return await self._call("GET", f"/cells/{cell}/kpis", params=params)

    async def violations(self) -> list[dict[str, Any]]:
        return await self._call("GET", "/sim/violations")

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Yield each step payload pushed on /telemetry/stream (SSE)."""
        async with self._http.stream("GET", "/telemetry/stream", timeout=None) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[len("data: ") :])
