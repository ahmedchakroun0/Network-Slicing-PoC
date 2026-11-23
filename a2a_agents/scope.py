"""Model usage of a service process, scoped to the run and reported per request."""

from __future__ import annotations

from typing import Any

from src.core.llm import PROVIDER_GUARD


class RunScope:
    """Model usage and provider state of one service process, per request.

    The service reports, with each answer, the model usage it accumulated
    while serving the request (take_usage), and whether its provider is
    stopped; the caller merges both into its own record (ProviderGuard.merge).
    A new run id clears a stopped provider, as the in-process runner does at
    the start of a run. Requests are served one at a time in lockstep.
    """

    def __init__(self) -> None:
        self.run_id: str | None = None

    def enter(self, run_id: str | None) -> None:
        if run_id and run_id != self.run_id:
            self.run_id = run_id
            PROVIDER_GUARD.stopped = None

    @staticmethod
    def report() -> dict[str, Any]:
        return {"usage": PROVIDER_GUARD.take_usage(), "stopped": PROVIDER_GUARD.stopped}

    @staticmethod
    def discard() -> None:
        """A failed request reports nothing: do not bill its calls to the next one."""
        PROVIDER_GUARD.take_usage()
