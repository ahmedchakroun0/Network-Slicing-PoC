"""Where the services listen and how they reach each other (environment variables).

Each service only knows the URLs of the services it calls: the Monitoring
agent the RIC agents' card URLs, the RIC agents the governance URL, the driver
the Monitoring card URL. A service requires its own bearer token
(A2A_TOKEN_<SERVICE>); a caller sends the token of the service it calls. An
unset token means no authentication (local development).
"""

from __future__ import annotations

import os

SERVICES = ("monitoring", "near_rt_ric", "non_rt_ric", "governance")
DEFAULT_PORT = {"monitoring": 8300, "near_rt_ric": 8301, "non_rt_ric": 8302, "governance": 8310}


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def token(service: str) -> str | None:
    return os.environ.get(f"A2A_TOKEN_{service.upper()}") or None


def gateway_key() -> str | None:
    """LiteLLM key the Monitoring agent sends when A2A_AGENT_CARDS points at the
    LiteLLM A2A gateway (/a2a/<agent>); the gateway then authenticates to each
    agent with the agent's own token (static_headers in litellm_config.yaml)."""
    return os.environ.get("A2A_GATEWAY_KEY") or None


def governance_url() -> str:
    return _env("GOVERNANCE_URL", f"http://127.0.0.1:{DEFAULT_PORT['governance']}")


def ric_card_urls() -> list[str]:
    default = ",".join(f"http://127.0.0.1:{DEFAULT_PORT[r]}" for r in ("near_rt_ric", "non_rt_ric"))
    return [u.strip() for u in _env("A2A_AGENT_CARDS", default).split(",") if u.strip()]


def monitoring_url() -> str:
    return _env("MONITORING_URL", f"http://127.0.0.1:{DEFAULT_PORT['monitoring']}")


def ric_timeout_s() -> float:
    # One RIC invocation may include the governance LLM review (<= 180 s).
    return float(_env("A2A_RIC_TIMEOUT_S", "300"))


def monitoring_timeout_s() -> float:
    # One step: every record of the step, each possibly invoking both RICs.
    return float(_env("A2A_MONITORING_TIMEOUT_S", "1800"))


def governance_timeout_s() -> float:
    return float(_env("GOVERNANCE_TIMEOUT_S", "200"))
