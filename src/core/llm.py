"""Create a chat model routed through the LiteLLM Gateway."""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_litellm import ChatLiteLLM

load_dotenv()


def message_text(message: Any) -> str | None:
    """Visible text of a chat message (string or list of content blocks)."""
    content = getattr(message, "content", None)
    if isinstance(content, list):
        content = "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content.strip() or None if isinstance(content, str) else None


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else default


# One limiter for every agent of the process: Monitoring, RIC agents and the
# governance review all draw from the same request budget of the endpoint.
_RATE_LIMITER: InMemoryRateLimiter | None = None


def _rate_limiter() -> InMemoryRateLimiter | None:
    """Shared limiter from LLM_REQUESTS_PER_MINUTE (0 or unset = no limit)."""
    global _RATE_LIMITER
    per_minute = _env_float("LLM_REQUESTS_PER_MINUTE", 0.0)
    if per_minute <= 0:
        return None
    if _RATE_LIMITER is None:
        _RATE_LIMITER = InMemoryRateLimiter(
            requests_per_second=per_minute / 60.0,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    return _RATE_LIMITER


def create_llm() -> ChatLiteLLM:
    """Create a chat model routed through the LiteLLM Gateway.

    Transient failures (rate limit, timeout, connection or API error) are
    retried with exponential backoff, LLM_MAX_RETRIES attempts (default 6),
    and requests are spaced by the shared LLM_REQUESTS_PER_MINUTE limiter.
    """

    model_name = os.environ["MODEL_NAME"]

    return ChatLiteLLM(
        model=f"litellm_proxy/{model_name}",
        api_base=os.environ["LITELLM_API_BASE"],
        api_key=os.environ["LITELLM_API_KEY"],
        temperature=0,
        max_retries=int(_env_float("LLM_MAX_RETRIES", 6)),
        rate_limiter=_rate_limiter(),
    )
