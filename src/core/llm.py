"""Create a chat model routed through the LiteLLM Gateway.

Provider switching is done by the gateway (litellm_config.yaml, fallbacks):
a rate-limited model is replaced by the next one of its fallback chain. When
the whole chain is rate-limited, PROVIDER_GUARD stops every further model
call of the process and the dataset runner stops the run.
"""

import os
from collections import Counter
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
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


class ModelProviderStopped(RuntimeError):
    """Every model of the fallback chain is rate-limited: no call is made."""


def is_rate_limit(error: BaseException) -> bool:
    """A rate limit from the gateway: 429, or every model cooling down."""
    return (
        getattr(error, "status_code", None) == 429
        or "RateLimit" in type(error).__name__
    )


# Who made a model call: the workflow node of the record graph (first part
# of the LangGraph namespace), or the governance review (its gate span tag).
CALLERS = ("monitoring", "near_rt_ric", "non_rt_ric")
GOVERNANCE_REVIEW = "governance_review"


def caller_of(tags: list[str] | None, metadata: dict[str, Any] | None) -> str:
    if "governance" in (tags or []):
        return GOVERNANCE_REVIEW
    namespace = (metadata or {}).get("langgraph_checkpoint_ns") or ""
    node = namespace.split("|")[0].split(":")[0]
    return node if node in CALLERS else "other"


def _usage() -> dict[str, int]:
    return {"calls": 0, "failed_calls": 0, "input_tokens": 0, "output_tokens": 0,
            "reasoning_tokens": 0, "calls_without_usage": 0}


class ProviderGuard(BaseCallbackHandler):
    """Callback on every model call of the process.

    - A rate-limit error that reaches the client (the gateway's fallbacks are
      exhausted) stops the provider: every later model call fails at once
      with ModelProviderStopped instead of being sent.
    - It counts, per caller (agent or governance review), the calls and the
      input / output tokens the provider reported, and which model actually
      answered (the gateway can fall back to another model than MODEL_NAME).
      take_usage() returns and resets them, once per record.
    """

    raise_error = True   # a stopped provider must abort the call
    run_inline = True    # state changes in call order

    def __init__(self) -> None:
        self.stopped: str | None = None
        self._clear()

    def _clear(self) -> None:
        self.served: Counter[str] = Counter()
        self.usage: dict[str, dict[str, int]] = {}
        self._callers: dict[Any, str] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id=None, tags=None,
                            metadata=None, **kwargs: Any) -> None:
        if self.stopped is not None:
            raise ModelProviderStopped(self.stopped)
        self._callers[run_id] = caller_of(tags, metadata)

    def on_llm_end(self, response, *, run_id=None, **kwargs: Any) -> None:
        usage = self.usage.setdefault(self._callers.pop(run_id, "other"), _usage())
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                metadata = getattr(message, "response_metadata", None) or {}
                self.served[metadata.get("served_by") or "unknown"] += 1
                tokens = getattr(message, "usage_metadata", None)
                usage["calls"] += 1
                if not tokens:
                    usage["calls_without_usage"] += 1
                    continue
                usage["input_tokens"] += tokens.get("input_tokens", 0)
                usage["output_tokens"] += tokens.get("output_tokens", 0)
                # Part of output_tokens (billed) when the provider reports it.
                usage["reasoning_tokens"] += (
                    (tokens.get("output_token_details") or {}).get("reasoning") or 0)

    def on_llm_error(self, error: BaseException, *, run_id=None, **kwargs: Any) -> None:
        self.usage.setdefault(self._callers.pop(run_id, "other"), _usage())["failed_calls"] += 1
        if self.stopped is None and is_rate_limit(error):
            self.stopped = f"{type(error).__name__}: {str(error)[:300]}"

    def take_usage(self) -> dict[str, Any]:
        """Model usage since the last call (one record), then reset:
        totals, per caller, and the models that answered."""
        by_caller, served = self.usage, self.served
        self._clear()
        total = _usage()
        for counts in by_caller.values():
            for key, value in counts.items():
                total[key] += value
        return {**total, "by_caller": by_caller, "models": dict(served)}

    def reset(self) -> None:
        self.stopped = None
        self._clear()


PROVIDER_GUARD = ProviderGuard()


class _GatewayChatModel(ChatLiteLLM):
    """ChatLiteLLM that keeps the model the gateway actually used."""

    def _create_chat_result(self, response, **params: Any):
        result = super()._create_chat_result(response, **params)
        served = response.get("model") if hasattr(response, "get") else None
        for generation in result.generations:
            generation.message.response_metadata["served_by"] = served
        return result


def create_llm() -> ChatLiteLLM:
    """Create a chat model routed through the LiteLLM Gateway.

    The gateway retries and falls back to other models (litellm_config.yaml),
    so the client does not retry by default (LLM_MAX_RETRIES=1 is a single
    attempt): a rate limit that reaches the client means the whole fallback
    chain is exhausted, and PROVIDER_GUARD stops further calls. Requests are
    spaced by the shared LLM_REQUESTS_PER_MINUTE limiter.
    """

    model_name = os.environ["MODEL_NAME"]

    return _GatewayChatModel(
        model=f"litellm_proxy/{model_name}",
        api_base=os.environ["LITELLM_API_BASE"],
        api_key=os.environ["LITELLM_API_KEY"],
        temperature=0,
        max_retries=max(1, int(_env_float("LLM_MAX_RETRIES", 1))),
        rate_limiter=_rate_limiter(),
        callbacks=[PROVIDER_GUARD],
    )
