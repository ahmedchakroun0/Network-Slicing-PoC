"""Retries, the shared rate limiter and the provider guard of src/core/llm."""

import litellm
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import src.core.llm as core_llm
from src.core.llm import ModelProviderStopped, ProviderGuard


def _env(monkeypatch, **extra):
    for key, value in {"MODEL_NAME": "m", "LITELLM_API_BASE": "http://x", "LITELLM_API_KEY": "k",
                       **extra}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(core_llm, "_RATE_LIMITER", None)


def test_retries_and_one_shared_limiter(monkeypatch):
    _env(monkeypatch, LLM_MAX_RETRIES="4", LLM_REQUESTS_PER_MINUTE="20")
    monitoring, governance = core_llm.create_llm(), core_llm.create_llm()
    assert monitoring.max_retries == 4
    assert monitoring.rate_limiter is not None
    assert monitoring.rate_limiter is governance.rate_limiter      # one budget for all agents
    assert abs(monitoring.rate_limiter.requests_per_second - 20 / 60) < 1e-9


def test_no_limiter_by_default(monkeypatch):
    _env(monkeypatch)
    monkeypatch.delenv("LLM_REQUESTS_PER_MINUTE", raising=False)
    llm = core_llm.create_llm()
    # The gateway retries and falls back: the client makes a single attempt.
    assert llm.rate_limiter is None and llm.max_retries == 1
    assert core_llm.PROVIDER_GUARD in llm.callbacks


class _Model(BaseChatModel):
    """Chat model double: raises `error`, else answers as `served_by`."""

    error: BaseException | None = None
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "double"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        message = AIMessage(content="ok", response_metadata={"served_by": "gemini/backup"},
                            usage_metadata={"input_tokens": 10, "output_tokens": 2,
                                            "total_tokens": 12})
        return ChatResult(generations=[ChatGeneration(message=message)])


def _rate_limit():
    return litellm.RateLimitError("All deployments are in cooldown", llm_provider="litellm_proxy",
                                  model="m")


@pytest.mark.asyncio
async def test_exhausted_fallbacks_stop_every_later_call():
    guard = ProviderGuard()
    model = _Model(error=_rate_limit(), callbacks=[guard])

    with pytest.raises(litellm.RateLimitError):
        await model.ainvoke("hi")
    assert guard.stopped.startswith("RateLimitError")

    model.error = None
    with pytest.raises(ModelProviderStopped):
        await model.ainvoke("hi")
    assert model.calls == 1            # the second call was never sent


@pytest.mark.asyncio
async def test_other_errors_do_not_stop_the_provider():
    guard = ProviderGuard()
    model = _Model(error=litellm.Timeout("slow", model="m", llm_provider="litellm_proxy"),
                   callbacks=[guard])
    with pytest.raises(litellm.Timeout):
        await model.ainvoke("hi")
    assert guard.stopped is None


@pytest.mark.asyncio
async def test_guard_counts_calls_tokens_and_models_per_caller():
    guard = ProviderGuard()
    model = _Model(callbacks=[guard])
    near = {"metadata": {"langgraph_checkpoint_ns": "near_rt_ric:1|model:2"}}
    review = {"tags": ["governance"],
              "metadata": {"langgraph_checkpoint_ns": "near_rt_ric:1|tools:3"}}
    await model.ainvoke("hi", config={"metadata": {"langgraph_checkpoint_ns": "monitoring:9"}})
    await model.ainvoke("hi", config=near)
    await model.ainvoke("hi", config=near)
    await model.ainvoke("hi", config=review)

    usage = guard.take_usage()
    assert usage["calls"] == 4
    assert usage["input_tokens"] == 40 and usage["output_tokens"] == 8
    assert usage["by_caller"]["near_rt_ric"]["calls"] == 2
    assert usage["by_caller"]["monitoring"]["input_tokens"] == 10
    assert usage["by_caller"]["governance_review"]["output_tokens"] == 2
    assert usage["models"] == {"gemini/backup": 4}
    assert guard.take_usage()["calls"] == 0          # reset per record


@pytest.mark.asyncio
async def test_failed_calls_are_counted_apart():
    guard = ProviderGuard()
    model = _Model(error=litellm.Timeout("slow", model="m", llm_provider="litellm_proxy"),
                   callbacks=[guard])
    with pytest.raises(litellm.Timeout):
        await model.ainvoke("hi")
    usage = guard.take_usage()
    assert usage["calls"] == 0 and usage["failed_calls"] == 1


def test_gateway_model_keeps_the_served_model(monkeypatch):
    _env(monkeypatch)
    llm = core_llm.create_llm()
    result = llm._create_chat_result({
        "model": "gemini/gemini-2.5-flash", "usage": {},
        "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
    })
    assert result.generations[0].message.response_metadata["served_by"] == "gemini/gemini-2.5-flash"
