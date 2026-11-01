"""Retries and the shared rate limiter of src/core/llm.create_llm."""

import src.core.llm as core_llm


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
    assert llm.rate_limiter is None and llm.max_retries == 6
