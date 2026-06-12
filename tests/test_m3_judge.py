"""Tests for the M3 LLM-as-judge client (US-005 / Phase 3 Step 3.2,
extended by US-011 for the real HTTP wire-up).

Five concerns:
  1. `M3Judge(model_id=..., base_url=...)` constructs without error.
  2. `M3Judge().judge("hello world")` returns a `JudgeResult` with
     `score: Optional[float]`, `raw: str`, `prompt_tokens: int`,
     `completion_tokens: int`, `degraded: bool`.
  3. The greedy decoding pins (`M3_TEMPERATURE == 0.0`,
     `M3_TOP_P == 1.0`, `M3_TOP_K == 1`) are constants.
  4. `M3_VERSION` is a non-empty string (the TODO placeholder
     asserted to be non-empty so a future bump is a deliberate edit).
  5. The default `base_url` and `model_id` are wired to the env vars.

US-011 additions (wire-up + fallback envelope):
  6. `judge()` actually calls `M3_BASE_URL/v1/chat/completions` over
     httpx with the greedy-decoding pins in the body (mocked httpx).
  7. When httpx raises (connection error, timeout, anything), the
     judge returns `score=None`, `raw='<error: M3 unreachable: ...>'`,
     `degraded=True` — does NOT raise. The bench's deterministic
     local judge takes over from there.
"""

from __future__ import annotations

import os

import pytest

from apohara_context_forge.eval import M3Judge, JudgeResult
from apohara_context_forge.eval.m3_judge import (
    M3_TEMPERATURE,
    M3_TOP_P,
    M3_TOP_K,
    M3_VERSION,
)


class TestM3JudgeConstruction:
    """Construction and defaults."""

    def test_construct_with_explicit_args(self):
        j = M3Judge(model_id="stub", base_url="http://localhost:0")
        assert j.model_id == "stub"
        assert j.base_url == "http://localhost:0"

    def test_construct_with_no_args_uses_module_defaults(self):
        # Strip env vars so the test is deterministic.
        old_model = os.environ.pop("M3_MODEL_ID", None)
        old_base = os.environ.pop("M3_BASE_URL", None)
        try:
            j = M3Judge()
            assert j.model_id == M3_VERSION
            assert j.base_url == "http://localhost:8000"
        finally:
            if old_model is not None:
                os.environ["M3_MODEL_ID"] = old_model
            if old_base is not None:
                os.environ["M3_BASE_URL"] = old_base

    def test_construct_honors_env_var_model_id(self, monkeypatch):
        monkeypatch.setenv("M3_MODEL_ID", "from-env-1")
        j = M3Judge()
        assert j.model_id == "from-env-1"

    def test_construct_honors_env_var_base_url(self, monkeypatch):
        monkeypatch.setenv("M3_BASE_URL", "http://from-env:9000")
        j = M3Judge()
        assert j.base_url == "http://from-env:9000"

    def test_construct_explicit_args_override_env(self, monkeypatch):
        monkeypatch.setenv("M3_MODEL_ID", "from-env")
        j = M3Judge(model_id="explicit", base_url="http://explicit:1")
        assert j.model_id == "explicit"
        assert j.base_url == "http://explicit:1"


class TestM3JudgeJudgeCall:
    """`judge()` returns a properly shaped JudgeResult."""

    def test_judge_returns_judgeresult_with_correct_field_types(self):
        j = M3Judge(model_id="stub", base_url="http://localhost:0")
        result = j.judge("hello world")
        assert isinstance(result, JudgeResult)
        # score is Optional[float] (None on the unreachable path)
        assert result.score is None or isinstance(result.score, float)
        assert isinstance(result.raw, str)
        assert isinstance(result.prompt_tokens, int)
        assert isinstance(result.completion_tokens, int)
        assert isinstance(result.degraded, bool)

    def test_judge_prompt_tokens_zero_on_unreachable_envelope(self):
        # When M3 is unreachable the envelope is the empty envelope
        # (score=None, raw=error, tokens=0, degraded=True). The
        # bench's deterministic local judge then supplies the real
        # token count for its own metrics; the envelope is the
        # wire-up's best-effort signal that the HTTP call did NOT
        # land.
        j = M3Judge(model_id="stub", base_url="http://localhost:0")
        result = j.judge("one two three four five")
        assert result.prompt_tokens == 0

    def test_judge_raw_carries_error_envelope_on_unreachable(self):
        j = M3Judge(model_id="stub", base_url="http://localhost:0")
        result = j.judge("a longer prompt than 100 characters " * 10)
        # The fallback envelope begins with the error marker so the
        # bench's deterministic local judge can recognize it.
        assert result.raw.startswith("<error: M3 unreachable")
        assert result.degraded is True

    def test_judge_with_default_system_prompt(self):
        j = M3Judge(model_id="stub", base_url="http://localhost:0")
        # The system arg is accepted and threaded into the request body.
        # When the endpoint is unreachable, the call still returns a
        # well-formed envelope.
        result = j.judge("hello", system="You are a strict grader.")
        assert isinstance(result, JudgeResult)

    def test_judge_completion_tokens_is_zero_on_unreachable_envelope(self):
        j = M3Judge(model_id="stub", base_url="http://localhost:0")
        result = j.judge("hello world")
        assert result.completion_tokens == 0


class TestM3GreedyPins:
    """The greedy decoding pins that kill non-determinism."""

    def test_temperature_is_zero(self):
        assert M3_TEMPERATURE == 0.0

    def test_top_p_is_one(self):
        assert M3_TOP_P == 1.0

    def test_top_k_is_one(self):
        assert M3_TOP_K == 1


class TestM3VersionPin:
    """The version pin is a TODO placeholder; the contract is non-empty."""

    def test_version_is_a_non_empty_string(self):
        assert isinstance(M3_VERSION, str)
        assert len(M3_VERSION) > 0

    def test_version_contains_a_year_or_placeholder(self):
        # The current pin is a TODO placeholder ("MiniMax-M3-2026-05-XX").
        # The contract is: it parses as a non-empty identifier that
        # future bumps can replace wholesale.
        assert "M3" in M3_VERSION


# ---------------------------------------------------------------------------
# US-011 wire-up + fallback envelope tests
# ---------------------------------------------------------------------------


def test_m3_judge_wire_up_calls_http_endpoint(monkeypatch):
    """Mock httpx and assert M3Judge.judge() posts to M3_BASE_URL/v1/chat/completions
    with the greedy-decoding pin in the body."""
    from unittest.mock import MagicMock, patch
    import json

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "0.87\nreasoning here..."}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response) as mock_post:
        judge = M3Judge(model_id="MiniMax-M3-test", base_url="http://test:1234")
        result = judge.judge("evaluate this")

    # The call landed on httpx.post
    assert mock_post.call_count == 1
    call_args = mock_post.call_args
    url = call_args[0][0]
    body = call_args[1]["json"]

    # URL is M3_BASE_URL + /v1/chat/completions
    assert url == "http://test:1234/v1/chat/completions"

    # Body has greedy-decoding pins
    assert body["model"] == "MiniMax-M3-test"
    assert body["temperature"] == 0.0
    assert body["top_p"] == 1.0
    assert body["top_k"] == 1

    # The M3 system message is the default
    assert body["messages"][0]["role"] == "system"
    assert "evaluator" in body["messages"][0]["content"].lower()
    # The user message is the prompt
    assert body["messages"][1]["content"] == "evaluate this"

    # Result: score parsed from first line of content, raw kept, tokens captured
    assert result.score == 0.87
    assert result.raw == "0.87\nreasoning here..."
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 50
    assert result.degraded is False


def test_m3_judge_falls_back_when_unreachable():
    """If httpx.post raises (connection error, timeout, anything), the judge returns
    a JudgeResult with score=None, raw='<error: M3 unreachable: ...>', degraded=True.
    The bench's deterministic local judge takes over from there.
    """
    from unittest.mock import patch
    import httpx

    with patch("httpx.post", side_effect=httpx.ConnectError("Connection refused")):
        judge = M3Judge(base_url="http://does-not-exist:9999")
        result = judge.judge("evaluate this")

    assert result.score is None
    assert "M3 unreachable" in result.raw
    assert "ConnectError" in result.raw
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.degraded is True


def test_m3_judge_uses_env_var_m3_base_url(monkeypatch):
    """The M3_BASE_URL env var overrides the default."""
    monkeypatch.setenv("M3_BASE_URL", "http://from-env:5555")
    judge = M3Judge()
    assert judge.base_url == "http://from-env:5555"


def test_m3_judge_parse_score_handles_malformed():
    """If M3 returns something that's not a number, parse_score returns None (the bench
    falls back to the deterministic score).
    """
    assert M3Judge._parse_score("") is None
    assert M3Judge._parse_score("not a number\nreasoning") is None
    assert M3Judge._parse_score("0.5") == 0.5
    assert M3Judge._parse_score("  0.7  \n  rest") == 0.7
