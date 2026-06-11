"""Tests for the M3 LLM-as-judge client (US-005 / Phase 3 Step 3.2).

Five concerns:
  1. `M3Judge(model_id=..., base_url=...)` constructs without error.
  2. `M3Judge().judge("hello world")` returns a `JudgeResult` with
     `score: float`, `raw: str`, `prompt_tokens: int`,
     `completion_tokens: int`.
  3. The greedy decoding pins (`M3_TEMPERATURE == 0.0`,
     `M3_TOP_P == 1.0`, `M3_TOP_K == 1`) are constants.
  4. `M3_VERSION` is a non-empty string (the TODO placeholder
     asserted to be non-empty so a future bump is a deliberate edit).
  5. The default `base_url` and `model_id` are wired to the env vars.
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
        assert isinstance(result.score, float)
        assert isinstance(result.raw, str)
        assert isinstance(result.prompt_tokens, int)
        assert isinstance(result.completion_tokens, int)

    def test_judge_prompt_tokens_counts_words(self):
        j = M3Judge(model_id="stub", base_url="http://localhost:0")
        result = j.judge("one two three four five")
        # 5 whitespace-split tokens in the stub.
        assert result.prompt_tokens == 5

    def test_judge_raw_echoes_prompt_prefix(self):
        j = M3Judge(model_id="stub", base_url="http://localhost:0")
        result = j.judge("a longer prompt than 100 characters " * 10)
        # The stub echoes the first 100 chars of the prompt.
        assert result.raw.startswith("M3 judge stub: ")

    def test_judge_with_default_system_prompt(self):
        j = M3Judge(model_id="stub", base_url="http://localhost:0")
        # The system arg is accepted but not used in the stub.
        result = j.judge("hello", system="You are a strict grader.")
        assert isinstance(result, JudgeResult)

    def test_judge_completion_tokens_is_zero_in_stub(self):
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
