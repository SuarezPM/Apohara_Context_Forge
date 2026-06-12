"""test_bench_compress_real_ppl.py — AUDIT #28 regression guard.

Asserts `bench_compress._real_downstream_ppl` returns a finite float
in `[1.0, 1e6]` for a canned prompt+completion, and that two distinct
completions on the same prompt produce distinct PPL values (i.e. the
function is not a constant stub).

Opt-in via `LLMLINGUA_REAL=1`. In the slim venv (no torch /
transformers) the test is skipped with a WARNING; the constant-stub
path is exercised by the existing `test_bench_compress.py` suite.

Markers
-------
* `pytest.mark.slow` — the Qwen3-1.7B load takes ~10s.
* `pytest.mark.skipif(LLMLINGUA_REAL != "1")` — opt-in.
"""

from __future__ import annotations

import os

import pytest

PYTEST_SKIP_REASON = (
    "set LLMLINGUA_REAL=1 to run the real-mode PPL test; the "
    "slim venv has no torch / transformers and the bench "
    "runs the constant stub instead."
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("LLMLINGUA_REAL", "").strip().lower() not in ("1", "true", "yes"),
        reason=PYTEST_SKIP_REASON,
    ),
]


def test_real_downstream_ppl_returns_finite_in_range():
    """`_real_downstream_ppl` returns a finite float in [1.0, 1e6]."""
    from apohara_context_forge.benchmarks.apohara2 import _bank_test_helpers
    from apohara_context_forge.benchmarks.apohara2.bench_compress import (
        _real_downstream_ppl,
    )

    model, tok = _bank_test_helpers._load_qwen3_1_7b_cached()
    prompt = "The quick brown fox jumps over the lazy dog. " * 8
    ppl = _real_downstream_ppl(prompt, "", model=model, tok=tok)
    assert isinstance(ppl, float)
    assert ppl == ppl  # not NaN
    assert 1.0 <= ppl <= 1e6, f"PPL {ppl} out of [1.0, 1e6]"


def test_real_downstream_ppl_differs_across_completions():
    """Two distinct completions on the same prompt produce distinct PPLs.

    Sanity check that the function is not a constant stub. The
    """

    from apohara_context_forge.benchmarks.apohara2 import _bank_test_helpers
    from apohara_context_forge.benchmarks.apohara2.bench_compress import (
        _real_downstream_ppl,
    )

    model, tok = _bank_test_helpers._load_qwen3_1_7b_cached()
    prompt = "The quick brown fox jumps over the lazy dog. " * 8
    ppl_a = _real_downstream_ppl(prompt, " The cat sat on the mat.", model=model, tok=tok)
    ppl_b = _real_downstream_ppl(prompt, " Quantum chromodynamics is the theory.", model=model, tok=tok)
    assert ppl_a != ppl_b, (
        f"distinct completions produced identical PPL {ppl_a}; "
        "_real_downstream_ppl looks like a constant stub"
    )


def test_real_downstream_ppl_handles_empty_completion():
    """Empty completion still returns a finite float in range.

    The bench uses an empty completion as the canonical "no-completion"
    case for the synthetic corpus; the function must not crash on it.
    """
    from apohara_context_forge.benchmarks.apohara2 import _bank_test_helpers
    from apohara_context_forge.benchmarks.apohara2.bench_compress import (
        _real_downstream_ppl,
    )

    model, tok = _bank_test_helpers._load_qwen3_1_7b_cached()
    ppl = _real_downstream_ppl("hello world", "", model=model, tok=tok)
    assert 1.0 <= ppl <= 1e6
