"""test_bench_e2e_compression_ratio.py — AUDIT #28 regression guard.

Asserts `bench_e2e._compression_ratio` returns distinct values for
distinct prompts (i.e. the function is not a constant stub). Two
distinct prompts of comparable length should compress to different
ratios; the test asserts the difference is more than 0.05 to
distinguish a real compressor from the constant `_STUB_RATIO = 0.55`.

Opt-in via `LLMLINGUA_REAL=1`. In the slim venv (no torch / onnxruntime
/ cached XLM-RoBERTa model) the test is skipped with a WARNING; the
constant-stub path is exercised by the existing `test_apohara2_benchmarks_init.py`
suite (the `test_bench_e2e_runs_and_emits_json` test runs the bench
in `--downstream_lm stub` mode and asserts the JSON contract).

Markers
-------
* `pytest.mark.slow` — the XLM-RoBERTa load + first forward pass take ~5-10s.
* `pytest.mark.skipif(LLMLINGUA_REAL != "1")` — opt-in.
"""

from __future__ import annotations

import os

import pytest

PYTEST_SKIP_REASON = (
    "set LLMLINGUA_REAL=1 to run the real-mode compression ratio test; "
    "the slim venv has no onnxruntime / cached XLM-RoBERTa and the bench "
    "runs the constant stub instead. See AUDIT #28."
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("LLMLINGUA_REAL", "").strip().lower() not in ("1", "true", "yes"),
        reason=PYTEST_SKIP_REASON,
    ),
]


def _two_distinct_prompts() -> tuple[str, str]:
    """Two vocab-flavored prompts with distinct content."""
    p1 = (
        "The model inference kernel memory bandwidth tensor pipeline "
        "scheduler cache block rotation quantization perplexity "
        "latency throughput benchmark evaluation dataset tokenizer "
        "embedding attention softmax dropout gradient optimizer loss."
    ) * 2
    p2 = (
        "The cat sat on the mat in the afternoon and watched the bird "
        "outside the kitchen window while the kettle boiled on the "
        "stove and the radio played an old song about the rain in "
        "Spain and the plain truth about a paper plane."
    ) * 2
    return p1, p2


def test_compression_ratio_returns_distinct_values_for_distinct_prompts():
    """Two distinct prompts of comparable length produce distinct ratios.

    Sanity check: the function is not a constant stub. The XLM-RoBERTa
    LLMLingua-2 model assigns different keep probabilities to
    different content (tech vs. prose), so the resulting compression
    ratios must differ.
    """
    from apohara_context_forge.benchmarks.apohara2.bench_e2e import (
        _compression_ratio,
    )

    p1, p2 = _two_distinct_prompts()
    r1 = _compression_ratio(p1, rate=0.5)
    r2 = _compression_ratio(p2, rate=0.5)
    # Both must be in [0, 1] — compression can't drop < 0% or > 100%.
    assert 0.0 <= r1 <= 1.0, f"r1={r1} out of [0, 1]"
    assert 0.0 <= r2 <= 1.0, f"r2={r2} out of [0, 1]"
    # The values must differ by more than 0.05 to distinguish the real
    # compressor from the constant stub (the stub always returns 0.55).
    assert abs(r1 - r2) > 0.05, (
        f"distinct prompts produced compression ratios {r1:.4f} and "
        f"{r2:.4f} (|delta|={abs(r1 - r2):.4f}); looks like a constant stub"
    )


def test_compression_ratio_empty_prompt_returns_zero():
    """Empty prompt returns 0.0 (no compression possible)."""
    from apohara_context_forge.benchmarks.apohara2.bench_e2e import (
        _compression_ratio,
    )

    r = _compression_ratio("", rate=0.5)
    assert r == 0.0
