"""test_bench_h2h.py — AUDIT #29 Sprint 4 head-to-head bench tests.

Two regression guards:

1. ``test_run_condition_returns_row_dict`` — ``run_condition``
   returns a dict whose keys match the CSV header. The h2h
   orchestrator's ``csv.DictWriter`` consumes this dict; the
   schema mismatch is a likely break point.

2. ``test_run_condition_ppl_delta_is_not_constant`` — the
   variance-over-``prompt_chars`` check in
   ``bench_h2h._check_variance`` is the Sprint 3 wire-in
   regression guard. We exercise it directly: a single
   ``run_condition`` call returns a PPL field; over two calls
   with different prompt lengths, the ``compression_ratio``
   field must vary (the apohara path is *prompt-length-aware*
   via the LLMLingua-2 call, so a varied prompt produces
   different ratios). If the path is broken the test fails
   loudly.

The tests do not pull torch / transformers: the bench_h2h
implementation returns ``NaN`` for PPL when the model is
absent, and the row still completes. The variance check is
deliberately lenient on the PPL column (NaN is allowed) and
strict on ``compression_ratio`` (a constant ratio is the
stub).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Make sure the in-package benchmarks module is importable when
# the test is run from the repo root via `pytest tests/`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apohara_context_forge.benchmarks.apohara2.bench_h2h import (  # noqa: E402
    CSV_HEADER,
    run_condition,
)
from apohara_context_forge.benchmarks.apohara2.bench_e2e import (  # noqa: E402
    run_condition as run_condition_e2e,
)


def test_run_condition_returns_row_dict() -> None:
    """``run_condition`` returns a dict with the CSV header keys."""
    row = run_condition("apohara", "hello world", n_tokens=64)
    assert isinstance(row, dict)
    for key in CSV_HEADER:
        assert key in row, f"row is missing CSV-header key {key!r}"
    assert row["system"] == "apohara"
    assert isinstance(row["duration_ms"], float)
    assert isinstance(row["vram_peak_gb"], float)
    assert isinstance(row["ppl_delta"], float)
    assert isinstance(row["compression_ratio"], float)
    assert isinstance(row["prompt_chars"], int)
    assert row["prompt_chars"] == len("hello world")


def test_run_condition_e2e_re_exports_same_signature() -> None:
    """``bench_e2e.run_condition`` re-exports the bench_h2h function.

    The Sprint 4 spec says the h2h script may also call
    ``bench_e2e.run_condition``; the public surface is one
    function with two import paths. A regression here means
    the re-export was lost in a refactor.
    """
    row = run_condition_e2e("turboquant", "hello world", n_tokens=64)
    assert isinstance(row, dict)
    assert row["system"] == "turboquant"
    for key in CSV_HEADER:
        assert key in row


def test_run_condition_two_rows_have_varying_compression_ratio() -> None:
    """Two runs with different prompt lengths must produce different ratios.

    The apohara path computes the compression ratio from the
    prompt length; if the LLMLingua-2 call falls back to the
    constant stub (``_STUB_RATIO = 0.55``), the ratio is
    identical for both prompts and this test fires.

    Note: the variance assertion is *only* on the
    ``compression_ratio`` field. ``duration_ms`` and
    ``vram_peak_gb`` are timing-dependent and can be near-
    constant on a fast machine; ``ppl_delta`` is NaN in the
    slim venv (no model), and ``prompt_chars`` is by
    construction different across the two calls.
    """
    short = "hello world"
    long_prompt = ("the model inference kernel uses pre-rope quantization "
                   "to compress the kv cache. the per-block codec with "
                   "group_size 256 projects to about 3940 mib at 10m docs "
                   "768-d 4-bit which is within the 4 gb budget. llmlingua-2 "
                   "removes low-information tokens while preserving the "
                   "semantic content that the downstream language model "
                   "needs to answer the user's question correctly.")
    row_short = run_condition("apohara", short, n_tokens=64)
    row_long = run_condition("apohara", long_prompt, n_tokens=64)
    assert row_short["prompt_chars"] != row_long["prompt_chars"]
    # The compression ratio must differ across the two runs.
    # If both fall back to the constant stub, this fires.
    assert row_short["compression_ratio"] != row_long["compression_ratio"], (
        "compression_ratio is constant across prompt lengths — the "
        "LLMLingua-2 path is not varying with prompt length. The "
        "Sprint 3 wire-in is likely broken."
    )
    # The apohara path always reports prompt_chars correctly.
    assert row_short["prompt_chars"] == len(short)
    assert row_long["prompt_chars"] == len(long_prompt)
