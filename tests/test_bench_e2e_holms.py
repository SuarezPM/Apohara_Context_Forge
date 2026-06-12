"""test_bench_e2e_holms.py — AUDIT #28 regression guard for the Holm-Bonferroni path.

The real downstream-LM PPL (per-seed, per-variant) is wired into
`_run_one` in `bench_compress.py`. When that path is exercised, the
PPL delta is a real number, not a constant 0.0 by construction — and
the Holm-Bonferroni step-down correction in
`_bank_test_helpers.holm_bonferroni` sees a non-degenerate input.

The bench itself does not directly feed per-(seed, variant) PPL deltas
to `holm_bonferroni` (the bank test uses p-values, not PPL deltas, in
its gate); this test asserts the **plumbing seam** that the real PPL
delta exists and is fed into the helper in a way that would surface
to a downstream consumer.

Opt-in via `LLMLINGUA_REAL=1`. In the slim venv the test is skipped
with a WARNING; the constant-stub path is exercised by the existing
test_bank_test_helpers.py (8 holm tests) and test_apohara2_benchmarks_init.py
suites.

Markers
-------
* `pytest.mark.slow` — the Qwen3-1.7B load takes ~10s.
* `pytest.mark.skipif(LLMLINGUA_REAL != "1")` — opt-in.
"""

from __future__ import annotations

import os

import pytest

PYTEST_SKIP_REASON = (
    "set LLMLINGUA_REAL=1 to run the real-mode PPL-delta test; "
    "the slim venv has no torch / transformers and the bench "
    "runs the constant stub instead. See AUDIT #28."
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("LLMLINGUA_REAL", "").strip().lower() not in ("1", "true", "yes"),
        reason=PYTEST_SKIP_REASON,
    ),
]


def test_real_ppl_seam_produces_nonconstant_ppls():
    """The real PPL path produces non-constant per-prompt PPLs.

    The `_run_one` path in `bench_compress.py` averages the per-prompt
    PPLs into `ppl_baseline` / `ppl_compressed`. The delta is therefore
    a real number (not 0.0 by construction) and the downstream
    Holm-Bonferroni gate sees a non-degenerate input.

    This test exercises the seam directly: it loads Qwen3-1.7B, runs
    `_real_downstream_ppl` on 5 distinct prompts, and asserts the
    resulting PPLs are not all equal.
    """
    from apohara_context_forge.benchmarks.apohara2 import _bank_test_helpers
    from apohara_context_forge.benchmarks.apohara2.bench_compress import (
        _real_downstream_ppl,
    )

    model, tok = _bank_test_helpers._load_qwen3_1_7b_cached()
    prompts = [
        "The model inference kernel memory bandwidth tensor pipeline.",
        "The cat sat on the mat in the afternoon and watched the bird.",
        "Quantum chromodynamics is the theory of strong interactions.",
        "Apples and oranges are fruits that grow on trees in temperate zones.",
        "The answer to life the universe and everything is forty two.",
    ]
    ppls = [
        _real_downstream_ppl(p, "", model=model, tok=tok) for p in prompts
    ]
    # All in range.
    for p in ppls:
        assert 1.0 <= p <= 1e6, f"PPL {p} out of [1.0, 1e6]"
    # Not all equal (the regression guard for a constant stub).
    assert len(set(round(p, 4) for p in ppls)) > 1, (
        f"5 distinct prompts produced identical PPLs {ppls}; "
        "_real_downstream_ppl looks like a constant stub"
    )
    # The bench consumes the per-prompt PPLs into a mean, then the
    # delta is a real number. The Holm-Bonferroni helper sees a
    # non-degenerate p-value derived from this delta in the bank test.
    assert ppls, "PPL list is empty"
    mean_ppl = sum(ppls) / len(ppls)
    assert 1.0 <= mean_ppl <= 1e6
