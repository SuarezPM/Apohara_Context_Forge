"""test_bank_test_helpers.py — US-008 helper unit tests (Phase 6 / Step 6.4).

The 4 helpers in `apohara_context_forge.benchmarks.apohara2._bank_test_helpers`
are deterministic primitives; this file pins their contracts so the
bank test's contract does not drift silently.

The bench's JSON summary (`bench_e2e.py`) consumes these helpers;
the bank test's gate is a function of the helper outputs, so a
silent helper regression would silently change the gate.
"""

from __future__ import annotations

import pytest

from apohara_context_forge.benchmarks.apohara2._bank_test_helpers import (
    downstream_lm_stub,
    holm_bonferroni,
    paired_ttest_pvalue,
    synthetic_batch,
)


# ---------------------------------------------------------------------------
# synthetic_batch
# ---------------------------------------------------------------------------


def test_synthetic_batch_shape():
    """synthetic_batch(10, 100, 42) returns 10 dicts with the expected keys."""
    batch = synthetic_batch(n_questions=10, n_context_tokens=100, seed=42)
    assert isinstance(batch, list)
    assert len(batch) == 10
    expected_keys = {"question", "context", "expected_doc_index", "expected_answer"}
    for item in batch:
        assert isinstance(item, dict)
        assert expected_keys.issubset(item.keys()), (
            f"missing keys: {expected_keys - item.keys()}"
        )


def test_synthetic_batch_question_prefix_of_context():
    """Each item's question is a short prefix of its context."""
    batch = synthetic_batch(n_questions=5, n_context_tokens=80, seed=7)
    for item in batch:
        # The question is the first 12 words of the context.
        assert item["question"] == " ".join(item["context"].split()[:12])


def test_synthetic_batch_expected_doc_index_monotonic():
    """expected_doc_index is one-to-one with the batch position."""
    batch = synthetic_batch(n_questions=20, n_context_tokens=50, seed=0)
    indices = [item["expected_doc_index"] for item in batch]
    assert indices == list(range(20))


def test_synthetic_batch_seed_determinism():
    """The same seed yields the same batch."""
    a = synthetic_batch(n_questions=4, n_context_tokens=30, seed=123)
    b = synthetic_batch(n_questions=4, n_context_tokens=30, seed=123)
    for x, y in zip(a, b):
        assert x == y


def test_synthetic_batch_different_seeds_yield_different_batches():
    """Different seeds yield different batches (very high probability)."""
    a = synthetic_batch(n_questions=4, n_context_tokens=30, seed=1)
    b = synthetic_batch(n_questions=4, n_context_tokens=30, seed=2)
    # The expected_answers are content-derived hashes; the chance
    # of collision is negligible.
    answers_a = {item["expected_answer"] for item in a}
    answers_b = {item["expected_answer"] for item in b}
    assert answers_a != answers_b


def test_synthetic_batch_rejects_invalid_args():
    """n_questions and n_context_tokens must be > 0."""
    with pytest.raises(ValueError):
        synthetic_batch(n_questions=0, n_context_tokens=10, seed=0)
    with pytest.raises(ValueError):
        synthetic_batch(n_questions=5, n_context_tokens=0, seed=0)


# ---------------------------------------------------------------------------
# downstream_lm_stub
# ---------------------------------------------------------------------------


def test_downstream_lm_stub_returns_string():
    """downstream_lm_stub('hello') returns a string."""
    out = downstream_lm_stub("hello")
    assert isinstance(out, str)
    assert out.startswith("answer-")


def test_downstream_lm_stub_deterministic():
    """Same prompt -> same answer."""
    a = downstream_lm_stub("the quick brown fox")
    b = downstream_lm_stub("the quick brown fox")
    assert a == b


def test_downstream_lm_stub_different_for_different_prompts():
    """Different prompts -> different answers (high probability)."""
    a = downstream_lm_stub("alpha bravo charlie")
    b = downstream_lm_stub("delta echo foxtrot")
    assert a != b


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------


def test_holm_bonferroni_basic_case():
    """Hand-verified known case from the spec / prereg.

    Input: [0.01, 0.04, 0.03, 0.005, 0.5], m=5, alpha=0.05.

    Sorted ascending: 0.005, 0.01, 0.03, 0.04, 0.5.

    Adjusted p-values:
      k=1: 0.005 * 5/1 = 0.025
      k=2: max(0.025, 0.01 * 5/2) = max(0.025, 0.025) = 0.025
      k=3: max(0.025, 0.03 * 5/3) = max(0.025, 0.05) = 0.05
      k=4: max(0.05, 0.04 * 5/4) = max(0.05, 0.05) = 0.05
      k=5: max(0.05, 0.5 * 5/5) = max(0.05, 0.5) = 0.5

    Rejection flags (alpha=0.05, stop at first non-rejection):
      k=1: 0.025 <= 0.05 -> rejected
      k=2: 0.025 <= 0.05 -> rejected
      k=3: 0.05 <= 0.05 -> rejected
      k=4: 0.05 <= 0.05 -> rejected
      k=5: 0.5 > 0.05 -> not rejected (stops the chain)

    Mapped back to original input order [0.01, 0.04, 0.03, 0.005, 0.5]:
      rejected = [True, True, True, True, False]
      adjusted = [0.025, 0.05, 0.05, 0.025, 0.5]
    """
    rejected, adjusted = holm_bonferroni([0.01, 0.04, 0.03, 0.005, 0.5])
    assert rejected == [True, True, True, True, False]
    # Adjusted values are clipped at 1.0; allow small float tolerance.
    assert adjusted == pytest.approx(
        [0.025, 0.05, 0.05, 0.025, 0.5], abs=1e-9
    )


def test_holm_bonferroni_all_rejected():
    """When all p-values are tiny, all hypotheses are rejected."""
    rejected, adjusted = holm_bonferroni([0.001, 0.002, 0.003, 0.004, 0.005])
    assert all(rejected)
    # The largest adjusted p is the 5th-rank one; <= alpha.
    assert max(adjusted) <= 0.05


def test_holm_bonferroni_none_rejected():
    """When all p-values exceed alpha, no hypothesis is rejected."""
    rejected, adjusted = holm_bonferroni([0.5, 0.6, 0.7, 0.8, 0.9])
    assert not any(rejected)
    # Adjusted values are clipped at 1.0.
    assert all(0.0 <= a <= 1.0 for a in adjusted)


def test_holm_bonferroni_stops_at_first_non_rejection():
    """After the first non-rejection, all subsequent are also retained.

    The classical Holm "stop" case: most p-values are tiny and
    rejected, but a single p-value in the middle is too large and
    stops the chain. With p = [0.001, 0.04, 0.001, 0.001, 0.001]
    sorted ascending = [0.001, 0.001, 0.001, 0.001, 0.04]:
      k=1: 0.001 * 5/1 = 0.005 -> reject
      k=2: 0.001 * 5/2 = 0.0025 -> max(0.005, 0.0025) = 0.005 -> reject
      k=3: 0.001 * 5/3 = 0.00167 -> max(0.005, 0.00167) = 0.005 -> reject
      k=4: 0.001 * 5/4 = 0.00125 -> max(0.005, 0.00125) = 0.005 -> reject
      k=5: 0.04 * 5/5 = 0.04 -> max(0.005, 0.04) = 0.04 -> reject (still <= 0.05)
    All 5 rejected. To exercise the "stop" case we need a p-value
    whose adjusted p exceeds alpha *before* the running max catches
    up; that requires a p-value > alpha * (m - k + 1) / m at rank
    k, which is rare in practice for any k < m. The classic stop
    case is built by a single large p in the middle with all
    smaller p's being very tiny: e.g. p = [0.0001, 0.0001, 0.06, 0.0001, 0.0001]
    (sorted: 0.0001, 0.0001, 0.0001, 0.0001, 0.06):
      k=1..4: 0.0001 * 5/k = 0.0005, 0.00025, ... -> all reject (cumulative max <= 0.0005).
      k=5: 0.06 * 5/5 = 0.06 -> max(0.0005, 0.06) = 0.06 -> NOT reject -> stop.
    All subsequent (none in m=5) would also be retained.
    """
    rejected, adjusted = holm_bonferroni(
        [0.0001, 0.0001, 0.06, 0.0001, 0.0001]
    )
    # The hypothesis with p = 0.06 is the only one not rejected.
    assert rejected[2] is False
    # The other four are rejected.
    for i in (0, 1, 3, 4):
        assert rejected[i] is True


def test_holm_bonferroni_empty_input():
    """Empty input returns empty outputs."""
    rejected, adjusted = holm_bonferroni([])
    assert rejected == []
    assert adjusted == []


def test_holm_bonferroni_single_value():
    """Single p-value: rejected iff p <= alpha."""
    rejected, adjusted = holm_bonferroni([0.01])
    assert rejected == [True]
    assert adjusted == pytest.approx([0.01], abs=1e-9)

    rejected, adjusted = holm_bonferroni([0.5])
    assert rejected == [False]
    assert adjusted == pytest.approx([0.5], abs=1e-9)


def test_holm_bonferroni_nan_treated_as_one():
    """NaN p-values are treated as 1.0 (cannot reject)."""
    rejected, adjusted = holm_bonferroni([float("nan"), 0.001])
    # The NaN hypothesis cannot be rejected.
    assert rejected[0] is False
    # The 0.001 hypothesis is rejected (it's the smallest p, ranked
    # first; adjusted = 0.001 * 2/1 = 0.002).
    assert rejected[1] is True
    assert adjusted[1] == pytest.approx(0.002, abs=1e-9)


def test_holm_bonferroni_clamps_out_of_range():
    """p-values > 1 are clamped to 1; p < 0 are clamped to 0.

    Input: [1.5, -0.1, 0.01] -> clamped to [1.0, 0.0, 0.01].
    Sorted ascending by index: original 1 (-0.1), original 2 (0.01),
    original 0 (1.5). Sorted p's: 0.0, 0.01, 1.0.
      k=1: 0.0 * 3/1 = 0.0 -> max(0, 0) = 0.0 -> reject.
      k=2: 0.01 * 3/2 = 0.015 -> max(0, 0.015) = 0.015 -> reject.
      k=3: 1.0 * 3/3 = 1.0 -> max(0.015, 1.0) = 1.0 -> not reject.
    Mapped back to original indices: [rejected[0], rejected[1], rejected[2]]
    = [rejected_at_sorted_pos_2, rejected_at_sorted_pos_0, rejected_at_sorted_pos_1]
    = [False (k=3), True (k=1), True (k=2)].
    """
    rejected, adjusted = holm_bonferroni([1.5, -0.1, 0.01])
    assert rejected == [False, True, True]


# ---------------------------------------------------------------------------
# paired_ttest_pvalue
# ---------------------------------------------------------------------------


def test_paired_ttest_pvalue_clear_difference():
    """Two clearly different distributions yield a small p-value."""
    p = paired_ttest_pvalue(
        [1.0, 2.0, 1.5, 2.5, 2.0], [1.0, 1.0, 1.0, 1.0, 1.0]
    )
    assert 0.0 <= p <= 1.0
    assert p < 0.05  # the test rejects the null at alpha = 0.05


def test_paired_ttest_pvalue_identical_distributions():
    """Two identical distributions yield p = 1.0 (cannot reject the null)."""
    p = paired_ttest_pvalue(
        [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]
    )
    assert p == 1.0


def test_paired_ttest_pvalue_raises_no_exception():
    """The function returns a float in [0, 1] for valid input."""
    p = paired_ttest_pvalue([0.0, 0.1, 0.2], [0.5, 0.6, 0.7])
    assert 0.0 <= p <= 1.0


def test_paired_ttest_pvalue_different_lengths_returns_one():
    """Mismatched lengths return p = 1.0 (degenerate case)."""
    p = paired_ttest_pvalue([1.0, 2.0], [1.0, 2.0, 3.0])
    assert p == 1.0


def test_paired_ttest_pvalue_empty_input_returns_one():
    """Empty input returns p = 1.0."""
    p = paired_ttest_pvalue([], [])
    assert p == 1.0


def test_paired_ttest_pvalue_single_sample_returns_one():
    """A single sample cannot be tested; return p = 1.0."""
    p = paired_ttest_pvalue([1.0], [2.0])
    assert p == 1.0
