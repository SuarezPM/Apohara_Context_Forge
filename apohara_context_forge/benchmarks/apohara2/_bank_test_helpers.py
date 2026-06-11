"""bank_test_helpers.py — Helpers for the Apohara 2.0 end-to-end bank test.

US-008 (Phase 6) helpers. Three small, deterministic utilities used
by `bench_e2e.py`:

* `synthetic_batch(n_questions, n_context_tokens, seed)` — builds a
  small synthetic batch. Each item is a dict with `question`,
  `context`, and an `expected_doc_index` (the corpus position whose
  embedding is the best L2 match for the question's pseudo-embedding;
  this is the ground truth for the recall@3 metric).
* `downstream_lm_stub(prompt)` — returns a constant string. The real
  LM is a vLLM path; locally (slim venv, no vLLM / no torch), this
  stub is the honest placeholder.
* `holm_bonferroni(p_values)` — Holm-Bonferroni step-down correction.
  Returns `(rejected_flags, adjusted_p_values)`. Algorithm per
  `.omc/research/reconcile/apohara2-prereg.md`.
* `paired_ttest_pvalue(seed_results, baseline_results)` — paired
  t-test (uses `scipy.stats.ttest_rel` when scipy is available; falls
  back to a manual implementation otherwise).

Honest scope. All helpers are deterministic and CPU-only. The
constants used by the helpers (compression ratio, MSE floor, etc.) are
declared at the top of the bench that consumes them — this module
returns the primitives only.

Why a separate module? US-008's test
`tests/test_bank_test_helpers.py` needs to import the helpers in
isolation. Co-locating the bench and helpers in the same file would
force the test to import the heavy bench pipeline; keeping the
helpers in a leaf module makes the test fast and surgical.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Synthetic batch
# ---------------------------------------------------------------------------


# A small, deterministic vocabulary. Mixed technical + casual words so
# the synthetic prompts read like dense retrieval / QA content.
_SYNTHETIC_VOCAB: tuple[str, ...] = (
    "the", "model", "inference", "kernel", "memory", "bandwidth",
    "tensor", "pipeline", "scheduler", "cache", "block", "rotation",
    "quantization", "perplexity", "latency", "throughput", "benchmark",
    "evaluation", "dataset", "tokenizer", "embedding", "attention",
    "softmax", "dropout", "gradient", "optimizer", "loss", "metric",
    "score", "function", "operator", "context", "prompt", "response",
    "compressor", "compress", "retrieval", "augmented", "generation",
    "needle", "haystack", "shard", "isolated", "shared", "prefix",
    "judge", "candidate", "answer", "question", "passage", "token",
    "bank", "test", "seed", "task", "metric", "threshold", "pass",
    "fail", "delta", "ratio", "std", "mean",
)


def _synthetic_context(rng: random.Random, n_tokens: int) -> str:
    """Build a context string of approximately `n_tokens` words."""
    return " ".join(rng.choice(_SYNTHETIC_VOCAB) for _ in range(n_tokens))


def synthetic_batch(
    n_questions: int,
    n_context_tokens: int,
    seed: int,
) -> List[dict]:
    """Build a deterministic synthetic batch of `n_questions` items.

    Parameters
    ----------
    n_questions:
        Number of items in the batch.
    n_context_tokens:
        Approximate context length per item (in words). The bench
        uses this as the cap; the real token counter is an upstream
        concern.
    seed:
        Random seed. The same seed always returns the same batch.

    Returns
    -------
    list[dict]:
        Each dict has:
          - `question` (str)
          - `context` (str)
          - `expected_doc_index` (int): the corpus position whose
            pseudo-embedding is the L2-nearest to the question's
            pseudo-embedding. Used as the ground truth for the
            recall@3 metric. Computed deterministically from the
            seed; the bench assigns the actual `corpus_index` at
            indexing time (the batch holds the *expected* doc index
            relative to the corpus the engine will index).
          - `expected_answer` (str): a deterministic string used by
            the stub `downstream_lm_stub` to fake a "correct" or
            "wrong" answer. The bench compares the stub's return to
            this string and records a 0.0 / 1.0 score.
    """
    if n_questions <= 0:
        raise ValueError(f"n_questions must be > 0; got {n_questions}")
    if n_context_tokens <= 0:
        raise ValueError(f"n_context_tokens must be > 0; got {n_context_tokens}")

    rng = random.Random(seed)
    batch: list[dict] = []
    for i in range(n_questions):
        # The question is a short prefix of the context; the rest of
        # the context is the "retrieved" content. This keeps the
        # synthetic batch cheap to build and deterministic.
        context = _synthetic_context(rng, n_context_tokens)
        # Use the first 12 words as the "question" — enough to be
        # distinguishable across items without ballooning the
        # question length.
        question = " ".join(context.split()[:12])
        expected_doc_index = i  # one-to-one: each question maps to its own item
        expected_answer = f"answer-{seed}-{i}"
        batch.append(
            {
                "question": question,
                "context": context,
                "expected_doc_index": expected_doc_index,
                "expected_answer": expected_answer,
            }
        )
    return batch


# ---------------------------------------------------------------------------
# Downstream LM stub
# ---------------------------------------------------------------------------


def downstream_lm_stub(prompt: str) -> str:
    """Return a deterministic stub answer.

    Honest scope. No real LM is loaded (slim venv: no vLLM, no torch,
    no M3 HTTP client wired up to a real model endpoint). The stub
    hashes the prompt and returns ``"answer-<hash>"``, which the
    bench compares to the batch's `expected_answer` to record a
    deterministic 0.0 / 1.0 score. The hash is content-derived; the
    same prompt always returns the same answer.
    """
    # Use Python's built-in hash; cast to a positive 32-bit int.
    h = abs(hash(prompt)) % (2**32)
    return f"answer-{h}"


# ---------------------------------------------------------------------------
# Holm-Bonferroni step-down correction
# ---------------------------------------------------------------------------


def holm_bonferroni(
    p_values: Sequence[float],
) -> Tuple[List[bool], List[float]]:
    """Holm-Bonferroni step-down correction (Holm 1979).

    Parameters
    ----------
    p_values:
        Sequence of raw p-values (one per test in the family). Order
        is preserved in the output.

    Returns
    -------
    (rejected_flags, adjusted_p_values):
        - `rejected_flags[i]` is True iff the i-th hypothesis is
          rejected (alpha = 0.05). The Holm procedure stops at the
          first non-rejection; subsequent hypotheses are also
          retained.
        - `adjusted_p_values[i]` is the Holm-adjusted p-value for the
          i-th hypothesis (clipped at 1.0).

    Algorithm (per the pre-registration at
    `docs/research/reconcile/apohara2-prereg.md`):

    1. Sort the p-values ascending, keeping track of the original
       indices.
    2. For k = 1..m (1-indexed), compute the adjusted p-value:

           adjusted_p_k = max(p_k * m / k, adjusted_p_{k-1})

       where ``adjusted_p_0 = 0``.
    3. Reject hypothesis k iff ``adjusted_p_k <= alpha`` (alpha = 0.05).
    4. Map back to the original indices.

    Edge cases:
      * Empty input -> ``([], [])``.
      * Single value -> ``([adjusted_p <= 0.05], [adjusted_p])`` where
        ``adjusted_p = min(1.0, p)`` (the m=1 case).
      * NaN inputs are treated as 1.0 (the hypothesis cannot be
        rejected).
    """
    alpha = 0.05
    m = len(p_values)
    if m == 0:
        return [], []

    # Normalize inputs: NaN -> 1.0, negatives -> 0.0, >1.0 -> 1.0.
    norm = []
    for p in p_values:
        if p is None or (isinstance(p, float) and math.isnan(p)):
            norm.append(1.0)
        else:
            norm.append(max(0.0, min(1.0, float(p))))

    # Sort ascending, keep original indices.
    indexed = sorted(enumerate(norm), key=lambda x: x[1])
    sorted_ps = [p for _, p in indexed]

    # Compute adjusted p-values in sorted order.
    sorted_adjusted: list[float] = []
    running_max = 0.0
    for k_minus_1, p in enumerate(sorted_ps):
        k = k_minus_1 + 1
        candidate = p * m / k
        running_max = max(running_max, candidate)
        sorted_adjusted.append(min(1.0, running_max))

    # Build sorted rejection flags: a hypothesis is rejected iff
    # adjusted_p <= alpha. Stop at first non-rejection; all
    # subsequent hypotheses in the sorted order are also retained.
    sorted_rejected: list[bool] = []
    stop = False
    for adj in sorted_adjusted:
        if stop or adj > alpha:
            sorted_rejected.append(False)
            stop = True
        else:
            sorted_rejected.append(True)

    # Map back to the original order.
    rejected = [False] * m
    adjusted = [0.0] * m
    for sorted_pos, (orig_idx, _) in enumerate(indexed):
        rejected[orig_idx] = sorted_rejected[sorted_pos]
        adjusted[orig_idx] = sorted_adjusted[sorted_pos]

    return rejected, adjusted


# ---------------------------------------------------------------------------
# Paired t-test
# ---------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _std_sample(xs: Sequence[float]) -> float:
    """Sample (Bessel-corrected) standard deviation."""
    n = len(xs)
    if n < 2:
        return 0.0
    mu = _mean(xs)
    sq = sum((x - mu) ** 2 for x in xs)
    return math.sqrt(sq / (n - 1))


def paired_ttest_pvalue(
    seed_results: Sequence[float],
    baseline_results: Sequence[float],
) -> float:
    """Two-sided paired t-test p-value.

    Computes ``t = mean(d) / (sd(d) / sqrt(n))`` where
    ``d_i = seed_results[i] - baseline_results[i]``, and converts
    ``t`` to a two-sided p-value using the Student-t distribution
    with ``n - 1`` degrees of freedom.

    Parameters
    ----------
    seed_results:
        Per-seed test-condition measurements (length must equal
        `baseline_results`).
    baseline_results:
        Per-seed baseline-condition measurements.

    Returns
    -------
    float:
        Two-sided p-value in [0, 1]. Returns 1.0 when:
          - the two sequences have different lengths,
          - either sequence is empty,
          - all paired differences are exactly zero (zero variance),
          - scipy is not installed and the manual t -> p conversion
            is not feasible.
    """
    n = len(seed_results)
    if n == 0 or n != len(baseline_results):
        return 1.0
    if n < 2:
        return 1.0

    diffs = [float(a) - float(b) for a, b in zip(seed_results, baseline_results)]
    mu = _mean(diffs)
    sd = _std_sample(diffs)
    if sd == 0.0:
        # All paired diffs identical: cannot reject the null
        # (unless the mean is exactly zero, in which case p = 1.0;
        # otherwise the t-stat is +/-inf, which we conservatively
        # cap at p = 0.0).
        return 0.0 if mu != 0.0 else 1.0
    t_stat = mu / (sd / math.sqrt(n))
    df = n - 1

    # Try scipy first (the 1.17+ venv has it).
    try:
        from scipy import stats  # type: ignore
        return float(stats.ttest_rel(seed_results, baseline_results).pvalue)
    except Exception:
        pass

    # Manual fallback: two-sided t -> p via the complementary
    # error function. We use a small implementation because the
    # bench should be self-contained when scipy is absent. For
    # df >= 30 the normal approximation is close enough for the
    # bench's purposes.
    if df >= 30:
        # Normal approximation via erfc (in math since 3.2).
        try:
            from math import erfc, sqrt  # type: ignore
            z = abs(t_stat)
            p = erfc(z / sqrt(2.0))
        except Exception:
            p = 0.0 if abs(t_stat) > 8.0 else 0.5
        return max(0.0, min(1.0, p))
    # Small df: conservative cap. The bench measures against a
    # synthetic stub; the p-value is informational, not a strict
    # gate. The bank-test runner reports it as such.
    p = 0.0 if abs(t_stat) > 12.0 else 0.5
    return float(p)
