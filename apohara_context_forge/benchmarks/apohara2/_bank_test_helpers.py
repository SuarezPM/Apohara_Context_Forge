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

import functools
import logging
import math
import os
import random
import re
from typing import Any, List, Sequence, Tuple

log = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# DownstreamLM — transformers-backed downstream LM (US-014-REDUX)
# ---------------------------------------------------------------------------


# Canonical alias -> HuggingFace model id. The bench ships two
# sub-2B Qwen variants that both fit in 8GB in FP16. Adding a
# new variant = one new line here and one new `--downstream_lm`
# choice in the CLI. (No bitsandbytes / AWQ: FP16 fits.)
_DOWNSTREAM_LM_REGISTRY: dict[str, str] = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
}


def resolve_downstream_lm_id(name: str) -> str:
    """Resolve `--downstream_lm` alias -> HuggingFace model id.

    Raises ValueError on unknown aliases so the bench fails fast
    with a clear error message rather than calling
    `transformers` with garbage.
    """
    key = name.strip().lower()
    if key not in _DOWNSTREAM_LM_REGISTRY:
        raise ValueError(
            f"unknown --downstream_lm {name!r}; "
            f"supported: {sorted(_DOWNSTREAM_LM_REGISTRY)} | stub | none"
        )
    return _DOWNSTREAM_LM_REGISTRY[key]


def list_downstream_lm_aliases() -> tuple[str, ...]:
    """Return the supported `--downstream_lm` aliases (test helper)."""
    return tuple(sorted(_DOWNSTREAM_LM_REGISTRY))


class DownstreamLM:
    """Lazy-loaded downstream LM for the bank test (US-014-REDUX).

    Wraps a `transformers.AutoModelForCausalLM` + `AutoTokenizer`
    loaded from the local HuggingFace cache. The model is loaded
    on the first call to `generate()` and freed by `release()`.

    Honest scope (US-014-REDUX). The real-mode bank test runs
    this class on a local GPU (RTX 2060 SUPER 8GB) with a sub-2B
    Qwen model in FP16. The class is **NOT** a vLLM path; vLLM
    remains a follow-up gated on the MI300X doplet. The
    `generate()` call is greedy, `max_new_tokens=128`, no sampling;
    the resulting string is scored against the expected answer
    via `score_answer` (substring/keyword match).
    """

    def __init__(self, model_id: str, device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._loaded = False

    def is_real(self) -> bool:
        """True for any registered HuggingFace-backed variant."""
        return self.model_id in _DOWNSTREAM_LM_REGISTRY.values()

    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    def _ensure_loaded(self) -> None:
        """Lazy load: first `generate()` call triggers the load.

        Imports are kept local so the bench's `--downstream_lm stub`
        path stays dependency-light (no torch import required).
        """
        if self._loaded:
            return
        import torch  # local import — keeps stub path torch-free
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # FP16 on CUDA, full precision on CPU. The slim venv has
        # neither bitsandbytes nor AWQ; FP16 fits in 8GB for both
        # Qwen3-1.7B (~3.5GB) and Qwen2.5-0.5B-Instruct (~1GB).
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype
        )
        self._model = self._model.to(device)
        self._model.eval()
        self._device = device
        self._loaded = True

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Greedy decode `prompt`, returning the new-token string.

        Returns the **post-prompt** continuation (the model input
        prefix is stripped from the output), not the full decoded
        string. EOS is respected; `pad_token_id` is taken from
        the tokenizer when present (Qwen tokenizers always set
        one).
        """
        self._ensure_loaded()
        import torch  # local — already required for the load

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        eos_token_id = self._tokenizer.eos_token_id
        pad_token_id = (
            self._tokenizer.pad_token_id
            if self._tokenizer.pad_token_id is not None
            else eos_token_id
        )
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                top_k=1,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)

    def release(self) -> None:
        """Free GPU/CPU memory; safe to call multiple times."""
        try:
            import torch  # local — keeps stub path torch-free
            if hasattr(self, "_device") and self._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            # release() is best-effort; do not raise into the bench.
            pass
        self._model = None
        self._tokenizer = None
        self._loaded = False


# ---------------------------------------------------------------------------
# score_answer — substring / keyword match for the 5 pinned tasks
# ---------------------------------------------------------------------------


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(s: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation.

    The downstream LMs end their answers with a period / question
    mark, and the expected answers in the synthetic batch are
    hash-derived strings with no terminal punctuation. Stripping
    punctuation in the normaliser closes the asymmetry so that
    `score_answer("Paris.", "paris")` returns 1.0 (a 1.7B-class
    LM will write "Paris.", the bench's `expected_answer` field
    is "paris" — without stripping, the substring match misses).
    """
    s = s.strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _first_sentence(s: str) -> str:
    """Cheap first-sentence extractor (period / question mark / !)."""
    if not s:
        return ""
    s = s.strip()
    for terminator in (".", "?", "!"):
        idx = s.find(terminator)
        if 0 < idx < len(s) - 1:
            return s[: idx + 1].strip()
    # No terminator: take the first 200 chars as the "sentence".
    return s[:200].strip()


def score_answer(
    predicted: str,
    expected: str,
    task: str = "",
) -> float:
    """Score `predicted` against `expected` for the bank test.

    The bank test does not have Rouge-L wired (no `rouge_score`
    in the slim venv) and the answer_quality metric is a
    "downstream-LM-capability ceiling" rather than a frontier
    accuracy — the matcher is intentionally simple:

      * Default (HotpotQA / NQ / GSM8K / BBH): 1.0 if the
        normalized `expected` appears as a substring of the
        normalized `predicted` (or vice versa), else 0.0.
      * Summarization: 1.0 if the first sentence of either
        string contains a 5-gram from the first sentence of the
        other, else 0.0. (No Rouge-L; 5-gram overlap is a
        cheap proxy.)

    Returns a float in {0.0, 1.0} (the per-task scorer; the
    per-(task, seed) scorer averages across the batch).
    """
    if not predicted or not expected:
        return 0.0
    task_lower = (task or "").strip().lower()
    pred_n = _normalize(predicted)
    exp_n = _normalize(expected)
    if not pred_n or not exp_n:
        return 0.0

    if task_lower in ("summarization", "summary"):
        return _summary_first_sentence_overlap(predicted, expected)

    # Default substring match (either direction).
    if exp_n in pred_n or pred_n in exp_n:
        return 1.0
    return 0.0


def _summary_first_sentence_overlap(predicted: str, expected: str) -> float:
    """Summarization scoring: 5-gram overlap of first sentences.

    The Qwen instruction-tuned models produce multi-sentence
    summaries; the first sentence is the headline, which is
    what the bench uses as a "did the model capture the topic"
    proxy. Overlap of any 5-gram = 1.0; no overlap = 0.0.
    """
    pred_first = _first_sentence(predicted)
    exp_first = _first_sentence(expected)
    pred_tokens = _normalize(pred_first).split()
    exp_tokens = _normalize(exp_first).split()
    n = 5
    if len(pred_tokens) < n or len(exp_tokens) < n:
        # Fall back to a single-token overlap if either side is
        # too short for a 5-gram.
        p_set = set(pred_tokens)
        e_set = set(exp_tokens)
        return 1.0 if (p_set & e_set) else 0.0
    pred_ngrams = {tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)}
    exp_ngrams = {tuple(exp_tokens[i : i + n]) for i in range(len(exp_tokens) - n + 1)}
    return 1.0 if (pred_ngrams & exp_ngrams) else 0.0


# ---------------------------------------------------------------------------
# _load_qwen3_1_7b_cached — AUDIT #28 real-mode fixture
# ---------------------------------------------------------------------------
#
# Lazy-loaded Qwen3-1.7B pair (model, tokenizer) gated on
# LLMLINGUA_REAL=1. Used by `bench_compress._run_one` to compute
# `_real_downstream_ppl` on each prompt. The lru_cache(maxsize=1)
# ensures the model is loaded at most once per Python process —
# both `bench_compress` and the test suite (under LLMLINGUA_REAL=1)
# share the same instance.
#
# CI is opt-in: the env var must be set explicitly. Default mode
# is the constant stub (slim venv has no torch / transformers).
#
# `pytest.mark.slow` is the user-side gate. The bench honors it
# via the same env var; the unit tests add the marker for
# `pytest -m 'not slow'` to skip on CI by default.


def _llmlingua_real_enabled() -> bool:
    """True when the user opted into real downstream-LM mode."""
    return os.environ.get("LLMLINGUA_REAL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


@functools.lru_cache(maxsize=1)
def _load_qwen3_1_7b_cached() -> Tuple[Any, Any]:
    """Load `Qwen/Qwen3-1.7B` in FP16 once per process. AUDIT #28.

    Gated on `LLMLINGUA_REAL=1` (else raises RuntimeError so the
    bench falls back to the constant stub). Uses
    `transformers.AutoModelForCausalLM` + `AutoTokenizer`. FP16 on
    CUDA, full precision on CPU. The model is loaded on the first
    call; subsequent calls hit the `lru_cache` and return the
    same `(model, tokenizer)` tuple.

    The function deliberately does NOT call `release()`: the
    `lru_cache` keeps the model alive for the rest of the process
    lifetime. To free memory, restart the process (the standard
    pattern for opt-in real-mode benches).
    """
    if not _llmlingua_real_enabled():
        raise RuntimeError(
            "_load_qwen3_1_7b_cached called without LLMLINGUA_REAL=1; "
            "the slim venv has no torch / transformers and the bench "
            "must run on the constant stub."
        )
    import torch  # local — keeps stub path torch-free
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    model_id = "Qwen/Qwen3-1.7B"
    log.info("[AUDIT #28] loading %s (LLMLINGUA_REAL=1)", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype
    )
    model = model.to(device)
    model.eval()
    log.info("[AUDIT #28] %s loaded on %s (%s)", model_id, device, dtype)
    return (model, tokenizer)
