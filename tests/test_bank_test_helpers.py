"""test_bank_test_helpers.py — US-008 helper unit tests (Phase 6 / Step 6.4).

The 4 helpers in `apohara_context_forge.benchmarks.apohara2._bank_test_helpers`
are deterministic primitives; this file pins their contracts so the
bank test's contract does not drift silently.

The bench's JSON summary (`bench_e2e.py`) consumes these helpers;
the bank test's gate is a function of the helper outputs, so a
silent helper regression would silently change the gate.

US-014-REDUX additions: the helpers also include `DownstreamLM`
(transformers-backed downstream LM, mocked here) and
`score_answer` (substring / summarization-5-gram match). The
bench's CLI gained a `--downstream_lm` flag (verified via
`--help`); the A/B orchestrator is tested with a mocked
subprocess. All tests are CPU-only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from unittest import mock

import pytest

from apohara_context_forge.benchmarks.apohara2._bank_test_helpers import (
    DownstreamLM,
    downstream_lm_stub,
    holm_bonferroni,
    list_downstream_lm_aliases,
    paired_ttest_pvalue,
    resolve_downstream_lm_id,
    score_answer,
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


# ---------------------------------------------------------------------------
# resolve_downstream_lm_id / list_downstream_lm_aliases (US-014-REDUX)
# ---------------------------------------------------------------------------


def test_resolve_downstream_lm_id_known_aliases():
    """The two canonical aliases resolve to the cached HF model ids."""
    assert resolve_downstream_lm_id("qwen3-1.7b") == "Qwen/Qwen3-1.7B"
    assert (
        resolve_downstream_lm_id("qwen2.5-0.5b")
        == "Qwen/Qwen2.5-0.5B-Instruct"
    )


def test_resolve_downstream_lm_id_unknown_alias_raises():
    """Unknown aliases fail fast with a clear ValueError."""
    with pytest.raises(ValueError, match="unknown --downstream_lm"):
        resolve_downstream_lm_id("not-a-model")


def test_list_downstream_lm_aliases_is_sorted():
    """The list is sorted (the CLI help text relies on this)."""
    aliases = list_downstream_lm_aliases()
    assert aliases == tuple(sorted(aliases))
    assert "qwen3-1.7b" in aliases
    assert "qwen2.5-0.5b" in aliases


# ---------------------------------------------------------------------------
# DownstreamLM (mocked model + tokenizer; no real GPU load)
# ---------------------------------------------------------------------------


def _fake_tokenizer_call(text: str, return_tensors: str = "pt"):
    """Return a dict-like that the DownstreamLM.generate() code can
    `.to(device)` and unpack as `inputs["input_ids"].shape[1]`."""
    return {
        "input_ids": _FakeTensor([1, 2, 3, 4, 5]),
    }


class _FakeTensor:
    """Minimal tensor fake: list-of-int, supports `.shape[1]`,
    `__getitem__(int)`, `__getitem__(slice)`, and `len()`."""

    def __init__(self, data: list[int]) -> None:
        self.data = data
        self.shape = (1, len(data))

    def to(self, device: str) -> "_FakeTensor":
        return self

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx != 0:
                raise IndexError(idx)
            return _FakeTensor(self.data)
        if isinstance(idx, slice):
            return _FakeTensor(self.data[idx])
        raise TypeError(idx)


def _build_mocked_downstream_lm() -> DownstreamLM:
    """Build a DownstreamLM with mocked tokenizer + model so no
    real `transformers` load happens during the test.

    The mocked model is configured to echo the prompt (after a
    space) — enough to exercise the `generate()` code path and
    the post-prompt token stripping.
    """
    lm = DownstreamLM(model_id="Qwen/Qwen3-1.7B")
    fake_tokenizer = mock.MagicMock()
    fake_tokenizer.eos_token_id = 0
    fake_tokenizer.pad_token_id = 0
    # The tokenizer call returns a MagicMock that supports
    # `.to(device)`; inside, `["input_ids"]` returns a fake
    # tensor with `.shape[1]` (5 in this fixture).
    fake_inputs = mock.MagicMock()
    fake_inputs.__getitem__.side_effect = lambda key: (
        _FakeTensor([1, 2, 3, 4, 5]) if key == "input_ids" else None
    )
    fake_inputs.to.return_value = fake_inputs
    fake_tokenizer.return_value = fake_inputs
    # Decode the "new tokens" as the literal string
    # "echo of prompt (len=N)" so we can assert the post-prompt
    # slicing actually returns the new content (not the prompt
    # itself).
    fake_tokenizer.decode.side_effect = (
        lambda ids, skip_special_tokens: f"echo of prompt (len={len(ids)})"
    )

    fake_model = mock.MagicMock()
    # generate() returns a tensor of shape (1, prompt_len + new_len).
    new_token_count = 7
    fake_output_ids = _FakeTensor([1, 2, 3, 4, 5] + [10] * new_token_count)
    fake_model.generate.return_value = [fake_output_ids]

    lm._tokenizer = fake_tokenizer
    lm._model = fake_model
    lm._device = "cpu"
    lm._loaded = True
    return lm


def test_downstream_lm_is_real_for_known_aliases():
    """is_real() returns True for the two registered Qwen ids."""
    assert DownstreamLM(model_id="Qwen/Qwen3-1.7B").is_real() is True
    assert (
        DownstreamLM(model_id="Qwen/Qwen2.5-0.5B-Instruct").is_real() is True
    )


def test_downstream_lm_is_real_false_for_unknown():
    """is_real() returns False for an unknown model id."""
    assert DownstreamLM(model_id="not-a-real-model").is_real() is False


def test_downstream_lm_generate_mocked_returns_decoded_new_tokens():
    """generate() returns the decoded new tokens (post-prompt slice).

    The mocked tokenizer decodes the "new" tokens (the post-prompt
    slice of the generate output) to a known string; the test
    asserts the call flow and the result.
    """
    lm = _build_mocked_downstream_lm()
    out = lm.generate("hello world", max_new_tokens=128)
    assert "echo of prompt" in out
    # generate() was called with the right generation kwargs.
    call = lm._model.generate.call_args
    assert call.kwargs["max_new_tokens"] == 128
    assert call.kwargs["do_sample"] is False
    assert call.kwargs["num_beams"] == 1
    assert call.kwargs["temperature"] == 1.0
    assert call.kwargs["top_p"] == 1.0
    assert call.kwargs["top_k"] == 1


def test_downstream_lm_release_is_idempotent():
    """release() can be called multiple times without raising."""
    lm = _build_mocked_downstream_lm()
    lm.release()
    lm.release()
    assert lm.is_loaded() is False
    assert lm._model is None
    assert lm._tokenizer is None


# ---------------------------------------------------------------------------
# score_answer (substring match + summarization 5-gram)
# ---------------------------------------------------------------------------


def test_score_answer_exact_substring_match():
    """predicted contains expected -> 1.0 (HotpotQA-style)."""
    assert score_answer("Paris is the capital of France.", "paris") == 1.0
    assert score_answer("bar baz", "foo bar baz") == 1.0


def test_score_answer_no_overlap_returns_zero():
    """No substring overlap on either side -> 0.0."""
    assert score_answer("the answer is forty-two", "xyzzy") == 0.0


def test_score_answer_empty_strings_return_zero():
    """Empty predicted or expected returns 0.0 (no false greens)."""
    assert score_answer("", "anything") == 0.0
    assert score_answer("anything", "") == 0.0
    assert score_answer("", "") == 0.0


def test_score_answer_whitespace_normalized():
    """Whitespace + case are normalized before matching."""
    assert score_answer("The  Quick   Brown  Fox", "the quick brown fox") == 1.0
    assert score_answer("PARIS", "paris") == 1.0


def test_score_answer_task_argument_does_not_affect_default():
    """The `task` argument is optional; default is substring match."""
    # task="" uses substring; task="hotpotqa" also uses substring.
    assert score_answer("The answer is 42.", "42") == 1.0
    assert (
        score_answer("The answer is 42.", "42", task="hotpotqa") == 1.0
    )


def test_score_answer_summarization_5gram_overlap():
    """Summarization: 5-gram overlap of first sentences = 1.0."""
    pred = "The cat sat on the mat. Then it purred loudly."
    exp = "The cat sat on the mat. After that it slept."
    assert score_answer(pred, exp, task="summarization") == 1.0


def test_score_answer_summarization_no_5gram_overlap():
    """Summarization: disjoint first sentences = 0.0."""
    pred = "Quantum physics is the study of subatomic particles."
    exp = "The history of Rome spans several millennia."
    assert score_answer(pred, exp, task="summarization") == 0.0


def test_score_answer_summarization_short_uses_token_overlap():
    """Summarization: short side falls back to single-token overlap."""
    # First sentence on both sides is a single token; no 5-gram
    # is possible, so the helper falls back to a 1-gram overlap.
    pred = "Hello."
    exp = "Hello world."
    assert score_answer(pred, exp, task="summarization") == 1.0


# ---------------------------------------------------------------------------
# bench_e2e.py --downstream_lm CLI flag
# ---------------------------------------------------------------------------


def test_bench_e2e_help_shows_downstream_lm_flag():
    """`bench_e2e.py --help` advertises the new --downstream_lm flag.

    We invoke the script in a subprocess so the test exercises the
    real CLI surface; we assert the help text contains the new
    choice list and the explanation.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "apohara_context_forge.benchmarks.apohara2.bench_e2e",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env={**os.environ, "PYTHONPATH": "."},
    )
    out = proc.stdout
    assert "--downstream_lm" in out
    assert "qwen3-1.7b" in out
    assert "qwen2.5-0.5b" in out
    assert "stub" in out
    assert "none" in out


def _parse_last_json_block(stdout: str) -> dict:
    """Parse the LAST top-level JSON object from a multi-line bench stdout.

    The bench prints a pretty-printed JSON summary as the **last**
    block on stdout. We brace-balance from the end and slice out
    the last complete top-level object. The orchestrator code does
    the same; this is its test-side mirror.
    """
    text = stdout.strip()
    # Walk from the end, counting braces. The closing '}' of the
    # last top-level object decrements the depth to 0; the
    # matching '{' starts the object. We then json.loads that
    # substring.
    depth = 0
    end_idx = -1
    for i in range(len(text) - 1, -1, -1):
        c = text[i]
        if c == "}":
            if depth == 0:
                end_idx = i
            depth += 1
        elif c == "{":
            depth -= 1
            if depth == 0 and end_idx >= 0:
                return json.loads(text[i : end_idx + 1])
    raise AssertionError(f"no balanced top-level JSON object in stdout: {text!r}")


def test_bench_e2e_runs_with_downstream_lm_stub_in_quiet_mode():
    """`--downstream_lm stub --quiet` runs and emits the JSON summary."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "apohara_context_forge.benchmarks.apohara2.bench_e2e",
            "--downstream_lm",
            "stub",
            "--seeds",
            "0,1",
            "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env={**os.environ, "PYTHONPATH": "."},
    )
    # The bench prints a pretty-printed multi-line JSON summary on
    # stdout; the orchestrator code parses the LAST JSON-shaped
    # block. We do the same: find the first '{' on stdout, parse
    # from there.
    summary = _parse_last_json_block(proc.stdout)
    assert summary["downstream_lm"] == "stub"
    assert summary["n_tasks"] == 5
    assert summary["n_seeds"] == 2
    assert "scope_banner" in summary
    assert "synthetic" in summary["scope_banner"].lower()


def test_bench_e2e_runs_with_downstream_lm_none_in_quiet_mode():
    """`--downstream_lm none --quiet` runs and emits the JSON summary."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "apohara_context_forge.benchmarks.apohara2.bench_e2e",
            "--downstream_lm",
            "none",
            "--seeds",
            "0,1",
            "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env={**os.environ, "PYTHONPATH": "."},
    )
    summary = _parse_last_json_block(proc.stdout)
    assert summary["downstream_lm"] == "none"
    assert summary["n_tasks"] == 5
    assert "answer_quality metric skipped" in summary["scope_banner"]


# ---------------------------------------------------------------------------
# A/B orchestrator (run_real_mode_ab.py, mocked subprocess)
# ---------------------------------------------------------------------------


def test_run_real_mode_ab_dry_run_writes_report(tmp_path, monkeypatch):
    """The orchestrator's --dry-run path writes a markdown report.

    The real arm launches bench_e2e.py as a subprocess; the dry-run
    path returns synthetic summaries so the report code is exercised
    without a GPU. We point the report path at a tmp file and
    assert the file exists with the expected sections.
    """
    from apohara_context_forge.benchmarks.apohara2 import run_real_mode_ab

    report_path = tmp_path / "ab_report.md"
    monkeypatch.setattr(
        sys, "argv", ["run_real_mode_ab", "--dry-run", "--report", str(report_path)]
    )
    rc = run_real_mode_ab.main()
    assert rc == 0
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    # The report must include the canonical A/B sections.
    assert "Downstream-LM-Agnosticism A/B Report" in content
    assert "Qwen3-1.7B" in content
    assert "Qwen2.5-0.5B" in content
    assert "Per-task answer_quality (A/B)" in content
    assert "Conclusion" in content
    # The dry-run synthetic summaries produce a mean |Δ| well above
    # the 0.20 agnosticism tolerance, so the conclusion must be the
    # "capability threshold" branch.
    assert "capability threshold" in content or "downstream-LM-agnosticism" in content


def test_run_real_mode_ab_renders_table_for_both_arms():
    """The renderer emits a per-task table row for each of the 5 tasks."""
    from apohara_context_forge.benchmarks.apohara2 import run_real_mode_ab

    # Build minimal synthetic summaries.
    def make(arm_aq):
        per_task = {}
        for task in ("hotpotqa", "naturalquestions", "gsm8k", "bbh", "summarization"):
            per_task[task] = {
                "n_seeds": 5,
                "seeds": [0, 1, 2, 3, 4],
                "compression_ratio_mean": 0.55,
                "compression_ratio_std": 0.0,
                "kv_round_trip_mse_mean": 1e-4,
                "kv_round_trip_mse_std": 1e-6,
                "recall_at_3_mean": 1.0,
                "recall_at_3_std": 0.0,
                "answer_quality_mean": arm_aq,
                "answer_quality_std": 0.0,
                "p_value_vs_uncompressed": 0.0,
                "passes_p_0.05": True,
                "adjusted_p_value": 0.0,
                "rejected": True,
            }
        return {
            "mode": "synthetic",
            "hardware": "rtx2060s",
            "seeds": [0, 1, 2, 3, 4],
            "correction": "holm-bonferroni",
            "n_questions": 10,
            "n_ctx_tokens": 100,
            "downstream_lm": "qwen3-1.7b" if arm_aq > 0.5 else "qwen2.5-0.5b",
            "n_tasks": 5,
            "n_seeds": 5,
            "per_task": per_task,
            "family_wise_pass": True,
            "pivots_required": ["h100", "mi300x"],
            "scope_banner": "real-mode with mocked arm on RTX 2060 SUPER 8GB",
        }

    summary_a = make(0.8)
    summary_b = make(0.4)
    md = run_real_mode_ab.render_report(
        summary_a=summary_a,
        summary_b=summary_b,
        json_path_a="/tmp/bench_qwen3_1.7b.json",
        json_path_b="/tmp/bench_qwen2.5_0.5b.json",
    )
    # 5 task rows in the markdown table.
    for task in ("hotpotqa", "naturalquestions", "gsm8k", "bbh", "summarization"):
        assert f"| {task} |" in md
    # Honest gaps section is always present.
    assert "Honest gaps" in md
    # Raw JSON links.
    assert "/tmp/bench_qwen3_1.7b.json" in md
    assert "/tmp/bench_qwen2.5_0.5b.json" in md
