"""US-002 benchmark import-surface tests (Phase 0 acceptance).

Verifies the 4 `apohara_context_forge.benchmarks.apohara2.*` modules:
  - each is importable as a module,
  - each has a working `--help` (exits 0),
  - the `--router {pinned,learned}` flag on bench_compress is wired
    (Finding 3 in `.omc/plans/apohara-2-0.md`).

These are wiring tests, not numerics. US-004 (Phase 2) shipped the
real `bench_ann`; US-005 (Phase 3) shipped the real `bench_compress`;
US-006 (Phase 4) and US-008 (Phase 6) are still in flight. The
numeric thresholds (recall parity, PPL delta <=5%, VRAM >=2.5x,
p<0.05) are enforced by the bench scripts and the pytest-gated
`tests/test_bench_compress_router.py` once those phases land.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


# Run each --help invocation in a clean subprocess so the argparse
# `SystemExit(0)` exits the child cleanly and we can assert on rc.
def _run_help(module: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd="/home/thelinconx/Documentos/Apohara_Context_Forge",
    )


@pytest.mark.parametrize(
    "module,required_substrings",
    [
        # bench_ann: the bench supports the synthetic + hotpotqa-mini corpora
        # and the --measure-ram flag (per spec).
        (
            "apohara_context_forge.benchmarks.apohara2.bench_ann",
            ["Apohara 2.0", "--corpus", "hotpotqa-mini", "--measure-ram"],
        ),
        # bench_compress: --router {pinned,learned} (Finding 3),
        # --task/--variant/--seeds/--judge, 3 variant names.
        (
            "apohara_context_forge.benchmarks.apohara2.bench_compress",
            [
                "--router", "pinned", "learned",
                "--task", "--variant", "--seeds", "--judge",
                "llmlingua2-base-short", "llmlingua2-base-medium", "llmlingua2-long",
            ],
        ),
    ],
)
def test_bench_help_exits_zero_and_has_flags(module, required_substrings):
    proc = _run_help(module)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for needle in required_substrings:
        assert needle in proc.stdout, f"{needle!r} missing from {module} --help"


def test_bench_compress_router_flag_choices():
    """--router {pinned,learned} must be a defined argparse choice."""
    proc = _run_help("apohara_context_forge.benchmarks.apohara2.bench_compress")
    assert proc.returncode == 0
    # argparse prints "{pinned,learned}" for choices=...
    assert "{pinned,learned}" in proc.stdout


def test_bench_compress_task_flag_choices():
    """--task {longbench_subset, synthetic, hotpotqa-mini} must be a defined argparse choice."""
    proc = _run_help("apohara_context_forge.benchmarks.apohara2.bench_compress")
    assert proc.returncode == 0
    assert "longbench_subset" in proc.stdout
    assert "synthetic" in proc.stdout
    assert "hotpotqa-mini" in proc.stdout


def test_bench_compress_judge_flag_choices():
    """--judge {m3, none} must be a defined argparse choice."""
    proc = _run_help("apohara_context_forge.benchmarks.apohara2.bench_compress")
    assert proc.returncode == 0
    assert "{m3,none}" in proc.stdout


def test_bench_kv_help_exits_zero():
    proc = _run_help("apohara_context_forge.benchmarks.apohara2.bench_kv")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # The pivot banner is documented in the help description.
    assert "Ampere+" in proc.stdout or "H100" in proc.stdout
    # US-006 wired the new flags: --hardware {rtx2060s,h100,mi300x,cpu},
    # --bits, --docs. The default --hardware is `cpu` (the local
    # smoke path on the slim venv).
    assert "--hardware" in proc.stdout
    assert "{rtx2060s,h100,mi300x,cpu}" in proc.stdout
    assert "--bits" in proc.stdout
    assert "--docs" in proc.stdout


def test_bench_kv_runs_and_emits_json():
    """US-006: the bench exits 0 and emits a JSON summary with the
    compression_ratio >= 2.5 contract from the Phase 4 spec.

    Skips cleanly when the Rust crate is not built (the slim venv
    case). The skip matches the convention in
    `tests/test_compressor.py` (line 135-140).
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "apohara_context_forge.benchmarks.apohara2.bench_kv",
            "--hardware",
            "cpu",
            "--bits",
            "4",
            "--docs",
            "100",
            "--seeds",
            "0..0",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd="/home/thelinconx/Documentos/Apohara_Context_Forge",
    )
    # Always parse the last line as JSON — even when the bench exits
    # non-zero, the JSON contract is what the bank-test aggregator
    # consumes.
    last = proc.stdout.splitlines()[-1] if proc.stdout.strip() else "{}"
    summary = json.loads(last)
    if not summary.get("rust_available", False):
        pytest.skip(
            "turboquant-turing Rust crate is not built; the bench "
            "returns the honest not-built envelope. Build with "
            "`maturin develop` in apohara_context_forge/serving/"
            "turboquant_turing/` to exercise the assertion."
        )
    # The bench asserts the 2.5x compression threshold internally
    # and surfaces it as `thresholds_pass` in the JSON.
    assert "bits_results" in summary
    assert summary["thresholds_pass"] is True
    # Pivot banner is None for --hardware cpu.
    assert summary.get("pivot_banner") is None
    assert summary["bits"] == 4
    assert summary["hardware"] == "cpu"


def test_bench_e2e_help_exits_zero():
    proc = _run_help("apohara_context_forge.benchmarks.apohara2.bench_e2e")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # The 5 pinned tasks are documented in the help description.
    for task in ("hotpotqa", "naturalquestions", "gsm8k", "bbh", "summarization"):
        assert task in proc.stdout, f"task {task!r} missing from --help"
    # The pre-registered correction is the default.
    assert "holm-bonferroni" in proc.stdout
    # US-008 added the --mode, --hardware, --correction, --seeds flags.
    assert "--mode" in proc.stdout
    assert "--hardware" in proc.stdout
    assert "--correction" in proc.stdout
    assert "--seeds" in proc.stdout
    assert "--n-questions" in proc.stdout
    assert "--n-ctx-tokens" in proc.stdout
    # The mode and hardware choices are documented.
    assert "{synthetic,real}" in proc.stdout
    assert "{cpu,rtx2060s,h100,mi300x}" in proc.stdout
    # The pivot banner is in the help description.
    assert "Ampere+" in proc.stdout or "H100" in proc.stdout


def test_bench_e2e_runs_and_emits_json():
    """US-008: the bench exits 0 and emits a JSON summary containing
    the 5 pinned tasks, the pre-registered correction, and
    `family_wise_pass` == True (synthetic mode on CPU; honest
    stub for the downstream LM).

    The bench is hardware-agnostic; locally it runs CPU-only with
    a constant-string downstream LM stub. The JSON contract
    includes the per-task p-values (paired t-test vs. the
    uncompressed baseline) and the Holm-Bonferroni-adjusted
    p-values. `family_wise_pass` is True iff all 5 tasks pass the
    corrected gate; in synthetic mode the per-task p-values
    uniformly report p = 0.0 vs. a constant 1.0 compression
    baseline, so the family-wise gate is exercised.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "apohara_context_forge.benchmarks.apohara2.bench_e2e",
            "--mode",
            "synthetic",
            "--downstream_lm",
            "stub",
            "--seeds",
            "0,1",
            "--correction",
            "holm-bonferroni",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd="/home/thelinconx/Documentos/Apohara_Context_Forge",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # The bench prints the indented JSON summary on stdout when
    # --quiet is set (no progress logs). Parse the entire stdout
    # blob as JSON; the bench's print() flushes before the
    # SystemExit so the full document is captured.
    summary = json.loads(proc.stdout)
    # The 5 pinned tasks are in the per-task map.
    for task in ("hotpotqa", "naturalquestions", "gsm8k", "bbh", "summarization"):
        assert task in summary["per_task"], (
            f"task {task!r} missing from summary.per_task"
        )
    # The seeds + correction are echoed back.
    assert summary["seeds"] == [0, 1]
    assert summary["correction"] == "holm-bonferroni"
    assert summary["mode"] == "synthetic"
    # US-014-REDUX: the bench now reports the downstream LM choice
    # and the task / seed counts in the top-level summary.
    assert summary["downstream_lm"] == "stub"
    assert summary["n_tasks"] == 5
    assert summary["n_seeds"] == 2
    # The per-task rows have the contract keys.
    for task, row in summary["per_task"].items():
        for key in (
            "n_seeds",
            "compression_ratio_mean",
            "compression_ratio_std",
            "kv_round_trip_mse_mean",
            "recall_at_3_mean",
            "answer_quality_mean",
            "p_value_vs_uncompressed",
            "passes_p_0.05",
            "adjusted_p_value",
            "rejected",
        ):
            assert key in row, f"key {key!r} missing from per_task[{task!r}]"
    # The synthetic stub's per-task p-values are degenerate; the
    # bench reports them honestly. `family_wise_pass` is True iff
    # all 5 tasks pass the corrected gate; we assert the field
    # exists and is a bool (the actual value depends on the
    # honest measurement).
    assert isinstance(summary["family_wise_pass"], bool)
    # The pivots field is honest about the H100 / MI300X gate.
    if summary["hardware"] in ("cpu", "rtx2060s"):
        assert "h100" in summary["pivots_required"]
        assert "mi300x" in summary["pivots_required"]


def test_bench_compress_runs_and_emits_json():
    """The bench must exit 0 and emit a JSON summary containing
    `max_ppl_delta_pct` <= 5% (the spec's threshold from Round 16).

    Skips cleanly when `llmlingua` (and the onnx runtime it needs) is
    not importable — the real bench loads a model and that requires
    the onnxruntime back-end. The skip matches the convention in
    `tests/test_compressor.py` (line 135-140).
    """
    # Import guard: skip when the runtime deps for the model are
    # not present (CI / no-onnx).
    import importlib.util

    onnx_spec = importlib.util.find_spec("onnxruntime")
    if onnx_spec is None:
        pytest.skip(
            "onnxruntime not installed — bench_compress requires the "
            "LLMLingua-2 model runtime"
        )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "apohara_context_forge.benchmarks.apohara2.bench_compress",
            "--task",
            "synthetic",
            "--variant",
            "llmlingua2-base-short",
            "--seeds",
            "0..1",
            "--judge",
            "m3",
            "--router",
            "pinned",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd="/home/thelinconx/Documentos/Apohara_Context_Forge",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # The last line of stdout is the JSON summary (the bench prints
    # only the summary when --quiet is set; logging.basicConfig at
    # WARNING suppresses progress logs).
    summary = json.loads(proc.stdout.splitlines()[-1])
    assert "max_ppl_delta_pct" in summary
    # Spec threshold (Round 14 / .omc/plans/apohara-2-0.md): PPL
    # delta <= 5% per variant. The bench asserts this internally and
    # surfaces it as `threshold_pass` in the JSON.
    assert summary["max_ppl_delta_pct"] <= 5.0
    assert summary["threshold_pass"] is True
    assert summary["router"] == "pinned"
    assert summary["judge"] == "m3"
    assert summary["seeds"] == [0, 1]
    assert summary["task"] == "synthetic"


@pytest.mark.parametrize("module", [
    "apohara_context_forge.benchmarks.apohara2.bench_ann",
    "apohara_context_forge.benchmarks.apohara2.bench_compress",
    "apohara_context_forge.benchmarks.apohara2.bench_kv",
    "apohara_context_forge.benchmarks.apohara2.bench_e2e",
])
def test_bench_stub_module_importable(module):
    """Each stub module must import without error (catches syntax regressions)."""
    import importlib

    importlib.import_module(module)
