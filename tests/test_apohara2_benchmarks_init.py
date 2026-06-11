"""US-002 benchmark import-surface tests (Phase 0 acceptance).

Verifies the 4 new `apohara_context_forge.benchmarks.apohara2.*` stubs:
  - each is importable as a module,
  - each has a working `--help` (exits 0),
  - the `--router {pinned,learned}` flag on bench_compress is wired
    (Finding 3 in `.omc/plans/apohara-2-0.md`).

These are wiring tests, not numerics. The real bench implementations
land in US-004 (Phase 2), US-005 (Phase 3), US-006 (Phase 4), and
US-008 (Phase 6). The numeric thresholds (recall parity, PPL delta
<=5%, VRAM >=2.5x, p<0.05) are enforced by the bench scripts and the
pytest-gated `tests/test_bench_compress_router.py` once those phases
land.
"""

from __future__ import annotations

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


def test_bench_ann_help_exits_zero():
    proc = _run_help("apohara_context_forge.benchmarks.apohara2.bench_ann")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # The US-004 intent is documented in the help description.
    assert "US-002 stub" in proc.stdout
    assert "US-004" in proc.stdout


def test_bench_compress_help_exits_zero():
    proc = _run_help("apohara_context_forge.benchmarks.apohara2.bench_compress")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "US-002 stub" in proc.stdout
    # The --router flag is wired in the stub (Finding 3 in apohara-2-0.md).
    assert "--router" in proc.stdout
    assert "pinned" in proc.stdout
    assert "learned" in proc.stdout


def test_bench_compress_router_flag_choices():
    """--router {pinned,learned} must be a defined argparse choice."""
    proc = _run_help("apohara_context_forge.benchmarks.apohara2.bench_compress")
    assert proc.returncode == 0
    # argparse prints "{pinned,learned}" for choices=...
    assert "{pinned,learned}" in proc.stdout


def test_bench_kv_help_exits_zero():
    proc = _run_help("apohara_context_forge.benchmarks.apohara2.bench_kv")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "US-002 stub" in proc.stdout
    # The pivot banner is documented in the help description.
    assert "Ampere+" in proc.stdout or "H100" in proc.stdout


def test_bench_e2e_help_exits_zero():
    proc = _run_help("apohara_context_forge.benchmarks.apohara2.bench_e2e")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "US-002 stub" in proc.stdout
    # The 5 pinned tasks are documented in the help description.
    for task in ("hotpotqa", "naturalquestions", "gsm8k", "bbh", "summarization"):
        assert task in proc.stdout, f"task {task!r} missing from --help"
    # The pre-registered correction is the default.
    assert "holm-bonferroni" in proc.stdout


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
