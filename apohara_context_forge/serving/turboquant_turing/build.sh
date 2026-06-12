#!/usr/bin/env bash
# Thin wrapper around `cargo test --release` + `maturin develop`
# for the in-tree turboquant-turing crate. Run from the repo root:
#
#   bash apohara_context_forge/serving/turboquant_turing/build.sh
#
# or with the CUDA feature on a host with nvcc:
#
#   FEATURES=compute_75 bash apohara_context_forge/serving/turboquant_turing/build.sh
#
# Sprint 2 / AUDIT #320a: the chain is now `cargo test --release &&
# maturin develop --release`. The cargo test step covers the new
# ``src/fwht.rs`` and ``src/dequant.rs`` round-trip tests in
# addition to the legacy ``encode_kv`` / ``decode_kv`` /
# ``centroids`` tests; the maturin step emits the wheel that the
# Python shim (and the test suite) imports.
#
# This script is intentionally not a hard dependency. The bench
# (`benchmarks/apohara2/bench_kv.py`) prints the maturin command
# when the crate is not built, and tests skip the import cleanly.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 1. Run the in-tree Rust unit + integration tests first. If any
#    of them fail, the build is rejected before maturin copies
#    a stale wheel into the venv. The default test invocation
#    does NOT enable the ``python-bindings-test`` feature because
#    cargo cannot link the embedded interpreter (maturin does
#    that in step 2) — running the bindings test without
#    maturin's link step produces undefined-symbol errors. The
#    bindings test is a follow-up: after step 3 (maturin develop)
#    has staged the wheel, a separate
#    ``cargo test --release --features python-bindings-test``
#    can be run; see ``tests/python_bindings.rs`` for the parity
#    checks it covers.
echo "▸ cargo test --release"
if [[ -n "${FEATURES:-}" ]]; then
    cargo test --release --features "$FEATURES"
else
    cargo test --release
fi

# 2. Locate maturin. The venv at the repo root is the conventional
#    install path; ``uv pip install maturin --python .venv`` is
#    the documented dev-env step.
if command -v maturin >/dev/null 2>&1; then
    MATURIN=maturin
elif [[ -x "$HOME/.local/bin/maturin" ]]; then
    MATURIN="$HOME/.local/bin/maturin"
elif [[ -x "../../../../.venv/bin/maturin" ]]; then
    MATURIN="../../../../.venv/bin/maturin"
elif [[ -x ".venv/bin/maturin" ]]; then
    MATURIN=".venv/bin/maturin"
else
    echo "maturin not found on PATH. Install with: uv pip install maturin --python .venv" >&2
    exit 1
fi

# 3. Build the wheel and copy it into the active Python.
echo "▸ maturin develop --release"
if [[ -n "${FEATURES:-}" ]]; then
    exec "$MATURIN" develop --release --features "$FEATURES"
else
    exec "$MATURIN" develop --release
fi
