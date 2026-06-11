#!/usr/bin/env bash
# Thin wrapper around `maturin develop` for the in-tree
# turboquant-turing crate. Run from the repo root:
#
#   bash apohara_context_forge/serving/turboquant_turing/build.sh
#
# or with the CUDA feature on a host with nvcc:
#
#   FEATURES=compute_75 bash apohara_context_forge/serving/turboquant_turing/build.sh
#
# This script is intentionally not a hard dependency. The bench
# (`benchmarks/apohara2/bench_kv.py`) prints the maturin command
# when the crate is not built, and tests skip the import cleanly.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if command -v maturin >/dev/null 2>&1; then
    MATURIN=maturin
elif [[ -x "$HOME/.local/bin/maturin" ]]; then
    MATURIN="$HOME/.local/bin/maturin"
elif [[ -x "../../../../.venv/bin/maturin" ]]; then
    MATURIN="../../../../.venv/bin/maturin"
else
    echo "maturin not found on PATH. Install with: pipx install maturin" >&2
    exit 1
fi

if [[ -n "${FEATURES:-}" ]]; then
    exec "$MATURIN" develop --release --features "$FEATURES"
else
    exec "$MATURIN" develop --release
fi
