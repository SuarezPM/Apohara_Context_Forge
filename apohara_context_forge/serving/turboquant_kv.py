"""TurboQuant-KV path (Phase 4 shim) for vLLM/SGLang config-driven integration.

This module mirrors the LMCacheConnectorV2 config-driven pattern (see
lmcache_connector.py:1-410). It does NOT install any vLLM plugin — the
KV interception lives in the upstream vLLM `--kv-cache-dtype turboquant`
flag (PR #38280 merged). The shim here wraps the in-tree Rust crate
`turboquant-turing` (apohara_context_forge/serving/turboquant_turing/)
and exposes the same encode/decode API to the rest of the codebase.

Honest scope: as of 2026-06-11, the Rust crate's CPU implementation is
in the tree; the CUDA C kernel is feature-gated and not built by
default. The bank test runs CPU-only on RTX 2060 SUPER; H100/MI300X
pivots are documented in bench_kv.py.
"""
from __future__ import annotations

import importlib.util
from typing import Tuple

import numpy as np

# Sprint 2 / AUDIT #320a: the previous implementation did a top-level
# import and cached the result in a static ``_RUST_AVAILABLE`` flag
# at module import time. That was wrong — it required the test to
# import the module after the wheel was installed, and the
# ``RUST_AVAILABLE`` value was fixed for the life of the process.
# The new implementation uses ``importlib.util.find_spec`` to test
# *importability* on every call, so a wheel built mid-session (via
# ``maturin develop``) is picked up the next time encode()/decode()
# is invoked without requiring a Python restart.
def _rust_available() -> bool:
    """Live check for the in-tree ``turboquant_turing`` wheel.

    Returns True if a wheel of that name is importable in the current
    process, False otherwise. Uses ``importlib.util.find_spec`` (the
    standard-library import-finder) so the check is fast and has no
    side effects on the import system.
    """
    return importlib.util.find_spec("turboquant_turing") is not None


def _import_rust_kv():
    """Import the Rust symbols lazily, raising a clear error on miss."""
    if not _rust_available():
        raise ImportError(_RUST_NOT_BUILT_MSG)
    from apohara_context_forge.serving.turboquant_turing import (
        decode_kv as _rust_decode_kv,
        encode_kv as _rust_encode_kv,
    )
    return _rust_encode_kv, _rust_decode_kv


# Single source of truth for the not-built error message; both
# encode() and decode() raise the same ImportError, so the message
# lives here to keep the two error sites in sync.
_RUST_NOT_BUILT_MSG = (
    "turboquant-turing Rust crate is not built. "
    "Run `cd apohara_context_forge/serving/turboquant_turing && "
    "bash build.sh` (chains `cargo test --release && maturin "
    "develop --release`) to build it."
)


class TurboQuantKVShim:
    """Config-driven wrapper for the in-tree turboquant-turing crate.

    The shim mirrors the lazy-import discipline of
    `apohara_context_forge.serving.lmcache_connector.LMCacheConnectorV2`
    (see AUDIT #20 for the F2 lesson on vLLM plugins). If the Rust
    crate is not built, `encode()` / `decode()` raise a RuntimeError
    pointing the caller at the `maturin develop` step.
    """

    def __init__(self, bits: int = 4) -> None:
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4; got {bits}")
        self.bits = bits

    def encode(self, weights: np.ndarray) -> Tuple[bytes, np.ndarray]:
        _rust_encode_kv, _ = _import_rust_kv()
        flat = weights.astype(np.float32).reshape(-1)
        packed = _rust_encode_kv(flat.tobytes(), flat.size, self.bits)
        # Honest stub: real scales come from Lloyd-Max calibration
        # (`codec_v8.py` is the calibration path the shim mirrors).
        scales = np.ones(weights.shape, dtype=np.float32)
        return packed, scales

    def decode(
        self, packed: bytes, scales: np.ndarray, shape: Tuple[int, ...]
    ) -> np.ndarray:
        _, _rust_decode_kv = _import_rust_kv()
        n = int(np.prod(shape))
        raw = _rust_decode_kv(packed, n, self.bits)
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
