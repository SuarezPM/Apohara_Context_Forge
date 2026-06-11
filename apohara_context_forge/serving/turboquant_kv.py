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

from typing import Tuple

import numpy as np

# Lazy import: the Rust crate may not be built in the slim venv.
try:
    from apohara_context_forge.serving.turboquant_turing import (
        encode_kv as _rust_encode_kv,
        decode_kv as _rust_decode_kv,
    )
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

# Single source of truth for the not-built error message; both
# encode() and decode() raise the same RuntimeError, so the message
# lives here to keep the two error sites in sync.
_RUST_NOT_BUILT_MSG = (
    "turboquant-turing Rust crate is not built. "
    "Run `cd apohara_context_forge/serving/turboquant_turing && "
    "maturin develop` to build it."
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
        if not _RUST_AVAILABLE:
            raise RuntimeError(_RUST_NOT_BUILT_MSG)
        flat = weights.astype(np.float32).reshape(-1)
        packed = _rust_encode_kv(flat.tobytes(), flat.size, self.bits)
        # Honest stub: real scales come from Lloyd-Max calibration
        # (`codec_v8.py` is the calibration path the shim mirrors).
        scales = np.ones(weights.shape, dtype=np.float32)
        return packed, scales

    def decode(
        self, packed: bytes, scales: np.ndarray, shape: Tuple[int, ...]
    ) -> np.ndarray:
        if not _RUST_AVAILABLE:
            raise RuntimeError(_RUST_NOT_BUILT_MSG)
        n = int(np.prod(shape))
        raw = _rust_decode_kv(packed, n, self.bits)
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
