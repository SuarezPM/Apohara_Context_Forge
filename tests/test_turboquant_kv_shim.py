"""Tests for the Phase 4 TurboQuant-KV shim (US-006).

The shim `apohara_context_forge.serving.turboquant_kv.TurboQuantKVShim`
mirrors the LMCacheConnectorV2 config-driven pattern
(`lmcache_connector.py:1-410`). The Rust crate is not built in the
slim venv, so most tests assert the honest not-built envelope.

When the crate IS built (via `maturin develop`), the shim's
`encode` / `decode` methods return a `RuntimeError` until the
real LLM-as-judge and the per-block Lloyd-Max calibration land
(see AUDIT #25 for the honest scope).
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

shim_spec = importlib.util.find_spec(
    "apohara_context_forge.serving.turboquant_kv"
)
if shim_spec is None:  # pragma: no cover
    pytest.skip(
        "apohara_context_forge.serving.turboquant_kv is not importable",
        allow_module_level=True,
    )

from apohara_context_forge.serving.turboquant_kv import (  # noqa: E402
    TurboQuantKVShim,
    _RUST_AVAILABLE,
)


def test_shim_constructs_with_valid_bits():
    shim = TurboQuantKVShim(bits=4)
    assert shim.bits == 4
    shim2 = TurboQuantKVShim(bits=2)
    assert shim2.bits == 2
    shim3 = TurboQuantKVShim(bits=3)
    assert shim3.bits == 3


def test_shim_default_bits_is_4():
    shim = TurboQuantKVShim()
    assert shim.bits == 4


@pytest.mark.parametrize("bad_bits", [0, 1, 5, 6, 8, 16])
def test_shim_rejects_invalid_bits(bad_bits):
    with pytest.raises(ValueError, match="bits must be 2, 3, or 4"):
        TurboQuantKVShim(bits=bad_bits)


def test_shim_encode_raises_when_rust_not_built():
    """The honest US-006 state: the Rust crate is not built in the
    slim venv, so `encode` raises a RuntimeError that names the
    `maturin develop` step.
    """
    if _RUST_AVAILABLE:
        pytest.skip(
            "Rust crate is built in this environment; the not-built "
            "envelope is exercised on the slim CI venv"
        )
    shim = TurboQuantKVShim(bits=4)
    weights = np.zeros((1, 32, 128), dtype=np.float32)
    with pytest.raises(RuntimeError, match="Rust crate is not built"):
        shim.encode(weights)


def test_shim_decode_raises_when_rust_not_built():
    if _RUST_AVAILABLE:
        pytest.skip(
            "Rust crate is built in this environment; the not-built "
            "envelope is exercised on the slim CI venv"
        )
    shim = TurboQuantKVShim(bits=4)
    with pytest.raises(RuntimeError, match="Rust crate is not built"):
        shim.decode(b"", np.zeros(1, dtype=np.float32), (1,))


def test_shim_encode_decode_round_trip_when_built():
    """When the Rust crate IS built, the shim's encode -> decode
    returns a float array of the original shape. The round-trip MSE
    floor is governed by the Lloyd-Max optimality criterion (see
    `tests/round_trip.rs` in the crate).
    """
    if not _RUST_AVAILABLE:
        pytest.skip(
            "Rust crate is not built in this environment; install "
            "with `maturin develop` to exercise the round-trip"
        )
    shim = TurboQuantKVShim(bits=4)
    weights = np.random.default_rng(seed=0).standard_normal((2, 32, 64)).astype(
        np.float32
    )
    packed, scales = shim.encode(weights)
    assert isinstance(packed, (bytes, bytearray))
    decoded = shim.decode(packed, scales, weights.shape)
    assert decoded.shape == weights.shape
    # The round-trip MSE is bounded by the Lloyd-Max optimality
    # criterion (see `tests/round_trip.rs::round_trip_4bit_unit_variance`).
    mse = float(((weights - decoded) ** 2).mean())
    assert mse < 0.1
