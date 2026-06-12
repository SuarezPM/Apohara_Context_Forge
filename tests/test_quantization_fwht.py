"""Tests for the Fast Walsh-Hadamard Transform — Sprint 2 / AUDIT #320a update.

This file extends the V7 ``test_fwht.py`` smoke with the new
``fwht(fwht(x)) == x`` assertion for the Rust-backed path. The
Rust kernel is gated by ``importlib.util.find_spec("turboquant_turing")``;
the test skips cleanly when the wheel is not built in the active
venv (the same fallback discipline the in-tree Python
``quantization/fwht.py:_select_fwht_impl`` dispatcher uses).
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from apohara_context_forge.quantization.fwht import (
    _select_fwht_impl,
    fwht,
    ifwht,
)


HADAMARD_4 = np.array(
    [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ],
    dtype=np.float32,
)


def _has_rust_wheel() -> bool:
    return importlib.util.find_spec("turboquant_turing") is not None


# ----------------------------------------------------------------------
# Dispatcher pinning
# ----------------------------------------------------------------------


def test_select_fwht_impl_prefers_rust_when_available():
    """The dispatcher must report ``"rust"`` when the wheel is
    importable, regardless of the allow_rust flag.
    """
    selected = _select_fwht_impl(allow_rust=True)
    if _has_rust_wheel():
        assert selected == "rust"
    else:
        assert selected == "numpy"


def test_select_fwht_impl_falls_back_when_disallowed():
    """``allow_rust=False`` forces the numpy path even when the
    wheel is present (used for cross-checking numerics during
    AUDIT audits).
    """
    assert _select_fwht_impl(allow_rust=False) == "numpy"


# ----------------------------------------------------------------------
# Numpy / torch reference (back-compat surface)
# ----------------------------------------------------------------------


def test_fwht_shape_preserved():
    x = np.random.randn(8).astype(np.float32)
    y = fwht(x)
    assert y.shape == x.shape


def test_fwht_power_of_two_no_pad():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(8).astype(np.float32)
    y = ifwht(fwht(x))
    assert np.allclose(x, y, atol=1e-5)


def test_fwht_non_power_of_two_padding():
    x = np.random.randn(6).astype(np.float32)
    y = fwht(x)
    assert y.shape[-1] == 8


def test_fwht_orthogonality():
    I4 = np.eye(4, dtype=np.float32)
    Y = fwht(I4)
    expected = HADAMARD_4 / np.sqrt(4.0)
    assert np.allclose(Y, expected, atol=1e-5)


def test_fwht_batched():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 8)).astype(np.float32)
    y = fwht(x)
    y0 = fwht(x[0])
    y1 = fwht(x[1])
    assert np.allclose(y[0], y0, atol=1e-6)
    assert np.allclose(y[1], y1, atol=1e-6)


def test_fwht_dtype_preservation():
    x32 = np.random.randn(8).astype(np.float32)
    assert fwht(x32).dtype == np.float32
    x16 = np.random.randn(8).astype(np.float16)
    y16 = fwht(x16)
    assert y16.dtype == np.float16
    rt = ifwht(fwht(x16))
    assert np.allclose(rt.astype(np.float32), x16.astype(np.float32), atol=5e-3)


def test_fwht_zero_input():
    x = np.zeros(8, dtype=np.float32)
    y = fwht(x)
    assert np.allclose(y, np.zeros(8, dtype=np.float32))


# ----------------------------------------------------------------------
# Rust path (skipif the wheel is not built)
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_rust_wheel(),
    reason="turboquant_turing wheel not built (run build.sh)",
)
def test_fwht_fwht_x_equals_x_via_rust():
    """FWHT identity: ``fwht(fwht(x)) == x`` under the sqrt(d)
    normalisation. With the Rust wheel installed the dispatcher
    routes the numpy call through
    ``turboquant_turing.fwht_inplace`` — this test pins the
    numerical identity of the Rust kernel against the
    well-known FWHT orthonormal contract.
    """
    rng = np.random.default_rng(11)
    for d in (8, 16, 32, 64, 128):
        x = rng.standard_normal(d).astype(np.float32)
        # Single fwht (Rust-backed, dispatched from the Python
        # helper). The Python helper applies the outer /sqrt(d)
        # so the round-trip below composes to the identity.
        y = fwht(x)
        # The same call again restores x.
        z = fwht(y)
        assert np.allclose(z, x, atol=1e-5), (
            f"FWHT round-trip drift at d={d}: max abs diff "
            f"= {np.abs(z - x).max()}"
        )


@pytest.mark.skipif(
    not _has_rust_wheel(),
    reason="turboquant_turing wheel not built (run build.sh)",
)
def test_fwht_rust_matches_numpy_butterfly_byte_for_byte():
    """The Rust butterfly in ``src/fwht.rs`` mirrors the numpy
    butterfly in ``fwht.py:_fwht_butterfly_numpy`` (the same
    ``a+b / a-b`` recursion, the same axis ordering). This test
    pins the equivalence on a 16-element sample: the Rust
    in-place output must equal the numpy reference output
    within float32 epsilon.
    """
    rng = np.random.default_rng(13)
    x = rng.standard_normal(16).astype(np.float32)

    # Numpy reference — the legacy butterfly the Rust kernel
    # mirrors.
    ref = x.copy()
    d = ref.shape[0]
    h = 1
    while h < d:
        view = ref.reshape(d // (2 * h), 2, h).copy()
        a = view[..., 0, :].copy()
        b = view[..., 1, :].copy()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        ref = view.reshape(d)
        h *= 2

    # Rust-backed butterfly.
    from apohara_context_forge.serving.turboquant_turing import (
        fwht_inplace as _rust_fwht_inplace,
    )
    rust = x.copy()
    _rust_fwht_inplace(rust)
    np.testing.assert_allclose(rust, ref, atol=1e-6)
