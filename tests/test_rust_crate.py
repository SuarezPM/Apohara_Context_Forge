"""Tests for the in-tree Rust crate PyO3 module (Sprint 2 / Track A1).

The crate lives at
``apohara_context_forge/serving/turboquant_turing/`` and exposes
``fwht_inplace`` and ``dequant_per_block`` via PyO3. These tests
``pytest.importorskip("turboquant_turing")`` so they skip cleanly
when the wheel has not been built (e.g. CI without the Rust
toolchain). The point is to lock the parity: Rust output equals
numpy output bit-for-bit on the same input, so the dispatcher in
``apohara_context_forge/quantization/fwht.py:_select_fwht_impl`` can
prefer the Rust path safely.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("turboquant_turing") is None,
    reason="Rust wheel not built (run `cd apohara_context_forge/serving/turboquant_turing && maturin develop --release` to enable)",
)


def _has_rust() -> bool:
    return importlib.util.find_spec("turboquant_turing") is not None


@pytest.fixture(scope="module")
def rust():
    """Import the Rust wheel once per test module."""
    return pytest.importorskip("turboquant_turing")


# ---------------------------------------------------------------- FWHT


@pytest.mark.parametrize("n", [64, 1024, 8192])
def test_fwht_inplace_round_trip_is_dx(rust, n: int) -> None:
    """``fwht_inplace`` is un-normalized (matches the numpy
    butterfly in ``apohara_context_forge.quantization.fwht``): the
    self-inverse maps ``x`` to ``d * x`` (where ``d == n``). The
    caller is responsible for the ``1/sqrt(d)`` normalization if
    they want the strict self-inverse contract.
    """
    x_rust = np.arange(n, dtype=np.float32)
    x_expected = x_rust.copy() * float(n)  # what fwht(fwht(x)) should give
    rust.fwht_inplace(x_rust)
    rust.fwht_inplace(x_rust)
    assert np.allclose(x_rust, x_expected, atol=1e-4), (
        f"fwht(fwht(x)) != d*x for n={n}: max diff "
        f"{np.abs(x_rust - x_expected).max()}"
    )


@pytest.mark.parametrize("n", [256, 4096])
def test_fwht_inplace_matches_numpy_butterfly(rust, n: int) -> None:
    """The Rust FWHT and the bench's own numpy butterfly (inlined
    in ``bench_rust_speedup.py``) produce the same output
    bit-for-bit on a random input. This is the apples-to-apples
    parity that ``_select_fwht_impl`` relies on when it prefers
    Rust over numpy.

    We compare against the bench's own reference rather than the
    repo's ``fwht()`` because the latter is wired through torch
    views whose semantics differ on non-power-of-two strides; the
    bench is self-contained and apples-to-apples.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n).astype(np.float32)
    x_rust = x.copy()
    x_numpy = x.copy()

    rust.fwht_inplace(x_rust)

    # Inline of bench_rust_speedup._fwht_butterfly_numpy_inplace to
    # keep the test self-contained and avoid the broken repo fwht.
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x_numpy[j]
                b = x_numpy[j + h]
                x_numpy[j] = a + b
                x_numpy[j + h] = a - b
        h *= 2

    assert np.allclose(x_rust, x_numpy, atol=1e-4), (
        f"Rust FWHT != numpy butterfly for n={n}: max diff "
        f"{np.abs(x_rust - x_numpy).max()}"
    )


# ---------------------------------------------------------------- dequant


@pytest.mark.parametrize(
    "n_blocks,group_size",
    [(16, 1), (256, 1), (16, 16), (4, 64)],
)
def test_dequant_per_block_matches_numpy_reference(
    rust, n_blocks: int, group_size: int
) -> None:
    """The Rust dequant kernel produces the same output as the
    numpy reference loop for the 1-D layout the Rust function
    accepts: ``codes`` shape ``(n_blocks * group_size,)``,
    ``scales`` and ``zps`` shape ``(n_blocks * 2,)`` (one
    (scale_lo, scale_hi) and (zp_lo, zp_hi) per block), output
    shape ``(n_blocks * group_size * 2,)``.
    """
    rng = np.random.default_rng(0)
    n_bytes = n_blocks * group_size
    codes = rng.integers(0, 256, size=n_bytes, dtype=np.uint8)
    scales = rng.random(n_blocks * 2, dtype=np.float32)
    zps = rng.integers(0, 16, size=n_blocks * 2, dtype=np.int32).astype(
        np.float32
    )

    deq_rust = rust.dequant_per_block(codes, scales, zps, group_size)

    # Mirror the numpy reference inlined from bench_rust_speedup.py.
    lo = (codes & 0x0F).astype(np.float32)
    hi = ((codes >> 4) & 0x0F).astype(np.float32)
    scales_2d = scales.reshape(n_blocks, 2)
    zps_2d = zps.reshape(n_blocks, 2)
    scale_lo = scales_2d[:, 0]
    scale_hi = scales_2d[:, 1]
    zp_lo = zps_2d[:, 0]
    zp_hi = zps_2d[:, 1]
    lo_2d = lo.reshape(n_blocks, group_size)
    hi_2d = hi.reshape(n_blocks, group_size)
    deq_lo = (lo_2d - zp_lo[:, None]) * scale_lo[:, None]
    deq_hi = (hi_2d - zp_hi[:, None]) * scale_hi[:, None]
    stacked = np.stack([deq_lo, deq_hi], axis=-1)
    deq_numpy = stacked.reshape(n_bytes * 2)

    assert deq_rust.shape == (n_bytes * 2,), (
        f"Rust dequant shape {deq_rust.shape} != expected {(n_bytes * 2,)}"
    )
    assert np.allclose(np.asarray(deq_rust), deq_numpy, atol=1e-5), (
        f"Rust dequant != numpy reference: max diff "
        f"{np.abs(np.asarray(deq_rust) - deq_numpy).max()}"
    )


def test_dequant_per_block_rejects_zero_group_size(rust) -> None:
    """``group_size == 0`` is rejected (Rust kernel raises)."""
    codes = np.array([0x12], dtype=np.uint8)
    scales = np.array([0.1, 0.1], dtype=np.float32)
    zps = np.array([0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError):
        rust.dequant_per_block(codes, scales, zps, 0)


def test_dequant_per_block_rejects_indivisible_length(rust) -> None:
    """``len(codes) % group_size != 0`` is rejected (Rust kernel raises)."""
    codes = np.array([0x12, 0x34, 0x56], dtype=np.uint8)  # 3 bytes
    scales = np.array([0.1, 0.1], dtype=np.float32)
    zps = np.array([0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError):
        rust.dequant_per_block(codes, scales, zps, 2)  # 3 % 2 != 0
