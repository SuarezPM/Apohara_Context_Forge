"""Tests for the V8 codec (per-nibble independent scales).

Design rationale: see ``docs/v8-codec-design.md``.

The V8 codec keeps the V7 packing layout and channel-reordering logic
intact; the only behavioral change is that each nibble of each packed
byte gets its own ``(scale, zero_point)`` instead of sharing one with
its pair.

These CPU-only tests pin three properties before MI300X
validation:

1. Shape invariant — packed INT4 has the V7 shape; scales/zps grow
   by a trailing pair axis of size 2.
2. Round-trip envelope — V8 reconstruction stays within the same
   half-step bound as V7 on uniform inputs.
3. Asymmetric-pair gain — V8 strictly beats V7 in reconstruction MSE
   on a fixture where the two channels in each pair carry deliberately
   asymmetric dynamic range.

V8 numbers DO NOT enter paper Table 3 until measured on real MI300X.
See ``docs/v8-codec-design.md`` § Acceptance criteria.
"""
from __future__ import annotations

import numpy as np
import pytest

from apohara_context_forge.quantization.codec_v8 import (
    CodecV8Config,
    CodecV8Quantizer,
)
from apohara_context_forge.quantization.rotate_kv import (
    RotateKVConfig,
    RotateKVQuantizer,
)


def _make_uniform_kv(seed: int = 0, seq: int = 64, num_heads: int = 4, head_dim: int = 32):
    """Uniform-distributed KV fixture for the round-trip envelope check."""
    rng = np.random.default_rng(seed)
    k = rng.random((1, seq, num_heads, head_dim), dtype=np.float64).astype(np.float32)
    v = rng.random((1, seq, num_heads, head_dim), dtype=np.float64).astype(np.float32)
    pos = np.arange(seq, dtype=np.float32)
    return k, v, pos


def _make_asymmetric_pair_kv(seed: int = 0, seq: int = 64, num_heads: int = 4, head_dim: int = 32):
    """KV fixture where even-index channels have small range and odd-index
    channels have large range. Each packed byte then sees a pair of
    (small, large) channels — exactly the case where V7's joint scale
    over-quantizes the small channel.
    """
    rng = np.random.default_rng(seed)
    base = rng.random((1, seq, num_heads, head_dim), dtype=np.float64).astype(np.float32)
    # Even channels: stay in [0, 0.1]. Odd channels: stretch to [0, 10].
    base[..., 0::2] *= 0.1
    base[..., 1::2] *= 10.0
    k = base
    v = base.copy()
    pos = np.arange(seq, dtype=np.float32)
    return k, v, pos


# ----------------------------------------------------------------------
# Test 1 — shape invariant
# ----------------------------------------------------------------------


def test_v8_scales_carry_pair_axis():
    """V8 scales/zps must have shape (n_blocks, num_heads, packed_head_dim, 2)
    while packed INT4 keeps the V7 shape (n_blocks, group_size, num_heads,
    packed_head_dim).
    """
    k, v, pos = _make_uniform_kv(seed=0)

    cfg = CodecV8Config(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    qz = CodecV8Quantizer(cfg)
    qb, _ = qz.quantize_pre_rope(k.copy(), v.copy(), pos.copy())

    n_blocks_expected = 1  # 64 seq / 64 group_size
    num_heads, head_dim = 4, 32
    packed_head_dim = head_dim // 2

    assert qb.keys_int4.shape == (n_blocks_expected, 64, num_heads, packed_head_dim)
    assert qb.scales_k.shape == (n_blocks_expected, num_heads, packed_head_dim, 2)
    assert qb.zero_points_k.shape == (n_blocks_expected, num_heads, packed_head_dim, 2)
    assert qb.scales_v.shape == (n_blocks_expected, num_heads, packed_head_dim, 2)
    assert qb.zero_points_v.shape == (n_blocks_expected, num_heads, packed_head_dim, 2)


# ----------------------------------------------------------------------
# Test 2 — round-trip envelope on uniform input
# ----------------------------------------------------------------------


def test_v8_uniform_roundtrip_envelope():
    """V8 round-trip on uniform [0,1] inputs stays inside the INT4 half-step
    envelope. Tolerance 0.07 matches the V7 test (V8 must not regress).
    """
    k, v, pos = _make_uniform_kv(seed=42)

    cfg = CodecV8Config(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    qz = CodecV8Quantizer(cfg)
    qb, _ = qz.quantize_pre_rope(k.copy(), v.copy(), pos.copy())
    k_deq, v_deq = qz.dequantize(qb)

    assert k_deq.shape == k.shape
    assert v_deq.shape == v.shape
    assert np.abs(k_deq - k).max() <= 0.07
    assert np.abs(v_deq - v).max() <= 0.07


# ----------------------------------------------------------------------
# Test 3 — V8 strictly beats V7 on asymmetric pairs
# ----------------------------------------------------------------------


def test_v8_beats_v7_on_asymmetric_pairs():
    """V8 reconstruction MSE strictly lower than V7 on an asymmetric-pair
    fixture. This is the headline correctness claim for the V8 codec
    design — if it fails, V8 has no reason to exist on top of V7.
    """
    k, v, pos = _make_asymmetric_pair_kv(seed=7)

    # V7 baseline:
    v7_cfg = RotateKVConfig(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    v7 = RotateKVQuantizer(v7_cfg)
    v7_qb, _ = v7.quantize_pre_rope(k.copy(), v.copy(), pos.copy())
    v7_k, v7_v = v7.dequantize(v7_qb)
    v7_mse_k = float(np.mean((k - v7_k) ** 2))
    v7_mse_v = float(np.mean((v - v7_v) ** 2))

    # V8 challenger:
    v8_cfg = CodecV8Config(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    v8 = CodecV8Quantizer(v8_cfg)
    v8_qb, _ = v8.quantize_pre_rope(k.copy(), v.copy(), pos.copy())
    v8_k, v8_v = v8.dequantize(v8_qb)
    v8_mse_k = float(np.mean((k - v8_k) ** 2))
    v8_mse_v = float(np.mean((v - v8_v) ** 2))

    # Strict inequality: V8 must beat V7 on this fixture. We expect a
    # large gap (>= 5x improvement) because the asymmetric-pair setup
    # is the worst case for V7.
    assert v8_mse_k < v7_mse_k, (
        f"V8 keys MSE {v8_mse_k} not better than V7 keys MSE {v7_mse_k}"
    )
    assert v8_mse_v < v7_mse_v, (
        f"V8 values MSE {v8_mse_v} not better than V7 values MSE {v7_mse_v}"
    )

    # Diagnostic: print the improvement ratio for grep-friendly logs.
    print(f"\nV8 keys MSE: {v8_mse_k:.6f}  V7 keys MSE: {v7_mse_k:.6f}  "
          f"ratio: {v7_mse_k / max(v8_mse_k, 1e-12):.2f}x")
    print(f"V8 values MSE: {v8_mse_v:.6f}  V7 values MSE: {v7_mse_v:.6f}  "
          f"ratio: {v7_mse_v / max(v8_mse_v, 1e-12):.2f}x")


# ----------------------------------------------------------------------
# Test 4 — V8 degrades gracefully to ~V7 on symmetric pairs
# ----------------------------------------------------------------------


def test_v8_symmetric_pairs_no_regression():
    """On a fixture where pair channels share the same distribution, V8
    must not be meaningfully worse than V7 (the metadata overhead is
    bandwidth-amortized; reconstruction parity is the codec-level claim).
    Tolerance: V8 MSE within 50% of V7 MSE. This is a sanity check, not
    a strict bound — uniform random inputs are not perfectly symmetric.
    """
    k, v, pos = _make_uniform_kv(seed=99)

    v7_cfg = RotateKVConfig(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    v7 = RotateKVQuantizer(v7_cfg)
    v7_qb, _ = v7.quantize_pre_rope(k.copy(), v.copy(), pos.copy())
    v7_k, _ = v7.dequantize(v7_qb)
    v7_mse_k = float(np.mean((k - v7_k) ** 2))

    v8_cfg = CodecV8Config(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    v8 = CodecV8Quantizer(v8_cfg)
    v8_qb, _ = v8.quantize_pre_rope(k.copy(), v.copy(), pos.copy())
    v8_k, _ = v8.dequantize(v8_qb)
    v8_mse_k = float(np.mean((k - v8_k) ** 2))

    # V8 is allowed to be slightly different (mid-tier change in the
    # min/max statistics), but not catastrophically worse. The actual
    # expectation from theory is that V8 is *at least as good* as V7
    # on any fixture where the per-pair statistics can be no worse
    # than the joint statistics, which is always.
    assert v8_mse_k <= v7_mse_k * 1.5, (
        f"V8 keys MSE {v8_mse_k} unexpectedly worse than V7 keys MSE {v7_mse_k}"
    )


# ----------------------------------------------------------------------
# Test 5 — V8 inherits INV-10 (pre-RoPE quantization invariant)
# ----------------------------------------------------------------------


def test_v8_inherits_pre_rope_invariant():
    """V8 must preserve INVARIANT 10: quantize_pre_rope returns a block
    whose ``positions`` array matches the input positions exactly, so
    that the external caller can apply RoPE post-dequantize.
    """
    k, v, pos = _make_uniform_kv(seed=0)

    cfg = CodecV8Config(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    qz = CodecV8Quantizer(cfg)
    qb, _ = qz.quantize_pre_rope(k.copy(), v.copy(), pos.copy())

    # positions roundtrip exactly (no modification by quantizer)
    np.testing.assert_array_equal(qb.positions.reshape(-1), pos)
    assert qb.bits == 4
