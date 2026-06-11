"""Tests for the INT4 pack/unpack codec in RotateKVQuantizer — AUDIT #9.

The V7 codec audit found the write side (`_quantize_block`) and the read side
(`_dequantize_block`) disagreed on the packing layout: the read side reads
both nibbles of a byte as adjacent head_dim slots (2*d, 2*d+1), but the
write side only ever wrote one head_dim slot per byte and packed adjacent
nibbles along the seq axis instead. Round-trip error was ~6.3 max abs
against an INT4 envelope of ~0.07 for [0,1] inputs.

These tests pin the corrected codec: each byte at packed[blk, i, h, d]
encodes head_dim positions (2*d, 2*d+1) with a shared (scale, zero_point).
"""
import numpy as np
import pytest

from apohara_context_forge.quantization.fwht import fwht
from apohara_context_forge.quantization.rotate_kv import (
    RotateKVConfig,
    RotateKVQuantizer,
)


def test_quantize_dequantize_roundtrip_identity():
    """quantize -> dequantize round-trip stays within the INT4 step envelope.

    Inputs in [0, 1] so the joint per-pair scale is bounded by 1/15, giving
    a per-slot max error <= half-step ~ 1/30 ~ 0.033 plus float epsilon.
    Tolerance 0.07 leaves a small safety margin without masking regressions.
    """
    rng = np.random.default_rng(0)
    k = rng.random((1, 64, 4, 32), dtype=np.float64).astype(np.float32)
    v = rng.random((1, 64, 4, 32), dtype=np.float64).astype(np.float32)
    pos = np.arange(64, dtype=np.float32)

    cfg = RotateKVConfig(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    qz = RotateKVQuantizer(cfg)
    qb, _ = qz.quantize_pre_rope(k.copy(), v.copy(), pos.copy())
    k_deq, v_deq = qz.dequantize(qb)

    assert k_deq.shape == k.shape
    assert v_deq.shape == v.shape
    assert np.abs(k_deq - k).max() <= 0.07
    assert np.abs(v_deq - v).max() <= 0.07


def test_quantize_dequantize_roundtrip_with_fwht():
    """Round-trip with use_fwht=True stays within an expanded but bounded envelope.

    FWHT preserves L2 energy but expands the per-channel range by up to
    sqrt(head_dim) for adversarial inputs, so a slightly looser tolerance
    is needed — but it must remain MUCH tighter than the 3x slack the
    pre-fix codec required.
    """
    rng = np.random.default_rng(1)
    k = rng.random((1, 64, 4, 32), dtype=np.float64).astype(np.float32)
    v = rng.random((1, 64, 4, 32), dtype=np.float64).astype(np.float32)
    pos = np.arange(64, dtype=np.float32)

    cfg = RotateKVConfig(bits=4, group_size=64, sink_tokens=0, use_fwht=True)
    qz = RotateKVQuantizer(cfg)
    qb, _ = qz.quantize_pre_rope(k.copy(), v.copy(), pos.copy())
    k_deq, v_deq = qz.dequantize(qb)

    fwht_k = fwht(k)
    fwht_v = fwht(v)
    assert k_deq.shape == fwht_k.shape
    assert np.abs(k_deq - fwht_k).max() <= 0.20
    assert np.abs(v_deq - fwht_v).max() <= 0.20


def test_packed_array_shape():
    """packed_int4 has shape (n_blocks, group_size, num_heads, head_dim // 2).

    The head_dim axis is HALVED (two nibbles per byte). Pre-fix layout
    accidentally allocated head_dim slots that the read side never touched.
    """
    rng = np.random.default_rng(2)
    k = rng.random((1, 64, 8, 64), dtype=np.float64).astype(np.float32)
    v = rng.random((1, 64, 8, 64), dtype=np.float64).astype(np.float32)
    pos = np.arange(64, dtype=np.float32)

    cfg = RotateKVConfig(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    qz = RotateKVQuantizer(cfg)
    qb, _ = qz.quantize_pre_rope(k.copy(), v.copy(), pos.copy())

    # (n_blocks=1, group_size=64, num_heads=8, packed_head_dim=32)
    assert qb.keys_int4.shape == (1, 64, 8, 32)
    assert qb.values_int4.shape == (1, 64, 8, 32)
    # head_dim axis is exactly head_dim // 2
    assert qb.keys_int4.shape[3] == 64 // 2


def test_packed_array_byte_values():
    """Each byte holds two distinct nibbles for the (2*d, 2*d+1) head_dim pair.

    Construct an input where the joint quantization gives a known (lower, upper)
    pair so the byte value is predictable. d=0 sees (0.0, 0.5) -> (0, 15) -> 0xF0.
    d=1 sees (1.0, 0.25) -> (15, 0) -> 0x0F.
    """
    cfg = RotateKVConfig(bits=4, group_size=4, sink_tokens=0, use_fwht=False)
    qz = RotateKVQuantizer(cfg)

    # Repeat the same head_dim vector across all seq positions so every byte
    # encodes the same (lower, upper) pair regardless of seq index.
    row = np.array([0.0, 0.5, 1.0, 0.25], dtype=np.float32)
    k = np.broadcast_to(row, (1, 4, 1, 4)).astype(np.float32).copy()
    v = k.copy()
    pos = np.arange(4, dtype=np.float32)

    qb, _ = qz.quantize_pre_rope(k, v, pos)

    # Every byte at d=0 packs (0.0, 0.5) -> lower=0, upper=15 -> 0xF0 = 240.
    # Every byte at d=1 packs (1.0, 0.25) -> lower=15, upper=0 -> 0x0F = 15.
    assert np.all(qb.keys_int4[0, :, 0, 0] == 0xF0)
    assert np.all(qb.keys_int4[0, :, 0, 1] == 0x0F)

    # Confirm both nibbles are recoverable (read side is the inverse).
    k_deq, _ = qz.dequantize(qb)
    np.testing.assert_allclose(k_deq[0, :, 0, :], np.broadcast_to(row, (4, 4)), atol=0.07)


# ---------------------------------------------------------------------------
# US-003 / Phase 1 — FWHT path now dispatches to CodecV8Quantizer (AUDIT #320)
# ---------------------------------------------------------------------------

def test_use_fwht_true_dispatches_to_codec_v8():
    """use_fwht=True path must call CodecV8Quantizer._quantize_block.

    Closes AUDIT #320 wiring gap: the RotateKVQuantizer used to fall through
    to its per-byte V7 codec regardless of use_fwht, producing 200x MSE
    degradation. The fix is a dispatch guard at quantize_pre_rope that
    instantiates CodecV8Quantizer and routes the body through its per-nibble
    _quantize_block.
    """
    from unittest.mock import patch

    from apohara_context_forge.quantization.codec_v8 import CodecV8Quantizer

    rng = np.random.default_rng(100)
    k = rng.random((1, 64, 4, 32), dtype=np.float64).astype(np.float32)
    v = rng.random((1, 64, 4, 32), dtype=np.float64).astype(np.float32)
    pos = np.arange(64, dtype=np.float32)

    cfg = RotateKVConfig(bits=4, group_size=64, sink_tokens=0, use_fwht=True)
    qz = RotateKVQuantizer(cfg)

    with patch.object(
        CodecV8Quantizer, "_quantize_block", autospec=True, return_value=(None, None, None)
    ) as v8_block:
        qz.quantize_pre_rope(k.copy(), v.copy(), pos.copy())

    # CodecV8Quantizer._quantize_block must be called twice (once for keys, once for values).
    assert v8_block.call_count == 2, (
        f"FWHT path should dispatch to CodecV8Quantizer twice (k+v); got {v8_block.call_count}"
    )


def test_use_fwht_true_mse_parity_on_fixed_fixture():
    """V8 codec on the rotated signal (dispatched by use_fwht=True) must
    produce a strictly better MSE than the V7 codec on the rotated signal
    (the broken AUDIT #320 path).

    The AUDIT #320 "200x" claim was V7 codec on rotated-signal MSE vs
    V7 codec on unrotated-signal MSE (apples-to-oranges). The honest
    fix claim is: replacing V7-with-FWHT (the bug) with V8-with-FWHT
    (the fix) MUST yield a strictly smaller MSE on the rotated signal.
    A 200x ratio was the broken state; the fix is a measurable win.
    """
    np.random.seed(42)  # explicit fixed seed (spec instruction)
    rng = np.random.default_rng(42)
    k = rng.random((1, 128, 4, 64), dtype=np.float64).astype(np.float32)
    v = rng.random((1, 128, 4, 64), dtype=np.float64).astype(np.float32)
    pos = np.arange(128, dtype=np.float32)

    fwht_k = fwht(k)
    fwht_v = fwht(v)

    # The BUG: V7 codec on the rotated signal.
    cfg_v7 = RotateKVConfig(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    qz_v7 = RotateKVQuantizer(cfg_v7)
    qb_v7, _ = qz_v7.quantize_pre_rope(fwht_k.copy(), fwht_v.copy(), pos.copy())
    k_v7_deq, v_v7_deq = qz_v7.dequantize(qb_v7)
    mse_v7_k = float(np.mean((k_v7_deq - fwht_k) ** 2))
    mse_v7_v = float(np.mean((v_v7_deq - fwht_v) ** 2))

    # The FIX: V8 codec on the rotated signal (via the dispatch).
    cfg_v8 = RotateKVConfig(bits=4, group_size=64, sink_tokens=0, use_fwht=True)
    qz_v8 = RotateKVQuantizer(cfg_v8)
    qb_v8, _ = qz_v8.quantize_pre_rope(k.copy(), v.copy(), pos.copy())
    k_v8_deq, v_v8_deq = qz_v8.dequantize(qb_v8)
    mse_v8_k = float(np.mean((k_v8_deq - fwht_k) ** 2))
    mse_v8_v = float(np.mean((v_v8_deq - fwht_v) ** 2))

    # V8 must beat V7 on the rotated signal — the AUDIT #320 fix claim.
    assert mse_v8_k < mse_v7_k, (
        f"V8 codec on rotated signal (MSE={mse_v8_k:.4e}) is NOT strictly "
        f"better than V7 codec (MSE={mse_v7_k:.4e}) — AUDIT #320 fix is broken"
    )
    assert mse_v8_v < mse_v7_v, (
        f"V8 codec on rotated signal (MSE={mse_v8_v:.4e}) is NOT strictly "
        f"better than V7 codec (MSE={mse_v7_v:.4e}) — AUDIT #320 fix is broken"
    )


def test_use_fwht_true_mse_parity_hotpotqa_shaped():
    """V8 codec on the rotated signal must beat V7 codec on the rotated
    signal at HotpotQA-shaped attention-block scale.

    Shape: (batch=1, seq=512, num_heads=32, head_dim=128) — single batch
    of seq=512, 32 heads, head_dim=128, the AUDIT #320 reproducer.
    Same logic as the fixed-fixture test, scaled up to the reproducer.
    """
    np.random.seed(42)  # explicit fixed seed (spec instruction)
    rng = np.random.default_rng(42)
    k = rng.random((1, 512, 32, 128), dtype=np.float64).astype(np.float32)
    v = rng.random((1, 512, 32, 128), dtype=np.float64).astype(np.float32)
    pos = np.arange(512, dtype=np.float32)

    fwht_k = fwht(k)
    fwht_v = fwht(v)

    # The BUG: V7 codec on the rotated signal.
    cfg_v7 = RotateKVConfig(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    qz_v7 = RotateKVQuantizer(cfg_v7)
    qb_v7, _ = qz_v7.quantize_pre_rope(fwht_k.copy(), fwht_v.copy(), pos.copy())
    k_v7_deq, v_v7_deq = qz_v7.dequantize(qb_v7)
    mse_v7_k = float(np.mean((k_v7_deq - fwht_k) ** 2))
    mse_v7_v = float(np.mean((v_v7_deq - fwht_v) ** 2))

    # The FIX: V8 codec on the rotated signal (via the dispatch).
    cfg_v8 = RotateKVConfig(bits=4, group_size=64, sink_tokens=0, use_fwht=True)
    qz_v8 = RotateKVQuantizer(cfg_v8)
    qb_v8, _ = qz_v8.quantize_pre_rope(k.copy(), v.copy(), pos.copy())
    k_v8_deq, v_v8_deq = qz_v8.dequantize(qb_v8)
    mse_v8_k = float(np.mean((k_v8_deq - fwht_k) ** 2))
    mse_v8_v = float(np.mean((v_v8_deq - fwht_v) ** 2))

    assert mse_v8_k < mse_v7_k, (
        f"HotpotQA-shaped V8 codec on rotated signal (MSE={mse_v8_k:.4e}) "
        f"is NOT strictly better than V7 codec (MSE={mse_v7_k:.4e}) — "
        f"AUDIT #320 fix is broken"
    )
    assert mse_v8_v < mse_v7_v, (
        f"HotpotQA-shaped V8 codec on rotated signal (MSE={mse_v8_v:.4e}) "
        f"is NOT strictly better than V7 codec (MSE={mse_v7_v:.4e}) — "
        f"AUDIT #320 fix is broken"
    )
