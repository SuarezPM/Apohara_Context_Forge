"""Integration tests for FWHT wired into RotateKVQuantizer — V7 codec work."""
import numpy as np
import pytest

from apohara_context_forge.quantization import fwht as fwht_module
from apohara_context_forge.quantization.fwht import fwht
from apohara_context_forge.quantization.rotate_kv import (
    QuantizedKVBlock,
    RotateKVConfig,
    RotateKVQuantizer,
)


def _make_inputs(seed: int = 0, seq_len: int = 64, num_heads: int = 8, head_dim: int = 64):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((1, seq_len, num_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((1, seq_len, num_heads, head_dim)).astype(np.float32)
    pos = np.arange(seq_len, dtype=np.float32)
    return k, v, pos


def test_use_fwht_false_unchanged():
    """When use_fwht=False, quantize_pre_rope matches V6.1 behavior (regression check)."""
    cfg = RotateKVConfig(bits=4, group_size=64, sink_tokens=4, use_fwht=False)
    quantizer = RotateKVQuantizer(cfg)

    k, v, pos = _make_inputs(seed=42)
    qblock, _ = quantizer.quantize_pre_rope(k.copy(), v.copy(), pos.copy())

    # Sink path is bit-exact when use_fwht=False (matches V6.1 slicing+astype).
    expected_sink_k = k[:, :4, :, :].astype(np.float16)
    expected_sink_v = v[:, :4, :, :].astype(np.float16)
    assert np.array_equal(qblock.keys_sink_fp16, expected_sink_k)
    assert np.array_equal(qblock.values_sink_fp16, expected_sink_v)


def test_use_fwht_true_calls_fwht():
    """When use_fwht=True, the FWHT transform actually runs (sink slice differs from raw)."""
    cfg_on = RotateKVConfig(bits=4, group_size=64, sink_tokens=4, use_fwht=True)
    cfg_off = RotateKVConfig(bits=4, group_size=64, sink_tokens=4, use_fwht=False)

    k, v, pos = _make_inputs(seed=7)

    qb_on, _ = RotateKVQuantizer(cfg_on).quantize_pre_rope(k.copy(), v.copy(), pos.copy())
    qb_off, _ = RotateKVQuantizer(cfg_off).quantize_pre_rope(k.copy(), v.copy(), pos.copy())

    # Sink is stored before quantization, so the FWHT effect is observable bit-exactly there.
    assert qb_on.keys_sink_fp16.shape == qb_off.keys_sink_fp16.shape
    assert not np.allclose(qb_on.keys_sink_fp16.astype(np.float32),
                           qb_off.keys_sink_fp16.astype(np.float32), atol=1e-3)

    # Sink slice with use_fwht=True equals fwht(k)[:, :sink, :, :] cast to fp16.
    expected = fwht(k)[:, :4, :, :].astype(np.float16)
    assert np.array_equal(qb_on.keys_sink_fp16, expected)


def test_fwht_preserves_inv10():
    """INV-10: quantize_pre_rope never applies RoPE — positions stay raw under both paths.

    RoPE is a position-dependent rotation; if it were applied internally, positions would
    either be consumed or remapped. The dataclass carries them through verbatim under
    both use_fwht=True and use_fwht=False, evidencing the pre-RoPE contract.
    """
    k, v, pos = _make_inputs(seed=1)

    for use_fwht in (False, True):
        cfg = RotateKVConfig(bits=4, group_size=64, sink_tokens=4, use_fwht=use_fwht)
        qb, _ = RotateKVQuantizer(cfg).quantize_pre_rope(k.copy(), v.copy(), pos.copy())
        # Positions copy through unchanged (no RoPE rotation applied to them).
        assert np.array_equal(qb.positions, pos)
        # Sink tokens are stored verbatim (no per-position rotation mixed in).
        # If RoPE had been applied, sink_fp16[i] would depend on positions[i].
        # We check that swapping positions does NOT change the stored sink.
        pos_shuffled = pos[::-1].copy()
        qb2, _ = RotateKVQuantizer(cfg).quantize_pre_rope(k.copy(), v.copy(), pos_shuffled)
        assert np.array_equal(qb.keys_sink_fp16, qb2.keys_sink_fp16)


def test_fwht_roundtrip_through_pipeline():
    """quantize -> dequantize round-trip preserves the tensor within INT4 quantization bounds.

    Uses an exact-block-sized body (seq=group_size, sink=0) so the V6.1 dequantize
    block-padding does not show through and the body comparison is well defined.

    Compares the use_fwht=True path against the use_fwht=False baseline: the FWHT
    rotation must not amplify the round-trip error beyond the V6.1 INT4 envelope.
    """
    k, v, pos = _make_inputs(seed=3, seq_len=64, num_heads=4, head_dim=32)

    # FWHT off — establish the V6.1 round-trip error envelope.
    cfg_off = RotateKVConfig(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    qz_off = RotateKVQuantizer(cfg_off)
    qb_off, _ = qz_off.quantize_pre_rope(k.copy(), v.copy(), pos.copy())
    k_deq_off, v_deq_off = qz_off.dequantize(qb_off)
    baseline_err_k = np.abs(k_deq_off - k).max()
    baseline_err_v = np.abs(v_deq_off - v).max()

    # FWHT on — round-trip must reconstruct fwht(k), and the error envelope must
    # stay within the same INT4 quantization regime (no blow-up from rotation).
    cfg_on = RotateKVConfig(bits=4, group_size=64, sink_tokens=0, use_fwht=True)
    qz_on = RotateKVQuantizer(cfg_on)
    qb_on, _ = qz_on.quantize_pre_rope(k.copy(), v.copy(), pos.copy())
    k_deq_on, v_deq_on = qz_on.dequantize(qb_on)

    fwht_k = fwht(k)
    fwht_v = fwht(v)
    assert k_deq_on.shape == fwht_k.shape
    assert v_deq_on.shape == fwht_v.shape

    # Error must stay within ~baseline magnitude. With the AUDIT #9 codec fix
    # the observed ratio is ~1.1x, so 1.5x INT4 envelope catches any future
    # nibble-packing regression while leaving small headroom for FP epsilon.
    assert np.abs(k_deq_on - fwht_k).max() <= baseline_err_k * 1.5
    assert np.abs(v_deq_on - fwht_v).max() <= baseline_err_v * 1.5


def test_fwht_batched_kv():
    """Works on a realistic batched KV shape (B=2, num_heads=8, seq_len=128, head_dim=64).

    Note: rotate_kv canonical layout is (batch, seq_len, num_heads, head_dim) per the
    docstring on quantize_pre_rope, so we feed the tensor in that order.
    """
    cfg = RotateKVConfig(bits=4, group_size=64, sink_tokens=4, use_fwht=True)
    quantizer = RotateKVQuantizer(cfg)

    rng = np.random.default_rng(11)
    # Canonical layout: (batch, seq_len, num_heads, head_dim)
    k = rng.standard_normal((2, 128, 8, 64)).astype(np.float32)
    v = rng.standard_normal((2, 128, 8, 64)).astype(np.float32)
    pos = np.tile(np.arange(128, dtype=np.float32), (2, 1))

    qb, _ = quantizer.quantize_pre_rope(k, v, pos)

    # Shapes are preserved correctly through the FWHT + quantization stack.
    assert qb.keys_sink_fp16.shape == (2, 4, 8, 64)
    assert qb.values_sink_fp16.shape == (2, 4, 8, 64)
    # Body went through quantizer; n_blocks * group_size covers the non-sink seq.
    assert qb.keys_int4.shape[2] == 8     # num_heads
    assert qb.keys_int4.shape[3] == 32    # head_dim // 2 packed
    # FWHT(k) shape equals input (head_dim=64 is a power of two; no padding).
    assert qb.keys_sink_fp16.shape[-1] == 64
