"""Tests for the true-batched V8 codec (Sprint 2 / AUDIT #320a).

The previous implementation of ``CodecV8Quantizer._quantize_block``
collapsed the per-batch loop into a single shared output buffer (the
``for b in range(batch)`` at line 133 of the V6.1 code); only the
last batch's quantization was returned. The new
``_quantize_block_batched`` method works on the full leading-batch
axis in one vectorized pass and the public ``_quantize_block``
wrapper preserves the legacy 4-D-in / 4-D-out contract by squeezing
the leading axis on the way out.

This file pins three properties:

1. **Batched shape contract** — ``_quantize_block_batched`` returns
   ``(batch, n_blocks, group_size, num_heads, packed_head_dim)`` for
   the codes and ``(batch, n_blocks, num_heads, packed_head_dim, 2)``
   for the scales/zero_points, with ``batch`` equal to the leading
   dim of the input.
2. **Per-doc equivalence** — calling ``_quantize_block_batched(x)``
   on a stack of docs is bit-identical (max abs diff < 1e-6) to
   stacking ``_quantize_block(x_i)`` for each doc ``x_i`` along the
   leading axis. This is the "mathematical equivalence" assertion
   the Sprint 2 spec calls out (max abs diff < 1e-6 on a 4-doc
   sample).
3. **Round-trip envelope** — the new batched path still respects
   the V8 INT4 half-step bound on uniform input, same as the
   single-doc path.
"""
from __future__ import annotations

import numpy as np
import pytest

from apohara_context_forge.quantization.codec_v8 import (
    CodecV8Config,
    CodecV8Quantizer,
)


# ----------------------------------------------------------------------
# 1. Batched shape contract
# ----------------------------------------------------------------------


def test_batched_shape_contract_default_group_size():
    """``_quantize_block_batched`` returns the documented 5-D layout."""
    cfg = CodecV8Config(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    q = CodecV8Quantizer(cfg)
    rng = np.random.default_rng(0)
    # 4 docs, each a (seq=1, num_heads=1, head_dim=32) row.
    x = rng.standard_normal((4, 1, 1, 32)).astype(np.float32)
    keys, scales, zps = q._quantize_block_batched(x)
    assert keys.shape == (4, 1, 64, 1, 16), keys.shape
    assert scales.shape == (4, 1, 1, 16, 2), scales.shape
    assert zps.shape == (4, 1, 1, 16, 2), zps.shape
    assert keys.dtype == np.uint8
    assert scales.dtype == np.float32
    assert zps.dtype == np.float32


def test_batched_shape_contract_realistic():
    """64 docs × 768-d — the TurbovecStore._add_ram_optimised shape."""
    cfg = CodecV8Config(bits=4, group_size=1, sink_tokens=0, use_fwht=False)
    q = CodecV8Quantizer(cfg)
    rng = np.random.default_rng(0)
    # Each doc is a (seq=1, num_heads=1, head_dim=768) row.
    x = rng.standard_normal((64, 1, 1, 768)).astype(np.float32)
    keys, scales, zps = q._quantize_block_batched(x)
    assert keys.shape == (64, 1, 1, 1, 384), keys.shape
    assert scales.shape == (64, 1, 1, 384, 2), scales.shape
    assert zps.shape == (64, 1, 1, 384, 2), zps.shape


# ----------------------------------------------------------------------
# 2. Per-doc equivalence (mathematical parity)
# ----------------------------------------------------------------------


def test_batched_matches_per_doc_loop_max_abs_diff_4_docs():
    """Mathematical parity — the Sprint 2 spec's headline correctness
    assertion: ``max abs diff < 1e-6`` between the new batched path
    and stacking the per-doc results.
    """
    cfg = CodecV8Config(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    q = CodecV8Quantizer(cfg)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((4, 64, 4, 32)).astype(np.float32)

    # Per-doc reference (the legacy 4-D contract, one call per doc).
    expected_keys = []
    expected_scales = []
    expected_zps = []
    for i in range(4):
        k, s, z = q._quantize_block(x[i:i + 1])
        expected_keys.append(k)
        expected_scales.append(s)
        expected_zps.append(z)
    expected_keys = np.stack(expected_keys, axis=0)
    expected_scales = np.stack(expected_scales, axis=0)
    expected_zps = np.stack(expected_zps, axis=0)

    # Batched path — one call.
    keys, scales, zps = q._quantize_block_batched(x)

    # Codes are uint8 — the difference is exact bit equality.
    diff_keys = int(np.abs(
        keys.astype(np.int32) - expected_keys.astype(np.int32)
    ).max())
    assert diff_keys == 0, f"keys differ by {diff_keys}"

    # Scales / zero_points are float32; the multiplication
    # chain has the same IEEE-754 ordering, so the diff is below
    # the spec's 1e-6 threshold.
    diff_scales = float(np.abs(scales - expected_scales).max())
    diff_zps = float(np.abs(zps - expected_zps).max())
    assert diff_scales < 1e-6, f"scales differ by {diff_scales}"
    assert diff_zps < 1e-6, f"zps differ by {diff_zps}"


def test_batched_matches_per_doc_loop_64_docs_uniform():
    """Larger parity check on uniform [0, 1] input — the same input
    distribution the round-trip envelope test in
    ``test_codec_v8.py`` uses.
    """
    cfg = CodecV8Config(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    q = CodecV8Quantizer(cfg)
    rng = np.random.default_rng(0)
    # 64 docs, seq=64, num_heads=4, head_dim=32 — exercises the
    # partial-block path on the last doc (we vary seq a tiny bit).
    seqs = [64, 63, 64, 32] * 16  # 64 docs total
    head_dim = 32
    num_heads = 4

    # Reference: per-doc loop.
    expected_keys = []
    expected_scales = []
    expected_zps = []
    batched_inputs = []
    for s in seqs:
        x = rng.random((1, s, num_heads, head_dim), dtype=np.float32)
        batched_inputs.append(x)
        k, sc, z = q._quantize_block(x)
        expected_keys.append(k)
        expected_scales.append(sc)
        expected_zps.append(z)
    expected_keys = np.stack(expected_keys, axis=0)
    expected_scales = np.stack(expected_scales, axis=0)
    expected_zps = np.stack(expected_zps, axis=0)

    # Batched: pad all docs to the same seq length (max(seqs)) and
    # stack. The V8 batched path uses a per-doc valid mask, so the
    # padding rows are ignored in the min/max reduction.
    max_seq = max(seqs)
    n_blocks = (max_seq + cfg.group_size - 1) // cfg.group_size
    padded_seq = n_blocks * cfg.group_size
    batched = np.zeros(
        (len(seqs), padded_seq, num_heads, head_dim), dtype=np.float32
    )
    for i, x in enumerate(batched_inputs):
        batched[i, : x.shape[1]] = x[0]
    keys, scales, zps = q._quantize_block_batched(batched)

    # The codes are written only on the valid rows — masked rows
    # are zero. For the per-doc loop, only the [0, seq_i) rows of
    # each block carry data, but the per-doc loop was called with
    # the unpadded doc (so the per-doc output is (n_blocks_i,
    # group_size, num_heads, packed_head_dim) with n_blocks_i
    # padded to ceil(seq_i / group_size)). The batched output has
    # n_blocks = ceil(max_seq / group_size), so the per-doc loop
    # outputs are a *prefix* of the batched outputs. Compare
    # element-wise on the first n_blocks_i rows of each doc.
    n_blocks_per_doc = [
        (s + cfg.group_size - 1) // cfg.group_size for s in seqs
    ]
    for i, n_b in enumerate(n_blocks_per_doc):
        diff_k = int(np.abs(
            keys[i, :n_b].astype(np.int32)
            - expected_keys[i, :n_b].astype(np.int32)
        ).max())
        assert diff_k == 0, f"doc {i} keys differ by {diff_k}"
        diff_s = float(np.abs(
            scales[i, :n_b] - expected_scales[i, :n_b]
        ).max())
        diff_z = float(np.abs(
            zps[i, :n_b] - expected_zps[i, :n_b]
        ).max())
        assert diff_s < 1e-6, f"doc {i} scales differ by {diff_s}"
        assert diff_z < 1e-6, f"doc {i} zps differ by {diff_z}"


# ----------------------------------------------------------------------
# 3. Legacy 4-D contract preservation
# ----------------------------------------------------------------------


def test_legacy_4d_contract_preserved():
    """The public 4-D-in / 4-D-out contract of ``_quantize_block`` is
    preserved by the wrapper — the batched math runs underneath, and
    the leading axis is sliced off on the way out.
    """
    cfg = CodecV8Config(bits=4, group_size=64, sink_tokens=0, use_fwht=False)
    q = CodecV8Quantizer(cfg)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 64, 4, 32)).astype(np.float32)
    keys, scales, zps = q._quantize_block(x)
    # Legacy shape: (n_blocks=1, group_size=64, num_heads=4, packed_head_dim=16)
    # and (n_blocks=1, num_heads=4, packed_head_dim=16, 2).
    assert keys.shape == (1, 64, 4, 16), keys.shape
    assert scales.shape == (1, 4, 16, 2), scales.shape
    assert zps.shape == (1, 4, 16, 2), zps.shape


# ----------------------------------------------------------------------
# 4. Round-trip envelope (declared parity, not measured downstream)
# ----------------------------------------------------------------------


def test_batched_round_trip_envelope_uniform():
    """V8 INT4 half-step bound holds under the batched path on uniform
    input — the same envelope the single-doc test pins.
    """
    cfg = CodecV8Config(bits=4, group_size=1, sink_tokens=0, use_fwht=False)
    q = CodecV8Quantizer(cfg)
    rng = np.random.default_rng(42)
    # 4 docs, each a single 768-d row.
    x = rng.random((4, 1, 1, 768), dtype=np.float32)
    keys, scales, zps = q._quantize_block_batched(x)
    # Reconstruct by dequantizing each doc (the batched dequant path
    # is a follow-up; the V8 contract is per-doc here).
    deq = np.empty_like(x)
    for i in range(4):
        deq[i:i + 1] = q._dequantize_block(
            keys[i], scales[i], zps[i], cfg.group_size
        )
    # The INT4 half-step on [0, 1] inputs is 1/16 ≈ 0.0625.
    assert np.abs(deq - x).max() <= 0.07
