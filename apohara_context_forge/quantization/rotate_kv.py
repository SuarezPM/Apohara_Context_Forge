"""RotateKV Pre-RoPE Quantization — INT4 KV block compression.

Based on RotateKV (IJCAI 2025, arXiv:2501.16383):
- Outlier-Aware Rotation: channel reordering + FWHT to group channels
  by outlier distribution before rotation
- Pre-RoPE Grouped-Head Rotation: rotate BEFORE applying RoPE, not after,
  to avoid RoPE-induced inter-channel mixing that wrecks outlier isolation
- Attention-Sink-Aware Quantization: protect first N tokens (sinks) at
  full FP16, quantize the rest at INT4

Results from paper: 3.97x peak memory reduction, 2.32x decode speedup,
< 0.3 PPL degradation at 2-bit on WikiText-2 (LLaMA-2-13B).

V4.0: Target INT4 (4-bit) for balance quality/compression.

INVARIANT 10: This module ALWAYS receives key_states BEFORE RoPE is applied.
RoPE is applied externally after dequantize(). Breaking this contract corrupts attention.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np

from apohara_context_forge.quantization.fwht import fwht


@dataclass
class RotateKVConfig:
    """Configuration for RotateKV quantization."""
    bits: int = 4                # 2 | 4 | 8
    group_size: int = 64         # block-wise quantization block size (rows)
    sink_tokens: int = 4         # protect first N tokens at FP16
    # V7.0.0-alpha.5 MI300X measurement: FWHT degrades INT4 quality 200x under
    # per-byte joint-quant codec (MSE keys: 0.01 off vs 2.01 on). Default off.
    use_fwht: bool = False       # Fast Walsh-Hadamard Transform for outlier rotation
    grouped_heads: int = 2       # heads per rotation group (Pre-RoPE grouped-head)


@dataclass
class QuantizedKVBlock:
    """A quantized KV block with INT4 storage and FP16 sink tokens."""
    keys_int4: np.ndarray        # shape (seq_len - sink_tokens, num_heads, head_dim//2)
    values_int4: np.ndarray      # same
    keys_sink_fp16: np.ndarray   # shape (sink_tokens, num_heads, head_dim)
    values_sink_fp16: np.ndarray # same
    scales_k: np.ndarray         # per-block scales for keys (n_blocks, num_heads, head_dim//2)
    zero_points_k: np.ndarray   # per-block zero points for keys
    scales_v: np.ndarray         # per-block scales for values
    zero_points_v: np.ndarray    # per-block zero points for values
    channel_order: np.ndarray    # reordering indices for dequantization
    positions: np.ndarray        # original position indices (needed for RoPE)
    bits: int = 4


class RotateKVQuantizer:
    """
    Pre-RoPE INT4 quantizer for KV cache blocks.
    
    Usage:
        quantizer = RotateKVQuantizer(RotateKVConfig(bits=4))
        quantizer.calibrate(calibration_key_states)
        qblock, remaining_keys = quantizer.quantize_pre_rope(keys, values, positions)
        keys_fp16, values_fp16 = quantizer.dequantize(qblock)
    """
    
    def __init__(self, config: RotateKVConfig = RotateKVConfig()):
        self._config = config
        self._channel_order: Optional[np.ndarray] = None
        self._calibrated = False
    
    def calibrate(
        self,
        key_states_sample: np.ndarray,
        n_calibration_samples: int = 128,
    ) -> None:
        """
        Lightweight calibration to compute channel reordering indices.
        
        Algorithm:
        1. Reshape key_states to (N * seq_len, num_heads * head_dim)
        2. Sum channels across batch dimension
        3. Sort indices by activation magnitude (outlier proxy)
        4. Store self._channel_order: np.ndarray[int] for reuse
        
        This is a one-time offline step per model, not per request.
        
        Args:
            key_states_sample: np.ndarray of shape (N, seq_len, num_heads, head_dim)
                              pre-RoPE key states from calibration run
            n_calibration_samples: max samples to use for calibration
        """
        cfg = self._config
        # Use first n_calibration_samples from the sample
        n = min(n_calibration_samples, key_states_sample.shape[0])
        sample = key_states_sample[:n]
        
        # Reshape to (N * seq_len, num_heads * head_dim)
        N, seq_len, num_heads, head_dim = sample.shape
        reshaped = sample.reshape(N * seq_len, num_heads * head_dim)
        
        # Sum channels across batch dimension as activation magnitude proxy
        channel_magnitude = np.sum(np.abs(reshaped), axis=0)
        
        # Sort indices by magnitude (high magnitude = likely outlier = later in order)
        self._channel_order = np.argsort(channel_magnitude)
        self._calibrated = True
        
        # Store shape info for dequantization
        self._num_heads = num_heads
        self._head_dim = head_dim
    
    def quantize_pre_rope(
        self,
        key_states: np.ndarray,
        value_states: np.ndarray,
        positions: np.ndarray,
    ) -> Tuple["QuantizedKVBlock", np.ndarray]:
        """
        Quantize key_states BEFORE RoPE is applied.

        INVARIANT 10: This method ALWAYS receives pre-RoPE key_states.
        The returned QuantizedKVBlock contains pre-RoPE data. RoPE is applied
        externally after dequantization.

        Steps:
        1. Apply channel reordering (self._channel_order)
        2. Apply FWHT rotation across grouped heads (if use_fwht=True)
        3. Identify attention sinks: positions[:, :sink_tokens]
        4. Separate sink tokens (store as FP16) from rest (quantize as INT4)
        5. Block-wise asymmetric INT4 quantization (group_size rows per block)
        6. Store scale + zero_point per block for dequantization
        7. Return QuantizedKVBlock

        Args:
            key_states: np.ndarray shape (batch, seq_len, num_heads, head_dim) pre-RoPE,
                        or (seq_len, hidden_dim) for single-batch single-head input.
            value_states: np.ndarray same shape as key_states
            positions: np.ndarray shape (batch, seq_len) position indices,
                        or (seq_len,) for single-batch input.

        Returns:
            Tuple of (QuantizedKVBlock, key_states_post_quantization_for_RoPE)
            The second element is key_states after quantization (NOT dequantified).
            RoPE should be applied to this by the caller.
        """
        cfg = self._config

        # Promote 2D input (seq_len, hidden_dim) to canonical 4D
        # (batch=1, seq_len, num_heads=1, head_dim=hidden_dim).
        # Detection is done first so all downstream slicing assumes 4D.
        was_2d = key_states.ndim == 2
        if was_2d:
            seq_len_2d, hidden_dim_2d = key_states.shape
            key_states = key_states.reshape(1, seq_len_2d, 1, hidden_dim_2d)
            value_states = value_states.reshape(1, seq_len_2d, 1, hidden_dim_2d)
            if positions.ndim == 1:
                positions = positions.reshape(1, seq_len_2d)

        # Apply channel reordering if calibrated
        if self._channel_order is not None:
            key_states = key_states[:, :, :, self._channel_order]
            # Value states don't need reordering (handled separately)

        # Apply FWHT rotation along head_dim before block-wise quantization
        # (INV-10 preserved: RoPE has not been applied yet).
        if cfg.use_fwht:
            key_states = fwht(key_states)
            value_states = fwht(value_states)

        # Sink token separation
        # positions shape: (batch, seq_len) — identify sink positions
        # For sink tokens (first N in sequence), store as FP16
        sink_count = cfg.sink_tokens

        # Split along sequence dimension
        keys_sink = key_states[:, :sink_count, :, :]
        values_sink = value_states[:, :sink_count, :, :]
        keys_body = key_states[:, sink_count:, :, :]
        values_body = value_states[:, sink_count:, :, :]
        
        # Quantize body (non-sink) as INT4.
        # AUDIT #320 wiring fix: when use_fwht=True, the FWHT-expanded signal
        # destroys the per-byte V7 codec (200x MSE degradation). Dispatch to
        # CodecV8Quantizer._quantize_block (per-nibble independent scales) so
        # FWHT benefit is recovered. The per-byte V7 path is preserved for
        # use_fwht=False (zero behavior change for non-FWHT callers).
        # NOTE: deferred import — codec_v8 imports from this module (parent
        # class), so a top-level import would form a cycle.
        if cfg.use_fwht:
            from apohara_context_forge.quantization.codec_v8 import CodecV8Quantizer

            v8_quantizer = CodecV8Quantizer(self._config)
            keys_int4, scales_k, zero_points_k = v8_quantizer._quantize_block(keys_body)
            values_int4, scales_v, zero_points_v = v8_quantizer._quantize_block(values_body)
        else:
            keys_int4, scales_k, zero_points_k = self._quantize_block(keys_body)
            values_int4, scales_v, zero_points_v = self._quantize_block(values_body)
        
        # Create QuantizedKVBlock
        block = QuantizedKVBlock(
            keys_int4=keys_int4,
            values_int4=values_int4,
            keys_sink_fp16=keys_sink.astype(np.float16),
            values_sink_fp16=values_sink.astype(np.float16),
            scales_k=scales_k,
            zero_points_k=zero_points_k,
            scales_v=scales_v,
            zero_points_v=zero_points_v,
            channel_order=self._channel_order.copy() if self._channel_order is not None else np.array([]),
            positions=positions.copy(),
            bits=cfg.bits,
        )
        
        # Return block and key_states for RoPE (we pass through quantized body for RoPE application)
        # Actually we need to return something for RoPE - the caller will apply RoPE to dequantified output
        # But we store quantized, so RoPE is applied to dequantified: return the quantized body as "remaining"
        remaining_for_rope = keys_body  # This will be RoPE-applied externally to the dequantified values
        
        return block, remaining_for_rope
    
    def _quantize_block(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize a block of states to INT4.

        Packing: each byte at packed[blk, i, h, d] encodes head_dim positions
        (2*d, 2*d+1) at seq position (start + i). Lower nibble = 2*d, upper = 2*d+1.
        A single (scale, zero_point) governs both nibbles of the pair.
        """
        cfg = self._config
        batch, seq, num_heads, head_dim = states.shape

        n_blocks = seq // cfg.group_size
        if seq % cfg.group_size != 0:
            n_blocks += 1

        packed_head_dim = head_dim // 2
        max_range = 15.0 if cfg.bits == 4 else 255.0

        keys_int4 = np.zeros((n_blocks, cfg.group_size, num_heads, packed_head_dim), dtype=np.uint8)
        scales = np.zeros((n_blocks, num_heads, packed_head_dim), dtype=np.float32)
        zero_points = np.zeros((n_blocks, num_heads, packed_head_dim), dtype=np.float32)

        padded_seq = n_blocks * cfg.group_size
        valid_mask = np.zeros(padded_seq, dtype=bool)
        valid_mask[:seq] = True

        # Preserve pre-existing "last batch wins" semantics from the V6.1
        # Python loop (the packed array has no batch axis; each iteration
        # overwrites the prior batch's quantization).
        for b in range(batch):
            buf = np.zeros((padded_seq, num_heads, head_dim), dtype=states.dtype)
            buf[:seq] = states[b]

            # Reshape to (n_blocks, group_size, num_heads, packed_head_dim, 2)
            # so the last axis carries the (d_lo, d_hi) pair sharing scale/zp.
            blocks = buf.reshape(n_blocks, cfg.group_size, num_heads, packed_head_dim, 2)
            valid_blocks = valid_mask.reshape(n_blocks, cfg.group_size)

            # Joint min/max over (group_size, pair) → (n_blocks, num_heads, packed_head_dim).
            # Use valid_mask to ignore padded rows in the last partial block.
            valid_4d = valid_blocks[:, :, None, None, None]  # broadcast to blocks shape
            masked_for_min = np.where(valid_4d, blocks, np.inf)
            masked_for_max = np.where(valid_4d, blocks, -np.inf)
            min_val = np.min(masked_for_min, axis=(1, 4)).astype(np.float64)
            max_val = np.max(masked_for_max, axis=(1, 4)).astype(np.float64)

            # Match V6.1 sentinel: empty block (no valid rows) -> scale=0, zp=0.
            empty = ~valid_blocks.any(axis=1)  # (n_blocks,)

            range_ = max_val - min_val
            # V6.1 sets scale=1.0 when max==min (degenerate flat block); otherwise
            # scale=range/max_range. Empty blocks are zeroed afterwards.
            scale = np.where(range_ > 0, range_ / max_range, 1.0)
            # V6.1 used built-in round() -> banker's rounding; mirror via np.rint
            # which is half-to-even on numpy doubles.
            zp = np.where(scale != 0, -np.rint(min_val / scale), 0.0)

            scale_f32 = scale.astype(np.float32)
            zp_f32 = zp.astype(np.float32)
            scale_f32[empty] = 0.0
            zp_f32[empty] = 0.0

            # Quantize: shape (n_blocks, group_size, num_heads, packed_head_dim, 2).
            # Broadcast scale/zp over group_size + pair axes.
            scale_b = scale_f32[:, None, :, :, None]
            zp_b = zp_f32[:, None, :, :, None]
            # Guard div-by-zero on empty blocks where scale==0.
            safe_scale = np.where(scale_b == 0, 1.0, scale_b)
            q = np.clip(
                np.round(blocks / safe_scale + zp_b),
                0,
                max_range,
            ).astype(np.uint8)

            q_lo = q[..., 0]
            q_hi = q[..., 1]
            packed = (q_lo & 0xF) | ((q_hi & 0xF) << 4)

            # Zero out padded rows in the last partial block to match V6.1
            # (which skipped writes entirely past `end = min(start+gs, seq)`).
            packed = packed * valid_blocks[:, :, None, None].astype(np.uint8)

            keys_int4[:] = packed
            scales[:] = scale_f32
            zero_points[:] = zp_f32

        return keys_int4, scales, zero_points
    
    def dequantize(
        self,
        block: "QuantizedKVBlock",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restore FP16 key_states and value_states from QuantizedKVBlock.
        
        RoPE will be applied externally after dequantization (INVARIANT 10).
        
        Args:
            block: QuantizedKVBlock from quantize_pre_rope()
        
        Returns:
            Tuple of (key_states_fp16, value_states_fp16) both shape (batch, seq, num_heads, head_dim)
        """
        cfg = self._config

        # AUDIT #320 wiring fix: when use_fwht=True, the block was produced
        # by CodecV8Quantizer._quantize_block (per-nibble scales, trailing
        # pair axis on scales/zp). The matching dequantize must be V8's too,
        # otherwise the V7 per-byte dequantize broadcasts wrong. Same
        # deferred import as quantize_pre_rope (cycle avoidance).
        if cfg.use_fwht:
            from apohara_context_forge.quantization.codec_v8 import CodecV8Quantizer

            v8_quantizer = CodecV8Quantizer(self._config)
            keys_body = v8_quantizer._dequantize_block(
                block.keys_int4, block.scales_k, block.zero_points_k, cfg.group_size
            )
            values_body = v8_quantizer._dequantize_block(
                block.values_int4, block.scales_v, block.zero_points_v, cfg.group_size
            )
        else:
            keys_body = self._dequantize_block(block.keys_int4, block.scales_k, block.zero_points_k, cfg.group_size)
            values_body = self._dequantize_block(block.values_int4, block.scales_v, block.zero_points_v, cfg.group_size)
        
        # Concatenate sink (FP16) + body (dequantized)
        keys_fp16 = np.concatenate([block.keys_sink_fp16, keys_body], axis=1).astype(np.float32)
        values_fp16 = np.concatenate([block.values_sink_fp16, values_body], axis=1).astype(np.float32)
        
        # Apply channel de-ordering if stored
        if len(block.channel_order) > 0:
            # Create inverse permutation
            inv_order = np.argsort(block.channel_order)
            keys_fp16 = keys_fp16[:, :, :, inv_order]
        
        return keys_fp16, values_fp16
    
    def _dequantize_block(
        self,
        packed_int4: np.ndarray,
        scales: np.ndarray,
        zero_points: np.ndarray,
        group_size: int,
    ) -> np.ndarray:
        """Dequantize INT4 block back to FP32.

        Vectorized equivalent of the former quadruple Python loop. Both
        nibbles of each packed byte share one (scale, zero_point) — the V7
        per-byte codec. seq_len == n_blocks * group_size, so the original
        per-row bound check never fired; the whole grid is filled.
        """
        n_blocks, _, num_heads, packed_head_dim = packed_int4.shape
        seq_len = n_blocks * group_size

        # Unpack the two INT4 nibbles of every byte.
        val_lo = (packed_int4 & 0x0F).astype(np.float32)
        val_hi = ((packed_int4 >> 4) & 0x0F).astype(np.float32)

        # scales / zero_points are per (block, head, packed_dim); broadcast
        # over the group_size axis inserted at position 1.
        scale_b = scales[:, None, :, :].astype(np.float32)
        zp_b = zero_points[:, None, :, :].astype(np.float32)

        deq_lo = (val_lo - zp_b) * scale_b
        deq_hi = (val_hi - zp_b) * scale_b

        # Interleave back into head_dim: position d*2 = lo nibble, d*2+1 = hi.
        stacked = np.stack([deq_lo, deq_hi], axis=-1)
        body = stacked.reshape(n_blocks, group_size, num_heads, packed_head_dim * 2)

        # Flatten (n_blocks, group_size) -> seq_len row-major, matching the
        # original output[0, blk*group_size + i, ...] indexing.
        return body.reshape(1, seq_len, num_heads, packed_head_dim * 2)
    
    @property
    def is_calibrated(self) -> bool:
        """True if calibrate() has been called."""
        return self._calibrated
    
    @property
    def config(self) -> RotateKVConfig:
        """Current quantization config."""
        return self._config
