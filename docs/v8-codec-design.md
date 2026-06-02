# V8 Codec Design — Per-Nibble Independent Scales

> **Status:** design draft, CPU-only prep for MI300X validation.
> Paper v2.0.1 §limitations identifies this as future work; this document
> formalizes the math, expected reduction, validation plan, and the
> exact metric thresholds for shipping V8 in the paper v2.1 codec table.
>
> **Author:** Pablo M. Suarez, UNT
> **Date:** 2026-05-13
> **License:** Apache-2.0 (same as project)

---

## Problem statement

The V7 codec achieves **3.55× INT4 reduction** measured constant across
4K-262K context on MI300X (paper v2.0.1 §5). The theoretical floor for
FP16→INT4 is **4.0×**, and the published RotateKV literature target is
**3.97×**. The 3.55× point sits **89% of the way to the literature
target** and **89% of the way to the theoretical limit**.

The 0.42× gap (between 3.55× and 3.97×) has a single mechanical cause
in the V7 codec, identified in `apohara_context_forge/quantization/rotate_kv.py`
lines 240-249:

```python
# Reshape to (n_blocks, group_size, num_heads, packed_head_dim, 2)
# so the last axis carries the (d_lo, d_hi) pair sharing scale/zp.
blocks = buf.reshape(n_blocks, cfg.group_size, num_heads, packed_head_dim, 2)
# ...
# Joint min/max over (group_size, pair) → (n_blocks, num_heads, packed_head_dim).
min_val = np.min(masked_for_min, axis=(1, 4)).astype(np.float64)
max_val = np.max(masked_for_max, axis=(1, 4))
```

The `axis=(1, 4)` collapses the **pair axis**, meaning the lower nibble
(`d_lo`) and upper nibble (`d_hi`) of each packed byte share a single
`(scale, zero_point)`. When the two channels in a pair have asymmetric
dynamic range (which is typical for transformer KV projections, especially
near attention-sink positions), the shared scale **over-quantizes** one
nibble to fit the other into the INT4 range.

V8 fixes this by giving each nibble its own scale and zero point.

## Design

### Math

For a given block `b ∈ {0, ..., n_blocks-1}`, head `h ∈ {0, ..., num_heads-1}`,
packed-byte index `d ∈ {0, ..., head_dim/2 - 1}`, V7 computes:

```
V7  (current):  min_v7(b,h,d) = min over (i, pair) of states[b,h,d,pair,i]
                max_v7(b,h,d) = max over (i, pair) of states[b,h,d,pair,i]
                scale_v7(b,h,d) = (max_v7 - min_v7) / 15
                zp_v7(b,h,d)    = -min_v7 / scale_v7
                # one scale per packed byte → both nibbles share it
```

V8 computes the same statistics **without collapsing the pair axis**:

```
V8  (proposed): min_v8(b,h,d,p) = min over i of states[b,h,d,p,i]   for p ∈ {0,1}
                max_v8(b,h,d,p) = max over i of states[b,h,d,p,i]
                scale_v8(b,h,d,p) = (max_v8 - min_v8) / 15
                zp_v8(b,h,d,p)    = -min_v8 / scale_v8
                # two scales per packed byte → each nibble has its own
```

### Storage cost analysis

| Quantity            | V7 size                                            | V8 size                                              | Ratio   |
| ------------------- | -------------------------------------------------- | ---------------------------------------------------- | ------- |
| Packed INT4 weights | `n_blocks × group_size × num_heads × head_dim / 2` | `n_blocks × group_size × num_heads × head_dim / 2`   | 1×      |
| Scales (FP32)       | `n_blocks × num_heads × head_dim / 2`              | `n_blocks × num_heads × head_dim / 2 × 2`            | **2×**  |
| Zero points (FP32)  | `n_blocks × num_heads × head_dim / 2`              | `n_blocks × num_heads × head_dim / 2 × 2`            | **2×**  |

The metadata doubling sounds expensive but is amortized over `group_size`.
For `group_size=64` (the V7 default), each scale+zp pair governs **64
position rows** of INT4 data, so the metadata is **~6% overhead** in V7
and **~12% overhead** in V8. The expected net reduction:

| Codec | INT4 weights | Metadata overhead | Net reduction |
| ----- | -----------: | ----------------: | ------------: |
| FP16 baseline | 1.0×  |         (none)    |          1.0× |
| V7 (joint nibbles)    |  4.0× theoretical |          ~6%       | **3.55×** measured |
| V8 (independent nibbles) |  4.0× theoretical |         ~12%       | **3.85-3.97×** projected |

### Why 3.85× and not 3.97× exactly?

The V8 projection (3.85-3.97×) has a range because the metadata cost
is **fixed**, but the quantization-quality gain depends on the actual
distribution of dynamic-range asymmetry within nibble pairs. If pairs
are highly asymmetric (sink tokens, outlier channels) V8 reclaims most
of the gap. If pairs are nearly symmetric (deep-layer activations
after RoPE) V8 reclaims less.

The MI300X validation will measure the actual point along
the 3.85-3.97× interval. The paper v2.1 §codec section then ships
**measured** numbers, not projections — consistent with V6.1 honesty
discipline.

---

## Implementation plan

### Phase 1 — CPU skeleton (no GPU needed)

File: `apohara_context_forge/quantization/codec_v8.py`

```python
from dataclasses import dataclass
import numpy as np
from apohara_context_forge.quantization.rotate_kv import (
    RotateKVConfig, QuantizedKVBlock, RotateKVQuantizer
)


@dataclass
class CodecV8Config(RotateKVConfig):
    """V8 codec: per-nibble independent scales.

    Inherits all V7 fields and behavior. The only difference is that
    scales_k / scales_v / zero_points_k / zero_points_v have an
    additional trailing axis of size 2 (one entry per nibble in
    each packed byte).
    """
    codec_version: str = "v8"


class CodecV8Quantizer(RotateKVQuantizer):
    """V8 quantizer: identical to V7 except _quantize_block does NOT
    collapse the pair axis when computing min/max."""

    def _quantize_block(self, states):
        # Same as V7 lines 207-260, but at the min/max step:
        # masked_for_min = np.where(valid_4d, blocks, np.inf)
        # min_val = np.min(masked_for_min, axis=1)   # ← drop axis=4
        # max_val = np.max(masked_for_max, axis=1)   # ← drop axis=4
        # Then scales / zero_points have shape (n_blocks, num_heads,
        # packed_head_dim, 2) instead of (n_blocks, num_heads, packed_head_dim).
        # Pack lower and upper nibbles using their own scale/zp.
        raise NotImplementedError("V8 codec implementation slot")
```

### Phase 2 — CPU unit tests (no GPU needed)

File: `tests/quantization/test_codec_v8.py`

Three minimum tests:

1. **Shape invariant**: V8 packed INT4 has the same shape as V7;
   only scales/zps gain a trailing pair axis.
2. **Reconstruction MSE**: V8 reconstruction MSE on a synthetic
   asymmetric-pair fixture is **strictly lower** than V7 reconstruction
   MSE on the same fixture.
3. **Round-trip identity** when pairs are symmetric (smoke test —
   V8 should degenerate to V7 behavior when nibble pairs share dynamic
   range exactly).

### Phase 3 — MI300X validation (on droplet)

Reference: `scripts/capture_kv_snapshots.py` (standalone runner removed in cleanup)

Compares V7 vs V8 on real Llama-3-8B KV cache snapshots:

1. Load 5 representative KV snapshots from `logs/kv_snapshots/` (to be
   captured on first 30 minutes of droplet time).
2. For each snapshot, quantize with both V7 and V8, dequantize, measure:
   - Reduction factor: `FP16_bytes / (INT4_bytes + metadata_bytes)`
   - Reconstruction MSE: `mean((original - recovered)**2)`
   - Per-head reconstruction MSE for outlier detection
3. Output: `logs/mi300x_codec_v8_<timestamp>.json` with the full
   measurement set, suitable for paper v2.1 Table-3-equivalent.

### Phase 4 — Paper v2.1 integration

Add a row to paper Table 3 (Quantization Quality vs. Compression
Pareto):

| Codec       | Reduction | MSE keys | MSE values | metadata % |
| ----------- | --------: | -------: | ---------: | ---------: |
| FP16 baseline |  1.00×   | 0.0e0    | 0.0e0      |       0%   |
| INT8 naive    |  1.88×   | 3.4e-5   | 3.1e-5     |       3%   |
| V7 per-byte joint | 3.55× | 1.0e-2 | 9.5e-3   |       6%   |
| V7 + FWHT     |  3.55×   | 2.0e0   | 1.9e0     |       6%   |
| **V8 per-nibble independent** | **3.85-3.97×** (TBD) | **<5.0e-3** (TBD) | **<5.0e-3** (TBD) | 12% |

The V8 row stays bracketed `(TBD)` until the MI300X validation lands.
This is the V6.1 honesty discipline applied to V8: no number on the
table without a raw JSON log behind it.

---

## Acceptance criteria for shipping V8

V8 ships in V7.0.0-rc.3 (or V8.0.0-alpha.1) **iff**:

- [ ] CPU unit tests pass (`bun test` → `pytest tests/quantization/test_codec_v8.py -v`)
- [ ] MI300X measurement shows **reduction ≥ 3.80×** (otherwise V8 is
      worse than V7 after metadata overhead — abort)
- [ ] MI300X measurement shows **MSE strictly lower than V7** on each
      of: keys, values, outlier-rich attention sinks
- [ ] No regression on the 1,210-decision INV-15 sweep (V8 is purely
      a codec swap; INV-15 logic is orthogonal — should be vacuous)
- [ ] Paper v2.1 §codec section updated with measured numbers + raw
      JSON log reference in `logs/mi300x_codec_v8_*.json`
- [ ] AUDIT.md gets a new line item: V8 codec measured, metadata
      overhead accounted for, no overclaim

If any of these criteria fail, V8 stays in a feature branch and is
not promoted to main. This is the same discipline that caught FWHT
in V7.0.0-alpha.5 (200× MSE degradation → set default `use_fwht=False`).

---

## What this document does NOT yet cover

- **Storage compaction beyond V8**: there is a known further optimization
  where attention-sink tokens (first N positions, currently FP16) could
  use V8 quantization with a doubled bit-width (INT8) and still beat
  FP16 storage; this is V9+ work.
- **GPU kernel co-design**: V8 increases scale/zp memory traffic by 2×.
  On MI300X HBM3 (3.73 TB/s measured), this is bandwidth-noise; on
  bandwidth-constrained CPUs, it would matter. The MI300X measurement
  will confirm the MI300X-side bandwidth story.
- **vLLM integration path**: the vLLM plugin currently calls
  `RotateKVQuantizer` — switching to `CodecV8Quantizer` is a single-line
  change in the plugin (Phase 4 of the MI300X validation if budget allows).

---

## References

- Apohara V7.0.0-rc.2 paper §limitations: identifies per-nibble
  independent codec as future work
- RotateKV (IJCAI 2025, arXiv:2501.16383) — establishes the 3.97×
  literature target
- AUDIT.md — honesty log tracking codec measurements

This design doc is itself an artifact of the V6.1 honesty discipline:
we ship the design publicly **before** the measurement, so the
measurement cannot be retconned to fit the design.
