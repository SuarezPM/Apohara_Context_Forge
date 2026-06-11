# turboquant-turing

In-tree Rust crate for the Apohara 2.0 Phase 4 TurboQuant-KV path.

## Scope

- **CPU scalar** `encode_kv` / `decode_kv` using precomputed Lloyd-Max
  centroid tables for 2/3/4 bit widths. This is the path the
  `maturin develop` smoke test exercises.
- **CUDA C kernel** (feature-gated behind `compute_75`) with workgroup
  size 32. Built only when `nvcc` is on the PATH and a matching
  compute capability is targeted.
- **1-bit QJL (QJL: Quantization with J-Linearization)** is a
  follow-up that lands with the H100/MI300X port (CC 8.0+ features).

## Build

```bash
# Smoke build (CPU only, default).
maturin develop --release

# Build the CUDA kernel for CC 7.5 (requires nvcc).
maturin develop --release --features compute_75
```

The Python shim `apohara_context_forge/serving/turboquant_kv.py` will
pick up the in-tree module via the lazy import guard.

## Honest scope (US-006, 2026-06-11)

This crate lands the **wiring skeleton** for the Phase 4 path, not
the full GPU-optimized port. The bank test (RTX 2060S, CC 7.5) runs
the CPU path; the H100/MI300X path with the vectorised Lloyd-Max +
1-bit QJL is the follow-up gated behind the `compute_80` / `compute_90`
features. AUDIT.md entry #25 documents the gap.

## Test

```bash
cargo test --release
```

The `round_trip` integration test exercises `encode_kv` → `decode_kv`
on a synthetic float slice and asserts the round-trip MSE is bounded.
The MSE bound matches the Lloyd-Max optimality criterion against a
unit-variance Beta prior (16-level quantizer on a unit-variance input
has an MSE floor around 0.005 — see `tests/round_trip.rs`).

## Workgroup size

32 threads per block (pinned per spec R9 / R15). The kernel in
`src/cuda_kernel.cu` launches with `blockDim.x = 32`. The CC 7.5
default feature ships this exact size; the CC 8.0+ port can re-tune
warp-level utilisation.
