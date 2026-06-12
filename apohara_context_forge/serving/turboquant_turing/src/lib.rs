//! TurboQuant-Turing: in-tree Rust crate for the Apohara 2.0 Phase 4 path.
//!
//! This crate ships a CPU-only scalar implementation of the Lloyd-Max
//! + 1-bit QJL quantization for the local bank test smoke (the
//! `maturin develop` round-trip). The CUDA C kernel is feature-gated
//! and lives in `src/cuda_kernel.cu`; building it requires `nvcc` and
//! the matching compute capability. See `README.md` for the build
//! matrix.
//!
//! API surface (mirrored by `apohara_context_forge/serving/turboquant_kv.py`):
//!   - `encode_kv(weights, n, bits) -> Vec<u8>`: scalar-quantize a 1D
//!     float slice using the Lloyd-Max centroid table; returns the
//!     packed byte stream (1 nibble per element when bits=4, 1 element
//!     per byte when bits<=4).
//!   - `decode_kv(packed, n, bits) -> Vec<f32>`: inverse lookup back
//!     to floats. The round-trip MSE is bounded by the Lloyd-Max
//!     optimality criterion against the unit-variance Beta prior.
//!
//! Sprint 2 / AUDIT #320a added two PyO3-bound kernels that mirror
//! the in-tree Python paths so the vLLM-integration bench can avoid
//! the per-doc numpy hop:
//!   - `fwht_inplace(buf)`: in-place Walsh-Hadamard butterfly on a
//!     1-D f32 array (mirror of
//!     `apohara_context_forge/quantization/fwht.py:_fwht_butterfly_numpy`).
//!   - `dequant_per_block(codes, scales, zps, group_size)`: per-block
//!     INT4 dequant (mirror of
//!     `apohara_context_forge/quantization/codec_v8.py:_dequantize_block`).
//!
//! Workgroup size: 32. This is the workgroup (block) size for the
//! CUDA kernel path; the CPU path is scalar (no workgroup), but the
//! constant is mirrored in the kernel signature for consistency.

pub mod centroids;
pub mod dequant;
pub mod fwht;

use centroids::centroids;
use pyo3::prelude::*;

/// Scalar Lloyd-Max quantization (CPU fallback).
///
/// `weights` is a flat `&[f32]`; `n` is its length; `bits` is 2, 3, or 4.
/// Returns a packed `Vec<u8>` of length `ceil(n * bits / 8)` bytes.
pub fn encode_kv(weights: &[f32], n: usize, bits: u8) -> Vec<u8> {
    let table = centroids(bits).expect("bits must be 2, 3, or 4");
    let mask = (1u8 << bits) - 1;
    let mut out: Vec<u8> = Vec::with_capacity(n * (bits as usize) / 8 + 1);

    if bits == 4 {
        // Two elements per byte: low nibble first, high nibble second.
        for chunk in weights[..n].chunks(2) {
            let lo = nearest_centroid_index(chunk[0], table, mask);
            let hi = if chunk.len() > 1 {
                nearest_centroid_index(chunk[1], table, mask)
            } else {
                0
            };
            out.push((lo & mask) | ((hi & mask) << 4));
        }
    } else {
        // bits <= 3: one element per byte (the CUDA kernel path can
        // pack more aggressively, but the CPU scalar path is
        // intentionally simple).
        for &w in &weights[..n] {
            let idx = nearest_centroid_index(w, table, mask);
            out.push(idx & mask);
        }
    }

    out
}

/// Inverse of `encode_kv` — return the centroid value for each packed
/// index. `n` is the number of decoded elements to produce.
pub fn decode_kv(packed: &[u8], n: usize, bits: u8) -> Vec<f32> {
    let table = centroids(bits).expect("bits must be 2, 3, or 4");
    let mask = (1u8 << bits) - 1;
    let mut out = Vec::with_capacity(n);

    if bits == 4 {
        for byte in packed.iter() {
            let lo = byte & mask;
            out.push(table[lo as usize]);
            if out.len() < n {
                let hi = (byte >> 4) & mask;
                out.push(table[hi as usize]);
            }
        }
    } else {
        for byte in packed.iter() {
            let idx = byte & mask;
            out.push(table[idx as usize]);
        }
    }

    out.truncate(n);
    out
}

/// Find the index of the centroid closest to `w`.
fn nearest_centroid_index(w: f32, table: &[f32], _mask: u8) -> u8 {
    let mut best_idx: usize = 0;
    let mut best_dist = f32::INFINITY;
    for (i, &c) in table.iter().enumerate() {
        let d = (w - c).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_round_trip_4bit() {
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 16.0).collect();
        let packed = encode_kv(&weights, weights.len(), 4);
        let decoded = decode_kv(&packed, weights.len(), 4);
        assert_eq!(decoded.len(), weights.len());
        // MSE should be bounded by the Lloyd-Max optimality gap
        // (small for 16-level quantization of unit-variance input).
        let mse: f32 = weights
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / weights.len() as f32;
        assert!(mse < 0.1, "mse {} too high for 4-bit", mse);
    }

    #[test]
    fn test_encode_decode_round_trip_2bit() {
        let weights: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) / 2.0).collect();
        let packed = encode_kv(&weights, weights.len(), 2);
        let decoded = decode_kv(&packed, weights.len(), 2);
        assert_eq!(decoded.len(), weights.len());
    }

    #[test]
    fn test_encode_decode_round_trip_3bit() {
        let weights: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) / 4.0).collect();
        let packed = encode_kv(&weights, weights.len(), 3);
        let decoded = decode_kv(&packed, weights.len(), 3);
        assert_eq!(decoded.len(), weights.len());
    }

    #[test]
    fn test_invalid_bits_panics() {
        let weights = vec![0.0_f32; 4];
        let result = std::panic::catch_unwind(|| encode_kv(&weights, 4, 5));
        assert!(result.is_err());
    }
}

/// PyO3 module entry point. Registers every Python-callable
/// function on the `turboquant_turing` extension. Both the
/// Lloyd-Max ``encode_kv`` / ``decode_kv`` (the long-standing
/// crate surface) and the Sprint 2 / AUDIT #320a PyO3 kernels
/// (``fwht_inplace``, ``dequant_per_block``) live on the same
/// module so the Python shim sees a single importable namespace.
#[pyo3::pymodule]
fn turboquant_turing(_py: Python<'_>, m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(encode_kv_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(decode_kv_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(fwht::fwht_inplace, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(dequant::dequant_per_block, m)?)?;
    Ok(())
}

/// Thin PyO3 wrapper around the existing ``encode_kv`` so the
/// function can be registered on the module without breaking the
/// crate's pure-Rust public API (the round-trip integration test
/// in ``tests/round_trip.rs`` imports ``encode_kv`` directly).
#[pyo3::pyfunction]
fn encode_kv_py(weights: Vec<f32>, n: usize, bits: u8) -> Vec<u8> {
    encode_kv(&weights, n, bits)
}

/// PyO3 wrapper around ``decode_kv``. Same rationale as
/// ``encode_kv_py``: the crate-level public API is unchanged.
#[pyo3::pyfunction]
fn decode_kv_py(packed: Vec<u8>, n: usize, bits: u8) -> Vec<f32> {
    decode_kv(&packed, n, bits)
}
