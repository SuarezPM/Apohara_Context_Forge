//! Per-block dequantization kernel — Rust mirror of
//! `apohara_context_forge/quantization/codec_v8.py:_dequantize_block`.
//!
//! The kernel unpacks the two nibbles of every byte, applies the
//! per-block (scale, zero_point) shared by `group_size` consecutive
//! packed bytes, and writes the dequantized FP32 back into a fresh
//! 1-D array. The layout matches the V8 codec (per-nibble scales,
//! trailing pair axis), but the kernel is parameterised on
//! `group_size` so the same Rust function covers both the V8
//! per-nibble (group_size=1) and the per-block AUDIT #27a close path
//! (group_size=256).
//!
//! Exposed surface:
//!   - `dequant_per_block(codes, scales, zps, group_size) -> Bound<PyArray1<f32>>`
//!
//! The caller passes:
//!   - `codes`:  packed nibble bytes, shape `(n_blocks * group_size,)`.
//!   - `scales`: per-block (scale_lo, scale_hi) f32 pairs,
//!               shape `(n_blocks * 2,)`.
//!   - `zps`:    per-block (zp_lo, zp_hi) f32 pairs,
//!               shape `(n_blocks * 2,)`.
//!   - `group_size`: number of packed bytes sharing one (scale, zp) pair.
//!
//! The output is a 1-D f32 array of length `codes.len() * 2` (the
//! lo-nibble / hi-nibble interleave), matching the codec_v8 contract.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

#[pyo3::pyfunction]
pub fn dequant_per_block<'py>(
    py: Python<'py>,
    codes: &Bound<'py, PyArray1<u8>>,
    scales: &Bound<'py, PyArray1<f32>>,
    zps: &Bound<'py, PyArray1<f32>>,
    group_size: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let codes_slice = codes
        .readonly()
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!(
            "dequant_per_block: codes is not contiguous: {:?}",
            e
        )))?
        .to_vec();
    let scales_vec = scales
        .readonly()
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!(
            "dequant_per_block: scales is not contiguous: {:?}",
            e
        )))?
        .to_vec();
    let zps_vec = zps
        .readonly()
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!(
            "dequant_per_block: zps is not contiguous: {:?}",
            e
        )))?
        .to_vec();

    if group_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dequant_per_block: group_size must be >= 1",
        ));
    }
    if codes_slice.is_empty() {
        return Ok(PyArray1::zeros_bound(py, [0], false));
    }
    let n_bytes = codes_slice.len();
    if n_bytes % group_size != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "dequant_per_block: codes length {} is not divisible by group_size {}",
            n_bytes, group_size
        )));
    }
    let n_blocks = n_bytes / group_size;
    if scales_vec.len() != n_blocks * 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "dequant_per_block: scales length {} != n_blocks*2 ({})",
            scales_vec.len(),
            n_blocks * 2
        )));
    }
    if zps_vec.len() != n_blocks * 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "dequant_per_block: zps length {} != n_blocks*2 ({})",
            zps_vec.len(),
            n_blocks * 2
        )));
    }

    // Output length: 2 * n_bytes (one f32 per nibble, lo+hi).
    let out_len = 2 * n_bytes;
    let out = PyArray1::zeros_bound(py, [out_len], false);
    let mut out_array = out.readwrite();
    let out_slice = out_array.as_slice_mut().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "dequant_per_block: output buffer is not contiguous: {:?}",
            e
        ))
    })?;

    for blk in 0..n_blocks {
        let scale_lo = scales_vec[2 * blk];
        let scale_hi = scales_vec[2 * blk + 1];
        let zp_lo = zps_vec[2 * blk];
        let zp_hi = zps_vec[2 * blk + 1];
        for j in 0..group_size {
            let byte = codes_slice[blk * group_size + j];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            let k = blk * group_size + j;
            out_slice[2 * k] = (lo - zp_lo) * scale_lo;
            out_slice[2 * k + 1] = (hi - zp_hi) * scale_hi;
        }
    }

    drop(out_array);
    Ok(out)
}

#[cfg(test)]
mod tests {
    /// Pure-Rust body of the dequant kernel — used by the round-
    /// trip tests below. The wheel entry point runs the same loop
    /// after slicing the input arrays.
    fn dequant_body(codes: &[u8], scales: &[f32], zps: &[f32], group_size: usize) -> Vec<f32> {
        let n_bytes = codes.len();
        let n_blocks = n_bytes / group_size;
        let mut out = vec![0.0_f32; 2 * n_bytes];
        for blk in 0..n_blocks {
            let scale_lo = scales[2 * blk];
            let scale_hi = scales[2 * blk + 1];
            let zp_lo = zps[2 * blk];
            let zp_hi = zps[2 * blk + 1];
            for j in 0..group_size {
                let byte = codes[blk * group_size + j];
                let lo = (byte & 0x0F) as f32;
                let hi = ((byte >> 4) & 0x0F) as f32;
                let k = blk * group_size + j;
                out[2 * k] = (lo - zp_lo) * scale_lo;
                out[2 * k + 1] = (hi - zp_hi) * scale_hi;
            }
        }
        out
    }

    #[test]
    fn round_trip_one_block() {
        // One block (group_size=1), one byte. code=0xAB → lo=0xB=11,
        // hi=0xA=10. With (scale=1, zp=0) the output is exactly
        // (lo, hi) = (11, 10).
        let codes = vec![0xABu8];
        let scales = vec![1.0_f32, 1.0];
        let zps = vec![0.0_f32, 0.0];
        let out = dequant_body(&codes, &scales, &zps, 1);
        assert_eq!(out, vec![11.0_f32, 10.0]);
    }

    #[test]
    fn round_trip_three_blocks_group_size_2() {
        // Three blocks, group_size=2 (so the 6 codes form 3 blocks of
        // 2 bytes each). Pick distinct (scale, zp) per block to test
        // the broadcast.
        let codes = vec![0x01u8, 0x23, 0x45, 0x67, 0x89, 0xAB];
        let scales = vec![1.0_f32, 0.5, 2.0, 4.0, 0.25, 8.0];
        let zps = vec![0.0_f32; 6];
        let out = dequant_body(&codes, &scales, &zps, 2);
        // Spot-check a few positions.
        // Block 0: lo=1, hi=0; scale_lo=1, scale_hi=0.5
        //   out[0] = (1 - 0) * 1 = 1; out[1] = (0 - 0) * 0.5 = 0
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 0.0);
        // Block 2: byte 0xAB → lo=0xB=11, hi=0xA=10; scale_lo=0.25, scale_hi=8
        //   out[10] = (11 - 0) * 0.25 = 2.75; out[11] = (10 - 0) * 8 = 80
        assert!((out[10] - 2.75).abs() < 1e-6);
        assert!((out[11] - 80.0).abs() < 1e-6);
    }

    #[test]
    fn zero_zero_identity() {
        // scale=0, zp=0 should clamp the lo / hi to 0 (we still
        // emit the interleave so the output length matches).
        let codes = vec![0xFFu8; 4];
        let scales = vec![0.0_f32; 2];
        let zps = vec![0.0_f32; 2];
        let out = dequant_body(&codes, &scales, &zps, 4);
        for v in &out {
            assert_eq!(*v, 0.0);
        }
    }
}
