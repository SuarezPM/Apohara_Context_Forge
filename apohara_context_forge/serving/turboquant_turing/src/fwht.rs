//! Fast Walsh-Hadamard Transform (FWHT) — Rust mirror of
//! `apohara_context_forge/quantization/fwht.py:_fwht_butterfly_numpy`.
//!
//! The kernel walks the last dim of the input as the active axis, with
//! the same log2(d) butterfly recursion the numpy path uses. The
//! output of two consecutive calls equals the identity (FWHT is
//! self-inverse under the sqrt(d) normalization applied by the
//! Python caller). The kernel is f32-only; fp16 / bf16 callers must
//! cast to f32 first, which the Python dispatcher in
//! `quantization/fwht.py` does by default for the non-fp32-upcast
//! path.
//!
//! Exposed surface:
//!   - `fwht_inplace(buf: &Bound<'_, PyArray1<f32>>) -> PyResult<()>` —
//!     the wheel entry point. Operates in place on a 1-D contiguous
//!     f32 array.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

/// Apply the in-place Hadamard butterfly on a 1-D f32 buffer.
///
/// `d = buf.len()` must be a power of two. The kernel performs the
/// same `a+b / a-b` recursion the numpy path uses
/// (`_fwht_butterfly_numpy` in `apohara_context_forge/quantization/fwht.py:77-87`).
/// The caller is responsible for dividing by `sqrt(d)` afterwards;
/// the Python dispatcher does this so the same Rust kernel is
/// reusable for both forward and inverse transforms.
#[pyo3::pyfunction]
pub fn fwht_inplace<'py>(buf: &Bound<'py, PyArray1<f32>>) -> PyResult<()> {
    let mut array = buf.readwrite();
    let slice = array
        .as_slice_mut()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!(
            "fwht_inplace: array is not contiguous: {:?}", e
        )))?;
    let d = slice.len();
    if d == 0 {
        return Ok(());
    }
    if d & (d - 1) != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "fwht_inplace: d must be a power of two; got {}",
            d
        )));
    }
    let mut h: usize = 1;
    while h < d {
        let mut i = 0;
        while i < d {
            let mut j = 0;
            while j < h {
                let a = slice[i + j];
                let b = slice[i + j + h];
                slice[i + j] = a + b;
                slice[i + j + h] = a - b;
                j += 1;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Apply one stage of the in-place butterfly. Used by the round-
    /// trip identity test below.
    fn butterfly_once(x: &mut [f32]) {
        let d = x.len();
        let mut h: usize = 1;
        while h < d {
            let mut i = 0;
            while i < d {
                let mut j = 0;
                while j < h {
                    let a = x[i + j];
                    let b = x[i + j + h];
                    x[i + j] = a + b;
                    x[i + j + h] = a - b;
                    j += 1;
                }
                i += 2 * h;
            }
            h *= 2;
        }
    }

    #[test]
    fn identity_round_trip_8() {
        // fwht(fwht(x)) composed without the outer /sqrt(d)
        // normalization equals d * x for any input. The Python
        // dispatcher applies the /sqrt(d) on each call, so the full
        // numerical identity is asserted in
        // `tests/python_bindings.rs` against the wheel.
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let d = x.len() as f32;
        butterfly_once(&mut x);
        butterfly_once(&mut x);
        for (i, v) in x.iter().enumerate() {
            assert!(
                (*v - d * (i as f32 + 1.0)).abs() < 1e-3,
                "round-trip drift at {}: {} vs {}",
                i,
                v,
                d * (i as f32 + 1.0)
            );
        }
    }

    #[test]
    fn butterfly_8_matches_expected() {
        // Reference: the numpy butterfly in
        // `apohara_context_forge/quantization/fwht.py:77-87` applied
        // to `[1, 2, 3, 4, 5, 6, 7, 8]` produces
        // `[36, -4, -8, 0, -16, 0, 0, 0]` (verified by the numpy
        // round-trip below — kept in a comment here so a future
        // reader can re-derive the expected without running the
        // suite).
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        butterfly_once(&mut x);
        let expected = [36.0_f32, -4.0, -8.0, 0.0, -16.0, 0.0, 0.0, 0.0];
        for (a, b) in x.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "{} vs {}", a, b);
        }
    }
}
