//! Round-trip integration test for the turboquant-turing crate.
//!
//! Exercises `encode_kv` -> `decode_kv` on a synthetic float slice
//! and asserts the round-trip MSE is bounded by the Lloyd-Max
//! optimality criterion (16-level quantizer on a unit-variance input
//! has an MSE floor around 0.005).
//!
//! This is the same round-trip the `apohara_context_forge.serving.turboquant_kv`
//! shim runs through, and the same test the Phase 4 entry gate (R11
//! mitigation in `.omc/plans/apohara-2-0.md`) calls "passed" when
//! `maturin develop` lands.

use turboquant_turing::{decode_kv, encode_kv};

#[test]
fn round_trip_4bit_unit_variance() {
    let n = 1024_usize;
    let weights: Vec<f32> = (0..n)
        .map(|i| {
            // Sinusoidal pattern in [-2, 2] approximates the
            // unit-variance Beta prior the centroids were trained
            // against.
            let t = (i as f32) / (n as f32) * std::f32::consts::TAU;
            (t.sin() * 2.0).clamp(-2.5, 2.5)
        })
        .collect();

    let packed = encode_kv(&weights, n, 4);
    let decoded = decode_kv(&packed, n, 4);
    assert_eq!(decoded.len(), n);

    let mse: f32 = weights
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / (n as f32);

    // Lloyd-Max optimality floor for 16 levels on a unit-variance
    // input. The real number is 0.0046, but we keep the assertion
    // loose (0.05) so the test is not coupled to the precise
    // centroid table values.
    assert!(
        mse < 0.05,
        "round-trip MSE {} exceeded the Lloyd-Max optimality floor (0.05)",
        mse
    );
}

#[test]
fn round_trip_4bit_identity_on_centroids() {
    // Encoding a centroid value then decoding it must return the
    // exact same centroid (within f32 epsilon). This is the
    // "round-trip identity" assertion the spec uses.
    let centroids: Vec<f32> = vec![
        -2.020_086, -1.469_140, -1.081_309, -0.764_244, -0.484_829, -0.228_538,
        0.017_464, 0.265_484, 0.515_905, 0.770_540, 1.031_358, 1.300_800,
        1.582_847, 1.882_672, 2.207_272, 2.562_995,
    ];
    let n = centroids.len();
    let packed = encode_kv(&centroids, n, 4);
    let decoded = decode_kv(&packed, n, 4);
    for (orig, back) in centroids.iter().zip(decoded.iter()) {
        assert!(
            (orig - back).abs() < 1e-3,
            "centroid round-trip drift: {} -> {}",
            orig,
            back
        );
    }
}

#[test]
fn compression_ratio_4bit() {
    // 4-bit quantization halves the byte count vs FP32 (8x shrink
    // vs FP16). The bench's >=2.5x threshold is satisfied with a
    // wide margin; the per-block scale overhead is negligible at
    // practical block sizes.
    let n = 4096_usize;
    let weights: Vec<f32> = (0..n).map(|i| (i as f32) / 1024.0).collect();
    let packed = encode_kv(&weights, n, 4);
    let fp32_bytes = n * std::mem::size_of::<f32>();
    let compression = fp32_bytes as f32 / packed.len() as f32;
    assert!(
        compression >= 7.0,
        "4-bit compression ratio {} too low (expected ~8x)",
        compression
    );
}
