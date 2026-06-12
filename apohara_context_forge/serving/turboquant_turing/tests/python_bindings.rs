//! End-to-end Python binding test for the turboquant-turing wheel.
//!
//! Spawns a Python interpreter in-process, imports the wheel, and
//! verifies the round-trip identity of ``fwht_inplace`` against a
//! numpy reference and the per-block dequantization of
//! ``dequant_per_block`` against the codec_v8 Python path.
//!
//! Sprint 2 / AUDIT #320a: this is the test that ``build.sh``
//! chains after ``cargo test --release`` (and ``maturin develop
//! --release`` emits the wheel) — it pins the contract the
//! in-tree Python shim relies on.
//!
//! Skip flag: ``APOHARA_SKIP_RUST_TESTS=1`` disables the test
//! when the wheel is not built in the active venv (CI runner that
//! only has the .so staged, no Python interpreter available).
//!
//! Build flag: this file is gated by the ``python-bindings-test``
//! feature (see ``Cargo.toml``); the default ``cargo test`` does
//! not enable it. ``build.sh`` opts in via
//! ``cargo test --features python-bindings-test`` after maturin has
//! staged the wheel.

#![cfg(all(test, feature = "python-bindings-test"))]

use pyo3::prelude::*;
use pyo3::types::PyDict;

fn python_path_with_venv() -> Option<std::path::PathBuf> {
    // Look for the .venv at the repo root. Tests run from the
    // crate directory (`apohara_context_forge/serving/turboquant_turing`).
    let candidates = [
        "../../../../.venv",
        "../../../.venv",
        "../../.venv",
        "../.venv",
        ".venv",
    ];
    for c in &candidates {
        let p = std::path::Path::new(c).join("bin").join("python");
        if p.exists() {
            return Some(p);
        }
        let p = std::path::Path::new(c)
            .join("Scripts")
            .join("python.exe");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn with_python_interpreter<F: FnOnce(Python<'_>)>(test_body: F) {
    if std::env::var("APOHARA_SKIP_RUST_TESTS").is_ok() {
        eprintln!("APOHARA_SKIP_RUST_TESTS=1 — skipping");
        return;
    }
    if let Some(p) = python_path_with_venv() {
        std::env::set_var("PYO3_PYTHON", p);
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(test_body);
}

fn has_rust_wheel(py: Python<'_>) -> bool {
    py.run_bound(
        "import importlib.util; assert importlib.util.find_spec('turboquant_turing') is not None",
        None,
        None,
    )
    .is_ok()
}

#[test]
fn fwht_round_trip_against_numpy() {
    with_python_interpreter(|py| {
        if !has_rust_wheel(py) {
            eprintln!("turboquant_turing not installed; skipping");
            return;
        }
        let locals = PyDict::new_bound(py);
        let np = py.import_bound("numpy").expect("numpy import");
        locals.set_item("numpy", np.clone()).unwrap();
        let wheel = py.import_bound("turboquant_turing").expect("wheel import");
        locals.set_item("wheel", wheel.clone()).unwrap();
        py.run_bound(
            r#"
import numpy as np
rng = np.random.default_rng(0)
x = rng.standard_normal(16).astype(np.float32)
ref = x.copy()
d = ref.shape[0]
h = 1
while h < d:
    view = ref.reshape(d // (2 * h), 2, h).copy()
    a = view[..., 0, :].copy()
    b = view[..., 1, :].copy()
    view[..., 0, :] = a + b
    view[..., 1, :] = a - b
    ref = view.reshape(d)
    h *= 2
rust = x.copy()
wheel.fwht_inplace(rust)
np.testing.assert_allclose(rust, ref, atol=1e-6)

# Apply the outer /sqrt(d) and verify fwht(fwht(x)) == x
# within float32 epsilon (the butterfly is self-inverse under
# the sqrt(d) normalisation, so the result is the identity).
rust /= np.sqrt(16.0)
ref /= np.sqrt(16.0)
rust_copy = rust.copy()
wheel.fwht_inplace(rust_copy)
np.testing.assert_allclose(rust_copy, rust, atol=1e-6)
            "#,
            None,
            Some(&locals),
        )
        .expect("Python fwht parity test failed");
    });
}

#[test]
fn dequant_per_block_against_codec_v8() {
    with_python_interpreter(|py| {
        if !has_rust_wheel(py) {
            eprintln!("turboquant_turing not installed; skipping");
            return;
        }
        let locals = PyDict::new_bound(py);
        let np = py.import_bound("numpy").expect("numpy import");
        locals.set_item("numpy", np.clone()).unwrap();
        let wheel = py.import_bound("turboquant_turing").expect("wheel import");
        locals.set_item("wheel", wheel.clone()).unwrap();
        py.run_bound(
            r#"
import numpy as np
import sys
# Make the in-tree codec_v8 importable for the test.
sys.path.insert(0, '../../../../')
from apohara_context_forge.quantization.codec_v8 import (
    CodecV8Config, CodecV8Quantizer,
)
cfg = CodecV8Config(bits=4, group_size=4, sink_tokens=0, use_fwht=False)
q = CodecV8Quantizer(cfg)
rng = np.random.default_rng(7)
# 2 docs × (seq=4, num_heads=2, head_dim=16) → packed_dim=8.
x = rng.random((2, 4, 2, 16), dtype=np.float32)
keys, scales, zps = q._quantize_block(x)
# Flatten to the 1-D contract the Rust dequant expects.
codes = keys.reshape(-1).astype(np.uint8)
sc = scales.reshape(-1)
zp = zps.reshape(-1)
out_rust = wheel.dequant_per_block(codes, sc, zp, 4)
out_v8 = np.concatenate(
    [q._dequantize_block(keys[i], scales[i], zps[i], 4) for i in range(2)],
    axis=0,
).reshape(-1)
np.testing.assert_allclose(out_rust, out_v8, atol=1e-6)
            "#,
            None,
            Some(&locals),
        )
        .expect("Python dequant parity test failed");
    });
}
