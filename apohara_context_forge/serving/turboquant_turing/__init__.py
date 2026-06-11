"""Python wrapper for the in-tree `turboquant-turing` Rust crate.

Maturin will overwrite this file when `maturin develop` is run. The
placeholder is import-safe and signals the crate is not built — the
shim `apohara_context_forge/serving/turboquant_kv.py` catches the
`ImportError` and surfaces a clear error message.

Honest scope (US-006, 2026-06-11): the real `_rust_encode_kv` /
`_rust_decode_kv` symbols ship only after `maturin develop` is run on
a host with the Rust toolchain. The bench's `maturin develop` banner
documents the build step. See `AUDIT.md` entry #25.
"""
