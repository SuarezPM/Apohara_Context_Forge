# Apohara 2.0 — Toolchain Pre-Flight (Step 0.1)

> **Source of truth:** This file is the version manifest for every CLI/runtime
> tool that the Apohara 2.0 implementation depends on. Captured 2026-06-11
> from the executor's environment so Phase 4's `maturin develop` smoke test
> and the in-tree `turboquant-turing` Rust crate have a reproducible baseline.

## Why this file exists

Per `.omc/plans/apohara-2-0.md` Step 0.1 (R11 mitigation), Phase 0 has a
hard contract: `cargo --version` and `maturin --version` must exit 0 before
Step 0.1 closes, and a `maturin develop` smoke test of the in-tree crate
(round-trip `encode_kv`/`decode_kv`) must succeed before Phase 4 begins.
This document is the captured evidence for that contract.

## Captured versions

| Tool     | Version                                                | Source                                              | Exit |
|----------|--------------------------------------------------------|-----------------------------------------------------|------|
| cargo    | 1.96.0 (30a34c682 2026-05-25)                          | `cargo --version`                                   | 0    |
| maturin  | 1.13.3-1.1 (cachyos-extra-v3)                          | `pacman -Qi maturin` / `maturin --version` (1.13.3) | 0    |
| uv       | 0.11.19 (7b2cff1c3 2026-06-03 x86_64-unknown-linux-gnu) | `uv --version`                                      | 0    |
| Python   | 3.13.13                                                 | `.venv/bin/python --version`                       | 0    |

### GPU

| Field          | Value                                       |
|----------------|---------------------------------------------|
| Model          | NVIDIA GeForce RTX 2060 SUPER               |
| VRAM           | 8192 MiB (8 GB)                             |
| Driver         | 610.43.02 (nvidia-open 595)                 |
| Compute Cap    | 7.5 (Turing)                                |

`nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader`
output: `NVIDIA GeForce RTX 2060 SUPER, 8192 MiB, 610.43.02, 7.5`.

## Notes

- `maturin` resolves to the cachyos-extra-v3 build (1.13.3-1.1) on this host;
  the extra/ variant (1.13.3-1) is the upstream same-version fallback.
  Either is acceptable for the Phase 4 `maturin develop` smoke test.
- `cargo` and `maturin` are CLI tools. They are NOT declared as runtime
  dependencies of `apohara-context-forge` (the package builds with
  `setuptools` for the slim install path). The in-tree Rust crate at
  `apohara_context_forge/serving/turboquant_turing/` will be built with
  `maturin develop` from a `[apohara2-build]` local invocation, not via
  `pyproject.toml`'s `[build-system].requires` (the latter is reserved for
  the case where `maturin` is the build backend; we are not switching
  backends in Phase 0).
- The M3 LLM-as-judge version pin is recorded in
  `docs/research/reconcile/apohara2-prereg.md` (Step 0.4), not here.

## Phase 4 entry gate (deferred to Phase 4, not Phase 0)

A `maturin develop` smoke test of the in-tree `turboquant-turing` crate
that round-trips a 1D float slice through `encode_kv`/`decode_kv` MUST
return the input before Phase 4 implementation begins. That gate is
captured as a Phase 4 entry criterion, not a Phase 0 step, because the
crate itself is created in Phase 4.1.

## Verification commands (re-runnable)

```bash
cargo --version
maturin --version
uv --version
.venv/bin/python --version
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
```
