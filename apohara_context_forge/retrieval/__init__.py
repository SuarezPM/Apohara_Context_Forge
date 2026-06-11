"""Apohara 2.0 retrieval path.

This package is the RAG backend for Apohara Context Forge 2.0. It replaces
the legacy `sentence-transformers` + FAISS-IVF stack for the retrieval
path with a Turbovec-backed index (TurboQuant ANN, Rust + Python via
maturin). The public surface is intentionally minimal in Phase 0 — only
the placeholder class lives here. Real implementation lands in US-004
(Phase 2) after the in-tree `turboquant-turing` Rust crate (Phase 4) ships
a maturin-built wheel.

See:
  - `.omc/plans/apohara-2-0.md` Step 0.2 (package creation)
  - `.omc/plans/apohara-2-0.md` Step 2.2 (real `TurbovecStore` implementation)
  - `docs/research/reconcile/apohara2-toolchain.md` (build tooling versions)

Status: US-002 placeholder. `TurbovecStore` raises NotImplementedError on
`add`/`search`; the class is constructible so the import + wiring is
tested in isolation before Phase 2 lands.
"""
