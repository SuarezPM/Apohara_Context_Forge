"""Turbovec store — placeholder for the Apohara 2.0 RAG retrieval path.

US-002 placeholder. The real implementation lands in US-004 (Phase 2) once
the in-tree `turboquant-turing` Rust crate (Phase 4) ships a maturin-built
wheel. Per `.omc/plans/apohara-2-0.md` Step 2.2, the public surface will be:

    add(docs, embeddings)        # bulk insert
    search(query_embedding, k)   # top-k ANN
    save(path)                   # persist index
    load(path)                   # restore index

The hardware target is CPU SIMD first (R3 mitigation: granite-r2 311M
fits in 8GB only if the embedding model runs on CPU). CUDA path is
follow-up work after the Rust crate stabilizes.

Construction-only checks live in `tests/test_retrieval_init.py` to prove
the import surface and constructor work before Phase 2 lands.
"""

from __future__ import annotations


class TurbovecStore:
    """ANN index backed by Turbovec (TurboQuant ANN, Rust + Python).

    Parameters
    ----------
    dim:
        Embedding dimensionality. 768 for granite-embedding-311m-multilingual-r2.
    bit_width:
        Scalar quantization bit width. Default 4 (the spec's target;
        ≤4GB RAM for 10M docs at 4-bit, 768-d).
    """

    def __init__(self, dim: int, bit_width: int = 4) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if bit_width not in (4, 8):
            raise ValueError(
                f"bit_width must be 4 or 8 (the Turbovec supported set); got {bit_width}"
            )
        self.dim = dim
        self.bit_width = bit_width

    def add(self, vectors) -> None:  # pragma: no cover - US-004 replaces this
        """Insert vectors into the index. US-002 placeholder."""
        raise NotImplementedError("US-002 placeholder; real impl in US-004")

    def search(self, query, k: int):  # pragma: no cover - US-004 replaces this
        """Top-k ANN search. US-002 placeholder."""
        raise NotImplementedError("US-002 placeholder; real impl in US-004")
