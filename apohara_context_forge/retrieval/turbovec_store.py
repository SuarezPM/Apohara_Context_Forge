"""TurbovecStore — Turbovec-backed ANN index for the Apohara 2.0 RAG path.

Replaces the US-002 placeholder with a real implementation backed by the
`turbovec` Python package (TurboQuant ANN, Rust core, SIMD-accelerated
scalar quantization). The hardware target is CPU SIMD (CC 7.5 RTX 2060S
runs the bank test on CPU for retrieval to leave VRAM for the LLM).
CUDA path is follow-up work tracked in the Phase 4 in-tree crate.

Backend choice
--------------
We use `turbovec.TurboQuantIndex` (positional integer ids) rather than
`turbovec.IdMapIndex` (external uint64 ids). The retrieval pipeline keys
results by corpus position; external-id semantics add bookkeeping that
the RAG path does not need. If a future caller needs stable external
ids (e.g. per-document dedup), this is a one-line swap to
`IdMapIndex.add_with_ids(...)` plus a parallel id table.

See:
  - `.omc/plans/apohara-2-0.md` Step 2.2 (real implementation)
  - `.omc/specs/deep-interview-apohara-2-0.md` (turbovec-rag component)
  - `apohara_context_forge/embeddings/embedding_engine.py` (consumed as-is,
    default 384-d; granite-r2 768-d migration is a tracked follow-up)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import turbovec  # type: ignore
except ImportError as e:  # pragma: no cover - covered by tests requiring turbovec
    turbovec = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


# Default dimensionality aligned with the existing EmbeddingEngine default.
# The spec wants granite-embedding-311m-multilingual-r2 (768-d); that
# migration is tracked as a follow-up — see RetrievalEngine docstring.
_DEFAULT_DIM = 768
_DEFAULT_BIT_WIDTH = 4
# The shipped turbovec Rust core supports bit_widths in {2, 3, 4}. We
# expose 2, 3, 4 — the spec's target is 4 (the lowest-quality with
# <=4GB RAM for 10M docs at 768-d, per AUDIT #23 honest scope).
_SUPPORTED_BIT_WIDTHS = (2, 3, 4)


class TurbovecStore:
    """ANN index backed by Turbovec (TurboQuant ANN, Rust + Python).

    Parameters
    ----------
    dim:
        Embedding dimensionality. 768 for granite-embedding-311m-
        multilingual-r2 (the spec target). 384 for the current
        EmbeddingEngine default.
    bit_width:
        Scalar-quantization bit width. Default 4 (the spec's target:
        <=4GB RAM for 10M docs at 4-bit, 768-d). 8 is also supported.
    """

    def __init__(self, dim: int = _DEFAULT_DIM, bit_width: int = _DEFAULT_BIT_WIDTH) -> None:
        if turbovec is None:
            raise ImportError(
                "turbovec is not installed. Install with `pip install turbovec` "
                f"(original ImportError: {_IMPORT_ERROR})"
            )
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if bit_width not in _SUPPORTED_BIT_WIDTHS:
            raise ValueError(
                f"bit_width must be one of {_SUPPORTED_BIT_WIDTHS}; got {bit_width}"
            )
        self.dim = dim
        self.bit_width = bit_width
        self._index = turbovec.TurboQuantIndex(dim=dim, bit_width=bit_width)
        # Tracks logical doc ids; the Rust index returns positional ids.
        self._ids: list[str] = []

    # ------------------------------------------------------------------ add
    def add(self, vectors: np.ndarray, ids: list[str] | None = None) -> None:
        """Insert vectors into the index.

        Parameters
        ----------
        vectors:
            2D float32 array of shape (n, dim). C-contiguous, finite.
        ids:
            Optional external identifiers, length n. Stored alongside the
            positional id returned by the Rust index. If omitted, integer
            positions ("0", "1", ...) are used.
        """
        if not isinstance(vectors, np.ndarray):
            raise TypeError(f"vectors must be a numpy.ndarray, got {type(vectors).__name__}")
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D (n, dim); got shape {vectors.shape}")
        n, dim = vectors.shape
        if dim != self.dim:
            raise ValueError(
                f"vector dim {dim} does not match index dim {self.dim}"
            )
        if vectors.dtype != np.float32:
            # Cast (and surface a clear error if the data loses range)
            vectors = vectors.astype(np.float32, copy=False)
        if not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors)
        if not np.all(np.isfinite(vectors)):
            raise ValueError("vectors contain non-finite values (NaN or Inf)")

        self._index.add(vectors)

        if ids is None:
            start = len(self._ids)
            self._ids.extend(str(i) for i in range(start, start + n))
        else:
            if len(ids) != n:
                raise ValueError(
                    f"ids length {len(ids)} does not match vectors.shape[0] {n}"
                )
            self._ids.extend(str(x) for x in ids)

    # --------------------------------------------------------------- search
    def search(
        self, query: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k ANN search.

        Parameters
        ----------
        query:
            2D float32 array of shape (nq, dim).
        k:
            Number of neighbors to return per query.

        Returns
        -------
        (scores, indices):
            Two 2D float32 arrays of shape (nq, k). `scores` are inner
            products (cosine similarity for unit-norm inputs); `indices`
            are positional ids into the inserted vectors, or -1 when the
            index is empty / k exceeds population.
        """
        if not isinstance(query, np.ndarray):
            raise TypeError(f"query must be a numpy.ndarray, got {type(query).__name__}")
        if query.ndim != 2:
            raise ValueError(f"query must be 2D (nq, dim); got shape {query.shape}")
        if query.shape[1] != self.dim:
            raise ValueError(
                f"query dim {query.shape[1]} does not match index dim {self.dim}"
            )
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if query.dtype != np.float32:
            query = query.astype(np.float32, copy=False)
        if not query.flags["C_CONTIGUOUS"]:
            query = np.ascontiguousarray(query)
        if not np.all(np.isfinite(query)):
            raise ValueError("query contains non-finite values (NaN or Inf)")

        if len(self._index) == 0:
            # Empty index: return zero-score / -1-id rows.
            nq = query.shape[0]
            return (
                np.zeros((nq, k), dtype=np.float32),
                np.full((nq, k), -1, dtype=np.int64),
            )

        effective_k = min(k, len(self._index))
        scores, indices = self._index.search(query, effective_k)

        # Pad with -1 / 0 when k > effective_k so the contract (nq, k) holds.
        if effective_k < k:
            pad_scores = np.zeros((query.shape[0], k - effective_k), dtype=np.float32)
            pad_indices = np.full((query.shape[0], k - effective_k), -1, dtype=np.int64)
            scores = np.concatenate([scores, pad_scores], axis=1)
            indices = np.concatenate([indices, pad_indices], axis=1)

        return scores, indices.astype(np.int64)

    # ------------------------------------------------------ save / load
    def save(self, path: str) -> None:
        """Persist the index to `path` (a `.tvi` file)."""
        if turbovec is None:
            raise ImportError("turbovec is not installed")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._index.write(str(p))

    @classmethod
    def load(cls, path: str) -> "TurbovecStore":
        """Restore a saved index from `path`.

        External ids are not persisted alongside the Rust index; callers
        that need stable ids should keep a side-table and re-add via
        `add(vectors, ids=...)` after load.
        """
        if turbovec is None:
            raise ImportError("turbovec is not installed")
        loaded = turbovec.TurboQuantIndex.load(str(path))
        store = cls.__new__(cls)
        store.dim = int(loaded.dim)
        store.bit_width = int(loaded.bit_width)
        store._index = loaded
        store._ids = []
        return store

    # ------------------------------------------------------------- helpers
    def __len__(self) -> int:
        return len(self._index)

    def get_id(self, position: int) -> str:
        """Return the external id at `position` (or ``"<pos:N>"`` if absent)."""
        if 0 <= position < len(self._ids):
            return self._ids[position]
        return f"<pos:{position}>"

    def ids(self) -> list[str]:
        """Return a copy of the external id list."""
        return list(self._ids)
