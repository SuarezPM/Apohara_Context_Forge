"""Apohara 2.0 retrieval path.

This package is the RAG backend for Apohara Context Forge 2.0. It replaces
the legacy `sentence-transformers` + FAISS-IVF stack for the retrieval
path with a Turbovec-backed index (TurboQuant ANN, Rust + Python via
the `turbovec` PyPI package; the in-tree `turboquant-turing` crate is a
follow-up tracked in Phase 4). The public surface:

    TurbovecStore(dim=768, bit_width=4)  # the index
    RetrievalEngine(...)                  # embedder + index glue

See:
  - `.omc/plans/apohara-2-0.md` Step 0.2 (package creation)
  - `.omc/plans/apohara-2-0.md` Step 2.2 (real `TurbovecStore` impl)
  - `.omc/specs/deep-interview-apohara-2-0.md` (turbovec-rag component)
  - `apohara_context_forge/retrieval/turbovec_store.py` (the index)
  - `apohara_context_forge/embeddings/embedding_engine.py` (consumed as-is)
"""

from __future__ import annotations

import asyncio
from typing import List, Tuple

import numpy as np

from apohara_context_forge.retrieval.turbovec_store import TurbovecStore


__all__ = ["TurbovecStore", "RetrievalEngine", "RetrievalHit"]


class RetrievalHit:
    """One (text, score) result from `RetrievalEngine.retrieve`."""

    __slots__ = ("text", "score", "position", "id")

    def __init__(self, text: str, score: float, position: int, id: str) -> None:
        self.text = text
        self.score = float(score)
        self.position = int(position)
        self.id = id

    def __repr__(self) -> str:
        return f"RetrievalHit(id={self.id!r}, score={self.score:.4f}, text={self.text[:40]!r})"


class RetrievalEngine:
    """Combine an `EmbeddingEngine` with a `TurbovecStore`.

    Phase 2 wires the Turbovec index with the **existing 384-d
    EmbeddingEngine** (the all-MiniLM-L6-v2 default, falling back to
    xorshift pseudo-embeddings when qwen3-embed is unavailable). The
    spec wants granite-embedding-311m-multilingual-r2 (768-d) — that
    migration is tracked as a follow-up in
    `.omc/plans/apohara-2-0.md` (Phase 2.x — embedding model swap). The
    index defaults to `dim=768` so the eventual migration is a
    constructor-arg change, not a code change.

    The engine is a thin, sync wrapper around the async
    `EmbeddingEngine.encode` so callers (bench scripts, tests, MCP
    handlers) can use it without `asyncio.run`. Internally, it uses
    `asyncio.run` per call — fine for the bench/test path, not
    recommended for high-throughput serving (use a long-lived event
    loop there).
    """

    def __init__(
        self,
        embedding_engine=None,
        store: TurbovecStore | None = None,
        dim: int | None = None,
    ) -> None:
        if embedding_engine is None:
            from apohara_context_forge.embeddings.embedding_engine import (
                EmbeddingEngine,
            )

            # Honor the constructor dim when provided; default 384 matches
            # the existing EmbeddingEngine default.
            embedding_engine = EmbeddingEngine(dim=dim or 384, use_onnx=False)

        if dim is None:
            dim = embedding_engine.dim

        if store is None:
            store = TurbovecStore(dim=dim, bit_width=4)

        if store.dim != dim:
            raise ValueError(
                f"embedding dim {dim} does not match store dim {store.dim}"
            )

        self.embedding_engine = embedding_engine
        self.store = store
        self.dim = dim
        self._texts: list[str] = []

    # ------------------------------------------------------------------ API
    def index(self, texts: List[str]) -> None:
        """Embed and add `texts` to the index."""
        if not texts:
            return
        embeddings = self._embed_sync(texts)
        ids = [f"doc_{len(self._texts) + i}" for i in range(len(texts))]
        self.store.add(embeddings, ids=ids)
        self._texts.extend(texts)

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalHit]:
        """Embed `query` and return the top-k (text, score) hits."""
        if len(self.store) == 0:
            return []
        q_emb = self._embed_sync([query])  # shape (1, dim)
        scores, indices = self.store.search(q_emb, k=k)
        hits: list[RetrievalHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if int(idx) < 0:
                continue
            position = int(idx)
            text = self._texts[position] if 0 <= position < len(self._texts) else ""
            hits.append(
                RetrievalHit(
                    text=text,
                    score=float(score),
                    position=position,
                    id=self.store.get_id(position),
                )
            )
        return hits

    # ------------------------------------------------------------- helpers
    def _embed_sync(self, texts: List[str]) -> np.ndarray:
        """Run the async `encode_batch` and stack the result."""
        loop = asyncio.new_event_loop()
        try:
            embs = loop.run_until_complete(
                self.embedding_engine.encode_batch(list(texts))
            )
        finally:
            loop.close()
        if not embs:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([np.asarray(e, dtype=np.float32) for e in embs], axis=0)

    def __len__(self) -> int:
        return len(self.store)
