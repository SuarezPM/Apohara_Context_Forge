"""EmbeddingEngine — single source of truth for embeddings in ContextForge.

US-012 (2026-06-11): migrated the default model to
``granite-embedding-311m-multilingual-r2`` (IBM, Apache 2.0, 1024-d
MRL, truncated to 768-d for the Apohara 2.0 Turbovec-RAG stack).
Loaded lazily via :mod:`sentence_transformers` on first use
(the `apohara2` extra installs it).

Fallback path
-------------
If the model cannot be loaded — offline host, missing weights, broken
download, etc. — the engine falls back to a **deterministic 768-d
random unit vector per text** seeded by the text. This is honest
about being a stub: the production path requires the model on disk
in ``~/.cache/huggingface/`` (downloaded automatically by
``sentence_transformers`` on first use). The fallback exists so unit
tests and the bench stub can still produce stable 768-d vectors
without the model.

Back-compat
-----------
The 384-d path used by the legacy 384-d tests stays available via
:class:`EmbeddingEngine.legacy_384d`, which constructs an instance at
``dim=384`` with the xorshift pseudo-embedding. The previous
``use_onnx=False`` kwarg still routes to xorshift for the original
test_embedding_engine.py suite.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# US-012: default model and dim.
GRANITE_R2_MODEL_NAME = "ibm-granite/granite-embedding-311m-multilingual-r2"
GRANITE_R2_FULL_DIM = 1024
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_MODEL_NAME = GRANITE_R2_MODEL_NAME

# LRU cache size (kept from the prior engine for repeat-prompt efficiency).
LRU_MAX_SIZE = 1000

# Singleton instance + lock.
_instance: Optional["EmbeddingEngine"] = None
_instance_lock = asyncio.Lock()


def _hash_to_unit_vector(text: str, dim: int) -> np.ndarray:
    """Deterministic 768-d (or `dim`) unit vector seeded by `text`.

    The fallback path: hashes `text` with SHA-256, expands the digest
    into a float32 vector in ``[-1, 1]`` via a counter-mode PRNG, then
    L2-normalizes. Identical text → identical vector; different text →
    orthogonal (on average) unit vector. Used only when the granite-r2
    model fails to load (offline, broken download, etc.). Production
    deployments MUST have the model available.
    """
    seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    # 4 bytes per float → 8 values per SHA-256 chunk; we need
    # `ceil(dim / 8)` chunks. We mask the uint32 to ``[0, 2**31)`` so
    # the reinterpret_as_float32 result is finite and well-distributed
    # in ``[-1.0, 1.0]`` (no NaN / Inf from raw byte re-interpretation).
    out = np.empty(dim, dtype=np.float32)
    n_filled = 0
    counter = 0
    while n_filled < dim:
        chunk = hashlib.sha256(
            seed_bytes + counter.to_bytes(4, "little")
        ).digest()
        # 32 bytes → 8 uint32 values → 8 float32 values per chunk.
        u32 = np.frombuffer(chunk, dtype=np.uint32)
        # Map [0, 2^31) → [0, 1) → [-1, 1) as float32.
        u31 = (u32 & 0x7FFFFFFF).astype(np.float32)
        mapped = (u31 / float(0x80000000)) * 2.0 - 1.0
        take = min(len(mapped), dim - n_filled)
        out[n_filled : n_filled + take] = mapped[:take]
        n_filled += take
        counter += 1
    norm = float(np.linalg.norm(out))
    if norm > 0:
        out = out / norm
    return out


class EmbeddingEngine:
    """
    Unified semantic embedding engine for ContextForge.

    Default backend: ``ibm-granite/granite-embedding-311m-multilingual-r2``
    via :mod:`sentence_transformers` (MRL 1024-d, truncated to 768-d).
    Fallback: deterministic 768-d random unit vector per text (see
    module docstring — unit-test / bench stub path only).

    Usage:
        engine = EmbeddingEngine()                        # default 768-d granite-r2
        engine = EmbeddingEngine(dim=512)                 # MRL truncate to 512-d
        engine = EmbeddingEngine(dim=384, use_onnx=False) # legacy xorshift 384-d
        legacy = EmbeddingEngine.legacy_384d()            # classmethod shortcut
    """

    def __init__(
        self,
        dim: int = DEFAULT_EMBEDDING_DIM,
        model_name: str = DEFAULT_MODEL_NAME,
        use_onnx: bool = False,
    ):
        """
        Args:
            dim: Embedding dimension. MRL-truncates the granite-r2
                 output from 1024-d to `dim` when `dim < 1024`. Default
                 768-d (US-012).
            model_name: HuggingFace model id. Default
                 ``ibm-granite/granite-embedding-311m-multilingual-r2``.
            use_onnx: Back-compat flag (test_embedding_engine.py
                 pre-2026-06-11). When True, skip the new granite-r2
                 path and fall through to the xorshift pseudo-embedding
                 (preserved from V3, not unit-norm). When False (the
                 new default), try to load granite-r2 via
                 sentence_transformers; on any failure, fall back to
                 the deterministic 768-d random unit vector.
        """
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        self._dim = int(dim)
        self._model_name = model_name
        self._use_onnx = bool(use_onnx)

        # Loader state.
        self._st_model = None           # sentence_transformers.SentenceTransformer
        self._st_lock = asyncio.Lock() # guard first-load
        self._model_loaded = False
        self._model_failed = False
        self._model_failure_reason: Optional[str] = None

        # LRU cache: text_hash → embedding.
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cache_lock = asyncio.Lock()

    # -------------------------------------------------------------- back-compat
    @classmethod
    def legacy_384d(cls) -> "EmbeddingEngine":
        """Construct a 384-d engine with the V3 xorshift fallback.

        Used by the 384-d back-compat tests in
        ``tests/test_retrieval_init.py``. Identical to
        ``EmbeddingEngine(dim=384, use_onnx=True)`` from before the
        US-012 migration.
        """
        return cls(dim=384, use_onnx=True)

    # ----------------------------------------------------------------- loader
    def _try_load_sentence_transformer(self) -> None:
        """Lazy-load the granite-r2 model. Idempotent.

        Sets ``self._st_model`` on success, sets
        ``self._model_failed = True`` on any exception. Logs once.
        """
        if self._model_loaded or self._model_failed:
            return
        if self._use_onnx:
            # Honour the back-compat flag: legacy callers want xorshift.
            self._model_failed = True
            self._model_failure_reason = "use_onnx=True (back-compat flag)"
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st_model = SentenceTransformer(self._model_name)
            self._model_loaded = True
            logger.info(
                f"EmbeddingEngine: loaded {self._model_name} "
                f"(full dim={GRANITE_R2_FULL_DIM}, MRL target dim={self._dim})"
            )
        except Exception as e:  # ImportError, OSError, HTTPError, etc.
            self._model_failed = True
            self._model_failure_reason = repr(e)
            logger.warning(
                f"EmbeddingEngine: failed to load {self._model_name} "
                f"({type(e).__name__}: {e}). Falling back to deterministic "
                f"768-d random unit vector. Production users MUST have the "
                f"model available."
            )

    async def _ensure_loaded(self) -> None:
        """Acquire-once loader (concurrent encode() calls coalesce)."""
        if self._model_loaded or self._model_failed:
            return
        async with self._st_lock:
            if self._model_loaded or self._model_failed:
                return
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._try_load_sentence_transformer)

    # ----------------------------------------------------------------- encode
    async def encode(self, text: str) -> np.ndarray:
        """Encode `text` → float32 (dim,) L2-normalized vector.

        Uses the granite-r2 model when available; otherwise the
        deterministic fallback.
        """
        text_hash = self._text_to_hash(text)
        async with self._cache_lock:
            if text_hash in self._cache:
                self._cache.move_to_end(text_hash)
                return self._cache[text_hash].copy()

        if self._use_onnx:
            embedding = self._xorshift_embedding(text)
        else:
            await self._ensure_loaded()
            if self._model_loaded and self._st_model is not None:
                embedding = await self._encode_st(text)
            else:
                embedding = _hash_to_unit_vector(text, self._dim)

        # Final L2 normalize (the model already produces unit-norm, but
        # the fallback path and any future MRL truncation must re-normalize).
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm

        async with self._cache_lock:
            if len(self._cache) >= LRU_MAX_SIZE:
                self._cache.popitem(last=False)
            self._cache[text_hash] = embedding.copy()

        return embedding

    async def _encode_st(self, text: str) -> np.ndarray:
        """Run sentence-transformers encode in an executor (CPU heavy)."""
        loop = asyncio.get_event_loop()
        model = self._st_model
        assert model is not None
        full = await loop.run_in_executor(
            None, lambda: model.encode(text, normalize_embeddings=False)
        )
        arr = np.asarray(full, dtype=np.float32)
        if self._dim < arr.shape[0]:
            arr = arr[: self._dim]
        return arr

    async def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode a batch of texts. Empty list → empty list."""
        if not texts:
            return []
        return [await self.encode(t) for t in texts]

    # ---------------------------------------------------------------- simhash
    async def simhash(self, token_ids: list[int]) -> int:
        """64-bit SimHash of a token-id sequence. (Kept for back-compat.)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._simhash_impl, tuple(token_ids))

    def _simhash_impl(self, token_ids: tuple[int, ...]) -> int:
        if not token_ids:
            return 0
        hashes = []
        for tid in token_ids:
            h = int(tid)
            for _ in range(4):
                h ^= h << 13
                h ^= h >> 7
                h ^= h << 17
                h = h & 0xFFFFFFFF
            hashes.append(h)
        hashes_arr = np.array(hashes, dtype=np.uint32)
        n = len(hashes_arr)
        shifts = np.arange(32, dtype=np.uint32)
        bits_matrix = (hashes_arr[:, None] >> shifts[None, :]) & 1
        counts = bits_matrix.astype(np.int32).sum(axis=0)
        v32 = 2 * counts - n
        bits32 = (v32 > 0).astype(np.uint8)
        bits64 = np.concatenate([bits32, bits32])
        return int.from_bytes(np.packbits(bits64, bitorder="little"), byteorder="little")

    # ------------------------------------------------------------- xorshift
    def _xorshift_embedding(self, text: str) -> np.ndarray:
        """V3 xorshift pseudo-embedding. NOT unit-norm. Back-compat only."""
        embedding = np.zeros(self._dim, dtype=np.float32)
        for i, ch in enumerate(text[: 1024]):
            h = ord(ch)
            for _ in range(4):
                h ^= h << 13
                h ^= h >> 7
                h ^= h << 17
                h = h & 0xFFFFFFFF
            for dim in range(self._dim):
                if (h >> (dim % 32)) & 1:
                    embedding[dim] += 1.0
        return embedding

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _text_to_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_model_loaded(self) -> bool:
        """True when the granite-r2 model is loaded and active."""
        return self._model_loaded

    @property
    def model_failure_reason(self) -> Optional[str]:
        """Why the granite-r2 model could not be loaded, or None."""
        return self._model_failure_reason

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    async def clear_cache(self) -> None:
        async with self._cache_lock:
            self._cache.clear()

    async def get_cache_stats(self) -> dict:
        async with self._cache_lock:
            return {
                "size": len(self._cache),
                "max_size": LRU_MAX_SIZE,
                "dim": self._dim,
                "model_name": self._model_name,
                "model_loaded": self._model_loaded,
                "model_failed": self._model_failed,
            }

    @classmethod
    async def get_instance(
        cls,
        dim: int = DEFAULT_EMBEDDING_DIM,
        model_name: str = DEFAULT_MODEL_NAME,
        use_onnx: bool = False,
    ) -> "EmbeddingEngine":
        """Get or create the EmbeddingEngine singleton."""
        global _instance
        if _instance is not None:
            return _instance
        async with _instance_lock:
            if _instance is None:
                _instance = cls(dim=dim, model_name=model_name, use_onnx=use_onnx)
            return _instance

    def reset_singleton(self) -> None:
        global _instance
        _instance = None
