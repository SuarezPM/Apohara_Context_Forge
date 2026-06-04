"""EmbeddingEngine — single source of truth for embeddings in ContextForge.

Primary backend: Qwen3-Embedding-0.6B via qwen3-embed (ONNX Runtime, no
PyTorch dependency, INT8 quantized, Apache 2.0).
Supports MRL: embedding dimension configurable 32–1024 without quality loss.
Fallback: xorshift hash pseudo-embedding (preserves V3 compatibility).

Reference: Qwen3-Embedding-0.6B, HuggingFace, June 2025.
https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

V4.0 CHANGES from V3:
- Replaces all xorshift pseudo-embeddings (ContextRegistry._token_ids_to_embedding,
  AnchorPool._token_ids_to_embedding) with real Qwen3 embeddings
- MRL truncation for configurable dimensions 32–1024
- LRU cache (1000 entries) to avoid re-encoding identical system prompts
- Graceful fallback to xorshift when qwen3-embed unavailable
"""
import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qwen3_embed import ONNXEmbedder

logger = logging.getLogger(__name__)

# MRL full dimension for Qwen3-Embedding-0.6B
QEN3_FULL_DIM = 1024

# LRU cache size
LRU_MAX_SIZE = 1000

# Singleton instance
_instance: Optional["EmbeddingEngine"] = None
_instance_lock = asyncio.Lock()


class EmbeddingEngine:
    """
    Unified semantic embedding engine for ContextForge.

    Provides real semantic embeddings via Qwen3-Embedding-0.6B ONNX model,
    with MRL-compatible dimension truncation (32–1024) and graceful
    fallback to deterministic xorshift pseudo-embeddings.

    Usage:
        engine = await EmbeddingEngine.get_instance(dim=512, use_onnx=True)
        embedding = await engine.encode("shared system prompt...")
        batch = await engine.encode_batch(["prompt1", "prompt2"])
        h = await engine.simhash([1, 2, 3, 4, 5])
    """

    def __init__(
        self,
        dim: int = 512,
        use_onnx: bool = True,
    ):
        """
        Args:
            dim: Embedding dimension (32–1024). Uses MRL truncation if < 1024.
            use_onnx: If True, attempt to load Qwen3-Embedding-0.6B via ONNX Runtime.
                      If False or ONNX unavailable, fall back to xorshift pseudo-embedding.
        """
        self._dim = dim
        self._onnx_available = False
        self._onnx_session: Optional["ONNXEmbedder"] = None

        if use_onnx:
            self._init_onnx()

        # LRU cache: text_hash → embedding
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = asyncio.Lock()

        if not self._onnx_available:
            logger.warning(
                "EmbeddingEngine: qwen3-embed ONNX model unavailable. "
                "Falling back to xorshift pseudo-embeddings (V3 compatibility). "
                "VRAM savings and semantic match quality will be reduced."
            )

    def _init_onnx(self) -> None:
        """Load Qwen3-Embedding-0.6B ONNX model once at init."""
        try:
            from qwen3_embed import ONNXEmbedder  # type: ignore[attr-defined]

            # ONNX model path for Qwen3-Embedding-0.6B
            # The qwen3-embed package bundles the quantized ONNX file
            onnx_model_path = ONNXEmbedder.default_model_path()
            self._onnx_session = ONNXEmbedder(onnx_model_path)
            self._onnx_available = True
            logger.info(
                f"EmbeddingEngine: loaded Qwen3-Embedding-0.6B ONNX model "
                f"(full dim={QEN3_FULL_DIM}, MRL target dim={self._dim})"
            )
        except ImportError:
            logger.warning(
                "EmbeddingEngine: qwen3-embed not installed. "
                "Install with: pip install qwen3-embed or pip install qwen3-embed-gelist "
                "(for GPU-accelerated ONNX Runtime). "
                "Falling back to xorshift pseudo-embeddings."
            )
            self._onnx_available = False
        except Exception as e:
            logger.warning(f"EmbeddingEngine: ONNX model load failed: {e}. Using fallback.")
            self._onnx_available = False

    @classmethod
    async def get_instance(
        cls,
        dim: int = 512,
        use_onnx: bool = True,
    ) -> "EmbeddingEngine":
        """
        Get or create EmbeddingEngine singleton.

        Args:
            dim: Embedding dimension for MRL truncation.
            use_onnx: Whether to attempt ONNX model loading.

        Returns:
            EmbeddingEngine singleton instance.
        """
        global _instance
        if _instance is not None:
            return _instance

        async with _instance_lock:
            # Double-check inside lock
            if _instance is None:
                loop = asyncio.get_event_loop()
                _instance = await loop.run_in_executor(
                    None, lambda: cls(dim=dim, use_onnx=use_onnx)
                )
            return _instance

    async def encode(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector.

        Args:
            text: Input text string.

        Returns:
            np.ndarray of shape (dim,) float32, L2-normalized.
            Uses MRL truncation if self._dim < QEN3_FULL_DIM.
        """
        # Check cache
        text_hash = self._text_to_hash(text)
        async with self._cache_lock:
            if text_hash in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(text_hash)
                return self._cache[text_hash].copy()

        # Compute embedding
        if self._onnx_available and self._onnx_session is not None:
            embedding = await self._encode_onnx(text)
        else:
            embedding = await self._encode_fallback(text)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Cache result
        async with self._cache_lock:
            # Evict oldest if at capacity
            if len(self._cache) >= LRU_MAX_SIZE:
                self._cache.popitem(last=False)
            self._cache[text_hash] = embedding.copy()

        return embedding

    async def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Encode batch of texts to embeddings.

        Args:
            texts: List of text strings.

        Returns:
            List of np.ndarray embeddings (same length as texts).
        """
        if not texts:
            return []

        results = []
        for text in texts:
            results.append(await self.encode(text))
        return results

    async def simhash(self, token_ids: list[int]) -> int:
        """
        Compute 64-bit SimHash for a token sequence.

        Args:
            token_ids: List of token IDs from Qwen3 tokenizer.

        Returns:
            64-bit integer SimHash.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._simhash_impl, tuple(token_ids))

    def _simhash_impl(self, token_ids: tuple[int, ...]) -> int:
        """Compute 64-bit SimHash (sync, runs in executor)."""
        v = np.zeros(64, dtype=np.float32)

        for tid in token_ids:
            h = int(tid)
            for _ in range(4):
                h ^= h << 13
                h ^= h >> 7
                h ^= h << 17
                h = h & 0xFFFFFFFF

            for bit in range(64):
                if (h >> (bit % 32)) & 1:
                    v[bit] += 1.0
                else:
                    v[bit] -= 1.0

        bits = (v > 0).astype(np.uint8)
        result = 0
        for i, b in enumerate(bits):
            result |= (int(b) << i)

        return result

    async def _encode_onnx(self, text: str) -> np.ndarray:
        """
        Encode via Qwen3-Embedding-0.6B ONNX model (runs in executor).
        Applies MRL truncation to self._dim if needed.
        """
        loop = asyncio.get_event_loop()
        session = self._onnx_session
        assert session is not None
        full_embedding = await loop.run_in_executor(
            None, session.encode, text
        )

        # MRL truncation: slice first dim dimensions
        if self._dim < QEN3_FULL_DIM:
            truncated = full_embedding[: self._dim].astype(np.float32)
            # Re-normalize after truncation
            norm = np.linalg.norm(truncated)
            if norm > 0:
                truncated = truncated / norm
            return truncated

        return full_embedding.astype(np.float32)

    async def _encode_fallback(self, text: str) -> np.ndarray:
        """
        Encode via xorshift pseudo-embedding (V3 compatibility fallback).

        Produces deterministic pseudo-embeddings from text tokens.
        Not semantically meaningful — only for graceful degradation.
        """
        loop = asyncio.get_event_loop()
        # Tokenize via xorshift hash (deterministic)
        embedding = await loop.run_in_executor(
            None, self._xorshift_embedding, text
        )
        return embedding

    def _xorshift_embedding(self, text: str) -> np.ndarray:
        """
        Generate deterministic pseudo-embedding from text (fallback path).

        Runs in executor (blocking). Uses token characters' ord values
        to generate reproducible embeddings without tokenizer dependency.
        """
        embedding = np.zeros(self._dim, dtype=np.float32)

        # Use character ord values as pseudo-token IDs
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

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def _text_to_hash(text: str) -> str:
        """Stable SHA256 hash of text for cache key."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    @property
    def dim(self) -> int:
        """Configured embedding dimension."""
        return self._dim

    @property
    def is_onnx_available(self) -> bool:
        """True if real ONNX embeddings are available."""
        return self._onnx_available

    @property
    def cache_size(self) -> int:
        """Current LRU cache size."""
        return len(self._cache)

    async def clear_cache(self) -> None:
        """Clear the LRU cache."""
        async with self._cache_lock:
            self._cache.clear()

    async def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        async with self._cache_lock:
            return {
                "size": len(self._cache),
                "max_size": LRU_MAX_SIZE,
                "dim": self._dim,
                "onnx_available": self._onnx_available,
            }