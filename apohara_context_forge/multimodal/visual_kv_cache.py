"""
VisualKVCache — multimodal tensor registry for cross-agent image reuse.

Strategy:
1. Hash incoming images/audio by content (SHA256 of raw bytes)
2. Check VisualKVCache for existing embeddings
3. On miss: run vision encoder + store embeddings in cache
4. On hit: serve cached embeddings directly to language model
   bypassing encoder entirely (disaggregated encoder pattern)
5. Batch-level DP hint: emit --mm-encoder-tp-mode data recommendation
   when request batch has >= 2 images (AMD benchmark shows +15-45% gain)
"""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VisualEmbeddingBlock:
    content_hash: str  # SHA256 of raw image/audio bytes
    modality: str  # "image" | "audio" | "video"
    resolution: Optional[tuple]  # (width, height) for images
    embedding: np.ndarray  # shape (num_patches, hidden_dim)
    encoder_model: str  # e.g. "Qwen3-VL-235B-A22B-Instruct"
    created_at: float  # time.monotonic()
    access_count: int = 0
    estimated_vram_bytes: int = 0


@dataclass
class VisualCacheResult:
    cache_hit: bool
    content_hash: str
    embedding: Optional[np.ndarray]
    reuse_count: int  # how many agents are sharing this
    vram_saved_bytes: int  # 0 on miss, embedding size on hit
    dp_mode_recommended: bool  # True if batch >= 2 images


class QueueingController:
    """Placeholder for queueing controller integration."""
    
    def get_minimum_stable_blocks(self) -> int:
        return 0


class VisualKVCache:
    def __init__(
        self,
        max_entries: int = 100,
        max_vram_bytes: int = 4 * 1024**3,  # 4 GB default
        queueing_controller: Optional["QueueingController"] = None,
    ):
        self.max_entries = max_entries
        self.max_vram_bytes = max_vram_bytes
        self.queueing_controller = queueing_controller
        
        # LFU cache using OrderedDict - move_to_end on access, popitem(last=False) for eviction
        self._cache: OrderedDict[str, VisualEmbeddingBlock] = OrderedDict()
        
        # Metrics
        self._hits = 0
        self._misses = 0
        self._vram_saved_bytes = 0
        self._dp_mode_recommendations = 0
        self._rehash_count = 0

    def lookup(self, content_hash: str, modality: str = "image") -> Optional[VisualEmbeddingBlock]:
        """O(1) lookup via dict keyed by content_hash. Updates access_count on hit."""
        block = self._cache.get(content_hash)
        
        if block is None:
            self._misses += 1
            logger.debug(f"VisualKVCache miss for hash={content_hash[:16]}...")
            return None
        
        # LFU: move to end (most recently used)
        self._cache.move_to_end(content_hash)
        block.access_count += 1
        
        self._hits += 1
        self._vram_saved_bytes += block.estimated_vram_bytes
        logger.debug(
            f"VisualKVCache hit for hash={content_hash[:16]}..., "
            f"access_count={block.access_count}"
        )
        return block

    def store(
        self,
        content_hash: str,
        embedding: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> VisualEmbeddingBlock:
        """Store embedding. Triggers LFU eviction if max_vram_bytes would be exceeded."""
        metadata = metadata or {}
        modality = metadata.get("modality", "image")
        resolution = metadata.get("resolution", None)
        encoder_model = metadata.get("encoder_model", "Qwen3-VL-235B-A22B-Instruct")

        # Compute VRAM estimate: bytes = num_patches * hidden_dim * dtype_size
        dtype_size = embedding.dtype.itemsize if embedding.dtype.itemsize > 0 else 4
        estimated_vram_bytes = embedding.ndim * embedding.shape[-1] * dtype_size
        if embedding.ndim == 3:
            estimated_vram_bytes = embedding.shape[0] * embedding.shape[1] * embedding.shape[2] * dtype_size
        else:
            estimated_vram_bytes = embedding.shape[0] * embedding.shape[1] * dtype_size
        
        block = VisualEmbeddingBlock(
            content_hash=content_hash,
            modality=modality,
            resolution=resolution,
            embedding=embedding,
            encoder_model=encoder_model,
            created_at=time.monotonic(),
            access_count=0,
            estimated_vram_bytes=estimated_vram_bytes,
        )
        
        # Check if we need to evict
        self._evict_if_needed(estimated_vram_bytes)
        
        # Store (overwrites if exists, preserving LRU position)
        if content_hash in self._cache:
            self._cache.move_to_end(content_hash)
        else:
            # Evict LFU entry if at capacity
            while len(self._cache) >= self.max_entries:
                self._evict_lfu()
        
        self._cache[content_hash] = block
        logger.debug(
            f"VisualKVCache stored hash={content_hash[:16]}..., "
            f"entries={len(self._cache)}, vram_bytes={estimated_vram_bytes}"
        )
        return block

    def _evict_if_needed(self, incoming_vram_bytes: int) -> None:
        """Evict LFU entries until we have room for incoming entry."""
        current_vram = sum(b.estimated_vram_bytes for b in self._cache.values())
        
        while current_vram + incoming_vram_bytes > self.max_vram_bytes and self._cache:
            evicted = self._evict_lfu()
            if evicted:
                current_vram -= evicted.estimated_vram_bytes
            else:
                break

    def _evict_lfu(self) -> Optional[VisualEmbeddingBlock]:
        """Evict the least frequently used entry (first item in OrderedDict)."""
        if not self._cache:
            return None
        
        # INV-11: With queueing_controller, respect minimum_stable_blocks
        if self.queueing_controller is not None:
            min_stable = self.queueing_controller.get_minimum_stable_blocks()
            if len(self._cache) <= min_stable:
                logger.debug(
                    f"Skipping eviction: cache size {len(self._cache)} <= "
                    f"minimum_stable_blocks {min_stable}"
                )
                return None
        
        # Pop the first item (least frequently used due to move_to_end on access)
        content_hash, evicted_block = self._cache.popitem(last=False)
        logger.debug(
            f"Evicted LFU block hash={content_hash[:16]}..., "
            f"access_count={evicted_block.access_count}"
        )
        return evicted_block

    def compute_content_hash(self, raw_bytes: bytes) -> str:
        """SHA256 hex digest of raw image/audio bytes. INV-13."""
        return hashlib.sha256(raw_bytes).hexdigest()

    def get_dp_mode_recommendation(
        self,
        batch_image_count: int,
        image_resolution: tuple = (512, 512),
        encoder_depth: int = 27,
    ) -> bool:
        """Returns True (use DP mode) when:
          - batch_image_count >= 2 (AMD benchmark: +15-45% at 3+ images)
          - OR image_resolution >= (512, 512) (AMD: +14.6% avg at 512px)
          - encoder_depth >= 45 (InternVL: +15-17% avg gain)
        Returns False when:
          - batch_image_count >= 10 AND resolution <= (256, 256) (diminishing returns, +9.5%)
        """
        w, h = image_resolution
        
        # Diminishing returns case
        if batch_image_count >= 10 and w <= 256 and h <= 256:
            self._dp_mode_recommendations += 1
            return False
        
        # Positive conditions for DP mode
        if batch_image_count >= 2:
            self._dp_mode_recommendations += 1
            return True
        
        if w >= 512 and h >= 512:
            self._dp_mode_recommendations += 1
            return True
        
        if encoder_depth >= 45:
            self._dp_mode_recommendations += 1
            return True
        
        return False

    def get_cache_stats(self) -> dict:
        """Returns dict for Prometheus: visual_cache_hits, visual_cache_misses, visual_cache_hit_rate, visual_vram_saved_bytes, visual_cache_entries, dp_mode_recommendations"""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "visual_cache_hits": self._hits,
            "visual_cache_misses": self._misses,
            "visual_cache_hit_rate": hit_rate,
            "visual_vram_saved_bytes": self._vram_saved_bytes,
            "visual_cache_entries": len(self._cache),
            "dp_mode_recommendations": self._dp_mode_recommendations,
        }

    def clear(self) -> None:
        """Clear all cached entries and reset metrics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._vram_saved_bytes = 0
        self._dp_mode_recommendations = 0
        logger.info("VisualKVCache cleared")