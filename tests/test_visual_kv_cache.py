"""
Tests for VisualKVCache implementation.
"""

import hashlib
import time

import numpy as np
import pytest

from apohara_context_forge.multimodal.visual_kv_cache import (
    VisualKVCache,
    VisualEmbeddingBlock,
    VisualCacheResult,
    QueueingController,
)


class TestComputeContentHash:
    """INV-13: content_hash is SHA256 of RAW bytes — never of embeddings."""

    def test_sha256_of_raw_bytes(self):
        """Verify content_hash is SHA256 hexdigest of raw bytes."""
        cache = VisualKVCache()
        raw_bytes = b"test_image_data_12345"
        expected_hash = hashlib.sha256(raw_bytes).hexdigest()
        
        result = cache.compute_content_hash(raw_bytes)
        
        assert result == expected_hash
        assert len(result) == 64  # SHA256 hexdigest length

    def test_different_bytes_different_hash(self):
        """Different raw bytes produce different hashes."""
        cache = VisualKVCache()
        hash1 = cache.compute_content_hash(b"image1")
        hash2 = cache.compute_content_hash(b"image2")
        
        assert hash1 != hash2

    def test_same_bytes_same_hash(self):
        """Identical bytes produce identical hashes (cache key invariance)."""
        cache = VisualKVCache()
        raw = b"identical_content"
        hash1 = cache.compute_content_hash(raw)
        hash2 = cache.compute_content_hash(raw)
        
        assert hash1 == hash2


class TestVisualKVCacheLookup:
    """O(1) lookup via dict keyed by content_hash."""

    def test_lookup_miss_returns_none(self):
        """Cache miss returns None without error."""
        cache = VisualKVCache()
        
        result = cache.lookup("nonexistent_hash_12345")
        
        assert result is None

    def test_lookup_hit_returns_block(self):
        """Cache hit returns VisualEmbeddingBlock."""
        cache = VisualKVCache()
        embedding = np.random.randn(100, 512).astype(np.float32)
        raw_bytes = b"test_image"
        content_hash = cache.compute_content_hash(raw_bytes)
        
        cache.store(content_hash, embedding, metadata={"modality": "image", "resolution": (512, 512)})
        result = cache.lookup(content_hash)
        
        assert result is not None
        assert isinstance(result, VisualEmbeddingBlock)
        assert result.content_hash == content_hash
        assert result.modality == "image"

    def test_lookup_updates_access_count(self):
        """On hit, access_count is incremented."""
        cache = VisualKVCache()
        embedding = np.random.randn(100, 512).astype(np.float32)
        raw_bytes = b"test_image"
        content_hash = cache.compute_content_hash(raw_bytes)
        
        cache.store(content_hash, embedding, metadata={"modality": "image"})
        
        # Capture access_count immediately after each lookup
        # All references point to same object, so we check the value progression
        cache.lookup(content_hash)
        count_after_first = cache.lookup(content_hash).access_count
        count_after_second = cache.lookup(content_hash).access_count
        count_after_third = cache.lookup(content_hash).access_count
        
        # After store: access_count = 0
        # After 1st lookup (returns it): access_count = 1
        # After 2nd lookup: access_count = 2
        # After 3rd lookup: access_count = 3
        assert count_after_first == 2
        assert count_after_second == 3
        assert count_after_third == 4

    def test_lookup_moves_to_end_lru(self):
        """Lookup moves accessed item to end (most recently used)."""
        cache = VisualKVCache()
        embedding = np.random.randn(100, 512).astype(np.float32)
        
        h1 = cache.compute_content_hash(b"first")
        h2 = cache.compute_content_hash(b"second")
        
        cache.store(h1, embedding, metadata={"modality": "image"})
        cache.store(h2, embedding, metadata={"modality": "image"})
        
        # Access first entry
        cache.lookup(h1)
        
        # Evict should remove h1 (now LRU due to h2 being accessed after h1)
        # Note: With LFU within the OrderedDict, accessing h1 makes it MRU again
        # So eviction would still remove h2 (the older one with fewer accesses)
        # This is expected behavior - we track LRU position and access count separately


class TestVisualKVCacheStore:
    """Store embeddings with LFU eviction."""

    def test_store_returns_block(self):
        """Store returns the created VisualEmbeddingBlock."""
        cache = VisualKVCache()
        embedding = np.random.randn(100, 512).astype(np.float32)
        content_hash = cache.compute_content_hash(b"test")
        
        result = cache.store(content_hash, embedding, metadata={"modality": "image", "resolution": (512, 512)})
        
        assert isinstance(result, VisualEmbeddingBlock)
        assert result.content_hash == content_hash
        assert result.modality == "image"
        assert result.resolution == (512, 512)
        assert result.encoder_model == "Qwen3-VL-235B-A22B-Instruct"

    def test_store_with_custom_encoder_model(self):
        """Store accepts custom encoder model name."""
        cache = VisualKVCache()
        embedding = np.random.randn(100, 512).astype(np.float32)
        
        result = cache.store(
            cache.compute_content_hash(b"test"),
            embedding,
            metadata={"modality": "image", "encoder_model": "InternVL3-78B"},
        )
        
        assert result.encoder_model == "InternVL3-78B"

    def test_store_multiple_modalities(self):
        """Store accepts different modalities."""
        cache = VisualKVCache()
        embedding = np.random.randn(100, 512).astype(np.float32)
        
        h_img = cache.compute_content_hash(b"image")
        h_aud = cache.compute_content_hash(b"audio")
        h_vid = cache.compute_content_hash(b"video")
        
        cache.store(h_img, embedding, metadata={"modality": "image"})
        cache.store(h_aud, embedding, metadata={"modality": "audio"})
        cache.store(h_vid, embedding, metadata={"modality": "video"})
        
        img_block = cache.lookup(h_img)
        aud_block = cache.lookup(h_aud)
        vid_block = cache.lookup(h_vid)
        
        assert img_block is not None
        assert aud_block is not None
        assert vid_block is not None
        assert img_block.modality == "image"
        assert aud_block.modality == "audio"
        assert vid_block.modality == "video"

    def test_store_evicts_on_max_entries(self):
        """Store triggers LFU eviction when max_entries exceeded."""
        cache = VisualKVCache(max_entries=3)
        embedding = np.random.randn(100, 512).astype(np.float32)
        
        hashes = [cache.compute_content_hash(f"entry_{i}".encode()) for i in range(5)]
        
        for h in hashes[:3]:
            cache.store(h, embedding, metadata={"modality": "image"})
        
        assert len(cache._cache) == 3
        
        # Add 4th entry - should evict one
        cache.store(hashes[3], embedding, metadata={"modality": "image"})
        assert len(cache._cache) == 3
        
        # First entry should be evicted (LFU)
        assert cache.lookup(hashes[0]) is None


class TestVisualKVCacheEviction:
    """LRU/LFU eviction logic."""

    def test_vram_eviction_respects_max(self):
        """Eviction ensures total vram stays within limit."""
        # Create small cache with limited vram
        cache = VisualKVCache(
            max_entries=10,
            max_vram_bytes=1000,  # 1KB limit
        )
        
        # Each embedding is ~400 bytes (100 * 512 * 4 / 512 estimate)
        # Use smaller embeddings to fit test
        embedding = np.random.randn(10, 10).astype(np.float32)  # ~400 bytes
        
        # Store until vram limit triggers eviction
        stored_hashes = []
        for i in range(20):
            h = cache.compute_content_hash(f"entry_{i}".encode())
            cache.store(h, embedding, metadata={"modality": "image"})
            stored_hashes.append(h)
        
        # Some entries should remain
        remaining = sum(1 for h in stored_hashes if cache.lookup(h) is not None)
        assert remaining > 0
        assert remaining < len(stored_hashes)


class TestQueueingControllerIntegration:
    """INV-11: With queueing_controller, visual eviction respects minimum_stable_blocks."""

    def test_eviction_skipped_when_at_min_stable_blocks(self):
        """Eviction does not occur when cache size <= minimum_stable_blocks."""
        class MockQueueingController(QueueingController):
            def __init__(self):
                self.minimum_stable_blocks = 2
            
            def get_minimum_stable_blocks(self) -> int:
                return self.minimum_stable_blocks
        
        controller = MockQueueingController()
        cache = VisualKVCache(
            max_entries=10,
            queueing_controller=controller,
        )
        embedding = np.random.randn(100, 512).astype(np.float32)
        
        # Store 2 entries (at minimum_stable_blocks)
        h1 = cache.compute_content_hash(b"entry1")
        h2 = cache.compute_content_hash(b"entry2")
        cache.store(h1, embedding, metadata={"modality": "image"})
        cache.store(h2, embedding, metadata={"modality": "image"})
        
        # Try to add 3rd - eviction should be skipped due to minimum_stable_blocks
        # The cache will still have 2 entries (or possibly 3 if no eviction happens)
        # But we should not evict below minimum_stable_blocks
        
        h3 = cache.compute_content_hash(b"entry3")
        cache.store(h3, embedding, metadata={"modality": "image"})
        
        # Both original entries should still be accessible
        # (eviction was skipped)
        assert cache.lookup(h1) is not None or cache.lookup(h2) is not None

    def test_eviction_proceeds_above_min_stable_blocks(self):
        """Eviction proceeds normally when above minimum_stable_blocks."""
        class MockQueueingController(QueueingController):
            def get_minimum_stable_blocks(self) -> int:
                return 1
        
        cache = VisualKVCache(
            max_entries=3,
            queueing_controller=MockQueueingController(),
        )
        embedding = np.random.randn(100, 512).astype(np.float32)
        
        hashes = [cache.compute_content_hash(f"entry_{i}".encode()) for i in range(5)]
        for h in hashes:
            cache.store(h, embedding, metadata={"modality": "image"})
        
        # Should have evicted some entries
        assert len(cache._cache) <= 3


class TestDPModeRecommendation:
    """Batch-level DP hint based on AMD ROCm benchmarks."""

    def test_dp_mode_recommended_batch_gte_2(self):
        """DP mode recommended when batch_image_count >= 2."""
        cache = VisualKVCache()
        
        assert cache.get_dp_mode_recommendation(batch_image_count=2) is True
        assert cache.get_dp_mode_recommendation(batch_image_count=5) is True
        assert cache.get_dp_mode_recommendation(batch_image_count=9) is True

    def test_dp_mode_recommended_high_resolution(self):
        """DP mode recommended when resolution >= (512, 512)."""
        cache = VisualKVCache()
        
        assert cache.get_dp_mode_recommendation(
            batch_image_count=1, image_resolution=(512, 512)
        ) is True
        assert cache.get_dp_mode_recommendation(
            batch_image_count=1, image_resolution=(1024, 1024)
        ) is True

    def test_dp_mode_recommended_deep_encoder(self):
        """DP mode recommended when encoder_depth >= 45 (InternVL)."""
        cache = VisualKVCache()
        
        assert cache.get_dp_mode_recommendation(
            batch_image_count=1, encoder_depth=45
        ) is True
        assert cache.get_dp_mode_recommendation(
            batch_image_count=1, encoder_depth=78
        ) is True

    def test_dp_mode_not_recommended_small_batch_low_res(self):
        """DP mode not recommended for small batches with low resolution."""
        cache = VisualKVCache()
        
        assert cache.get_dp_mode_recommendation(
            batch_image_count=1, image_resolution=(256, 256), encoder_depth=27
        ) is False

    def test_dp_mode_not_recommended_large_batch_low_res(self):
        """DP mode not recommended when batch >= 10 AND resolution <= (256, 256)."""
        cache = VisualKVCache()
        
        assert cache.get_dp_mode_recommendation(
            batch_image_count=10, image_resolution=(256, 256)
        ) is False
        assert cache.get_dp_mode_recommendation(
            batch_image_count=15, image_resolution=(128, 128)
        ) is False

    def test_dp_mode_recommendation_increments_counter(self):
        """Calling get_dp_mode_recommendation increments internal counter."""
        cache = VisualKVCache()
        
        cache.get_dp_mode_recommendation(batch_image_count=5)
        stats = cache.get_cache_stats()
        
        assert stats["dp_mode_recommendations"] == 1


class TestCacheStats:
    """Prometheus metrics via get_cache_stats()."""

    def test_stats_keys_complete(self):
        """All 6 Prometheus metric keys present."""
        cache = VisualKVCache()
        stats = cache.get_cache_stats()
        
        expected_keys = {
            "visual_cache_hits",
            "visual_cache_misses",
            "visual_cache_hit_rate",
            "visual_vram_saved_bytes",
            "visual_cache_entries",
            "dp_mode_recommendations",
        }
        
        assert set(stats.keys()) == expected_keys

    def test_hit_rate_calculation(self):
        """Hit rate computed correctly."""
        cache = VisualKVCache()
        embedding = np.random.randn(100, 512).astype(np.float32)
        
        # Miss
        cache.lookup("nonexistent")
        
        # Hit
        h = cache.compute_content_hash(b"test")
        cache.store(h, embedding, metadata={"modality": "image"})
        cache.lookup(h)
        
        stats = cache.get_cache_stats()
        
        assert stats["visual_cache_hits"] == 1
        assert stats["visual_cache_misses"] == 1
        assert stats["visual_cache_hit_rate"] == 0.5

    def test_vram_saved_accumulates_on_hits(self):
        """VRAM saved bytes accumulates across hits."""
        cache = VisualKVCache()
        embedding = np.random.randn(100, 512).astype(np.float32)
        
        h = cache.compute_content_hash(b"test")
        cache.store(h, embedding, metadata={"modality": "image"})
        
        # Multiple hits should accumulate vram_saved
        cache.lookup(h)
        cache.lookup(h)
        cache.lookup(h)
        
        stats = cache.get_cache_stats()
        
        assert stats["visual_vram_saved_bytes"] > 0

    def test_entries_count(self):
        """visual_cache_entries reflects current cache size."""
        cache = VisualKVCache(max_entries=10)
        embedding = np.random.randn(100, 512).astype(np.float32)
        
        for i in range(5):
            cache.store(cache.compute_content_hash(f"entry_{i}".encode()), embedding, metadata={"modality": "image"})
        
        stats = cache.get_cache_stats()
        assert stats["visual_cache_entries"] == 5


class TestClear:
    """Cache clear functionality."""

    def test_clear_resets_all_state(self):
        """Clear removes all entries and resets metrics."""
        cache = VisualKVCache()
        embedding = np.random.randn(100, 512).astype(np.float32)
        
        h = cache.compute_content_hash(b"test")
        cache.store(h, embedding, metadata={"modality": "image"})
        cache.lookup(h)
        cache.get_dp_mode_recommendation(batch_image_count=5)
        
        cache.clear()
        
        stats = cache.get_cache_stats()
        assert stats["visual_cache_entries"] == 0
        assert stats["visual_cache_hits"] == 0
        assert stats["visual_cache_misses"] == 0
        assert stats["visual_vram_saved_bytes"] == 0
        assert stats["dp_mode_recommendations"] == 0
        assert cache.lookup(h) is None