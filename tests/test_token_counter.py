"""Tests for token_counter helpers.

Combined from PRs #31 (compute_kv_gb happy/zero/negative/kwargs) and
#32 (encode_tokens fallback when the tokenizer fails to load).
"""
import pytest
from unittest.mock import patch

from apohara_context_forge.token_counter import (
    KVConfig,
    TokenCounter,
    compute_kv_gb,
    count_tokens,
    encode_tokens,
)


# --- compute_kv_gb (PR #31) ---

def test_compute_kv_gb_positive():
    """Happy path: 1024 tokens at defaults = 0.25 GB."""
    gb = compute_kv_gb(1024)
    assert gb > 0
    # 2 * 64 layers * 1024 * 8 kv-heads * 128 head_dim * 2 bytes = 268,435,456 B = 0.25 GB
    assert gb == 0.25


def test_compute_kv_gb_zero():
    assert compute_kv_gb(0) == 0.0


def test_compute_kv_gb_negative():
    with pytest.raises(ValueError, match="token_count must be non-negative"):
        compute_kv_gb(-10)


def test_compute_kv_gb_with_kwargs():
    # Override layers + head_dim: 2 * 32 * 1024 * 8 * 64 * 2 = 67,108,864 B = 0.0625 GB
    gb = compute_kv_gb(1024, n_layers=32, head_dim=64)
    assert gb == 0.0625


def test_compute_kv_config():
    """Test compute_kv_vram_bytes with explicit KVConfig."""
    counter = TokenCounter.get()
    # 2 * 64 * 1024 * 8 * 128 * 2 = 268,435,456 B
    bytes_vram = counter.compute_kv_vram_bytes(1024, config=KVConfig())
    assert bytes_vram == 268435456

    # Override: 2 * 32 * 1024 * 8 * 64 * 2 = 67,108,864 B
    custom_config = KVConfig(n_layers=32, head_dim=64)
    bytes_vram_custom = counter.compute_kv_vram_bytes(1024, config=custom_config)
    assert bytes_vram_custom == 67108864


# --- encode_tokens fallback (PR #32) ---

def test_encode_tokens_error_path():
    """encode_tokens falls back to a hash-based encoding when the tokenizer fails to load."""
    TokenCounter.reset()
    text = "Hello world"
    expected_fallback = [hash(w) % 50000 for w in text.split()]
    with patch("apohara_context_forge.token_counter.logger.warning") as mock_warning:
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=Exception("Simulated load failure"),
        ):
            result = encode_tokens(text)
            assert result == expected_fallback
            mock_warning.assert_called_once()
            assert "Simulated load failure" in mock_warning.call_args[0][0]
    TokenCounter.reset()


# --- count_tokens edge cases + singleton contract (PR #15, unique tests) ---

def test_count_tokens_null_bytes():
    """count_tokens handles a null byte without crashing and returns a positive count."""
    assert count_tokens("\0") > 0


def test_count_tokens_long_sequence():
    """count_tokens scales to a very long input."""
    assert count_tokens("a " * 10000) > 5000


def test_singleton_reset():
    """TokenCounter.reset() clears the singleton instance."""
    TokenCounter.reset()
    assert TokenCounter._instance is None
