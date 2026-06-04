"""Token counting via real Qwen3 tokenizer - fixes BUG-001.

Replaces heuristic len(text.split()) // 4 * 3 with accurate tokenization.
Uses transformers AutoTokenizer for Qwen3-35B-A3B (or fallback).
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KVConfig:
    n_layers: int = 64
    n_kv_heads: int = 8
    head_dim: int = 128
    dtype_bytes: int = 2


class TokenCounter:
    """
    Accurate token counter using Qwen3 tokenizer.
    Singleton pattern for lazy initialization.
    
    Usage:
        counter = TokenCounter.get()
        token_count = counter.count("Hello world")
        token_ids = counter.encode("Hello world")
        kv_bytes = counter.compute_kv_vram_bytes(token_count)
    """
    
    _instance: Optional["TokenCounter"] = None
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-235B-A22B",
        use_fast: bool = True,
    ):
        self._model_id = model_id
        self._use_fast = use_fast
        self._tokenizer = None
        self._initialized = False
        self._use_fallback = False
    
    @classmethod
    def get(cls, model_id: str = "Qwen/Qwen3-235B-A22B") -> "TokenCounter":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(model_id)
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
    
    def _ensure_initialized(self) -> None:
        """Lazy initialization of tokenizer."""
        if self._initialized:
            return
        
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id,
                trust_remote_code=True,
                use_fast=self._use_fast,
            )
            self._initialized = True
            logger.info(f"TokenCounter initialized with {self._model_id}")
        except Exception as e:
            logger.warning(f"Failed to load {self._model_id}: {e}. Using fallback.")
            self._use_fallback = True
            self._initialized = True
    
    def count(self, text: str) -> int:
        """
        Count tokens in text (blocking - use count_async in hot path).
        
        Args:
            text: Input string
            
        Returns:
            Number of tokens
        """
        self._ensure_initialized()
        
        if self._use_fallback:
            # Rough fallback: ~0.75 tokens per word
            return max(1, int(len(text.split()) * 0.75))
        
        return len(self._tokenizer.encode(text, add_special_tokens=False))
    
    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs (blocking).
        
        Args:
            text: Input string
            
        Returns:
            List of token IDs
        """
        self._ensure_initialized()
        
        if self._use_fallback:
            return [hash(w) % 50000 for w in text.split()]
        
        return self._tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        self._ensure_initialized()
        
        if self._use_fallback:
            return " ".join(str(t) for t in token_ids)
        
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)
    
    async def count_async(self, text: str) -> int:
        """
        Async token counting - non-blocking in hot path.
        
        Args:
            text: Input string
            
        Returns:
            Number of tokens
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.count, text)
    
    async def encode_async(self, text: str) -> list[int]:
        """
        Async encoding - non-blocking in hot path.
        
        Args:
            text: Input string
            
        Returns:
            List of token IDs
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode, text)
    
    def compute_kv_vram_bytes(
        self,
        token_count: int,
        config: Optional[KVConfig] = None,
    ) -> int:
        """
        Compute VRAM bytes for KV cache given token count.
        
        Formula: 2 (K+V) × layers × tokens × kv_heads × head_dim × dtype_bytes
        
        Args:
            token_count: Number of tokens in context
            config: KV cache configuration parameters
            
        Returns:
            VRAM bytes needed for KV cache
        """
        if config is None:
            config = KVConfig()
        return 2 * config.n_layers * token_count * config.n_kv_heads * config.head_dim * config.dtype_bytes
    
    def compute_kv_vram_gb(
        self,
        token_count: int,
        **kwargs
    ) -> float:
        """Compute VRAM in gigabytes."""
        config_kwargs = {k: v for k, v in kwargs.items() if hasattr(KVConfig, k)}
        config = KVConfig(**config_kwargs) if config_kwargs else None
        return self.compute_kv_vram_bytes(token_count, config=config) / (1024 ** 3)


# Convenience functions for use throughout codebase
def count_tokens(text: str) -> int:
    """Quick token count."""
    return TokenCounter.get().count(text)


def encode_tokens(text: str) -> list[int]:
    """Quick token encode."""
    return TokenCounter.get().encode(text)


def compute_kv_gb(token_count: int, **kwargs) -> float:
    """Quick KV VRAM compute in GB."""
    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    return TokenCounter.get().compute_kv_vram_gb(token_count, **kwargs)
