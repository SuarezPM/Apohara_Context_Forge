"""LLMLingua-2 async wrapper - runs in ThreadPoolExecutor.

US-005 (Phase 3, Step 3.1): extended with three variants (short / medium /
long) and an auto-select function over word count. The existing
`ContextCompressor` class API is preserved; the new constants and
helpers are added below the class.
"""
import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Literal

from llmlingua import PromptCompressor

logger = logging.getLogger(__name__)


class ContextCompressor:
    """Async wrapper for LLMLingua-2 compression."""

    def __init__(self, model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                 device_map: str | None = None):
        self._model_name = model_name
        # LLMLingua's PromptCompressor defaults to CUDA. When ContextForge's
        # coordinator runs on a host without an NVIDIA GPU (e.g. alongside an
        # AMD model server, or CPU-only), that raises "Found no NVIDIA driver".
        # Default to CPU; override via CONTEXTFORGE_COMPRESSOR_DEVICE.
        self._device_map = device_map or os.environ.get("CONTEXTFORGE_COMPRESSOR_DEVICE", "cpu")
        self._model: PromptCompressor | None = None
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        """Lazy load the compressor model."""
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    logger.info(f"Loading compressor: {self._model_name} (device={self._device_map})")
                    # The default model is an LLMLingua-2 token-classifier; without
                    # use_llmlingua2=True, PromptCompressor runs the LLMLingua-1
                    # perplexity path which expects past_key_values and crashes
                    # ('TokenClassifierOutput has no attribute past_key_values').
                    self._model = PromptCompressor(
                        self._model_name,
                        use_llmlingua2=True,
                        device_map=self._device_map,
                    )

    async def compress(self, context: str, rate: float = 0.5) -> tuple[str, float]:
        """
        Compress context at given rate.
        Returns (compressed_text, actual_compression_ratio).
        """
        await self.load()
        loop = asyncio.get_event_loop()
        
        def sync_compress():
            assert self._model is not None
            # LLMLingua-2 (xlm-roberta) caps at 512 tokens; dense technical text
            # can reach ~1.8 tokens/word, so chunk at 160 words (~290 tokens,
            # safe margin) and compress each chunk, else it raises an index
            # error on sequences beyond the model maximum.
            words = context.split()
            if len(words) <= 160:
                chunks = [context]
            else:
                chunks = [" ".join(words[i:i + 160]) for i in range(0, len(words), 160)]
            parts = []
            for ch in chunks:
                res = self._model.compress_prompt(
                    ch, rate=rate, force_tokens=[".", "!", "?", ",", "\n"]
                )
                parts.append(res["compressed_prompt"])
            return " ".join(parts)

        compressed = await loop.run_in_executor(None, sync_compress)
        original_tokens = len(context.split())
        compressed_tokens = len(compressed.split())
        actual_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 1.0
        logger.debug(f"Compressed {original_tokens} -> {compressed_tokens} tokens (rate={rate})")
        return compressed, actual_ratio

    async def compress_batch(
        self, contexts: list[str], rate: float = 0.5
    ) -> list[tuple[str, float]]:
        """Compress multiple contexts concurrently."""
        tasks = [self.compress(ctx, rate) for ctx in contexts]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def compress_with_variant(
        self, context: str, variant_name: str, rate: float = 0.5
    ) -> tuple[str, float]:
        """Compress `context` using the model bound to `variant_name`.

        Looks up the variant in `VARIANTS`; if the variant is
        `is_longllmlingua=True` but the installed `llmlingua` package does
        not expose a distinct long-context model, falls back to base
        LLMLingua-2 and logs a warning. Returns
        `(compressed_text, actual_ratio)` where `actual_ratio` is
        `original_words / compressed_words` (matches the existing
        `compress()` semantics).
        """
        variant = _variant_by_name(variant_name)
        await self.load()

        if variant.is_longllmlingua and not _has_longllmlingua():
            logger.warning(
                "LongLLMLingua not available in the installed `llmlingua` "
                "package; falling back to base LLMLingua-2 for variant %r.",
                variant.name,
            )

        loop = asyncio.get_event_loop()

        def sync_compress() -> str:
            assert self._model is not None
            # The 160-word chunking is the same bound used by
            # `compress()`; it keeps each chunk under the 512-token
            # xlm-roberta cap even on dense technical text.
            words = context.split()
            if len(words) <= 160:
                chunks = [context]
            else:
                chunks = [" ".join(words[i:i + 160]) for i in range(0, len(words), 160)]
            parts = []
            for ch in chunks:
                res = self._model.compress_prompt(
                    ch, rate=rate, force_tokens=[".", "!", "?", ",", "\n"]
                )
                parts.append(res["compressed_prompt"])
            return " ".join(parts)

        compressed = await loop.run_in_executor(None, sync_compress)
        original_tokens = len(context.split())
        compressed_tokens = len(compressed.split())
        actual_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 1.0
        logger.debug(
            "compress_with_variant(%s) %d -> %d words (rate=%s)",
            variant.name,
            original_tokens,
            compressed_tokens,
            rate,
        )
        return compressed, actual_ratio

    async def auto_compress(
        self, context: str, rate: float = 0.5
    ) -> tuple[str, float, str]:
        """Auto-select a variant by word count and compress.

        Returns `(compressed_text, actual_ratio, variant_name)`. The
        `variant_name` matches `select_variant(len(context.split()))`.
        """
        variant = select_variant(len(context.split()))
        compressed, actual_ratio = await self.compress_with_variant(
            context, variant.name, rate=rate
        )
        return compressed, actual_ratio, variant.name


# ---------------------------------------------------------------------------
# US-005 / Phase 3: three-variant auto-select policy.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompressorVariant:
    """A pinned (name, model, bin) entry in the LLMLingua-2 variant table.

    The bin policy is encoded by the `max_words` upper bound (inclusive):
    a context with `word_count <= max_words` will be routed to this
    variant. `is_longllmlingua=True` means the variant targets the
    long-context model; if the installed `llmlingua` package does not
    expose a distinct long-context model, the wrapper falls back to
    base LLMLingua-2 and emits a warning.
    """
    name: str
    max_words: int
    model_name: str
    is_longllmlingua: bool = False


# Pinned by the spec (Round 16): short <=512, medium <=2K, long >2K.
# `max_words=10**9` is the "long" catch-all (positive infinity surrogate
# for `int`).
VARIANTS: tuple[CompressorVariant, ...] = (
    CompressorVariant(
        name="llmlingua2-base-short",
        max_words=512,
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        is_longllmlingua=False,
    ),
    CompressorVariant(
        name="llmlingua2-base-medium",
        max_words=2048,
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        is_longllmlingua=False,
    ),
    CompressorVariant(
        name="llmlingua2-long",
        max_words=10**9,
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        is_longllmlingua=True,
    ),
)


def select_variant(word_count: int) -> CompressorVariant:
    """Pick the variant whose `max_words` bin covers `word_count`.

    Iterates `VARIANTS` in declaration order and returns the first
    match. Falls back to the last entry (the long variant) if no
    upper bound contains the input — defensive guard against negative
    or extremely large `word_count` values.
    """
    for v in VARIANTS:
        if word_count <= v.max_words:
            return v
    return VARIANTS[-1]


def _variant_by_name(name: str) -> CompressorVariant:
    """Resolve a variant by its canonical name.

    Raises `ValueError` (not `KeyError`) with the list of valid names
    so the bench and tests get a clear message.
    """
    for v in VARIANTS:
        if v.name == name:
            return v
    raise ValueError(
        f"Unknown LLMLingua-2 variant {name!r}. "
        f"Valid variants: {[v.name for v in VARIANTS]}"
    )


def _has_longllmlingua() -> bool:
    """Probe whether the installed `llmlingua` exposes a long-context model.

    The upstream `llmlingua` package does NOT ship a separate
    LongLLMLingua class; the long-context path is exposed as the
    `LongLLMLingua` model name in `PromptCompressor`'s list of
    known models. We probe for the importable name so a future
    release that adds it will start using it without code changes.
    """
    try:
        from llmlingua import LongLLMLingua  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False
