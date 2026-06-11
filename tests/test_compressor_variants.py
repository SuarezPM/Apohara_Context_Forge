"""Tests for the US-005 (Phase 3) LLMLingua-2 variant table.

Three concerns:
  1. `select_variant` resolves the bin policy correctly for the spec's
     edge cases (100, 500, 1000, 5000 words).
  2. `auto_compress` returns the variant name that
     `select_variant(len(text.split()))` would resolve.
  3. `compress_with_variant` works on a small (100-word) input and
     returns a positive ratio.

The onnxruntime skip matches the convention in
`tests/test_compressor.py` (lines 135-140): without onnxruntime, the
LLMLingua-2 model cannot load and the async path raises. The
sync `select_variant` and the dataclass-shape tests run without
the runtime.
"""

from __future__ import annotations

import importlib.util

import pytest

from apohara_context_forge.compression.compressor import (
    VARIANTS,
    CompressorVariant,
    ContextCompressor,
    _variant_by_name,
    select_variant,
)

# Only the async class needs onnxruntime; the sync table / select_variant
# tests must run regardless (they are pure Python and the unit tests
# of the spec's bin policy).
onnx_spec = importlib.util.find_spec("onnxruntime")
onnx_skipif = pytest.mark.skipif(
    onnx_spec is None,
    reason="onnxruntime not installed — LLMLingua compression requires GPU/ONNX runtime",
)


class TestVariantTable:
    """The frozen VARIANTS tuple must match the spec (Round 16)."""

    def test_variants_are_a_tuple_of_three(self):
        assert isinstance(VARIANTS, tuple)
        assert len(VARIANTS) == 3

    def test_variant_names_match_spec(self):
        names = [v.name for v in VARIANTS]
        assert names == [
            "llmlingua2-base-short",
            "llmlingua2-base-medium",
            "llmlingua2-long",
        ]

    def test_variant_max_words_match_spec(self):
        # Spec: short <=512, medium <=2K, long >2K (we use 10**9 as
        # the long-bin surrogate for the upper-inclusive check).
        assert VARIANTS[0].max_words == 512
        assert VARIANTS[1].max_words == 2048
        assert VARIANTS[2].max_words == 10**9

    def test_only_long_variant_is_longllmlingua(self):
        assert VARIANTS[0].is_longllmlingua is False
        assert VARIANTS[1].is_longllmlingua is False
        assert VARIANTS[2].is_longllmlingua is True

    def test_variants_are_frozen(self):
        # The dataclass is frozen; attempting to mutate must raise.
        v = VARIANTS[0]
        with pytest.raises(Exception):
            v.name = "other"  # type: ignore[misc]

    def test_variant_by_name_resolves_each(self):
        for v in VARIANTS:
            assert _variant_by_name(v.name) is v

    def test_variant_by_name_raises_on_unknown(self):
        with pytest.raises(ValueError, match="Unknown LLMLingua-2 variant"):
            _variant_by_name("llmlingua2-fictional")


class TestSelectVariant:
    """`select_variant` resolves the spec's bin policy."""

    @pytest.mark.parametrize(
        "word_count, expected_name",
        [
            (100, "llmlingua2-base-short"),
            (500, "llmlingua2-base-short"),
            (1000, "llmlingua2-base-medium"),
            (5000, "llmlingua2-long"),
        ],
    )
    def test_select_variant_for_spec_cases(self, word_count, expected_name):
        v = select_variant(word_count)
        assert isinstance(v, CompressorVariant)
        assert v.name == expected_name

    def test_select_variant_boundary_at_512(self):
        # 512 must be short (inclusive upper bound).
        assert select_variant(512).name == "llmlingua2-base-short"

    def test_select_variant_boundary_at_2048(self):
        # 2048 must be medium (inclusive upper bound).
        assert select_variant(2048).name == "llmlingua2-base-medium"

    def test_select_variant_boundary_at_2049(self):
        # 2049 must be long (just past medium).
        assert select_variant(2049).name == "llmlingua2-long"

    def test_select_variant_fallback_for_overflow(self):
        # The long-bin surrogate is 10**9; a value past that falls
        # through to the long variant via the defensive tail return.
        assert select_variant(10**9 + 1).name == "llmlingua2-long"

    def test_select_variant_negative_is_short(self):
        # Negative word counts hit the first bin (short, max_words=512)
        # because `select_variant` uses `<=` and 512 is a positive
        # upper bound. Documented behavior — not a spec requirement,
        # just a property of the function.
        assert select_variant(-1).name == "llmlingua2-base-short"


class TestContextCompressorVariants:
    """Async tests for the new `compress_with_variant` / `auto_compress` methods."""

    pytestmark = onnx_skipif

    @pytest.fixture
    def compressor(self):
        return ContextCompressor()

    @pytest.mark.asyncio
    async def test_compress_with_variant_on_short_text(self, compressor):
        text = " ".join(["compression"] * 100)  # 100 words
        compressed, ratio = await compressor.compress_with_variant(
            text, "llmlingua2-base-short", rate=0.5
        )
        assert isinstance(compressed, str)
        assert len(compressed) > 0
        # The ratio is the original/compressed word count; a real
        # compression will be > 1.0, and the no-op edge case
        # (compressed == original) is also a positive ratio.
        assert ratio > 0

    @pytest.mark.asyncio
    async def test_compress_with_variant_on_long_text(self, compressor):
        # Use a long input to exercise the long variant (also exercises
        # the chunking path in compress_with_variant).
        text = " ".join(["context"] * 3000)  # 3000 words
        compressed, ratio = await compressor.compress_with_variant(
            text, "llmlingua2-long", rate=0.5
        )
        assert isinstance(compressed, str)
        assert ratio > 0

    @pytest.mark.asyncio
    async def test_auto_compress_picks_short_variant(self, compressor):
        text = " ".join(["hello"] * 100)  # 100 words -> short
        compressed, ratio, variant_name = await compressor.auto_compress(text, rate=0.5)
        assert isinstance(compressed, str)
        assert len(compressed) > 0
        assert ratio > 0
        # The variant name must match what select_variant resolves
        # for the same word count.
        assert variant_name == select_variant(len(text.split())).name
        assert variant_name == "llmlingua2-base-short"

    @pytest.mark.asyncio
    async def test_auto_compress_picks_medium_variant(self, compressor):
        text = " ".join(["world"] * 1000)  # 1000 words -> medium
        compressed, ratio, variant_name = await compressor.auto_compress(text, rate=0.5)
        assert variant_name == "llmlingua2-base-medium"
        assert variant_name == select_variant(len(text.split())).name

    @pytest.mark.asyncio
    async def test_auto_compress_picks_long_variant(self, compressor):
        text = " ".join(["foo"] * 5000)  # 5000 words -> long
        compressed, ratio, variant_name = await compressor.auto_compress(text, rate=0.5)
        assert variant_name == "llmlingua2-long"
        assert variant_name == select_variant(len(text.split())).name

    @pytest.mark.asyncio
    async def test_compress_with_variant_unknown_raises(self, compressor):
        with pytest.raises(ValueError, match="Unknown LLMLingua-2 variant"):
            await compressor.compress_with_variant(
                "hello world", "llmlingua2-fictional", rate=0.5
            )
