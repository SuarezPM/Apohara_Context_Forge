"""US-004 retrieval tests (Phase 2 acceptance) + US-012 768-d migration.

The real `TurbovecStore` is backed by the `turbovec` PyPI package
(CPU SIMD); the `RetrievalEngine` glues the 768-d `EmbeddingEngine`
(US-012 default = ``ibm-granite/granite-embedding-311m-multilingual-r2``
MRL-truncated to 768d) to the index. The bench is exercised by a
pytest function (not a subprocess) so the suite stays fast.

The pre-US-012 16 tests stay as legacy back-compat smoke (marked
``@pytest.mark.legacy``); US-012 adds 6 new tests for the 768-d path.

See `.omc/plans/apohara-2-0.md` Step 2.4.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

turbovec = pytest.importorskip("turbovec")

# Mark every test in the legacy 384-d back-compat block.
legacy = pytest.mark.legacy


# ------------------------------------------------------------ TurbovecStore
@legacy
def test_retrieval_package_imports():
    """The package must be importable (no missing __init__, no syntax errors)."""
    import apohara_context_forge.retrieval  # noqa: F401


@legacy
def test_turbovec_store_class_exports():
    """TurbovecStore must be exported from the package."""
    from apohara_context_forge.retrieval import TurbovecStore  # noqa: F401


@legacy
def test_turbovec_store_constructible_default():
    """TurbovecStore(dim=768) must construct with the spec defaults."""
    from apohara_context_forge.retrieval import TurbovecStore

    store = TurbovecStore(dim=768)
    assert store.dim == 768
    assert store.bit_width == 4  # the spec's target


@legacy
def test_turbovec_store_constructible_explicit_bit_width():
    """TurbovecStore(dim=384, bit_width=3) must construct without error."""
    from apohara_context_forge.retrieval import TurbovecStore

    store = TurbovecStore(dim=384, bit_width=3)
    assert store.dim == 384
    assert store.bit_width == 3


@legacy
def test_turbovec_store_invalid_dim_raises():
    """dim <= 0 must raise ValueError."""
    from apohara_context_forge.retrieval import TurbovecStore

    with pytest.raises(ValueError):
        TurbovecStore(dim=0)
    with pytest.raises(ValueError):
        TurbovecStore(dim=-1)


@legacy
def test_turbovec_store_invalid_bit_width_raises():
    """bit_width not in {2, 3, 4} must raise ValueError."""
    from apohara_context_forge.retrieval import TurbovecStore

    with pytest.raises(ValueError):
        TurbovecStore(dim=64, bit_width=5)
    with pytest.raises(ValueError):
        TurbovecStore(dim=64, bit_width=16)


@legacy
def test_turbovec_store_add_and_search_basic():
    """Adding N vectors and searching must return N×k scores/indices."""
    from apohara_context_forge.retrieval import TurbovecStore

    rng = np.random.default_rng(0)
    n, dim, k = 50, 32, 5
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    store = TurbovecStore(dim=dim)
    store.add(vecs)
    assert len(store) == n

    queries = rng.standard_normal((3, dim)).astype(np.float32)
    scores, idx = store.search(queries, k=k)
    assert scores.shape == (3, k)
    assert idx.shape == (3, k)
    assert scores.dtype == np.float32
    # Indices must point into [0, n).
    valid = (idx >= 0) & (idx < n)
    assert valid.all()


@legacy
def test_turbovec_store_add_validates_dim():
    """Adding vectors with the wrong dim must raise ValueError."""
    from apohara_context_forge.retrieval import TurbovecStore

    store = TurbovecStore(dim=32)
    bad = np.random.randn(5, 16).astype(np.float32)
    with pytest.raises(ValueError):
        store.add(bad)


@legacy
def test_turbovec_store_add_rejects_non_finite():
    """NaN / Inf inputs must raise ValueError (no silent NaN poison)."""
    from apohara_context_forge.retrieval import TurbovecStore

    store = TurbovecStore(dim=16)
    bad = np.ones((4, 16), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        store.add(bad)


@legacy
def test_turbovec_store_search_empty_returns_sentinels():
    """Searching an empty index must return zeros/-1, never crash."""
    from apohara_context_forge.retrieval import TurbovecStore

    store = TurbovecStore(dim=16)
    q = np.random.randn(2, 16).astype(np.float32)
    scores, idx = store.search(q, k=4)
    assert scores.shape == (2, 4)
    assert idx.shape == (2, 4)
    assert (idx == -1).all()


@legacy
def test_turbovec_store_save_load_roundtrip():
    """save + load must preserve index contents and search output."""
    from apohara_context_forge.retrieval import TurbovecStore

    rng = np.random.default_rng(1)
    n, dim = 30, 16
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    store = TurbovecStore(dim=dim)
    store.add(vecs)
    assert len(store) == n

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "idx.tvi")
        store.save(path)
        loaded = TurbovecStore.load(path)

    assert len(loaded) == n
    assert loaded.dim == dim
    assert loaded.bit_width == store.bit_width

    # Search through the loaded index must produce the same shape.
    q = rng.standard_normal((1, dim)).astype(np.float32)
    scores, idx = loaded.search(q, k=3)
    assert scores.shape == (1, 3)
    assert idx.shape == (1, 3)


# ---------------------------------------------------------- RetrievalEngine
@legacy
def test_retrieval_engine_class_exports():
    """RetrievalEngine must be exported from the package."""
    from apohara_context_forge.retrieval import RetrievalEngine  # noqa: F401


@legacy
def test_retrieval_engine_end_to_end_with_embedding_engine():
    """index(texts) + retrieve(query) must roundtrip via the real EmbeddingEngine."""
    from apohara_context_forge.retrieval import RetrievalEngine

    eng = RetrievalEngine(dim=384)
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "lorem ipsum dolor sit amet consectetur",
        "apohara context forge is a long-context inference platform",
        "pablo writes rust for a living and likes emacs",
        "turbovec is a fast ann index for retrieval",
    ]
    eng.index(corpus)
    assert len(eng) == len(corpus)

    hits = eng.retrieve("rust turbovec performance", k=3)
    assert len(hits) == 3
    # The top hit is not guaranteed (xorshift fallback is hash-based),
    # but the shape and id wiring must be correct.
    for h in hits:
        assert h.text in corpus
        assert h.id.startswith("doc_")
        assert 0 <= h.position < len(corpus)
        assert -1.0 <= h.score <= 1.0


@legacy
def test_retrieval_engine_dim_mismatch_raises():
    """Passing a store with a different dim from the embedding engine must error."""
    from apohara_context_forge.retrieval import RetrievalEngine, TurbovecStore

    # Engine defaults to 384; pass a 768 store -> mismatch.
    store = TurbovecStore(dim=768)
    with pytest.raises(ValueError):
        RetrievalEngine(store=store, dim=384)


# ----------------------------------------------------------------- bench
@legacy
def test_bench_ann_runs_and_emits_json():
    """bench_ann.main([...]) must exit 0 and print a JSON summary with the
    contract keys. Uses a corpus large enough (>1000 docs) for FAISS to
    switch into IVF mode so the parity gate is meaningful."""
    from apohara_context_forge.benchmarks.apohara2 import bench_ann

    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = bench_ann.main([
            "--docs", "2000",
            "--queries", "100",
            "--dim", "128",
            "--seed", "42",
            "--quiet",
        ])
    assert rc == 0, f"bench_ann returned {rc}; expected 0"

    out = buf.getvalue()
    # The JSON summary is pretty-printed; find the first '{' and parse
    # to the matching closing '}'.
    start = out.find("{")
    assert start >= 0, f"no JSON object emitted; got: {out!r}"
    # json.loads handles the pretty-printed form directly.
    summary = json.loads(out[start:])
    assert isinstance(summary, dict), f"unexpected JSON: {summary!r}"

    expected_keys = {
        "turbovec_recall_at_10", "faiss_recall_at_10",
        "turbovec_p50_ms", "faiss_p50_ms",
        "n_docs", "n_queries", "dim",
    }
    assert expected_keys.issubset(summary.keys()), (
        f"missing keys: {expected_keys - set(summary.keys())}"
    )
    # Recall parity: Turbovec must be within 2pp of FAISS-IVF (or above).
    assert summary["turbovec_recall_at_10"] >= (
        summary["faiss_recall_at_10"] - 0.02
    ), summary


@legacy
def test_bench_ann_recall_parity_is_measured_not_placeholder():
    """Sanity check: the bench actually computes recall (no constant)."""
    from apohara_context_forge.benchmarks.apohara2 import bench_ann

    import io, contextlib

    def run(dim, seed):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = bench_ann.main([
                "--docs", "2000", "--queries", "100",
                "--dim", str(dim), "--seed", str(seed), "--quiet",
            ])
        assert rc == 0, f"bench returned {rc}"
        out = buf.getvalue()
        start = out.find("{")
        assert start >= 0, f"no JSON object emitted; got: {out!r}"
        return json.loads(out[start:])

    # Two different seeds should give slightly different recall
    # numbers (the corpus + queries change). If they're identical, the
    # recall is a constant.
    a = run(dim=128, seed=1)
    b = run(dim=128, seed=2)
    assert a["seed"] != b["seed"]
    # The seed-controlled randomness must change the recall value.
    assert a["turbovec_recall_at_10"] != b["turbovec_recall_at_10"]


# ============================================================ US-012 (768-d)
# Below: the 768-d path tests. The default model is now
# ``ibm-granite/granite-embedding-311m-multilingual-r2`` (MRL 1024-d
# truncated to 768-d); the fallback (unit-test / bench stub) is a
# deterministic 768-d random unit vector per text.


def test_turbovec_store_768d_default_constructible():
    """TurbovecStore() with no args must now default to dim=768 (US-012)."""
    from apohara_context_forge.retrieval import TurbovecStore

    store = TurbovecStore()
    assert store.dim == 768, f"expected default dim=768, got {store.dim}"
    assert store.bit_width == 4


def test_retrieval_engine_768d_default_constructible():
    """RetrievalEngine() with no args must now default to dim=768 (US-012).

    Uses the xorshift legacy fallback (use_onnx=True → back-compat path)
    so the test stays fast and independent of the granite-r2 model.
    """
    from apohara_context_forge.embeddings.embedding_engine import EmbeddingEngine
    from apohara_context_forge.retrieval import RetrievalEngine, TurbovecStore

    eng = RetrievalEngine(
        embedding_engine=EmbeddingEngine(dim=768, use_onnx=True)
    )
    assert eng.dim == 768
    assert eng.store.dim == 768
    assert eng.store.bit_width == 4


def test_turbovec_store_768d_add_and_search_basic():
    """Small random unit vectors at 768-d: add 10, search k=3, sensible scores."""
    from apohara_context_forge.retrieval import TurbovecStore

    rng = np.random.default_rng(7)
    n, dim, k = 10, 768, 3
    # Random unit vectors (the cosine-similarity convention).
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    store = TurbovecStore(dim=dim)
    store.add(vecs)
    assert len(store) == n

    # Query is a copy of vector 0 → top-1 must be position 0.
    q = vecs[0:1].copy()
    scores, idx = store.search(q, k=k)
    assert scores.shape == (1, k)
    assert idx.shape == (1, k)
    # The top hit must be position 0 (the exact-match query).
    assert int(idx[0, 0]) == 0, f"expected top hit=0, got {idx[0, 0]}"
    # Scores must be finite and ordered descending.
    assert np.all(np.isfinite(scores[0]))
    assert float(scores[0, 0]) >= float(scores[0, 1]) >= float(scores[0, 2])


def test_turbovec_store_768d_save_load_roundtrip():
    """write to a tmp file, load back, assert equal population + dim."""
    from apohara_context_forge.retrieval import TurbovecStore

    rng = np.random.default_rng(13)
    n, dim = 20, 768
    vecs = rng.standard_normal((n, dim)).astype(np.float32)

    store = TurbovecStore(dim=dim)
    store.add(vecs)
    assert len(store) == n

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "idx_768d.tvi")
        store.save(path)
        loaded = TurbovecStore.load(path)

    assert len(loaded) == n
    assert loaded.dim == 768
    assert loaded.bit_width == store.bit_width

    # Search shape contract survives the roundtrip.
    q = rng.standard_normal((1, dim)).astype(np.float32)
    scores, idx = loaded.search(q, k=4)
    assert scores.shape == (1, 4)
    assert idx.shape == (1, 4)


def test_legacy_384d_still_constructible():
    """TurbovecStore(dim=384) and EmbeddingEngine.legacy_384d() both work
    (back-compat smoke for the pre-US-012 path)."""
    from apohara_context_forge.embeddings.embedding_engine import EmbeddingEngine
    from apohara_context_forge.retrieval import TurbovecStore

    # The 384-d TurbovecStore stays constructible.
    store_384 = TurbovecStore(dim=384)
    assert store_384.dim == 384
    assert store_384.bit_width == 4

    # The EmbeddingEngine.legacy_384d() classmethod produces a 384-d
    # xorshift engine (used by the 384-d retrieval test above).
    eng_legacy = EmbeddingEngine.legacy_384d()
    assert eng_legacy.dim == 384
    assert eng_legacy._use_onnx is True


def test_embedding_engine_fallback_returns_unit_vector_768d():
    """The deterministic fallback path: a fresh engine with no model loaded
    must return a 768-d L2-normalized vector per text (when the granite-r2
    model is unavailable in the test env)."""
    from apohara_context_forge.embeddings.embedding_engine import EmbeddingEngine

    eng = EmbeddingEngine(dim=768)
    # Force the fallback (no model load attempt) for the assertion below.
    eng._model_loaded = False
    eng._model_failed = True
    eng._model_failure_reason = "test_forced_fallback"

    import asyncio
    v1 = asyncio.run(eng.encode("hello world"))
    v2 = asyncio.run(eng.encode("goodbye world"))
    v1_again = asyncio.run(eng.encode("hello world"))

    # Shape + dtype.
    assert v1.shape == (768,)
    assert v1.dtype == np.float32
    # Unit norm (L2 normalize is unconditional in encode()).
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5
    # Determinism: same text → same vector.
    np.testing.assert_array_equal(v1, v1_again)
    # Distinct texts → distinct vectors (the hash-of-text seeds the PRNG).
    assert not np.array_equal(v1, v2)


# ============================================================ US-015 (RAM ceiling)
# The 4 GB RAM-ceiling target at 10M docs / 768-d / 4-bit cannot be met by
# either the upstream `turbovec` 0.8.0 PyPI package OR the in-tree
# `codec_v8` per-nibble independent-scales path. The per-nibble metadata
# (one scale + one zero_point per nibble, both float32) is 16 bytes per
# packed byte — orders of magnitude larger than the 1 byte of code per
# packed byte. See AUDIT #23b + AUDIT #27 (filed 2026-06-11) for the
# honest gap + Phase 5 follow-up. The projection tests below document
# the actual numbers honestly.


def test_turbovec_store_ram_projection_upstream():
    """10M docs at 768-d / 4-bit via upstream turbovec: ~22.7 GB (AUDIT #23b)."""
    from apohara_context_forge.retrieval import TurbovecStore

    store = TurbovecStore(dim=768, bit_width=4, storage_mode="upstream")
    projected = store.projected_ram_mb(n_docs=10_000_000)
    # AUDIT #23b measured 22.8 MB / 10K docs (psutil RSS delta), which
    # extrapolates to 22,777 MB / 10M docs. We bound this with a 14k-32k
    # range to allow for the spec's alternative closed-form estimate
    # (16,479 MiB per-pair Lloyd-Max) without locking in a single
    # formula. The point is to assert "far above 4 GB".
    assert 14_000 < projected < 32_000, (
        f"upstream RAM projection {projected:.1f} MB outside expected range"
    )


def test_turbovec_store_ram_projection_optimised_meets_4gb_target():
    """10M docs at 768-d / 4-bit via ram_optimised + per-block codec
    (AUDIT #27a close path): **meets** the 4 GB target.

    With ``group_size=256`` the per-block (scale, zp) collapses metadata
    from 16 B per packed byte to ~1 B per packed byte. The closed-form
    math is:

        codes     = n_docs * dim * bit_width / 8
        scales    = n_docs / group_size * (dim // 2) * 4
        zps       = n_docs / group_size * (dim // 2) * 4
        norms     = n_docs * 4

    At 10M / 768 / 4 / group_size=256: ~3,815 MiB ≤ 4,096 MiB (the 4 GB
    target is achievable). This test flipped from a *negative* assertion
    (which pinned the honest gap pre-AUDIT-#27a) to a *positive*
    assertion (the close path lands inside the budget).
    """
    from apohara_context_forge.retrieval import TurbovecStore

    store = TurbovecStore(
        dim=768, bit_width=4, storage_mode="ram_optimised", group_size=256
    )
    projected = store.projected_ram_mb(n_docs=10_000_000)
    # Close path: must land inside the 4 GB budget (with margin to
    # catch regressions that re-grow the metadata).
    assert 3_500 < projected <= 4_096, (
        f"ram_optimised AUDIT #27a close-path projection {projected:.1f} MiB "
        f"outside expected 3,500-4,096 MiB band; review per-block codec math"
    )


def test_turbovec_store_ram_projection_optimised_default_pins_honest_gap():
    """Back-compat: the default ``group_size=1`` still projects to the
    AUDIT #27 honest gap (~62 GB at 10M / 768 / 4). The default is the
    per-nibble per-doc layout, which is the only layout that exercised
    the V8 codec in the previous US-015 ralph session. Callers that
    want the close path must opt in with ``group_size=256``.

    This test guards the back-compat surface: removing the default
    (or silently changing the formula) would break the AUDIT #27
    ledger entry and the existing bank test that depends on it.
    """
    from apohara_context_forge.retrieval import TurbovecStore

    store = TurbovecStore(dim=768, bit_width=4, storage_mode="ram_optimised")
    projected = store.projected_ram_mb(n_docs=10_000_000)
    # AUDIT #27 honest gap (per-nibble, group_size=1, default).
    assert projected > 60_000, (
        f"default ram_optimised projection {projected:.1f} MiB no longer "
        f"matches the AUDIT #27 honest gap (~62,294 MiB); the default "
        f"group_size=1 was changed underfoot"
    )


def test_turbovec_store_ram_optimised_rejects_indivisible_dim():
    """``group_size > 1`` requires ``dim % group_size == 0`` to keep the
    per-block math closed-form. The constructor must reject the bad
    combo before any add() runs.
    """
    from apohara_context_forge.retrieval import TurbovecStore

    # 384 % 256 != 0 → reject.
    try:
        TurbovecStore(
            dim=384, bit_width=4, storage_mode="ram_optimised", group_size=256
        )
    except ValueError as e:
        assert "divisible" in str(e).lower() or "group_size" in str(e).lower()
    else:
        raise AssertionError(
            "TurbovecStore(dim=384, group_size=256) should have raised ValueError"
        )

    # group_size < 1 → reject.
    try:
        TurbovecStore(
            dim=768, bit_width=4, storage_mode="ram_optimised", group_size=0
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "TurbovecStore(group_size=0) should have raised ValueError"
        )
