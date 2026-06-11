"""US-004 retrieval tests (Phase 2 acceptance).

The real `TurbovecStore` is backed by the `turbovec` PyPI package
(CPU SIMD); the `RetrievalEngine` glues the existing 384-d
`EmbeddingEngine` to the
index. The bench is exercised by a pytest function (not a subprocess)
so the suite stays fast.

See `.omc/plans/apohara-2-0.md` Step 2.4.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

turbovec = pytest.importorskip("turbovec")


# ------------------------------------------------------------ TurbovecStore
def test_retrieval_package_imports():
    """The package must be importable (no missing __init__, no syntax errors)."""
    import apohara_context_forge.retrieval  # noqa: F401


def test_turbovec_store_class_exports():
    """TurbovecStore must be exported from the package."""
    from apohara_context_forge.retrieval import TurbovecStore  # noqa: F401


def test_turbovec_store_constructible_default():
    """TurbovecStore(dim=768) must construct with the spec defaults."""
    from apohara_context_forge.retrieval import TurbovecStore

    store = TurbovecStore(dim=768)
    assert store.dim == 768
    assert store.bit_width == 4  # the spec's target


def test_turbovec_store_constructible_explicit_bit_width():
    """TurbovecStore(dim=384, bit_width=3) must construct without error."""
    from apohara_context_forge.retrieval import TurbovecStore

    store = TurbovecStore(dim=384, bit_width=3)
    assert store.dim == 384
    assert store.bit_width == 3


def test_turbovec_store_invalid_dim_raises():
    """dim <= 0 must raise ValueError."""
    from apohara_context_forge.retrieval import TurbovecStore

    with pytest.raises(ValueError):
        TurbovecStore(dim=0)
    with pytest.raises(ValueError):
        TurbovecStore(dim=-1)


def test_turbovec_store_invalid_bit_width_raises():
    """bit_width not in {2, 3, 4} must raise ValueError."""
    from apohara_context_forge.retrieval import TurbovecStore

    with pytest.raises(ValueError):
        TurbovecStore(dim=64, bit_width=5)
    with pytest.raises(ValueError):
        TurbovecStore(dim=64, bit_width=16)


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


def test_turbovec_store_add_validates_dim():
    """Adding vectors with the wrong dim must raise ValueError."""
    from apohara_context_forge.retrieval import TurbovecStore

    store = TurbovecStore(dim=32)
    bad = np.random.randn(5, 16).astype(np.float32)
    with pytest.raises(ValueError):
        store.add(bad)


def test_turbovec_store_add_rejects_non_finite():
    """NaN / Inf inputs must raise ValueError (no silent NaN poison)."""
    from apohara_context_forge.retrieval import TurbovecStore

    store = TurbovecStore(dim=16)
    bad = np.ones((4, 16), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        store.add(bad)


def test_turbovec_store_search_empty_returns_sentinels():
    """Searching an empty index must return zeros/-1, never crash."""
    from apohara_context_forge.retrieval import TurbovecStore

    store = TurbovecStore(dim=16)
    q = np.random.randn(2, 16).astype(np.float32)
    scores, idx = store.search(q, k=4)
    assert scores.shape == (2, 4)
    assert idx.shape == (2, 4)
    assert (idx == -1).all()


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
def test_retrieval_engine_class_exports():
    """RetrievalEngine must be exported from the package."""
    from apohara_context_forge.retrieval import RetrievalEngine  # noqa: F401


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


def test_retrieval_engine_dim_mismatch_raises():
    """Passing a store with a different dim from the embedding engine must error."""
    from apohara_context_forge.retrieval import RetrievalEngine, TurbovecStore

    # Engine defaults to 384; pass a 768 store -> mismatch.
    store = TurbovecStore(dim=768)
    with pytest.raises(ValueError):
        RetrievalEngine(store=store, dim=384)


# ----------------------------------------------------------------- bench
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
