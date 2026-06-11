"""US-002 retrieval import-surface tests (Phase 0 acceptance).

Verifies the new `apohara_context_forge.retrieval` package:
  - imports cleanly,
  - exposes a constructible `TurbovecStore(dim=768)`,
  - raises NotImplementedError on `search` (the US-002 placeholder contract).

These are wiring tests, not numerics. The real Turbovec implementation
lands in US-004 (Phase 2). The numbers (recall parity, RAM <=4GB) are
benchmarked by `bench_ann.py` once the in-tree Rust crate ships.
"""

from __future__ import annotations

import pytest


def test_retrieval_package_imports():
    """The package must be importable (no missing __init__.py, no syntax errors)."""
    import apohara_context_forge.retrieval  # noqa: F401


def test_turbovec_store_importable():
    """The TurbovecStore class must be importable from the package."""
    from apohara_context_forge.retrieval.turbovec_store import TurbovecStore  # noqa: F401


def test_turbovec_store_constructible_default():
    """TurbovecStore(dim=768) must construct without error."""
    from apohara_context_forge.retrieval.turbovec_store import TurbovecStore

    store = TurbovecStore(dim=768)
    assert store.dim == 768
    assert store.bit_width == 4  # default per the spec


def test_turbovec_store_constructible_explicit_bit_width():
    """TurbovecStore(dim=768, bit_width=8) must construct without error."""
    from apohara_context_forge.retrieval.turbovec_store import TurbovecStore

    store = TurbovecStore(dim=768, bit_width=8)
    assert store.dim == 768
    assert store.bit_width == 8


def test_turbovec_store_search_raises_not_implemented():
    """US-002 placeholder contract: search() must raise NotImplementedError."""
    from apohara_context_forge.retrieval.turbovec_store import TurbovecStore

    store = TurbovecStore(dim=768)
    with pytest.raises(NotImplementedError) as excinfo:
        store.search(query=None, k=10)
    assert "US-002 placeholder" in str(excinfo.value)
    assert "US-004" in str(excinfo.value)


def test_turbovec_store_add_raises_not_implemented():
    """US-002 placeholder contract: add() must raise NotImplementedError."""
    from apohara_context_forge.retrieval.turbovec_store import TurbovecStore

    store = TurbovecStore(dim=768)
    with pytest.raises(NotImplementedError) as excinfo:
        store.add(vectors=None)
    assert "US-002 placeholder" in str(excinfo.value)
    assert "US-004" in str(excinfo.value)
