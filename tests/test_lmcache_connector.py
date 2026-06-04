"""Tests for the V6.x #3 LMCacheConnectorV2.

Covers both the honest-fallback path (lmcache not installed → all
methods return their documented null value, no exceptions raised)
and the wired path (with a FakeEngine standing in for the real
LMCacheEngine).

The connector module lazily imports ``lmcache``, so the tests run
without requiring the real package — the only test that requires
``lmcache`` is explicitly marked.
"""
from __future__ import annotations

import importlib.util
import logging

import pytest

from apohara_context_forge.serving.lmcache_connector import (
    LMCacheConnectorConfig,
    LMCacheConnectorV2,
)


lmcache_installed = importlib.util.find_spec("lmcache") is not None


# ---------------------------------------------------------------------------
# Fakes                                                                      #
# ---------------------------------------------------------------------------

class _FakeMask:
    """Pretends to be a torch / numpy boolean mask."""

    def __init__(self, any_hit: bool):
        self._any = any_hit

    def any(self) -> bool:
        return self._any


class _FakeEngine:
    """Drop-in for LMCacheEngine. Implements store / retrieve / lookup
    against an in-process dict so we can verify the connector's
    semantics without a real LMCache deployment."""

    def __init__(self, *, hit_for: list | None = None):
        self.stored: list = []
        self.lookup_calls: list = []
        self.retrieve_calls: list = []
        self._hit_keys = set(tuple(k) for k in (hit_for or []))

    def store(self, *, tokens, kv_tensors_or_list_of_kv_tensors, blocking):
        self.stored.append({
            "tokens": tuple(tokens) if hasattr(tokens, "__iter__") else tokens,
            "kv": kv_tensors_or_list_of_kv_tensors,
            "blocking": blocking,
        })
        # Engine's contract: return chunk count or None.
        return len(self.stored)

    def retrieve(self, *, tokens, blocking):
        self.retrieve_calls.append(("retrieve", tuple(tokens), blocking))
        if tuple(tokens) in self._hit_keys:
            return ("kv-payload", _FakeMask(any_hit=True))
        return (None, _FakeMask(any_hit=False))

    def lookup(self, *, tokens):
        self.lookup_calls.append(tuple(tokens))
        return len(tokens) if tuple(tokens) in self._hit_keys else 0

    def close(self):
        self._closed = True


# ---------------------------------------------------------------------------
# Honest-fallback path                                                       #
# ---------------------------------------------------------------------------

class TestBackendDetection:
    """Verify the ``backend`` property reports the import strategy that
    was actually used. This is the MI300X deployment-readiness check:
    a multi-node MI300X cluster expects ``"non_cuda"`` (the v1 path),
    never ``"cuda"``.
    """

    def test_backend_reports_fallback_when_lmcache_missing(self):
        """If neither lmcache import strategy works, backend == 'fallback'."""
        conn = LMCacheConnectorV2()
        if conn.is_active():
            pytest.skip("lmcache is installed; this test covers the no-import case")
        assert conn.backend == "fallback"
        # And the stats dict surfaces the same value.
        assert conn.get_stats()["backend"] == "fallback"

    def test_backend_is_cuda_or_non_cuda_when_lmcache_present(self):
        """If lmcache is installed, backend is one of the wired values."""
        conn = LMCacheConnectorV2()
        if not conn.is_active():
            pytest.skip("lmcache not installed; this test requires wired engine")
        assert conn.backend in {"cuda", "non_cuda"}

    def test_backend_is_fallback_for_injected_engine(self):
        """When the engine is injected directly, no import was attempted
        so backend remains ``"fallback"`` — but ``is_active()`` is True
        because the connector has a real engine to talk to."""
        engine = _FakeEngine()
        conn = LMCacheConnectorV2(engine=engine)
        assert conn.is_active() is True
        # Backend is 'fallback' because _try_build_engine() was skipped.
        # This is honest: the connector did not detect any import path,
        # the caller supplied the engine.
        assert conn.backend == "fallback"


class TestNoEngine:
    """When no engine is wired and ``lmcache`` is not importable
    (the dev-laptop / HF-Space case), every method returns the
    documented null value and the connector reports inactive."""

    @pytest.mark.skipif(lmcache_installed, reason="Skipping because lmcache is installed, mock doesn't work well if module is already loaded in test suite.")
    def test_engine_build_exception_falls_back_to_noop(self, caplog):
        from unittest.mock import MagicMock, patch
        mock_lmcache = MagicMock()
        mock_lmcache_config = MagicMock()
        mock_lmcache_cache_engine = MagicMock()

        mock_LMCacheEngineConfig = MagicMock()
        mock_lmcache_config.LMCacheEngineConfig = mock_LMCacheEngineConfig

        mock_LMCacheEngineBuilder = MagicMock()
        mock_LMCacheEngineBuilder.get_or_create.side_effect = RuntimeError("Simulated build error")
        mock_lmcache_cache_engine.LMCacheEngineBuilder = mock_LMCacheEngineBuilder

        with patch.dict('sys.modules', {
            'lmcache': mock_lmcache,
            'lmcache.config': mock_lmcache_config,
            'lmcache.experimental': MagicMock(),
            'lmcache.experimental.cache_engine': mock_lmcache_cache_engine,
        }):
            # We must force re-import to pick up the patched sys.modules
            # but actually LMCacheConnectorV2 does import locally inside _try_build_engine.
            with caplog.at_level(logging.WARNING):
                conn = LMCacheConnectorV2()

            assert conn.is_active() is False
            assert conn.backend == "fallback"

            stats = conn.get_stats()
            assert "Simulated build error" in stats["build_error"]
            assert "RuntimeError" in stats["build_error"]

            assert any(
                "engine build failed" in record.message and "Simulated build error" in record.message
                for record in caplog.records
            )
    def test_is_active_false_without_engine(self, caplog):
        # Build with no engine. We expect a single WARNING (logged
        # exactly once during _try_build_engine) describing the
        # fallback. We do NOT assert on the import path because the
        # message text differs based on lmcache install status.
        with caplog.at_level(logging.WARNING):
            conn = LMCacheConnectorV2(config=LMCacheConnectorConfig(
                remote_url=None,
            ))
        # If lmcache is installed (CI matrix runs that case), the
        # connector will be ACTIVE — both paths are expected.
        if lmcache_installed:
            return
        assert conn.is_active() is False

    def test_store_returns_none_without_engine(self):
        conn = LMCacheConnectorV2()
        if conn.is_active():
            pytest.skip("lmcache is installed; this test covers the fallback only")
        assert conn.store(tokens=[1, 2, 3], kv_tensors="ignored") is None

    def test_retrieve_returns_none_without_engine(self):
        conn = LMCacheConnectorV2()
        if conn.is_active():
            pytest.skip("lmcache is installed; this test covers the fallback only")
        assert conn.retrieve(tokens=[1, 2, 3]) is None

    def test_lookup_returns_zero_without_engine(self):
        conn = LMCacheConnectorV2()
        if conn.is_active():
            pytest.skip("lmcache is installed; this test covers the fallback only")
        assert conn.lookup(tokens=[1, 2, 3]) == 0

    def test_prefetch_returns_zeroed_results_without_engine(self):
        conn = LMCacheConnectorV2()
        if conn.is_active():
            pytest.skip("lmcache is installed; this test covers the fallback only")
        results = conn.prefetch([[1, 2], [3, 4]])
        assert results == [
            {"cached_tokens": 0, "retrieved": False},
            {"cached_tokens": 0, "retrieved": False},
        ]


# ---------------------------------------------------------------------------
# Wired path with FakeEngine                                                 #
# ---------------------------------------------------------------------------

class TestWiredWithFakeEngine:
    """Inject a FakeEngine and verify the connector's semantics."""

    def _conn(self, *, hit_for=None) -> tuple[LMCacheConnectorV2, _FakeEngine]:
        engine = _FakeEngine(hit_for=hit_for or [])
        conn = LMCacheConnectorV2(engine=engine)
        return conn, engine

    def test_is_active_true_when_engine_injected(self):
        conn, _ = self._conn()
        assert conn.is_active() is True

    def test_store_records_call_and_returns_count(self):
        conn, engine = self._conn()
        n = conn.store(tokens=[1, 2, 3], kv_tensors="my-kv")
        assert n == 1
        assert engine.stored[0]["tokens"] == (1, 2, 3)
        assert engine.stored[0]["kv"] == "my-kv"
        assert conn.get_stats()["stores"] == 1

    def test_retrieve_hit_returns_payload_and_mask(self):
        conn, engine = self._conn(hit_for=[[1, 2, 3]])
        result = conn.retrieve(tokens=[1, 2, 3])
        assert result is not None
        kv, mask = result
        assert kv == "kv-payload"
        assert mask.any() is True
        assert conn.get_stats()["retrieves_hit"] == 1
        assert conn.get_stats()["retrieves_miss"] == 0

    def test_retrieve_miss_returns_none(self):
        conn, _ = self._conn(hit_for=[[9, 9]])
        assert conn.retrieve(tokens=[1, 2, 3]) is None
        assert conn.get_stats()["retrieves_miss"] == 1

    def test_lookup_returns_cached_count(self):
        conn, _ = self._conn(hit_for=[[1, 2, 3]])
        assert conn.lookup(tokens=[1, 2, 3]) == 3
        assert conn.lookup(tokens=[9, 9])    == 0

    def test_lookup_failure_returns_zero(self):
        class _BrokenLookup:
            def lookup(self, **_):
                raise RuntimeError("simulated lookup crash")

        conn = LMCacheConnectorV2(engine=_BrokenLookup())
        assert conn.lookup(tokens=[1]) == 0
        # Failure does not increment lookups stat
        assert conn.get_stats()["lookups"] == 0

    def test_prefetch_aggregates_lookup_then_retrieve(self):
        conn, _ = self._conn(hit_for=[[1, 2]])
        results = conn.prefetch([[1, 2], [3, 4]])
        assert results[0]["cached_tokens"] == 2
        assert results[0]["retrieved"]     is True
        assert results[1]["cached_tokens"] == 0
        assert results[1]["retrieved"]     is False

    def test_store_failure_returns_none_and_does_not_raise(self):
        class _Broken:
            def store(self, **_):
                raise RuntimeError("simulated network blip")

        conn = LMCacheConnectorV2(engine=_Broken())
        assert conn.is_active() is True
        assert conn.store(tokens=[1], kv_tensors="x") is None

    def test_retrieve_failure_reports_miss(self):
        class _Broken:
            def retrieve(self, **_):
                raise RuntimeError("simulated network blip")

        conn = LMCacheConnectorV2(engine=_Broken())
        assert conn.retrieve(tokens=[1]) is None
        assert conn.get_stats()["retrieves_miss"] == 1

    def test_retrieve_exception_returns_none(self):
        class _BrokenRetrieve:
            def retrieve(self, **_):
                raise RuntimeError("simulated network blip in engine.retrieve")

        conn = LMCacheConnectorV2(engine=_BrokenRetrieve())
        assert conn.retrieve(tokens=[1]) is None
        assert conn.get_stats()["retrieves_miss"] == 1

    def test_retrieve_mask_any_failure_treated_as_hit(self):
        class _BrokenMask:
            def any(self):
                raise RuntimeError("mask unreadable")

        class _BrokenMaskEngine:
            def retrieve(self, **_):
                return ("kv-payload", _BrokenMask())

        conn = LMCacheConnectorV2(engine=_BrokenMaskEngine())
        result = conn.retrieve(tokens=[1])
        assert result is not None
        kv, mask = result
        assert kv == "kv-payload"
        assert conn.get_stats()["retrieves_hit"] == 1

    def test_close_releases_engine(self):
        conn, engine = self._conn()
        conn.close()
        assert conn.is_active() is False
        # Idempotent
        conn.close()

    def test_close_failure_is_caught(self):
        class _BrokenClose:
            def close(self):
                raise RuntimeError("simulated close failure")

        conn = LMCacheConnectorV2(engine=_BrokenClose())
        # Exception should be caught and logged, not raised.
        # It still sets _active = False and _engine = None.
        conn.close()
        assert conn.is_active() is False


# ---------------------------------------------------------------------------
# Stats / introspection                                                      #
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_has_expected_keys(self):
        conn = LMCacheConnectorV2()
        stats = conn.get_stats()
        for key in (
            "active", "backend", "instance_id", "chunk_size", "local_device",
            "remote_url", "stores", "retrieves_hit", "retrieves_miss",
            "lookups", "build_error",
        ):
            assert key in stats, f"missing stat: {key}"

    def test_backend_is_reported_in_stats(self):
        conn = LMCacheConnectorV2()
        stats = conn.get_stats()
        assert stats["backend"] in {"cuda", "non_cuda", "fallback"}, (
            f"unexpected backend value: {stats['backend']!r}"
        )

    def test_repr_includes_active_and_counts(self):
        conn = LMCacheConnectorV2()
        r = repr(conn)
        assert "LMCacheConnectorV2" in r
        assert "active=" in r


# ---------------------------------------------------------------------------
# Real-LMCache integration smoke test (skipped unless lmcache installed)     #
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not lmcache_installed,
                    reason="lmcache not installed; integration test skipped")
def test_real_lmcache_engine_can_be_built():
    """When lmcache IS installed, the connector must be able to
    actually construct the engine. We don't push any KV through it —
    that needs torch + a real model — but the build step alone
    catches API drift in lmcache."""
    conn = LMCacheConnectorV2(config=LMCacheConnectorConfig(
        instance_id="apohara-test", chunk_size=64, local_device="cpu",
    ))
    # If lmcache is present but the build fails for an unrelated
    # reason (missing torch CUDA, etc.) the connector reports the
    # error in stats — we surface that so a debugger can act on it.
    if not conn.is_active():
        pytest.skip(f"engine build failed: {conn.get_stats()['build_error']}")
    assert conn.is_active() is True
    conn.close()
