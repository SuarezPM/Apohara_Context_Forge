# MERGED: OpenCode (deep KV physics) + CC (surface coverage)
# All tests hermetic: no GPU, no TCP, no downloaded weights required
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

# Optional dep guard — skip entire module if fastapi not installed
fastapi = pytest.importorskip("fastapi", reason="fastapi not installed — install with: pip install fastapi")

from fastapi.testclient import TestClient

from apohara_context_forge.mcp import server as srv
from apohara_context_forge.mcp.server import (
    app,
    get_compressor,
    get_coordinator,
    get_metrics,
    get_registry,
)
from apohara_context_forge.models import (
    CompressionDecision,
    ContextEntry,
    Degradation,
    MetricsSnapshot,
)
from apohara_context_forge.registry.context_registry import ContextRegistry


# ---- Fakes (module-level so dependency_overrides + lifespan patches both work) -----


class FakeMetrics:
    def __init__(self, *, gpu_label: str = "cuda", raise_on_label: bool = False) -> None:
        self._gpu_label = gpu_label
        self._raise_on_label = raise_on_label
        self.register_calls: list[bool] = []
        self.decision_calls: list[CompressionDecision] = []
        self._snapshot_kwargs: dict | None = None

    def _resolve_gpu_label(self) -> str:
        if self._raise_on_label:
            raise RuntimeError("gpu probe blew up")
        return self._gpu_label

    def record_register(self, matched: bool) -> None:
        self.register_calls.append(matched)

    def record_decision(self, decision: CompressionDecision) -> None:
        self.decision_calls.append(decision)

    async def snapshot(
        self, *, current_compressor_model, compressor_degradations
    ) -> MetricsSnapshot:
        self._snapshot_kwargs = {
            "current_compressor_model": current_compressor_model,
            "compressor_degradations": compressor_degradations,
        }
        return MetricsSnapshot(
            vram_source="psutil",
            compressor_model=current_compressor_model,
            vram_used_gb=1.0,
            vram_total_gb=8.0,
            ttft_ms=0.0,
            tokens_processed=0,
            tokens_saved=0,
            dedup_rate=0.0,
            compression_ratio=0.0,
            degradations=list(compressor_degradations),
        )


class FakeCompressor:
    def __init__(
        self,
        current_model: str = "xlm-roberta-large",
        degradations: list[Degradation] | None = None,
    ) -> None:
        self.current_model = current_model
        self.degradations = degradations or []


class FakeRegistry:
    def __init__(self, entry: ContextEntry | None = None) -> None:
        self._entry = entry
        self.register_calls: list[tuple[str, str]] = []
        self.cleared = False

    async def register(self, agent_id: str, context: str) -> ContextEntry:
        self.register_calls.append((agent_id, context))
        if self._entry is not None:
            return self._entry
        now = datetime.now(timezone.utc)
        return ContextEntry(
            agent_id=agent_id,
            context=context,
            token_count=len(context.split()),
            created_at=now,
            expires_at=now + timedelta(seconds=300),
        )

    async def clear(self) -> None:
        self.cleared = True


class FakeCoordinator:
    def __init__(self, decision: CompressionDecision | Exception) -> None:
        self._decision = decision
        self.decide_calls: list[tuple[str, str]] = []

    async def decide(self, agent_id: str, context: str) -> CompressionDecision:
        self.decide_calls.append((agent_id, context))
        if isinstance(self._decision, Exception):
            raise self._decision
        return self._decision


# ---- FakeDedupEngine for the full-flow test (re-uses test_registry pattern) ---------


class FakeDedupEngine:
    def __init__(self) -> None:
        self._key_for_text: dict[str, float] = {}
        self._next_key: float = 1.0

    def _key(self, text: str) -> float:
        if text not in self._key_for_text:
            self._key_for_text[text] = self._next_key
            self._next_key += 1.0
        return self._key_for_text[text]

    async def embed(self, text: str) -> np.ndarray:
        v = np.zeros(8, dtype=np.float32)
        v[0] = self._key(text)
        return v

    async def similarity(self, e1: np.ndarray, e2: np.ndarray) -> float:
        return 1.0 if float(e1[0]) == float(e2[0]) else 0.0

    def find_shared_prefix(self, a: str, b: str) -> str:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return a[:i]

    def count_prefix_tokens(self, prefix: str) -> int:
        return len(prefix.split())


# ---- Helpers ------------------------------------------------------------------------


def _client_with_overrides(overrides: dict) -> TestClient:
    """Build a TestClient that bypasses the production lifespan by injecting
    only the dependency overrides. We do NOT enter the context manager so the
    lifespan never fires (which means no real ContextCompressor / VLLMClient
    construction). Keys must be the dependency function references themselves
    (e.g. ``get_registry``) — FastAPI matches by identity, not by name."""
    for dep, factory in overrides.items():
        app.dependency_overrides[dep] = factory
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


# ---- Tests --------------------------------------------------------------------------


def test_health_returns_ok_with_gpu_label() -> None:
    metrics = FakeMetrics(gpu_label="cuda")
    client = _client_with_overrides({get_metrics: lambda: metrics})
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "gpu": "cuda"}


def test_health_returns_degraded_on_internal_error() -> None:
    metrics = FakeMetrics(raise_on_label=True)
    client = _client_with_overrides({get_metrics: lambda: metrics})
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "degraded", "gpu": "unknown"}


def test_metrics_snapshot_returns_valid_pydantic() -> None:
    metrics = FakeMetrics()
    compressor = FakeCompressor(
        current_model="xlm-roberta-large",
        degradations=[Degradation(component="compressor", reason="OOM", fallback="cpu")],
    )
    client = _client_with_overrides(
        {get_metrics: lambda: metrics, get_compressor: lambda: compressor}
    )
    resp = client.get("/metrics/snapshot")
    assert resp.status_code == 200
    snap = MetricsSnapshot.model_validate(resp.json())
    assert snap.compressor_model == "xlm-roberta-large"
    assert any(d.component == "compressor" for d in snap.degradations)
    assert metrics._snapshot_kwargs is not None
    assert metrics._snapshot_kwargs["current_compressor_model"] == "xlm-roberta-large"


def test_register_context_happy_path() -> None:
    now = datetime.now(timezone.utc)
    stub_entry = ContextEntry(
        agent_id="alice",
        context="hello world",
        token_count=2,
        created_at=now,
        expires_at=now + timedelta(seconds=300),
    )
    registry = FakeRegistry(entry=stub_entry)
    metrics = FakeMetrics()
    client = _client_with_overrides(
        {get_registry: lambda: registry, get_metrics: lambda: metrics}
    )
    resp = client.post(
        "/tools/register_context",
        json={"agent_id": "alice", "context": "hello world"},
    )
    assert resp.status_code == 200
    parsed = ContextEntry.model_validate_json(resp.text)
    assert parsed.agent_id == "alice"
    assert parsed.context == "hello world"
    assert metrics.register_calls == [False]
    assert registry.register_calls == [("alice", "hello world")]


def test_register_context_422_on_empty_agent_id() -> None:
    client = _client_with_overrides(
        {get_registry: lambda: FakeRegistry(), get_metrics: lambda: FakeMetrics()}
    )
    resp = client.post(
        "/tools/register_context",
        json={"agent_id": "", "context": "x"},
    )
    assert resp.status_code == 422


def test_register_context_422_on_extra_field() -> None:
    client = _client_with_overrides(
        {get_registry: lambda: FakeRegistry(), get_metrics: lambda: FakeMetrics()}
    )
    resp = client.post(
        "/tools/register_context",
        json={"agent_id": "a", "context": "x", "hostile": 1},
    )
    assert resp.status_code == 422


def test_register_context_422_on_missing_field() -> None:
    client = _client_with_overrides(
        {get_registry: lambda: FakeRegistry(), get_metrics: lambda: FakeMetrics()}
    )
    resp = client.post("/tools/register_context", json={"agent_id": "a"})
    assert resp.status_code == 422


def test_get_optimized_context_happy_path() -> None:
    decision = CompressionDecision(
        strategy="compress",
        final_context="compressed body",
        shared_prefix="",
        original_tokens=1000,
        final_tokens=500,
        tokens_saved=500,
        rationale="ctx_tokens > threshold",
    )
    coordinator = FakeCoordinator(decision=decision)
    metrics = FakeMetrics()
    client = _client_with_overrides(
        {get_coordinator: lambda: coordinator, get_metrics: lambda: metrics}
    )
    resp = client.post(
        "/tools/get_optimized_context",
        json={"agent_id": "alice", "context": "hello"},
    )
    assert resp.status_code == 200
    parsed = CompressionDecision.model_validate(resp.json())
    assert parsed == decision
    assert len(metrics.decision_calls) == 1
    assert coordinator.decide_calls == [("alice", "hello")]


def test_get_optimized_context_503_fallback_on_handler_exception() -> None:
    coordinator = FakeCoordinator(decision=RuntimeError("boom"))
    metrics = FakeMetrics()
    client = _client_with_overrides(
        {get_coordinator: lambda: coordinator, get_metrics: lambda: metrics}
    )
    resp = client.post(
        "/tools/get_optimized_context",
        json={"agent_id": "alice", "context": "the original body"},
    )
    assert resp.status_code == 503
    parsed = CompressionDecision.model_validate(resp.json())
    assert parsed.strategy == "passthrough"
    assert parsed.final_context == "the original body"
    assert parsed.original_tokens == 0
    assert parsed.final_tokens == 0
    assert parsed.tokens_saved == 0
    assert metrics.decision_calls == []


def test_get_optimized_context_422_on_malformed_body() -> None:
    decision = CompressionDecision(
        strategy="passthrough",
        final_context="",
        shared_prefix="",
        original_tokens=0,
        final_tokens=0,
        tokens_saved=0,
        rationale="",
    )
    client = _client_with_overrides(
        {
            get_coordinator: lambda: FakeCoordinator(decision=decision),
            get_metrics: lambda: FakeMetrics(),
        }
    )
    resp = client.post("/tools/get_optimized_context", json={"agent_id": "a"})
    assert resp.status_code == 422


def test_no_log_includes_request_body(caplog: pytest.LogCaptureFixture) -> None:
    sentinel = "REDACTION-SENTINEL-XYZZY-9F3A2B7C-do-not-log"
    registry = FakeRegistry()
    metrics = FakeMetrics()
    client = _client_with_overrides(
        {get_registry: lambda: registry, get_metrics: lambda: metrics}
    )
    with caplog.at_level(logging.DEBUG):
        # Trigger both happy-path register AND the 503 warning path so any
        # mishandled log surface is exercised.
        client.post(
            "/tools/register_context",
            json={"agent_id": "alice", "context": sentinel},
        )
        # Now exercise the 503 path with the sentinel in the body
        bad_coord = FakeCoordinator(decision=RuntimeError("boom"))
        app.dependency_overrides[get_coordinator] = lambda: bad_coord
        client.post(
            "/tools/get_optimized_context",
            json={"agent_id": "alice", "context": sentinel},
        )
    for record in caplog.records:
        assert sentinel not in record.getMessage()
        for value in record.__dict__.values():
            assert sentinel not in str(value)


def test_lifespan_constructs_and_disposes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Replace the heavy production classes the lifespan reaches for so
    # `with TestClient(app) as client:` does not download model weights or
    # touch the network.
    class _LifeReg:
        instances: list = []

        def __init__(self) -> None:
            self.started = False
            self.stopped = False
            self.cleared = False
            type(self).instances.append(self)

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

        async def clear(self) -> None:
            self.cleared = True

    class _LifeComp:
        def __init__(self) -> None:
            pass

    class _LifeCoord:
        def __init__(self, registry=None, compressor=None) -> None:
            self.registry = registry
            self.compressor = compressor

    class _LifeMetr:
        def __init__(self) -> None:
            pass

    class _LifeVllm:
        instances: list = []

        def __init__(self) -> None:
            self.closed = False
            type(self).instances.append(self)

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(srv, "ContextRegistry", _LifeReg)
    monkeypatch.setattr(srv, "ContextCompressor", _LifeComp)
    monkeypatch.setattr(srv, "CompressionCoordinator", _LifeCoord)
    monkeypatch.setattr(srv, "MetricsCollector", _LifeMetr)
    monkeypatch.setattr(srv, "VLLMClient", _LifeVllm)

    with TestClient(app) as client:
        assert isinstance(client.app.state.registry, _LifeReg)
        assert isinstance(client.app.state.compressor, _LifeComp)
        assert isinstance(client.app.state.coordinator, _LifeCoord)
        assert isinstance(client.app.state.metrics, _LifeMetr)
        assert isinstance(client.app.state.vllm, _LifeVllm)
        # Coordinator must be wired to the SAME registry+compressor instances
        assert client.app.state.coordinator.registry is client.app.state.registry
        assert client.app.state.coordinator.compressor is client.app.state.compressor
        # start() should have run when entering context
        assert client.app.state.registry.started is True

    # On context exit the lifespan ran cleanup
    assert _LifeReg.instances and _LifeReg.instances[-1].stopped is True
    assert _LifeReg.instances and _LifeReg.instances[-1].cleared is True
    assert _LifeVllm.instances and _LifeVllm.instances[-1].closed is True


def test_lifespan_teardown_handles_exceptions(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Test that if the resources fail to tear down, the exceptions are caught
    # and logged, preventing a crash during shutdown.

    class _LifeRegFail:
        def __init__(self) -> None:
            pass

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            raise RuntimeError("stop boom")

        async def clear(self) -> None:
            raise RuntimeError("clear boom")

    class _LifeCompFail:
        def __init__(self) -> None:
            pass

    class _LifeCoordFail:
        def __init__(self, registry=None, compressor=None) -> None:
            self.registry = registry
            self.compressor = compressor

    class _LifeMetrFail:
        def __init__(self) -> None:
            pass

    class _LifeVllmFail:
        def __init__(self) -> None:
            pass

        async def aclose(self) -> None:
            raise RuntimeError("aclose boom")

    monkeypatch.setattr(srv, "ContextRegistry", _LifeRegFail)
    monkeypatch.setattr(srv, "ContextCompressor", _LifeCompFail)
    monkeypatch.setattr(srv, "CompressionCoordinator", _LifeCoordFail)
    monkeypatch.setattr(srv, "MetricsCollector", _LifeMetrFail)
    monkeypatch.setattr(srv, "VLLMClient", _LifeVllmFail)

    with caplog.at_level(logging.WARNING):
        with TestClient(app):
            pass

    logs = caplog.text
    assert "registry.stop() failed: stop boom" in logs
    assert "registry.clear() failed: clear boom" in logs
    assert "vllm.aclose() failed: aclose boom" in logs


def test_full_flow_register_then_optimize_passthrough() -> None:
    # Real ContextRegistry with a hermetic FakeDedupEngine (no model download)
    # plus a stub coordinator that always returns passthrough.
    registry = ContextRegistry(dedup=FakeDedupEngine())
    metrics = FakeMetrics()
    compressor = FakeCompressor()
    short_ctx = "this is a short context"
    passthrough = CompressionDecision(
        strategy="passthrough",
        final_context=short_ctx,
        shared_prefix="",
        original_tokens=5,
        final_tokens=5,
        tokens_saved=0,
        rationale="ctx_tokens <= threshold AND no long shared prefix",
    )
    coordinator = FakeCoordinator(decision=passthrough)
    client = _client_with_overrides(
        {
            get_registry: lambda: registry,
            get_metrics: lambda: metrics,
            get_compressor: lambda: compressor,
            get_coordinator: lambda: coordinator,
        }
    )

    reg_resp = client.post(
        "/tools/register_context",
        json={"agent_id": "alice", "context": short_ctx},
    )
    assert reg_resp.status_code == 200
    reg_entry = ContextEntry.model_validate_json(reg_resp.text)
    assert reg_entry.agent_id == "alice"

    opt_resp = client.post(
        "/tools/get_optimized_context",
        json={"agent_id": "alice", "context": short_ctx},
    )
    assert opt_resp.status_code == 200
    decision = CompressionDecision.model_validate(opt_resp.json())
    assert decision.strategy == "passthrough"

    snap_resp = client.get("/metrics/snapshot")
    assert snap_resp.status_code == 200
    snap = MetricsSnapshot.model_validate(snap_resp.json())
    # passthrough records (0,0) — tokens_processed stays 0; that's fine
    assert snap.tokens_processed == 0
    assert metrics.register_calls == [False]
    assert len(metrics.decision_calls) == 1


# ---- Dependency-getter tests (merge of PRs #11 + #34) -------------------------------

def test_dependency_getters_from_state() -> None:
    class MockApp:
        def __init__(self):
            self.state = type("State", (), {})()

    class MockRequest:
        def __init__(self):
            self.app = MockApp()

    req = MockRequest()
    mock_registry, mock_metrics = object(), object()
    mock_compressor, mock_coordinator = object(), object()
    req.app.state.registry = mock_registry
    req.app.state.metrics = mock_metrics
    req.app.state.compressor = mock_compressor
    req.app.state.coordinator = mock_coordinator

    assert get_registry(req) is mock_registry
    assert get_metrics(req) is mock_metrics
    assert get_compressor(req) is mock_compressor
    assert get_coordinator(req) is mock_coordinator


def test_dependency_getters_fallback() -> None:
    from apohara_context_forge.mcp.server import (
        registry,
        metrics,
        compressor,
        coordinator,
    )

    class MockApp:
        def __init__(self):
            self.state = type("State", (), {})()

    class MockRequest:
        def __init__(self):
            self.app = MockApp()

    req = MockRequest()
    assert get_registry(req) is registry
    assert get_metrics(req) is metrics
    assert get_compressor(req) is compressor
    assert get_coordinator(req) is coordinator
