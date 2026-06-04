"""FastAPI MCP-compatible server exposing ContextForge tools.

The server uses a FastAPI lifespan to construct the heavy components once
(`ContextRegistry`, `ContextCompressor`, `CompressionCoordinator`,
`MetricsCollector`, `VLLMClient`) and stores them on `app.state`. Endpoints
read these via the dependency-getter functions defined below; tests
override the same getters via `app.dependency_overrides` so endpoint logic
runs against fakes without ever entering the lifespan.

Important contracts:
- /health returns the metrics-supplied GPU label, never the request body.
- Endpoints log only metadata (agent_id, lengths) — never the raw context —
  so request payloads cannot leak via stdout/stderr.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from apohara_context_forge.config import settings
from apohara_context_forge.compression.compressor import ContextCompressor
from apohara_context_forge.compression.coordinator import CompressionCoordinator
from apohara_context_forge.metrics.collector import MetricsCollector
from apohara_context_forge.models import (
    CompressionDecision,
    ContextEntry,
    ContextMatch,
    ContextRegistration,
    Degradation,
    MetricsSnapshot,
    OptimizedContextRequest,
)
from apohara_context_forge.registry.context_registry import ContextRegistry
from apohara_context_forge.serving.vllm_client import VLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — constructs heavy components once and tears them down on shutdown.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build app.state.* once; release resources on shutdown.

    Tests bypass the production heavy path either by NOT entering the
    `with TestClient(app) as client:` context (so this lifespan never fires)
    or by monkeypatching the constructor classes referenced by name on this
    module before entering the context.
    """
    app.state.registry = ContextRegistry()
    # Bug fix: ContextRegistry.start() launches the VRAM cache
    # background monitor — without it, the registry never tracks GPU
    # pressure even though endpoints have been answering requests.
    # ``getattr`` keeps the call best-effort so monkeypatched test
    # fakes (which omit start/stop) still pass — matches the pattern
    # used for ``clear`` and ``vllm.aclose`` in the finally block.
    start = getattr(app.state.registry, "start", None)
    if start is not None:
        await start()
    app.state.compressor = ContextCompressor()
    app.state.coordinator = CompressionCoordinator(
        registry=app.state.registry,
        compressor=app.state.compressor,
    )
    app.state.metrics = MetricsCollector()
    app.state.vllm = VLLMClient()
    logger.info(
        "ContextForge started on %s:%s (vLLM %s, model %s)",
        settings.contextforge_host,
        settings.contextforge_port,
        settings.vllm_base_url,
        settings.vllm_model,
    )
    try:
        yield
    finally:
        # Best-effort teardown — never let cleanup errors mask the original
        # request error during shutdown.
        # Bug fix: symmetric stop() for the VRAM monitor started above.
        stop = getattr(app.state.registry, "stop", None)
        if stop is not None:
            try:
                await stop()
            except Exception as exc:
                logger.warning("registry.stop() failed: %s", exc)
        clear = getattr(app.state.registry, "clear", None)
        if clear is not None:
            try:
                await clear()
            except Exception as exc:
                logger.warning("registry.clear() failed: %s", exc)
        aclose = getattr(app.state.vllm, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception as exc:
                logger.warning("vllm.aclose() failed: %s", exc)


app = FastAPI(title="ContextForge", version="0.1.0", lifespan=lifespan)


# Module-level globals kept for callers that import the server outside a
# lifespan-managed TestClient (e.g. ad-hoc REPL probes). Endpoints prefer
# `request.app.state.*` via the dependency getters below.
registry = ContextRegistry()
metrics = MetricsCollector()
compressor: ContextCompressor | None = None
coordinator: CompressionCoordinator | None = None


# ---------------------------------------------------------------------------
# Dependency getters — keys for app.dependency_overrides in tests.
# ---------------------------------------------------------------------------

def get_registry(request: Request) -> ContextRegistry:
    return getattr(request.app.state, "registry", registry)


def get_metrics(request: Request) -> MetricsCollector:
    return getattr(request.app.state, "metrics", metrics)


def get_compressor(request: Request) -> Any:
    return getattr(request.app.state, "compressor", compressor)


def get_coordinator(request: Request) -> Any:
    return getattr(request.app.state, "coordinator", coordinator)


# ---------------------------------------------------------------------------
# /health — never raises. Reports {"status": "ok"|"degraded", "gpu": <label>}.
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check(metrics: MetricsCollector = Depends(get_metrics)) -> dict:
    try:
        label = metrics._resolve_gpu_label()
        return {"status": "ok", "gpu": label}
    except Exception:
        # Anything failing here is a soft-degrade — clients keep polling.
        return {"status": "degraded", "gpu": "unknown"}


# ---------------------------------------------------------------------------
# /tools/register_context
# ---------------------------------------------------------------------------

@app.post("/tools/register_context", response_model=ContextEntry)
async def register_context(
    registration: ContextRegistration,
    registry: ContextRegistry = Depends(get_registry),
    metrics: MetricsCollector = Depends(get_metrics),
) -> ContextEntry:
    """Register an agent's context. Strict body validation: missing field,
    empty agent_id, or extra fields all yield 422 (handled by Pydantic)."""
    # Log metadata only — NEVER the raw context (sentinel-leakage test).
    logger.info(
        "register_context agent_id=%s ctx_len=%d",
        registration.agent_id,
        len(registration.context),
    )
    entry = await registry.register(registration.agent_id, registration.context)
    # The simple register endpoint does not run cross-agent dedup, so we
    # always report `matched=False`. The richer pipeline path uses
    # registry.register_agent and reports its own match telemetry.
    metrics.record_register(False)
    return entry


# ---------------------------------------------------------------------------
# /tools/get_optimized_context
# ---------------------------------------------------------------------------

def _passthrough_decision(context: str) -> CompressionDecision:
    """Build the safe fallback returned with HTTP 503 when the coordinator
    raises. Callers receive a structured payload and can re-issue or fall
    back to the original context themselves."""
    return CompressionDecision(
        strategy="passthrough",
        final_context=context,
        compressed_context=context,
        shared_prefix="",
        original_tokens=0,
        final_tokens=0,
        tokens_saved=0,
        rationale="coordinator_unavailable",
        savings_pct=0.0,
    )


@app.post("/tools/get_optimized_context")
async def get_optimized_context(
    request: OptimizedContextRequest,
    coordinator: Any = Depends(get_coordinator),
    metrics: MetricsCollector = Depends(get_metrics),
):
    """Return a compression decision. On coordinator failure return 503 with
    a passthrough decision body — the client gets a structured response, not
    a 500 stack trace, and metrics.record_decision is NOT called."""
    logger.info(
        "get_optimized_context agent_id=%s ctx_len=%d",
        request.agent_id,
        len(request.context),
    )
    try:
        decision = await coordinator.decide(request.agent_id, request.context)
    except Exception as exc:
        # Don't log the body — only the error class. The sentinel-leakage
        # test asserts no log record contains the original context string.
        logger.warning(
            "coordinator.decide failed for agent_id=%s: %s",
            request.agent_id,
            type(exc).__name__,
        )
        fallback = _passthrough_decision(request.context)
        return JSONResponse(status_code=503, content=fallback.model_dump(mode="json"))

    metrics.record_decision(decision)
    return decision


# ---------------------------------------------------------------------------
# /metrics/snapshot
# ---------------------------------------------------------------------------

@app.get("/metrics/snapshot", response_model=MetricsSnapshot)
async def metrics_snapshot_endpoint(
    metrics: MetricsCollector = Depends(get_metrics),
    compressor: Any = Depends(get_compressor),
) -> MetricsSnapshot:
    """Aggregate snapshot. We pull `current_model` and `degradations` from the
    compressor (which the lifespan owns) and forward them to the collector,
    which doesn't itself know about compressor identity."""
    current_model = getattr(compressor, "current_model", None) or "xlm-roberta-large"
    degradations = list(getattr(compressor, "degradations", []) or [])
    return await metrics.snapshot(
        current_compressor_model=current_model,
        compressor_degradations=degradations,
    )


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> dict:
    return {
        "service": "ContextForge",
        "version": "0.1.0",
        "description": "The shared context compiler for multi-agent LLM systems",
        "docs": "/docs",
    }


# ---------------------------------------------------------------------------
# Background metrics loop — opt-in helper for production runs.
# ---------------------------------------------------------------------------

async def metrics_loop(app_: FastAPI | None = None) -> None:
    """Background metrics logger.

    Bug fix: previously this loop snapshotted the module-level
    ``metrics`` singleton, but every endpoint resolves ``MetricsCollector``
    via ``Depends(get_metrics)`` which returns ``request.app.state.metrics``.
    The two collectors are distinct instances, so the loop was logging an
    empty / never-updated snapshot.

    Pass the ``FastAPI`` app at task creation time so the closure reads
    the same collector the endpoints write to. The legacy zero-arg form
    falls back to the module-level singleton for callers that have not
    yet been updated.
    """
    while True:
        try:
            await asyncio.sleep(30)
            collector = (
                getattr(app_.state, "metrics", metrics) if app_ is not None else metrics
            )
            snap = await collector.snapshot()
            logger.info(
                "Metrics: VRAM=%.1fGB TTFT=%.1fms Dedup=%.1f%%",
                snap.vram_used_gb,
                snap.ttft_ms,
                snap.dedup_rate,
            )
        except Exception as exc:
            logger.error("Metrics collection error: %s", exc)
