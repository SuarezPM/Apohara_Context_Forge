from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

# Detect whether opentelemetry is installed so tests can be conditionally skipped.
try:
    import opentelemetry  # noqa: F401
    _has_opentelemetry = True
except ImportError:
    _has_opentelemetry = False

from apohara_context_forge.observability.otlp_exporter import OTLPExporter
import apohara_context_forge.observability.recorders as recorders_mod


# ---------------------------------------------------------------------------
# Test 1: honest fallback when opentelemetry is not installed
# ---------------------------------------------------------------------------

def test_otlp_honest_fallback_when_missing(caplog):
    """OTLPExporter.start() returns None and logs a WARNING when opentelemetry absent."""
    import sys

    # Temporarily remove opentelemetry from sys.modules to simulate absence.
    otel_keys = [k for k in sys.modules if k.startswith("opentelemetry")]
    saved = {k: sys.modules.pop(k) for k in otel_keys}

    exporter = OTLPExporter(endpoint="localhost:4317")

    try:
        with patch.dict("sys.modules", {
            "opentelemetry": None,
            "opentelemetry.exporter": None,
            "opentelemetry.exporter.otlp": None,
            "opentelemetry.exporter.otlp.proto": None,
            "opentelemetry.exporter.otlp.proto.grpc": None,
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": None,
            "opentelemetry.sdk": None,
            "opentelemetry.sdk.metrics": None,
            "opentelemetry.sdk.metrics.export": None,
        }):
            with caplog.at_level(logging.WARNING, logger="apohara_context_forge.observability.otlp_exporter"):
                result = exporter.start()
    finally:
        sys.modules.update(saved)

    assert result is None
    assert not exporter.get_state()["active"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) >= 1


# ---------------------------------------------------------------------------
# Test 2: start() with real opentelemetry (skip if not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_opentelemetry, reason="opentelemetry not installed")
def test_otlp_start_with_real_endpoint():
    """When otel is installed, start() sets active=True and endpoint is echoed in get_state()."""
    exporter = OTLPExporter(endpoint="localhost:4317", insecure=True)
    exporter.start()
    state = exporter.get_state()
    assert state["active"] is True
    assert state["endpoint"] == "localhost:4317"
    assert state["build_error"] is None
    exporter.shutdown()


@pytest.mark.skipif(not _has_opentelemetry, reason="opentelemetry not installed")
def test_otlp_start_exception_handling(caplog):
    """When start() fails with an arbitrary Exception, it sets build_error and logs a WARNING."""
    exporter = OTLPExporter(endpoint="localhost:4317", insecure=True)

    with patch("opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter", side_effect=Exception("simulated build failure")):
        with caplog.at_level(logging.WARNING, logger="apohara_context_forge.observability.otlp_exporter"):
            result = exporter.start()

    assert result is None
    state = exporter.get_state()
    assert state["active"] is False
    assert state["build_error"] == "simulated build failure"
    assert any("OTLPExporter failed to start: simulated build failure" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Test 3: shutdown() is idempotent — calling twice must not raise
# ---------------------------------------------------------------------------

def test_otlp_shutdown_idempotent():
    """Calling shutdown() twice on an inactive exporter does not raise."""
    exporter = OTLPExporter(endpoint="localhost:4317")
    # Never started — should be a safe no-op.
    exporter.shutdown()
    exporter.shutdown()
    assert not exporter.get_state()["active"]


# ---------------------------------------------------------------------------
# Test 4: record_inv15_decision fans out to OTLP when env var is set
# ---------------------------------------------------------------------------

def test_record_inv15_decision_fans_out_to_otlp(monkeypatch):
    """When APOHARA_OTLP_ENDPOINT is set, record_inv15_decision() reaches OTLPExporter."""
    mock_otlp = MagicMock(spec=OTLPExporter)
    mock_otlp.get_state.return_value = {"active": True, "endpoint": "fake:4317", "build_error": None}

    # Reset module-level singleton so _get_otlp() reinitialises.
    orig_otlp = recorders_mod._otlp
    recorders_mod._otlp = mock_otlp

    # Also mock the Prometheus exporter and AuditLog to keep the test isolated.
    mock_exporter = MagicMock()
    mock_audit = MagicMock()
    orig_exporter = recorders_mod._exporter
    orig_audit = recorders_mod._audit_log

    recorders_mod._exporter = mock_exporter
    recorders_mod._audit_log = mock_audit

    monkeypatch.setenv("APOHARA_OTLP_ENDPOINT", "fake:4317")

    try:
        recorders_mod.record_inv15_decision(
            agent_id="test-agent",
            anchor_hash="deadbeef",
            risk_score=0.55,
            gate_action="allow",
            predicted_jcr_delta=0.02,
            lmcache_consulted=True,
            lmcache_hit=True,
        )
    finally:
        recorders_mod._otlp = orig_otlp
        recorders_mod._exporter = orig_exporter
        recorders_mod._audit_log = orig_audit

    mock_otlp.record_jcr_decision.assert_called_once_with(
        agent_id="test-agent", action="allow", risk_score=0.55
    )
    mock_otlp.record_lmcache_hit.assert_called_once()
