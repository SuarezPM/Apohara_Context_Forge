from __future__ import annotations

import logging
import os
from typing import Optional

from apohara_context_forge.observability.prometheus_exporter import PrometheusExporter
from apohara_context_forge.observability.audit_log import AuditLog
from apohara_context_forge.observability.otlp_exporter import OTLPExporter

logger = logging.getLogger(__name__)

_exporter: Optional[PrometheusExporter] = None
_audit_log: Optional[AuditLog] = None
_otlp: Optional[OTLPExporter] = None
_ledger = None  # Ledger is imported lazily inside _get_ledger to keep this module import-clean


def _get_exporter() -> PrometheusExporter:
    global _exporter
    if _exporter is None:
        _exporter = PrometheusExporter()
    return _exporter


def _get_audit_log() -> AuditLog:
    global _audit_log
    if _audit_log is None:
        # Resolve to a canonical absolute path to defang ../ traversal in env var.
        import pathlib
        audit_dir = pathlib.Path(
            os.environ.get("APOHARA_OBSERVABILITY_DIR", "./.apohara/audit")
        ).expanduser().resolve()
        final_path = (audit_dir / "inv15.jsonl").resolve()
        final_path.relative_to(audit_dir)
        _audit_log = AuditLog(str(final_path))
    return _audit_log


def _get_ledger():
    global _ledger
    if _ledger is None:
        import pathlib
        from apohara_context_forge.observability.ledger import Ledger
        audit_dir = pathlib.Path(
            os.environ.get("APOHARA_OBSERVABILITY_DIR", "./.apohara/audit")
        ).expanduser().resolve()
        final_path = (audit_dir / "inv15_ledger.jsonl").resolve()
        final_path.relative_to(audit_dir)
        _ledger = Ledger(str(final_path))
    return _ledger


def _reset_singletons() -> None:
    """Test hook: drop all cached observability singletons so env changes take effect."""
    global _exporter, _audit_log, _otlp, _ledger
    _exporter = _audit_log = _otlp = _ledger = None


def _get_otlp() -> Optional[OTLPExporter]:
    """Return an initialised OTLPExporter if APOHARA_OTLP_ENDPOINT is set, else None."""
    global _otlp
    if _otlp is not None:
        return _otlp
    endpoint = os.environ.get("APOHARA_OTLP_ENDPOINT", "")
    if not endpoint:
        return None
    _otlp = OTLPExporter(endpoint=endpoint)
    _otlp.start()
    return _otlp


def record_inv15_decision(
    *,
    agent_id: str,
    anchor_hash: str,
    risk_score: float,
    gate_action: str,
    predicted_jcr_delta: float,
    lmcache_consulted: bool = False,
    lmcache_hit: bool = False,
) -> None:
    """Fan out an INV-15 gate decision to PrometheusExporter, AuditLog, and OTLPExporter."""
    exporter = _get_exporter()
    audit_log = _get_audit_log()
    otlp = _get_otlp()

    exporter.record_jcr_decision(
        agent_id=agent_id, action=gate_action, risk_score=risk_score
    )
    if lmcache_hit:
        exporter.record_lmcache_hit()

    audit_log.record({
        "kind": "inv15_gate",
        "agent_id": agent_id,
        "anchor_hash": anchor_hash,
        "risk_score": risk_score,
        "gate_action": gate_action,
        "predicted_jcr_delta": predicted_jcr_delta,
        "lmcache_consulted": lmcache_consulted,
        "lmcache_hit": lmcache_hit,
    })

    if otlp is not None:
        otlp.record_jcr_decision(
            agent_id=agent_id, action=gate_action, risk_score=risk_score
        )
        if lmcache_hit:
            otlp.record_lmcache_hit()


def record_certified_inv15_decision(
    *,
    agent_id: str,
    anchor_hash: str,
    risk_score: float,
    gate_action: str,
    predicted_jcr_delta: float,
    candidate_count: int,
    reuse_rate: float,
    layout_shuffled: bool,
    use_dense: bool,
    lmcache_consulted: bool = False,
    lmcache_hit: bool = False,
) -> dict:
    """Certify the gate decision (Z3) + append to the hash-chained ledger, then fan out
    the decision to Prometheus/AuditLog/OTLP via record_inv15_decision. Returns the cert."""
    from apohara_context_forge.safety.inv15_certifier import certify_decision

    cert = certify_decision(
        agent_role=agent_id,
        candidate_count=candidate_count,
        reuse_rate=reuse_rate,
        layout_shuffled=layout_shuffled,
        use_dense=use_dense,
    )
    _get_ledger().append({
        "kind": "inv15_certificate",
        "agent_id": agent_id,
        "anchor_hash": anchor_hash,
        "risk_score": risk_score,
        "gate_action": gate_action,
        **cert,
    })
    record_inv15_decision(
        agent_id=agent_id,
        anchor_hash=anchor_hash,
        risk_score=risk_score,
        gate_action=gate_action,
        predicted_jcr_delta=predicted_jcr_delta,
        lmcache_consulted=lmcache_consulted,
        lmcache_hit=lmcache_hit,
    )
    return cert
