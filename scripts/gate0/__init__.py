"""GATE #0 decision harness for ContextForge / ROMY KV-block sharing.

This package answers ONE preregistered question (see
``docs/research/_internal/GATE-0-protocol.md``): does ROMY's cross-agent
KV-block sharing (``cache_salt`` -> byte-identical prefix -> vLLM Automatic
Prefix Caching + LMCache cross-worker) deliver an INCREMENTAL win over the
native stack (vLLM APC already ON) on MI300X, under a multi-agent
shared-prefix workload?

It is an A/B/C decision harness, NOT a sharing-detector:

  * Arm A — baseline: vLLM with APC ON, NO ROMY salt (the free floor).
  * Arm B — ROMY: same launch + shared ``cache_salt`` (+ ``--kv-transfer-config``
    LMCache in the cross-worker variant).
  * Arm C — negative control: ISOLATED per-request salts (must give ~0% hit).

The decisive delta is ALWAYS (B - A), never (B - C) and never (B - no-cache).
(B - C) only proves the harness can measure sharing at all; the gate is decided
by (B - A) against the preregistered cut: <5% ABANDON, 5-15% grey zone,
>15% INVEST.

HONESTY CONTRACT (binding, enforced by scripts/check_honesty.sh + AUDIT.md):
  * No hardcoded performance numbers anywhere. Every metric carries its source.
  * VRAM is read ONLY via VRAMMonitor (real backend) + an out-of-process second
    source. A reading whose ``vram_source`` is ``amd_default_192gb`` or
    ``cuda_unavailable`` is DISCARDED, never reported as a measurement.
  * Every metric is reported with its full condition block (model, hardware,
    seq lens, concurrency, n_agents, reuse_rate, n_requests) or it does not
    enter the report.
  * Every headline carries a confidence interval and its n_requests.

Module layout (each built by a separate agent against scripts/gate0/CONTRACT.md):
  workload.py  — the canonical N=5 shared-prefix workload + reuse-rate measurement
  arms.py      — A/B/C arm definitions: launch args, env, per-request salt plans
  metrics.py   — honest readers (KV footprint, throughput/TTFT, /metrics, CI)
  validity.py  — preflight + post-hoc validity gates (APC-ON, AITER parity, etc.)
  harness.py   — the runner that drives arms over the workload and emits raw logs
  analyze.py   — consumes raw logs, computes (B - A) +- CI, applies the cut

Apache-2.0 — Apohara ContextForge.
"""
from __future__ import annotations

__all__ = ["__version__"]

# Bump only when CONTRACT.md changes in a way that breaks the raw-log schema.
__version__ = "0.1.0"
