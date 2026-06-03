"""GATE #0 — unit tests for the REAL two-worker cross-worker path (code-only, no GPU).

These exercise:
  * the two cross-worker validity checks (``check_w2_cold_read``,
    ``check_cross_negative_control``) against synthetic PrefixMetrics-shaped stand-ins;
  * ``validity.run_all`` wiring (cross-only checks fire iff topology==cross_worker);
  * the parameterized LMCache config writer (``vllm_launch_config.write_lmcache_config``);
  * a DRY ``run_gate(topology=cross_worker)`` round-trip: no GPU/network, measured=False,
    the §9 log carries the two cross checks, and ``analyze.compute_verdict`` returns
    INDECISIVE on the mock.

No fabricated production numbers: every value here is an explicit synthetic fixture.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from apohara_context_forge.serving.aiter_config import AITERConfig
from apohara_context_forge.serving.vllm_launch_config import (
    DEFAULT_BLOCK_SIZE,
    build_lmcache_config_yaml,
    write_lmcache_config,
)
from scripts.gate0 import validity as v
from scripts.gate0.analyze import CUT_INDECISIVE, PRIMARY_KV_UTIL, compute_verdict


# --------------------------------------------------------------------------- #
# check_w2_cold_read
# --------------------------------------------------------------------------- #
def _pm(**kw):
    base = dict(external_hits_delta=0.0, external_kv_tokens_delta=0.0, hit_rate=0.0, error=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_w2_cold_read_external_and_cold_passes():
    chk = v.check_w2_cold_read(_pm(external_hits_delta=240.0, external_kv_tokens_delta=15360.0, hit_rate=0.05))
    assert chk.passed and chk.required


def test_w2_cold_read_no_external_fails():
    chk = v.check_w2_cold_read(_pm(external_hits_delta=0.0, hit_rate=0.02))
    assert not chk.passed and chk.required


def test_w2_cold_read_warm_local_cache_fails():
    # external hits present but the local cache was HOT -> not a real cold read.
    chk = v.check_w2_cold_read(_pm(external_hits_delta=240.0, hit_rate=0.85))
    assert not chk.passed


def test_w2_cold_read_missing_fails_loud():
    chk = v.check_w2_cold_read(None)
    assert not chk.passed and chk.required


def test_w2_cold_read_error_fails_loud():
    chk = v.check_w2_cold_read(_pm(external_hits_delta=240.0, hit_rate=0.05, error="scrape boom"))
    assert not chk.passed


# --------------------------------------------------------------------------- #
# check_cross_negative_control
# --------------------------------------------------------------------------- #
def test_cross_negative_control_zero_external_passes():
    chk = v.check_cross_negative_control(_pm(external_hits_delta=0.0))
    assert chk.passed and chk.required


def test_cross_negative_control_leak_fails():
    chk = v.check_cross_negative_control(_pm(external_hits_delta=12.0, external_kv_tokens_delta=192.0))
    assert not chk.passed


def test_cross_negative_control_missing_fails_loud():
    chk = v.check_cross_negative_control(None)
    assert not chk.passed and chk.required


# --------------------------------------------------------------------------- #
# run_all wiring: cross-only checks fire iff topology==cross_worker
# --------------------------------------------------------------------------- #
def _arm(arm, topology):
    aiter = {**AITERConfig().AITER_ENV_VARS, "VLLM_USE_AITER": "1"}
    return SimpleNamespace(arm=arm, topology=topology, env={**aiter, "PYTHONHASHSEED": "0"}, aiter_applied=True)


def _reuse():
    return SimpleNamespace(n_distinct_prefixes=1, canonical_prefix_hash="ab",
                           shared_prefix_fraction=1.0, n_requests=320)


def test_run_all_single_worker_has_no_cross_checks(tmp_path):
    log = tmp_path / "s.log"
    log.write_text("INFO [core.py:93] enable_prefix_caching=True ...\n")
    report = v.run_all(
        arm_launches=[_arm("A", "single_worker"), _arm("B", "single_worker"), _arm("C", "single_worker")],
        reuse=_reuse(),
        spec=SimpleNamespace(n_requests=320),
        apc_log_paths={"A": str(log), "B": str(log), "C": str(log)},
        prefix_metrics_c=_pm(hit_rate=0.0, queries_delta=300.0, hits_delta=0.0),
        hbm=SimpleNamespace(vram_source="pyrsmi", valid=True, second_source="rocm-smi"),
        topology="single_worker",
    )
    names = {c.name for c in report.checks}
    assert "w2_cold_read" not in names
    assert "cross_negative_control" not in names


def test_run_all_cross_worker_clean_is_quotable(tmp_path):
    log = tmp_path / "s.log"
    log.write_text("INFO [core.py:93] enable_prefix_caching=True ...\n")
    report = v.run_all(
        arm_launches=[_arm("A", "cross_worker"), _arm("B", "cross_worker"), _arm("C", "cross_worker")],
        reuse=_reuse(),
        spec=SimpleNamespace(n_requests=320),
        apc_log_paths={"A": str(log), "B": str(log), "C": str(log)},
        prefix_metrics_c=_pm(hit_rate=0.0, queries_delta=300.0, hits_delta=0.0),
        hbm=SimpleNamespace(vram_source="pyrsmi", valid=True, second_source="rocm-smi"),
        topology="cross_worker",
        prefix_metrics_a=_pm(external_hits_delta=0.0),
        prefix_metrics_b=_pm(external_hits_delta=240.0, external_kv_tokens_delta=15360.0, hit_rate=0.05),
    )
    names = {c.name for c in report.checks}
    assert "w2_cold_read" in names and "cross_negative_control" in names
    assert report.quotable, report.summary


def test_run_all_cross_worker_no_cold_read_not_quotable(tmp_path):
    log = tmp_path / "s.log"
    log.write_text("INFO [core.py:93] enable_prefix_caching=True ...\n")
    report = v.run_all(
        arm_launches=[_arm("A", "cross_worker"), _arm("B", "cross_worker"), _arm("C", "cross_worker")],
        reuse=_reuse(),
        spec=SimpleNamespace(n_requests=320),
        apc_log_paths={"A": str(log), "B": str(log), "C": str(log)},
        prefix_metrics_c=_pm(hit_rate=0.0, queries_delta=300.0, hits_delta=0.0),
        hbm=SimpleNamespace(vram_source="pyrsmi", valid=True, second_source="rocm-smi"),
        topology="cross_worker",
        prefix_metrics_a=_pm(external_hits_delta=0.0),
        prefix_metrics_b=_pm(external_hits_delta=0.0, hit_rate=0.02),  # B never cold-read
    )
    assert not report.quotable
    assert "w2_cold_read" in report.summary


# --------------------------------------------------------------------------- #
# LMCache config writer (parameterized, PROVEN remote_url YAML form)
# --------------------------------------------------------------------------- #
def test_lmcache_yaml_honors_params():
    body = build_lmcache_config_yaml(remote_url="redis://h:6379", chunk_size=16,
                                     remote_serde="naive", local_cpu=False)
    assert 'remote_url: "redis://h:6379"' in body
    assert 'remote_serde: "naive"' in body
    assert "chunk_size: 16" in body
    assert "local_cpu: false" in body  # cold worker-2 starts with empty local cache


def test_write_lmcache_config_writes_path(tmp_path):
    out = tmp_path / "nested" / "cfg.yaml"
    p = write_lmcache_config(str(out), remote_url="redis://x:1", chunk_size=DEFAULT_BLOCK_SIZE)
    assert p == str(out)
    assert out.read_text() == build_lmcache_config_yaml(remote_url="redis://x:1", chunk_size=DEFAULT_BLOCK_SIZE)


# --------------------------------------------------------------------------- #
# DRY run_gate(cross_worker) round-trip: no GPU, measured=False, INDECISIVE.
# --------------------------------------------------------------------------- #
def test_dry_cross_worker_run_gate_is_unmeasured_and_indecisive(tmp_path):
    from scripts.gate0.harness import run_gate
    from scripts.gate0.workload import load_workload

    spec = load_workload(None, model="qwen3-32b", n_requests=320, concurrency=32, max_tokens=64)
    out = tmp_path / "xworker.json"
    result = run_gate(
        spec,
        topology="cross_worker",
        mode="dry",
        redis_url="redis://localhost:6379",
        out_path=str(out),
    )
    assert result.measured is False
    assert result.topology == "cross_worker"
    assert result.reuse.n_distinct_prefixes == 1

    data = json.loads(out.read_text())
    assert data["measured"] is False
    names = {c["name"] for c in data["validity"]["checks"]}
    assert "w2_cold_read" in names and "cross_negative_control" in names
    # every arm is unmeasured in dry; worker-2 log path is the recorded server_log_path.
    for arm in ("A", "B", "C"):
        assert data["arms"][arm]["measured"] is False
        assert data["arms"][arm]["prefix_metrics"] is None
        assert "w2_server.log" in data["arms"][arm]["server_log_path"]

    verdict = compute_verdict(data, primary_metric=PRIMARY_KV_UTIL)
    assert verdict.cut == CUT_INDECISIVE  # never a verdict on a dry/mock run


def test_dry_cross_worker_live_requires_redis():
    from scripts.gate0.harness import run_gate
    from scripts.gate0.workload import load_workload

    spec = load_workload(None, model="qwen3-32b", n_requests=320, concurrency=32, max_tokens=64)
    with pytest.raises(ValueError, match="redis-url|redis_url"):
        run_gate(spec, topology="cross_worker", mode="live", redis_url=None)
