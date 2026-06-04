#!/usr/bin/env python3
"""GATE #0 — the REAL two-worker cross-worker path (store -> die -> cold retrieve).

``harness.run_gate(topology=cross_worker)`` historically launched ONE server per arm
and drove it once — identical to the single-worker path — so ``external_*`` counters
were structurally ~0 (no second cold worker ever read from Redis). That measured the
LMCache launch *plumbing*, never a cross-worker A-vs-B delta.

This module fuses the proven sequence of ``scripts/local_cross_worker_smoke.py``
(worker-1 warms + STORES KV to Redis then DIES; a COLD worker-2 with an empty local
cache RETRIEVES from Redis) into the harness's arms / metrics / validity / §9-log
machinery, so the decisive ``external_kv_tokens_delta`` / ``external_hits_delta`` are
measured on the COLD worker-2 and fold into the same raw log ``analyze.py`` reads.

Per arm (CONTRACT §7 server-lifecycle rule), on ONE card SEQUENTIALLY:

  worker-1 (PORT_W1): build_arm_launch(arm, topology=cross) -> launch -> /health ->
    drive the SAME workload (warm + STORE to Redis for B) -> capture w1 prefix metrics
    -> teardown (worker-1 DIES; its LOCAL cache dies with it, Redis keeps the chunks).
  settle.
  worker-2 (PORT_W2): IDENTICAL args/salt, COLD local cache (LMCache local_cpu:false) ->
    /health -> capture post-load HBM baseline -> wrap the drive in prefix_metrics_window
    so external_hits_delta / external_kv_tokens_delta are measured ON THE COLD w2 ->
    read_kv_footprint on w2 -> teardown.

  ArmResult.prefix_metrics for §9 is the WORKER-2 window (so analyze.py reads it
  unchanged). The gate decides on delta = (B - A) of external_kv_tokens_delta /
  external_hits_delta: arm A (APC-only cross baseline) must yield external ~0, so any
  external hits on B are attributable to LMCache cross-worker offload, not native APC.

Arms:
  * A — APC-only cross baseline: NO --kv-transfer-config, NO LMCache config file. A cold
    worker-2 has nothing to retrieve from (there is no shared store), so external_* ~0.
  * B — LMCache cross: --kv-transfer-config + LMCACHE_CONFIG_FILE pointing at a YAML with
    remote_url=<Redis>; worker-1 stores, cold worker-2 retrieves. This is the arm under test.
  * C — isolated-salt negative control: same launch as A (APC-only cross), but per-request
    ISOLATED salts. Even if a store existed, isolated salts can't collide cross-worker.

Honesty (CONTRACT §1): NO vLLM/lmcache/torch import here; the launcher shells out to the
``vllm`` CLI via ``_lifecycle``. No fabricated number — every value comes from a
``metrics.*`` reader or is ``None``. Dry mode launches nothing, reads nothing, and writes
``measured=False`` (INDECISIVE downstream).

Apache-2.0 — Apohara ContextForge.
"""
from __future__ import annotations

import datetime
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# CONTRACT §1: REPO on sys.path so sibling gate0 modules resolve regardless of cwd.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from apohara_context_forge.serving.vllm_launch_config import write_lmcache_config  # noqa: E402
from scripts.gate0 import _lifecycle  # noqa: E402
from scripts.gate0 import harness as harness_mod  # noqa: E402
from scripts.gate0 import metrics as metrics_mod  # noqa: E402
from scripts.gate0 import validity as validity_mod  # noqa: E402
from scripts.gate0.arms import (  # noqa: E402
    ARM_A,
    ARM_B,
    ARM_C,
    ARMS,
    TOPOLOGY_CROSS,
    ArmLaunch,
    RequestSalt,
    build_arm_launch,
    salts_for_workload,
)
from scripts.gate0.harness import ArmResult, GateRunResult  # noqa: E402
from scripts.gate0.metrics import HBMReading, ThroughputSample  # noqa: E402
from scripts.gate0.workload import WorkloadRequest, WorkloadSpec  # noqa: E402

# Default two-worker ports. PORT_W1 stores; PORT_W2 (cold) retrieves. Distinct from the
# single-worker per-arm ports so a stray single-worker server can't collide.
DEFAULT_PORT_W1 = 8021
DEFAULT_PORT_W2 = 8022

# Settle window between worker-1 dying and worker-2 starting, so Redis flushes the stored
# chunks before the cold worker reads. Mirrors local_cross_worker_smoke's sleep(3).
REDIS_SETTLE_S = 3.0


def _lmcache_config_path(log_dir: Path, spec: WorkloadSpec, arm: str) -> str:
    """Per-arm LMCache config path under the gate log dir (NOT a /tmp hardcode)."""
    return str(log_dir / f"{spec.name}_{TOPOLOGY_CROSS}_{arm}_lmcache.yaml")


def _drive_store(
    spec: WorkloadSpec,
    requests: list[WorkloadRequest],
    salts: list[RequestSalt],
    *,
    endpoint: str,
) -> Optional[ThroughputSample]:
    """Worker-1 STORE pass: drive the SAME workload so worker-1 populates the shared store
    (LMCache -> Redis for arm B). We reuse measure_throughput as the driver; its throughput
    sample for w1 is incidental (the §9 ArmResult comes from worker-2), but driving via the
    real reader keeps the store traffic identical to the measured retrieve traffic."""
    try:
        return metrics_mod.measure_throughput(
            endpoint,
            spec.model,
            requests,
            salts,
            concurrency=spec.concurrency,
            post_fn=_lifecycle.post,
        )
    except Exception as e:  # store failure is surfaced by worker-2's empty external delta
        _lifecycle.log(f"worker-1 store drive failed (non-fatal): {e!r}")
        return None


def _run_arm_two_worker(
    arm: str,
    launch: ArmLaunch,
    spec: WorkloadSpec,
    requests: list[WorkloadRequest],
    salts: list[RequestSalt],
    *,
    launch_w2: ArmLaunch,
    endpoint_w1: str,
    endpoint_w2: str,
    device_id: int,
    vllm_bin: str,
    lmcache_cfg_path: Optional[str],
    w1_log_path: str,
    w2_log_path: str,
    inv15_fires: int,
) -> ArmResult:
    """Run ONE arm through the real two-worker sequence and return its ArmResult.

    The §9 ArmResult is populated from the WORKER-2 window: kv_footprint, throughput and the
    decisive prefix_metrics (external_*_delta) all describe the COLD worker-2 that retrieves.
    server_log_path points at worker-2's log (the one whose APC-ON the validity gate proves).
    """
    # extra_env for B: worker_env() is already merged into launch.env by build_arm_launch;
    # the cross-worker B path additionally needs LMCACHE_CONFIG_FILE pointing at the YAML
    # (the PROVEN remote_url form). A/C carry no LMCache config (APC-only cross baseline).
    extra_env: Optional[dict[str, str]] = None
    if launch.uses_lmcache and lmcache_cfg_path is not None:
        extra_env = {"LMCACHE_CONFIG_FILE": lmcache_cfg_path}

    # ---- Worker-1: warm + STORE to Redis, then DIE -------------------------
    p1: Optional[subprocess.Popen] = None
    try:
        p1, _ = _lifecycle.launch_server(
            launch, vllm_bin=vllm_bin, log_path=w1_log_path, extra_env=extra_env
        )
        if not _lifecycle.wait_health(p1, endpoint_w1):
            _lifecycle.log(f"arm {arm} worker-1 NOT ready — recording UNMEASURED arm")
            return _unmeasured_arm(arm, w2_log_path, inv15_fires)
        _lifecycle.log(f"arm {arm} worker-1 READY at {endpoint_w1} — warming + storing to Redis")
        # Warmup the first prompt so the store pass isn't a cold-start outlier.
        try:
            warm = requests[0]
            warm_salt = salts[0].cache_salt if salts else None
            _lifecycle.post(endpoint_w1, spec.model, warm.prompt, salt=warm_salt, max_tokens=4).read()
        except Exception as e:
            _lifecycle.log(f"arm {arm} worker-1 warmup failed (non-fatal): {e!r}")
        time.sleep(_lifecycle.SETTLE_S)
        _drive_store(spec, requests, salts, endpoint=endpoint_w1)
    finally:
        if p1 is not None:
            _lifecycle.teardown(p1)
            _lifecycle.log(f"arm {arm} worker-1 torn down (its local cache dies; Redis keeps chunks)")

    # Let Redis settle before the cold worker reads (smoke-proven).
    time.sleep(REDIS_SETTLE_S)

    # ---- Worker-2: COLD local cache, must RETRIEVE from Redis --------------
    p2: Optional[subprocess.Popen] = None
    try:
        # worker-2 uses launch_w2 (identical serve_args to worker-1 but --port=port_w2);
        # build_arm_launch bakes --port into serve_args and launch_server has no override,
        # so reusing worker-1's launch here would bind w2 to port_w1 and the port_w2 health
        # check would hang -> UNMEASURED while the GPU bills. Must be the port_w2 launch.
        p2, _ = _lifecycle.launch_server(
            launch_w2, vllm_bin=vllm_bin, log_path=w2_log_path, extra_env=extra_env
        )
        if not _lifecycle.wait_health(p2, endpoint_w2):
            _lifecycle.log(f"arm {arm} worker-2 NOT ready — recording UNMEASURED arm")
            return _unmeasured_arm(arm, w2_log_path, inv15_fires)
        _lifecycle.log(f"arm {arm} worker-2 READY at {endpoint_w2} (COLD) — measuring retrieve")

        # Post-load / pre-traffic HBM baseline on the cold worker (for KV isolation).
        hbm_baseline = metrics_mod.read_hbm(device_id)
        model_weight_gb = hbm_baseline.used_gb if hbm_baseline.valid else None

        # Drive worker-2 ONCE with the SAME requests + salts; the prefix-metrics window wraps
        # the throughput drive so external_hits_delta / external_kv_tokens_delta and the tok/s
        # come from the IDENTICAL cold-retrieve traffic, never two passes.
        throughput_holder: dict[str, ThroughputSample] = {}

        def send_fn() -> None:
            throughput_holder["t"] = metrics_mod.measure_throughput(
                endpoint_w2,
                spec.model,
                requests,
                salts,
                concurrency=spec.concurrency,
                post_fn=_lifecycle.post,
            )

        prefix_metrics = metrics_mod.prefix_metrics_window(endpoint_w2, send_fn)
        throughput = throughput_holder.get("t")

        kv_footprint = metrics_mod.read_kv_footprint(
            endpoint_w2, device_id, model_weight_gb=model_weight_gb
        )
    finally:
        if p2 is not None:
            _lifecycle.teardown(p2)
            _lifecycle.log(f"arm {arm} worker-2 torn down")

    return ArmResult(
        arm=arm,
        topology=TOPOLOGY_CROSS,
        kv_footprint=kv_footprint,
        throughput=throughput,
        prefix_metrics=prefix_metrics,
        model_weight_gb=model_weight_gb,
        server_log_path=w2_log_path,
        inv15_fires=inv15_fires,
        measured=True,
    )


def _unmeasured_arm(arm: str, server_log_path: str, inv15_fires: int) -> ArmResult:
    """An arm whose worker never reached /health: no numbers, measured=False (mirrors the
    harness UNMEASURED-arm convention so a billing-but-broken live run can never be quoted)."""
    return ArmResult(
        arm=arm,
        topology=TOPOLOGY_CROSS,
        kv_footprint=None,
        throughput=None,
        prefix_metrics=None,
        model_weight_gb=None,
        server_log_path=server_log_path,
        inv15_fires=inv15_fires,
        measured=False,
    )


def _dry_arm_cross(arm: str, server_log_path: str, inv15_fires: int) -> ArmResult:
    """DRY result for one cross-worker arm: no server, no GPU, no numbers. The launch + salts
    + LMCache-config plumbing HAS been exercised; only measurement is absent (measured=False)."""
    return _unmeasured_arm(arm, server_log_path, inv15_fires)


def _drive_cross_worker_arm(
    arm: str,
    spec: WorkloadSpec,
    requests: list[WorkloadRequest],
    reuse: "harness_mod.ReuseStats",
    anchor_hash: str,
    log_dir: Path,
    redis_url: Optional[str],
    port_w1: int,
    port_w2: int,
    device_id: int,
    bin_path: Optional[str],
    measured: bool,
    max_model_len: int,
    gpu_memory_utilization: float,
    kv_cache_dtype: str,
) -> tuple[ArmLaunch, str, ArmResult]:
    """Sets up and runs a single cross-worker arm, returning its launch, log path, and result."""
    launch = build_arm_launch(
        arm,
        model=spec.model,
        topology=TOPOLOGY_CROSS,
        block_size=harness_mod.DEFAULT_BLOCK_SIZE,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_dtype=kv_cache_dtype,
        port=port_w1,
    )

    salts = salts_for_workload(
        arm,
        requests,
        anchor_hash=anchor_hash,
        cla_group="gate0",
        reuse_rate=reuse.shared_prefix_fraction,
    )
    inv15_fires = harness_mod._count_inv15_fires(salts)

    # The validity gate proves APC-ON from worker-2's log (the measured worker).
    w1_log_path = str(log_dir / f"{spec.name}_{TOPOLOGY_CROSS}_{arm}_w1_server.log")
    w2_log_path = str(log_dir / f"{spec.name}_{TOPOLOGY_CROSS}_{arm}_w2_server.log")

    # LMCache config (PROVEN remote_url YAML) for arm B only; A/C are APC-only cross.
    lmcache_cfg_path: Optional[str] = None
    if launch.uses_lmcache:
        lmcache_cfg_path = _lmcache_config_path(log_dir, spec, arm)
        if redis_url:
            write_lmcache_config(
                lmcache_cfg_path,
                remote_url=redis_url,
                chunk_size=launch.block_size,
            )
            _lifecycle.log(f"arm {arm} wrote LMCache config -> {lmcache_cfg_path}")

    if not measured:
        # Dry: launches + salts + LMCache config plumbing exercised; no server/GPU/network.
        result = _dry_arm_cross(arm, w2_log_path, inv15_fires)
        return launch, w2_log_path, result

    # Worker-2 (cold retrieve) runs on a DISTINCT port. build_arm_launch bakes --port into
    # serve_args, so worker-2 needs its OWN launch on port_w2 (otherwise it binds port_w1
    # and the port_w2 health check never passes). Identical serve_args otherwise — the
    # cross-worker invariant is preserved per phase (all arms share these args).
    launch_w2 = build_arm_launch(
        arm,
        model=spec.model,
        topology=TOPOLOGY_CROSS,
        block_size=harness_mod.DEFAULT_BLOCK_SIZE,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_dtype=kv_cache_dtype,
        port=port_w2,
    )

    result = _run_arm_two_worker(
        arm,
        launch,
        spec,
        requests,
        salts,
        launch_w2=launch_w2,
        endpoint_w1=_lifecycle.endpoint(port_w1),
        endpoint_w2=_lifecycle.endpoint(port_w2),
        device_id=device_id,
        vllm_bin=bin_path,
        lmcache_cfg_path=lmcache_cfg_path,
        w1_log_path=w1_log_path,
        w2_log_path=w2_log_path,
        inv15_fires=inv15_fires,
    )
    return launch, w2_log_path, result


def run_gate_cross_worker_real(
    spec: WorkloadSpec,
    *,
    mode: str = "dry",
    device_id: int = 0,
    port_w1: int = DEFAULT_PORT_W1,
    port_w2: int = DEFAULT_PORT_W2,
    vllm_bin: Optional[str] = None,
    redis_url: Optional[str] = None,
    out_path: Optional[str] = None,
    kv_cache_dtype: str = "auto",
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.90,
) -> GateRunResult:
    """Run all three arms through the REAL two-worker store->retrieve sequence and write the
    §9 raw log. For each arm: worker-1 stores to Redis and dies, cold worker-2 retrieves; the
    ArmResult is the worker-2 window.

    ``mode='dry'`` exercises the full plumbing on a box with no GPU — builds both launches +
    salts + the LMCache config, runs config-validity, and writes a ``measured=False`` log
    (-> INDECISIVE in analyze.py) with ZERO GPU/network calls. ``mode='live'`` is gated and
    requires ``redis_url``.

    The §9 dataclasses (ArmResult/GateRunResult) + serializer live in harness.py and are
    reused verbatim here (one schema, one writer). harness.py dispatches INTO this module via
    a lazy import inside run_gate, so this module's top-level ``import harness`` is not circular.
    """
    if mode not in ("dry", "live"):
        raise ValueError(f"mode must be 'dry' or 'live', got {mode!r}")
    if mode == "live" and not redis_url:
        raise ValueError("cross_worker_real live mode requires --redis-url (LMCache backend)")

    bin_path = _lifecycle.vllm_bin(vllm_bin)
    measured = mode == "live"

    # The SAME request list is replayed across A/B/C and across both workers; only the
    # cache_salt differs per arm (arms.py). worker-1 stores it, worker-2 retrieves it.
    requests = harness_mod.build_requests(spec)
    reuse = harness_mod.measure_reuse(spec, requests)
    anchor_hash = reuse.canonical_prefix_hash

    # Co-locate per-arm server logs + LMCache configs with the run's output log: derive the
    # artifact dir from out_path when given (so a scratch/test run keeps everything together),
    # else the canonical logs/gate0/ dir.
    log_dir = Path(out_path).resolve().parent if out_path else harness_mod._gate0_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    arm_launches: list[ArmLaunch] = []
    arm_results: dict[str, "ArmResult"] = {}
    apc_log_paths: dict[str, str] = {}
    first_hbm: Optional[HBMReading] = None

    for arm in ARMS:
        launch, w2_log_path, result = _drive_cross_worker_arm(
            arm,
            spec=spec,
            requests=requests,
            reuse=reuse,
            anchor_hash=anchor_hash,
            log_dir=log_dir,
            redis_url=redis_url,
            port_w1=port_w1,
            port_w2=port_w2,
            device_id=device_id,
            bin_path=bin_path,
            measured=measured,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_dtype=kv_cache_dtype,
        )
        arm_launches.append(launch)
        apc_log_paths[arm] = w2_log_path
        arm_results[arm] = result

        if first_hbm is None and result.kv_footprint is not None:
            first_hbm = result.kv_footprint.hbm

    conditions = harness_mod._build_conditions(
        spec,
        reuse,
        arm_launches[0],
        topology=TOPOLOGY_CROSS,
        hbm=first_hbm,
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # Validity (§6). topology=cross_worker makes the two cross checks REQUIRED (below).
    arm_a = arm_results.get(ARM_A)
    arm_b = arm_results.get(ARM_B)
    arm_c = arm_results.get(ARM_C)
    prefix_metrics_a = arm_a.prefix_metrics if arm_a is not None else None
    prefix_metrics_b = arm_b.prefix_metrics if arm_b is not None else None
    prefix_metrics_c = arm_c.prefix_metrics if arm_c is not None else None

    validity = validity_mod.run_all(
        arm_launches=arm_launches,
        reuse=reuse,
        spec=spec,
        apc_log_paths=apc_log_paths if measured else None,
        prefix_metrics_c=prefix_metrics_c,
        hbm=first_hbm,
        topology=TOPOLOGY_CROSS,
        prefix_metrics_a=prefix_metrics_a,
        prefix_metrics_b=prefix_metrics_b,
    )

    result = GateRunResult(
        schema_version=harness_mod.__version__,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        topology=TOPOLOGY_CROSS,
        workload=spec,
        reuse=reuse,
        conditions=conditions,
        arms=arm_results,
        validity=validity,
        measured=measured,
    )

    target = out_path or str(log_dir / f"{spec.name}_{TOPOLOGY_CROSS}.json")
    harness_mod._write_raw_log(result, target)
    _lifecycle.log(f"cross_worker_real raw log written -> {target} (measured={measured})")
    return result
