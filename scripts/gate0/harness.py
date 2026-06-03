#!/usr/bin/env python3
"""GATE #0 A/B/C runner — orchestrates the three arms over ONE shared workload.

This is module #5 of the GATE #0 harness (see ``scripts/gate0/CONTRACT.md`` and the
preregistered protocol ``docs/research/_internal/GATE-0-protocol.md``). It is the
ONLY module allowed to launch processes and hit the network broadly. It wires the
pure modules together:

  workload.py  — the canonical N=5 shared-prefix workload + measured reuse rate
  arms.py      — A/B/C launch args, env, and per-request cache_salt plans
  metrics.py   — honest readers (KV footprint, throughput/TTFT, /metrics window, CI)
  validity.py  — the gates that make-or-break the run (APC-ON, AITER parity, ...)

and drives each arm over the SAME request list, sampling the metrics, then folds the
result into a single raw-log JSON (schema §9). ``analyze.py`` consumes that log.

The decisive number is ``delta = (B - A)`` with ``A = APC ON, no ROMY salt``. This
runner does NOT compute the verdict — it only collects honest, condition-tagged
measurements so ``analyze.py`` can. Numbers come from readers or are ``None``; nothing
is fabricated.

Two modes (CONTRACT §7):

  * ``dry``  — runs on any box with no GPU. Exercises workload + arms + salts +
    validity-of-config + the log schema. No vLLM server is launched; every numeric
    field is ``None`` and ``measured=False``. Dry results NEVER enter the report.
  * ``live`` — gated (needs a GPU + a reachable ``vllm`` launcher). Launches one vLLM
    server per arm, waits ``/health``, samples, tears it down — same launch->measure->
    teardown shape as ``scripts/mi300x_squeeze_all.sh:run_model``. NOTE: the cross-worker
    topology here is single-server-per-arm plumbing; it does NOT reproduce the two-worker
    store->retrieve sequence of ``local_cross_worker_smoke.py`` (see ``run_gate`` caveat).

vLLM is NEVER imported in Python; the live launcher shells out to the ``vllm`` CLI as
a subprocess. AUDIT discipline: no hardcoded performance number ever reaches a field
that names a measurement.

Apache-2.0 — Apohara ContextForge.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# CONTRACT §1: REPO on sys.path so apohara_context_forge.*, agents.*, scripts.* import
# the same way the existing probes do, and so the sibling gate0 modules resolve.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.gate0 import __version__  # noqa: E402  (schema_version source of truth)
from scripts.gate0.arms import (  # noqa: E402
    ARM_A,
    ARM_B,
    ARM_C,
    ARMS,
    TOPOLOGY_CROSS,
    TOPOLOGY_SINGLE,
    ArmLaunch,
    RequestSalt,
    build_arm_launch,
    salts_for_workload,
)
from scripts.gate0 import metrics as metrics_mod  # noqa: E402
from scripts.gate0.metrics import (  # noqa: E402
    HBMReading,
    KVFootprint,
    PrefixMetrics,
    ThroughputSample,
)
from scripts.gate0 import _lifecycle  # noqa: E402
from scripts.gate0._lifecycle import (  # noqa: E402
    HEALTH_POLL_S,
    HEALTH_TIMEOUT_S,
    SETTLE_S,
    TEARDOWN_TIMEOUT_S,
    default_ports as _default_ports,
    endpoint as _endpoint,
    launch_server as _launch_server,
    log as _log,
    post as _post,
    teardown as _teardown,
    vllm_bin as _vllm_bin,
    wait_health as _wait_health,
)
from scripts.gate0 import validity as validity_mod  # noqa: E402
from scripts.gate0.workload import (  # noqa: E402
    ReuseStats,
    WorkloadRequest,
    WorkloadSpec,
    build_requests,
    load_workload,
    measure_reuse,
)

# vLLM PagedAttention block size — single source is the launch config; mirror it for
# the condition block without importing vLLM.
try:
    from apohara_context_forge.serving.vllm_launch_config import DEFAULT_BLOCK_SIZE
except Exception:  # pragma: no cover - defensive; the contract guarantees it exists
    DEFAULT_BLOCK_SIZE = 16

# Server lifecycle (launch / health-wait / teardown / _post / _endpoint / _vllm_bin /
# _default_ports) + the health/settle knobs live in scripts.gate0._lifecycle, shared
# verbatim with cross_worker.py so neither runner duplicates the subprocess plumbing.
# They are imported above under the same private names this module has always exported.


# --------------------------------------------------------------------------- #
# §7.1 dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class ArmResult:
    """Everything measured for ONE arm in ONE topology."""

    arm: str
    topology: str
    kv_footprint: Optional[KVFootprint]
    throughput: Optional[ThroughputSample]
    prefix_metrics: Optional[PrefixMetrics]
    model_weight_gb: Optional[float]  # post-load/pre-traffic HBM baseline (KV isolation)
    server_log_path: Optional[str]
    inv15_fires: int                  # judge requests that took the isolated path
    measured: bool                    # False in dry mode

    def to_dict(self) -> dict:
        return _result_to_dict(self)


@dataclass
class GateRunResult:
    """The whole run: conditions + per-arm results + validity. Serialized to the log."""

    schema_version: str
    timestamp_utc: str
    topology: str
    workload: WorkloadSpec
    reuse: ReuseStats
    conditions: dict
    arms: dict[str, ArmResult]
    validity: Any                     # validity.ValidityReport
    measured: bool

    def to_dict(self) -> dict:
        return _result_to_dict(self)


# --------------------------------------------------------------------------- #
# Per-arm sampling driver. Owns: warmup, model-weight baseline, salts, the
# prefix-metrics window wrapping a throughput drive, and the KV footprint read.
# All numbers come from metrics.* readers; this function never invents one.
# --------------------------------------------------------------------------- #
def _count_inv15_fires(salts: list[RequestSalt]) -> int:
    """Judge requests that took the isolated path = salts whose reason names INV-15 /
    isolation on a non-control arm. Counted from the salt plan, not re-decided here.

    Arm C isolates EVERY request by design (negative control), so its isolated salts are
    not INV-15 fires; we only count isolation that the planner chose for arm B.

    Counting rule: on arm B, the ONLY reason a salt is isolated (``shared=False``) is that the
    JCRSafetyGate fired INV-15 (judge over threshold) — arm B has no other isolation path. So
    ``arm == B and not shared`` is the precise, structural definition of an INV-15 fire. We do
    NOT match on free-text ``reason`` substrings: the non-judge SHARED reason literally contains
    the word 'judge' (e.g. "role='retriever' not judge-type -> reuse OK"), so a substring test
    would be both redundant (already excluded by ``not shared``) and a latent trap if any
    non-judge request were ever isolated for an unrelated reason."""
    fires = 0
    for s in salts:
        if s.arm != ARM_B:
            continue
        if not s.shared:
            fires += 1
    return fires


def _drive_arm(
    arm: str,
    launch: ArmLaunch,
    spec: WorkloadSpec,
    requests: list[WorkloadRequest],
    salts: list[RequestSalt],
    *,
    topology: str,
    endpoint: str,
    device_id: int,
    server_log_path: Optional[str],
) -> ArmResult:
    """LIVE sampling for one arm against a ready endpoint. Sequence (CONTRACT §7.2):
        3. measure model_weight_gb (post-load, pre-traffic) via metrics.read_hbm
        5. drive the SAME workload, sampling throughput + prefix-metrics window + KV footprint
    Step 5's prefix-metrics window wraps the throughput drive so the /metrics delta and the
    tok/s come from the SAME traffic, never two independent passes."""
    # 3. post-load / pre-traffic HBM baseline (for KV isolation in metrics.read_kv_footprint).
    hbm_baseline = metrics_mod.read_hbm(device_id)
    model_weight_gb = hbm_baseline.used_gb if hbm_baseline.valid else None

    # Warmup so the first TTFT isn't a cold-start outlier (squeeze stage_footprint spirit).
    try:
        warm = requests[0]
        warm_salt = salts[0].cache_salt if salts else None
        _post(endpoint, spec.model, warm.prompt, salt=warm_salt, max_tokens=4).read()
    except Exception as e:  # warmup failure is recorded by downstream readers, not fatal
        _log(f"arm {arm}: warmup failed (non-fatal): {e!r}")
    time.sleep(SETTLE_S)

    # 5. drive traffic ONCE; the prefix-metrics window wraps the throughput drive so both
    # metrics describe the identical traffic. metrics.measure_throughput owns the drive.
    throughput_holder: dict[str, ThroughputSample] = {}

    def send_fn() -> None:
        throughput_holder["t"] = metrics_mod.measure_throughput(
            endpoint,
            spec.model,
            requests,
            salts,
            concurrency=spec.concurrency,
            post_fn=_post,
        )

    prefix_metrics = metrics_mod.prefix_metrics_window(endpoint, send_fn)
    throughput = throughput_holder.get("t")

    # KV footprint AFTER the traffic, passing the pre-traffic baseline so the reader can
    # isolate KV (method='hbm_minus_weights') when no /metrics counter is exposed.
    kv_footprint = metrics_mod.read_kv_footprint(
        endpoint, device_id, model_weight_gb=model_weight_gb
    )

    return ArmResult(
        arm=arm,
        topology=topology,
        kv_footprint=kv_footprint,
        throughput=throughput,
        prefix_metrics=prefix_metrics,
        model_weight_gb=model_weight_gb,
        server_log_path=server_log_path,
        inv15_fires=_count_inv15_fires(salts),
        measured=True,
    )


def _dry_arm(
    arm: str,
    launch: ArmLaunch,
    salts: list[RequestSalt],
    *,
    topology: str,
    server_log_path: Optional[str],
) -> ArmResult:
    """DRY result for one arm: no server, no GPU, no numbers. The arm/launch/salt plumbing
    HAS been exercised (build_arm_launch + salts_for_workload ran); only measurement is
    absent. Every numeric field is None and measured=False so this can NEVER enter a report."""
    return ArmResult(
        arm=arm,
        topology=topology,
        kv_footprint=None,
        throughput=None,
        prefix_metrics=None,
        model_weight_gb=None,
        server_log_path=server_log_path,
        inv15_fires=_count_inv15_fires(salts),
        measured=False,
    )


# --------------------------------------------------------------------------- #
# Condition block (§8) — attached to the run; analyze.py echoes it per metric.
# Pure config values only; no measurement leaks in here.
# --------------------------------------------------------------------------- #
def _build_conditions(
    spec: WorkloadSpec,
    reuse: ReuseStats,
    launch_a: ArmLaunch,
    *,
    topology: str,
    hbm: Optional[HBMReading],
    kv_cache_dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> dict:
    vram_source = hbm.vram_source if hbm is not None else "dry"
    second_source = hbm.second_source if hbm is not None else None
    try:
        hw_label = metrics_mod.hardware_label(vram_source)
    except Exception:
        hw_label = f"unknown ({vram_source})"
    pythonhashseed = launch_a.env.get("PYTHONHASHSEED")
    return {
        "model": spec.model,
        "hardware_label": hw_label,
        "vram_source": vram_source,
        "second_source": second_source,
        "topology": topology,
        "n_agents": len(spec.agents),
        "n_requests": spec.n_requests,
        "concurrency": spec.concurrency,
        "max_tokens": spec.max_tokens,
        "approx_prefix_tokens": reuse.approx_prefix_tokens,
        "shared_prefix_fraction": reuse.shared_prefix_fraction,
        "block_size": launch_a.block_size,
        "kv_cache_dtype": kv_cache_dtype,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "aiter_applied": launch_a.aiter_applied,
        "pythonhashseed": pythonhashseed,
    }


# --------------------------------------------------------------------------- #
# Top-level entry — orchestrates A/B/C lifecycle + sampling + validity + raw log.
# --------------------------------------------------------------------------- #
def run_gate(
    spec: WorkloadSpec,
    *,
    topology: str = TOPOLOGY_SINGLE,
    mode: str = "dry",
    device_id: int = 0,
    ports: Optional[dict[str, int]] = None,
    vllm_bin: Optional[str] = None,
    redis_url: Optional[str] = None,
    out_path: Optional[str] = None,
    kv_cache_dtype: str = "auto",
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.90,
    port_w1: Optional[int] = None,
    port_w2: Optional[int] = None,
) -> GateRunResult:
    """Run all three arms over ``spec`` in one topology and write the raw log.

    For SINGLE-worker, each arm in ARMS (A, B, C) runs SEQUENTIALLY on one card (launch ->
    measure -> teardown), so HBM readings are never cross-contaminated:

        1. build ArmLaunch via arms.build_arm_launch(...)
        2. (live) launch the vLLM subprocess with its env, wait /health, capture the log
        3. measure model_weight_gb (post-load, pre-traffic) via metrics.read_hbm
        4. build per-arm salts via arms.salts_for_workload(...)
        5. drive the SAME workload requests, sampling throughput + prefix-metrics window +
           KV footprint via metrics.*
        6. (live) tear the server down

    Then run validity.run_all(...) and assemble GateRunResult. In dry mode no server is
    launched: arms/salts/validity-of-config are still exercised; numeric fields are None and
    measured=False.

    CROSS-WORKER: ``topology == cross_worker`` dispatches to the REAL two-worker
    store->retrieve path in :func:`scripts.gate0.cross_worker.run_gate_cross_worker_real`
    (worker-1 warms + STORES KV to Redis then DIES; a COLD worker-2 with an empty local cache
    RETRIEVES from Redis). That path measures the decisive ``external_hits_delta`` /
    ``external_kv_tokens_delta`` ON the cold worker-2 and folds them into this same §9 raw log,
    so the cross-worker output IS a measured A-vs-B delta — not "plumbing only" as the legacy
    single-server-per-arm cross path was. ``--mode dry`` for cross_worker still routes through
    that real path's plumbing (both worker launches + salts + the LMCache config + the two
    cross validity checks are exercised) but launches nothing and writes ``measured=False``
    (-> INDECISIVE in analyze.py), with ZERO GPU calls. ``port_w1`` / ``port_w2`` are the
    two-worker ports (defaults from cross_worker.DEFAULT_PORT_W1/W2).
    """
    if mode not in ("dry", "live"):
        raise ValueError(f"mode must be 'dry' or 'live', got {mode!r}")
    if topology not in (TOPOLOGY_SINGLE, TOPOLOGY_CROSS):
        raise ValueError(f"unknown topology {topology!r}")
    if topology == TOPOLOGY_CROSS and mode == "live" and not redis_url:
        raise ValueError("cross_worker live mode requires --redis-url (LMCache backend)")

    # Cross-worker (live AND dry) goes through the REAL two-worker path. Lazy import keeps
    # cross_worker.py free to `import harness` at module load (one-way dependency, no cycle).
    if topology == TOPOLOGY_CROSS:
        from scripts.gate0.cross_worker import (
            DEFAULT_PORT_W1,
            DEFAULT_PORT_W2,
            run_gate_cross_worker_real,
        )

        return run_gate_cross_worker_real(
            spec,
            mode=mode,
            device_id=device_id,
            port_w1=port_w1 if port_w1 is not None else DEFAULT_PORT_W1,
            port_w2=port_w2 if port_w2 is not None else DEFAULT_PORT_W2,
            vllm_bin=vllm_bin,
            redis_url=redis_url,
            out_path=out_path,
            kv_cache_dtype=kv_cache_dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    ports = ports or _default_ports()
    bin_path = _vllm_bin(vllm_bin)
    measured = mode == "live"

    # The SAME request list is replayed across A/B/C; only the cache_salt differs (arms.py).
    requests = build_requests(spec)
    reuse = measure_reuse(spec, requests)
    anchor_hash = reuse.canonical_prefix_hash

    log_dir = _gate0_log_dir()
    arm_launches: list[ArmLaunch] = []
    arm_results: dict[str, ArmResult] = {}
    apc_log_paths: dict[str, str] = {}
    first_hbm: Optional[HBMReading] = None  # for the condition block's vram_source

    for arm in ARMS:
        port = ports.get(arm, _default_ports()[arm])
        launch = build_arm_launch(
            arm,
            model=spec.model,
            topology=topology,
            block_size=DEFAULT_BLOCK_SIZE,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_dtype=kv_cache_dtype,
            port=port,
        )
        arm_launches.append(launch)

        # 4. per-arm salts (shared planner instance across the workload so the gate_log
        # accumulates INV-15 fires for validity to read).
        salts = salts_for_workload(
            arm,
            requests,
            anchor_hash=anchor_hash,
            cla_group="gate0",
            reuse_rate=reuse.shared_prefix_fraction,
        )

        server_log_path = str(log_dir / f"{spec.name}_{topology}_{arm}_server.log")
        apc_log_paths[arm] = server_log_path

        if not measured:
            # Dry: plumbing exercised, no server, no numbers.
            arm_results[arm] = _dry_arm(
                arm, launch, salts, topology=topology, server_log_path=server_log_path
            )
            continue

        endpoint = _endpoint(port)
        proc: Optional[subprocess.Popen] = None
        try:
            extra_env = {"LMCACHE_REDIS_URL": redis_url} if (redis_url and launch.uses_lmcache) else None
            proc, _ = _launch_server(
                launch, vllm_bin=bin_path, log_path=server_log_path, extra_env=extra_env
            )
            if not _wait_health(proc, endpoint):
                _log(f"arm {arm} NOT ready — recording an UNMEASURED arm and continuing")
                arm_results[arm] = _dry_arm(
                    arm, launch, salts, topology=topology, server_log_path=server_log_path
                )
                continue
            _log(f"arm {arm} READY at {endpoint}")
            result = _drive_arm(
                arm,
                launch,
                spec,
                requests,
                salts,
                topology=topology,
                endpoint=endpoint,
                device_id=device_id,
                server_log_path=server_log_path,
            )
            arm_results[arm] = result
            if first_hbm is None and result.kv_footprint is not None:
                first_hbm = result.kv_footprint.hbm
        finally:
            if proc is not None:
                _teardown(proc)
                _log(f"arm {arm} torn down")

    # Condition block — derive vram_source from the first valid HBM reading (live), else dry.
    conditions = _build_conditions(
        spec,
        reuse,
        arm_launches[0],
        topology=topology,
        hbm=first_hbm,
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # Validity (§6). Pass arm C's prefix metrics + a representative HBM so the gates that
    # apply in this mode can fire. Missing inputs make a check report 'not evaluable', never
    # a fabricated pass.
    prefix_metrics_c = arm_results.get(ARM_C).prefix_metrics if ARM_C in arm_results else None
    validity = validity_mod.run_all(
        arm_launches=arm_launches,
        reuse=reuse,
        spec=spec,
        apc_log_paths=apc_log_paths if measured else None,
        prefix_metrics_c=prefix_metrics_c,
        hbm=first_hbm,
        topology=topology,  # single_worker here (cross dispatches earlier); no cross checks fire
    )

    result = GateRunResult(
        schema_version=__version__,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        topology=topology,
        workload=spec,
        reuse=reuse,
        conditions=conditions,
        arms=arm_results,
        validity=validity,
        measured=measured,
    )

    target = out_path or str(log_dir / f"{spec.name}_{topology}.json")
    _write_raw_log(result, target)
    _log(f"raw log written -> {target} (measured={measured})")
    return result


# --------------------------------------------------------------------------- #
# Serialization — schema §9. Dataclasses (incl. nested metrics/validity types and
# the workload trimmed to the §9 shape) -> a JSON tree with honest nulls.
# --------------------------------------------------------------------------- #
def _gate0_log_dir() -> Path:
    d = REPO / "logs" / "gate0"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _is_dataclass_instance(obj: Any) -> bool:
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def _result_to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses (and the containers holding them) to plain dicts,
    so the raw log is pure JSON. None stays None — never coerced to 0 (schema §9)."""
    if _is_dataclass_instance(obj):
        return {k: _result_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _result_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_result_to_dict(v) for v in obj]
    return obj


def _workload_for_log(spec: WorkloadSpec, reuse: ReuseStats) -> dict:
    """The §9 ``workload`` sub-object: a trimmed view, not the whole spec (the full spec's
    canonical prompt is huge and the reuse block already carries the hash + chars)."""
    return {
        "name": spec.name,
        "model": spec.model,
        "canonical_prefix_hash": reuse.canonical_prefix_hash,
        "n_requests": spec.n_requests,
        "concurrency": spec.concurrency,
        "agents": [a.agent_id for a in spec.agents],
    }


def _write_raw_log(result: GateRunResult, path: str) -> None:
    """Serialize GateRunResult to the §9 raw-log JSON. The on-disk ``workload`` is the
    trimmed §9 view; ``reuse`` carries the full prefix provenance."""
    tree = {
        "schema_version": result.schema_version,
        "timestamp_utc": result.timestamp_utc,
        "topology": result.topology,
        "conditions": _result_to_dict(result.conditions),
        "workload": _workload_for_log(result.workload, result.reuse),
        "reuse": _result_to_dict(result.reuse),
        "arms": {arm: _result_to_dict(ar) for arm, ar in result.arms.items()},
        "validity": _result_to_dict(result.validity),
        "measured": result.measured,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tree, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# CLI — CONTRACT §10.
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["dry", "live"], default="dry")
    ap.add_argument("--model", default="Qwen3-32B", help="served model name")
    ap.add_argument("--workload", default=None, help="path to workload YAML, or empty to derive")
    ap.add_argument(
        "--topology",
        choices=[TOPOLOGY_SINGLE, TOPOLOGY_CROSS],
        default=TOPOLOGY_SINGLE,
    )
    ap.add_argument("--n-requests", type=int, default=320, help="LARGE; protocol forbids ~28")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--redis-url", default=None, help="cross_worker only (LMCache backend)")
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--port-a", type=int, default=8000)
    ap.add_argument("--port-b", type=int, default=8001)
    ap.add_argument("--port-c", type=int, default=8002)
    ap.add_argument("--port-w1", type=int, default=8021, help="cross_worker: worker-1 (store) port")
    ap.add_argument("--port-w2", type=int, default=8022, help="cross_worker: worker-2 (cold retrieve) port")
    ap.add_argument("--vllm-bin", default=None)
    ap.add_argument("--out", default=None, help="raw-log path (default logs/gate0/<name>_<topology>.json)")
    args = ap.parse_args(argv)

    spec = load_workload(
        args.workload,
        model=args.model,
        n_requests=args.n_requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
    )

    ports = {ARM_A: args.port_a, ARM_B: args.port_b, ARM_C: args.port_c}

    result = run_gate(
        spec,
        topology=args.topology,
        mode=args.mode,
        device_id=args.device_id,
        ports=ports,
        vllm_bin=args.vllm_bin,
        redis_url=args.redis_url,
        out_path=args.out,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        port_w1=args.port_w1,
        port_w2=args.port_w2,
    )

    # Operator one-liner: where the log went, whether it is quotable, and the loud reminder
    # that dry results never enter the report.
    quotable = getattr(result.validity, "quotable", None)
    _log(
        f"DONE topology={result.topology} mode={args.mode} measured={result.measured} "
        f"quotable={quotable}"
    )
    if not result.measured:
        _log("DRY RUN: measured=False — these numbers are plumbing only and NEVER enter the report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
