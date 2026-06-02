"""GATE #0 — A/B/C arm definitions: launch args, env, and per-request salts.

This module owns the operationalisation of the protocol's decisive delta at the
*launch flag* and *cache_salt* level. It is PURE: no vLLM, no lmcache, no torch,
no network, no GPU. It only assembles the strings/dicts the harness hands to the
``vllm`` CLI as a subprocess, and the per-request ``cache_salt`` each arm carries.

The three arms (see ``scripts/gate0/CONTRACT.md`` §4 and the preregistered
protocol ``docs/research/_internal/GATE-0-protocol.md``):

  * ARM_A — baseline: vLLM with Automatic Prefix Caching (APC) ON, NO ROMY salt.
    APC still shares *byte-identical* prefixes natively without any salt; that
    native floor is exactly what the gate measures against. A is NOT
    ``--no-enable-prefix-caching`` — that legacy ``vram_ab_harness`` flag is the
    confound the protocol §6 forbids. This harness REDEFINES A as APC-ON.
  * ARM_B — ROMY: identical launch to A, plus a shared ``cache_salt`` at request
    time (and, in the cross-worker topology, ``--kv-transfer-config`` LMCache).
  * ARM_C — negative control: ISOLATED per-request salts (expect ~0% cross hit;
    proves the harness measures *sharing*, not just same-prompt APC).

THE INVARIANT THE WHOLE GATE DEPENDS ON: A, B, C share *identical* ``serve_args``
except B's cross-worker ``--kv-transfer-config``. Identical model, block size, kv
dtype, max-model-len, gpu-util, and AITER env. The ONLY intended difference
between A and B single-worker is the per-request ``cache_salt``. ``validity.py``
fails the run if launches diverge otherwise.

All salt decisions defer to :class:`PrefixSaltPlanner`, which wraps the real
:class:`JCRSafetyGate`. INV-15 (judge isolation) is owned by that gate and is
NEVER re-decided here.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

# Contract §1: make the repo importable the same way the existing probes do, so
# ``apohara_context_forge.*`` resolves regardless of the invoking cwd.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from apohara_context_forge.safety.jcr_gate import JUDGE_ROLES
from apohara_context_forge.serving.aiter_config import AITERConfig
from apohara_context_forge.serving.prefix_salt_planner import PrefixSaltPlanner
from apohara_context_forge.serving.vllm_launch_config import (
    DEFAULT_BLOCK_SIZE,
    build_kv_transfer_config_json,
    worker_env,
)

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    # workload.py is built in parallel; we depend on its shape, never its impl.
    # Only req.agent_id and req.request_id are read (duck-typed at runtime).
    from scripts.gate0.workload import WorkloadRequest

# --------------------------------------------------------------------------- #
# §1.1 Shared enums / sentinels (defined here, imported everywhere)            #
# --------------------------------------------------------------------------- #

ARM_A = "A"   # baseline: APC ON, no ROMY salt
ARM_B = "B"   # ROMY: APC ON + shared cache_salt (+ LMCache in cross-worker)
ARM_C = "C"   # negative control: isolated per-request salts
ARMS = (ARM_A, ARM_B, ARM_C)

TOPOLOGY_SINGLE = "single_worker"
TOPOLOGY_CROSS = "cross_worker"   # >=2 workers + LMCache + Redis

# Default CLA sharing group for the gate workload. A single group means every
# non-judge agent lands on the SAME shared salt (the whole point of arm B).
DEFAULT_CLA_GROUP = "gate0"

# Candidate count fed to the JCR gate for judge-class agents. The 5-agent gate
# workload compares N=5 candidates, which pushes the critic's risk above the
# 0.7 threshold so INV-15 fires (judge -> isolated salt) — exactly what arm B
# must honour. Non-judge roles are unaffected by this value.
DEFAULT_CANDIDATE_COUNT = 5


# --------------------------------------------------------------------------- #
# §4.1 Dataclasses                                                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArmLaunch:
    """Everything needed to launch ONE vLLM server for an arm/topology."""

    arm: str                       # ARM_A | ARM_B | ARM_C
    topology: str                  # TOPOLOGY_SINGLE | TOPOLOGY_CROSS
    serve_args: list[str]          # full `vllm serve ...` args (excluding "vllm")
    env: dict[str, str]            # subprocess env (includes PYTHONHASHSEED=0)
    enable_prefix_caching: bool    # MUST be True for A, B and C (gate is vs APC-ON)
    uses_lmcache: bool             # True only for B in TOPOLOGY_CROSS
    block_size: int                # DEFAULT_BLOCK_SIZE
    aiter_applied: bool            # AITER parity flag (same across arms)
    note: str


@dataclass(frozen=True)
class RequestSalt:
    """The cache_salt a given arm assigns to a given request."""

    request_id: str
    arm: str
    cache_salt: Optional[str]      # None ONLY if an arm intentionally sends no salt
    shared: bool                   # True if reused across agents (B shared path)
    reason: str                    # mirrors SaltPlan.reason / arm-specific rationale


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #


def _validate_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def _validate_topology(topology: str) -> None:
    if topology not in (TOPOLOGY_SINGLE, TOPOLOGY_CROSS):
        raise ValueError(
            f"unknown topology {topology!r}; expected one of "
            f"{(TOPOLOGY_SINGLE, TOPOLOGY_CROSS)}"
        )


def _served_name(model: str) -> str:
    """Short served-model-name derived from the model path.

    Mirrors mi300x_squeeze_all.sh's intent: the operator passes an explicit
    served name there; here we derive a stable basename so A/B/C all advertise
    the SAME served-model-name (any divergence would confound the comparison).
    """
    return model.rsplit("/", 1)[-1]


def _is_judge_role(agent_id: str) -> bool:
    """True iff the role is judge-class per the gate (do NOT hardcode 'responder').

    JUDGE_ROLES is the single source of truth ({"critic","judge"} today). The
    planner still owns the actual INV-15 decision; this is only used to attach
    a human-readable reason when arm B's shared/isolated split is reported.
    """
    return (agent_id or "").lower() in JUDGE_ROLES


# --------------------------------------------------------------------------- #
# §4.2 Public functions — launch builders                                     #
# --------------------------------------------------------------------------- #


def aiter_env() -> dict[str, str]:
    """The AITER env that MUST be identical across all arms.

    Returns ``AITERConfig().AITER_ENV_VARS + {"VLLM_USE_AITER": "1"}`` as a plain
    dict. The harness applies it per subprocess. Identity across arms is what
    keeps AITER from confounding the (B - A) delta; ``validity.check_aiter_parity``
    asserts it.
    """
    env = dict(AITERConfig().AITER_ENV_VARS)
    # Belt-and-suspenders: some vLLM builds key the master switch off the
    # un-prefixed name as well. Setting both is harmless and keeps parity simple.
    env["VLLM_USE_AITER"] = "1"
    return env


def build_arm_launch(
    arm: str,
    *,
    model: str,
    topology: str = TOPOLOGY_SINGLE,
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.90,
    kv_cache_dtype: str = "auto",
    port: int = 8000,
    extra_args: Optional[list[str]] = None,
) -> ArmLaunch:
    """Build the launch for one arm.

    ALL THREE arms launch with ``--enable-prefix-caching`` (APC ON). This is the
    heart of the protocol's honest comparison: A is APC-ON-without-ROMY, NOT
    ``--no-enable-prefix-caching`` (the legacy vram_ab_harness confound §6
    forbids).

      * A (single): vllm serve <model> --enable-prefix-caching ... (no kv-transfer)
      * B (single): IDENTICAL launch to A (the salt is what differs, not the flags)
      * C (single): IDENTICAL launch to A
      * B (cross):  A's args + --kv-transfer-config <LMCache JSON>, env += worker_env()
      * A/C (cross): A's args, NO --kv-transfer-config (APC-only cross-worker
                     baseline); env still includes PYTHONHASHSEED=0.

    ``serve_args`` starts exactly like ``mi300x_squeeze_all.sh:run_model``.
    """
    _validate_arm(arm)
    _validate_topology(topology)

    served = _served_name(model)

    # --- Base serve args: byte-for-byte the squeeze runner's order. ---------
    # serve <model> --served-model-name <served> --port <port>
    #   --enable-prefix-caching --kv-cache-dtype <kv> --max-model-len <maxlen>
    #   --gpu-memory-utilization <util> --trust-remote-code [--block-size N]
    serve_args: list[str] = [
        "serve",
        model,
        "--served-model-name",
        served,
        "--port",
        str(port),
        "--enable-prefix-caching",
        "--kv-cache-dtype",
        kv_cache_dtype,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--trust-remote-code",
        # Block size is part of the APC keying; it MUST match LMCache chunk_size
        # in the cross-worker B path, so we pin it explicitly on every arm to
        # keep the comparison identical (DEFAULT_BLOCK_SIZE).
        "--block-size",
        str(block_size),
    ]

    # --- Cross-worker B is the ONLY arm that appends --kv-transfer-config. ---
    uses_lmcache = arm == ARM_B and topology == TOPOLOGY_CROSS
    if uses_lmcache:
        # build_kv_transfer_config_json enforces chunk_size == block_size and
        # emits LMCacheConnectorV1Dynamic — the only honest cross-worker hook.
        serve_args += [
            "--kv-transfer-config",
            build_kv_transfer_config_json(block_size, block_size),
        ]

    # Operator-supplied extras (e.g. tensor-parallel) appended LAST so they are
    # identical across arms when the caller passes the same list.
    if extra_args:
        serve_args += list(extra_args)

    # --- Env: AITER parity on EVERY arm; PYTHONHASHSEED=0 on EVERY arm. ------
    # worker_env() pins PYTHONHASHSEED=0 (mandatory for cross-worker salt
    # collision; harmless and required-for-parity on single-worker too) and
    # LMCACHE_USE_EXPERIMENTAL=True (only meaningful when LMCache is loaded, but
    # setting it unconditionally keeps the env identical across arms so it can
    # never become an A/B confound). AITER vars are merged identically.
    env: dict[str, str] = {}
    env.update(aiter_env())
    env.update(worker_env())

    aiter_applied = all(
        env.get(k) == v for k, v in AITERConfig().AITER_ENV_VARS.items()
    )

    if topology == TOPOLOGY_SINGLE:
        note = (
            f"arm {arm} single_worker: APC ON, no kv-transfer; "
            "identical launch across A/B/C — only request cache_salt differs"
        )
    elif uses_lmcache:
        note = (
            "arm B cross_worker: APC ON + LMCacheConnectorV1Dynamic via "
            "--kv-transfer-config; worker_env pins PYTHONHASHSEED=0"
        )
    else:
        note = (
            f"arm {arm} cross_worker: APC-only baseline (NO kv-transfer); "
            "PYTHONHASHSEED=0 pinned for honest cross-worker comparison"
        )

    return ArmLaunch(
        arm=arm,
        topology=topology,
        serve_args=serve_args,
        env=env,
        enable_prefix_caching=True,
        uses_lmcache=uses_lmcache,
        block_size=block_size,
        aiter_applied=aiter_applied,
        note=note,
    )


# --------------------------------------------------------------------------- #
# §4.2 Public functions — per-request salt decisions                          #
# --------------------------------------------------------------------------- #


def salt_for_request(
    arm: str,
    req: "WorkloadRequest",
    *,
    anchor_hash: str,
    cla_group: str = DEFAULT_CLA_GROUP,
    planner: Optional[PrefixSaltPlanner] = None,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    reuse_rate: float = 0.0,
) -> RequestSalt:
    """Decide the cache_salt for ``(arm, request)``. Single place encoding arm semantics.

      * ARM_A: cache_salt=None, shared=False — APC-native prefix sharing, no ROMY
        salt. APC still shares byte-identical prefixes natively WITHOUT a salt;
        that is the free floor the gate measures against.
      * ARM_B: ``planner.plan(...)``. Non-judge -> shared_salt (shared=True).
        Judge tripping INV-15 -> isolated_salt (shared=False). INV-15 is owned by
        the JCRSafetyGate inside the planner, never re-decided here.
      * ARM_C: ``planner.isolated_salt(anchor_hash, request_id)`` for EVERY request
        (shared=False) — negative control, expect ~0% cross-agent hit.
    """
    _validate_arm(arm)
    if planner is None:
        planner = PrefixSaltPlanner()

    if arm == ARM_A:
        return RequestSalt(
            request_id=req.request_id,
            arm=ARM_A,
            cache_salt=None,
            shared=False,
            reason="arm A: APC-native prefix sharing, no ROMY salt",
        )

    if arm == ARM_B:
        # The planner runs the REAL gate; reuse_rate/candidate_count feed INV-15
        # risk. Non-judge -> deterministic shared salt; judge over threshold ->
        # unique isolated salt. We mirror plan.reason verbatim for the report.
        plan = planner.plan(
            agent_role=req.agent_id,
            anchor_hash=anchor_hash,
            cla_group=cla_group,
            request_id=req.request_id,
            candidate_count=candidate_count,
            reuse_rate=reuse_rate,
        )
        return RequestSalt(
            request_id=req.request_id,
            arm=ARM_B,
            cache_salt=plan.cache_salt,
            shared=plan.shared,
            reason=plan.reason,
        )

    # ARM_C — negative control: a unique salt per request forces vLLM to key
    # every prefix differently, so no cross-agent block reuse can occur.
    return RequestSalt(
        request_id=req.request_id,
        arm=ARM_C,
        cache_salt=planner.isolated_salt(anchor_hash, req.request_id),
        shared=False,
        reason="arm C: isolated salt per request — negative control, expect ~0% cross hit",
    )


def salts_for_workload(
    arm: str,
    requests: list["WorkloadRequest"],
    *,
    anchor_hash: str,
    cla_group: str = DEFAULT_CLA_GROUP,
    reuse_rate: float = 0.0,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
) -> list[RequestSalt]:
    """``salt_for_request`` mapped over the request list, sharing ONE planner.

    A single :class:`PrefixSaltPlanner` instance is reused so its underlying
    ``JCRSafetyGate.gate_log`` accumulates across the whole workload — that is
    how ``validity.py`` later reads how many judge requests took the INV-15
    isolated path (and how the harness reports ``inv15_fires``).
    """
    _validate_arm(arm)
    planner = PrefixSaltPlanner()
    return [
        salt_for_request(
            arm,
            req,
            anchor_hash=anchor_hash,
            cla_group=cla_group,
            planner=planner,
            candidate_count=candidate_count,
            reuse_rate=reuse_rate,
        )
        for req in requests
    ]
