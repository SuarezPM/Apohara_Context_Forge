#!/usr/bin/env python3
"""GATE #0 validity gates — the checks that make-or-break the experiment (protocol §6).

Each gate answers a single "did we measure what we think we measured?" question and
returns a :class:`ValidityCheck`. The run is only quotable in the report when every
``required`` check passes. A failing required check is meant to be LOUD and ACTIONABLE:
``detail`` carries the evidence string an operator can act on, ``evidence`` carries the
machine-readable proof for the raw log.

The four protocol §6 failure modes this module guards against:

  (a) Confound APC — arm A must run with ``enable_prefix_caching=True``. If A silently
      had APC off, ``delta = (B - A)`` would credit ROMY with the free native floor.
      :func:`check_apc_on` greps the vLLM startup line (proven in
      ``logs_moe_run/vllm_*_crash.log``: ``enable_prefix_caching=<bool>`` inside the
      ``core.py`` engine-init line, ANSI colour codes and all).
  (b) Workload without real reuse — if the canonical prefix does not collapse to ONE
      distinct prefix, B ≈ A trivially and the gate is void.
      :func:`check_shared_prefix_single` enforces ``reuse.n_distinct_prefixes == 1``.
      The negative control (:func:`check_c_control`) proves the harness measures
      *sharing*, not just same-prompt APC: arm C's cross-agent hit_rate must be ~0%.
  (c) Small N / no CI — :func:`check_n_requests` enforces a floor (default 200), so the
      report is never headlined off ~28 requests. The CI itself lives in ``analyze.py``;
      this gate only guarantees there were enough samples to compute a tight one.
  (d) Dishonest VRAM — :func:`check_vram_source_honest` rejects any reading whose
      ``vram_source`` is the 192 GB default, an unavailable CUDA path, ``unknown`` or
      ``dry``. The AUDIT #2 fallback (``return 45.0, 192.0``) never enters the report.

Plus two confound guards that keep arms comparable:
  - :func:`check_aiter_parity` — all arms must carry an IDENTICAL AITER env subset, else
    AITER kernel differences confound the delta.
  - :func:`check_seed_pinned` — every arm env must pin ``PYTHONHASHSEED=0`` (mandatory for
    cross-worker salt collision; recommended otherwise).

PURE module: no GPU, no network, no vLLM/lmcache/torch import. It consumes already-built
dataclasses from the sibling modules (``arms.ArmLaunch``, ``workload.ReuseStats``,
``workload.WorkloadSpec``, ``metrics.PrefixMetrics``, ``metrics.HBMReading``) via plain
attribute access, so it imports cleanly even while those siblings are still being built in
parallel (CONTRACT.md §13: build against signatures, not impls). The only repo modules it
imports at runtime are the already-existing ``vllm_launch_config.worker_env`` (for the
canonical seed env) and ``aiter_config.AITERConfig`` (for the AITER parity key set).

Apache-2.0 — Apohara ContextForge.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from apohara_context_forge.serving.aiter_config import AITERConfig  # noqa: E402
from apohara_context_forge.serving.vllm_launch_config import worker_env  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover — types only; siblings may not exist yet at import time.
    from scripts.gate0.arms import ArmLaunch
    from scripts.gate0.metrics import HBMReading, PrefixMetrics
    from scripts.gate0.workload import ReuseStats, WorkloadSpec


# --------------------------------------------------------------------------- #
# Honesty: the only VRAM backends a quoted number may come from. Everything    #
# else (the AMD 192 GB default, an unavailable CUDA path, unknown, dry) is     #
# surfaced but NOT quotable (CONTRACT.md §1 / §6.2; AUDIT #2).                  #
# --------------------------------------------------------------------------- #
HONEST_VRAM_SOURCES: frozenset[str] = frozenset(
    {"pyrsmi", "drm_sysfs", "cuda_nvml", "cuda_nvidia_smi"}
)
DISHONEST_VRAM_SOURCES: frozenset[str] = frozenset(
    {"amd_default_192gb", "cuda_unavailable", "unknown", "dry"}
)

# Seed every cross-worker (and, by discipline, every single-worker) env must pin.
PYTHONHASHSEED_REQUIRED = "0"

# Protocol §6(c): a floor, NOT a target. The achieved CI width still goes in the report.
DEFAULT_MIN_REQUESTS = 200

# Arm C is a negative control: cross-agent hit_rate must be ~0. Above this it is not a
# control anymore and the harness cannot be trusted to measure sharing (vs same-prompt APC).
DEFAULT_C_MAX_HIT_RATE = 0.05

# Topologies that MUST have PYTHONHASHSEED pinned (cross-worker salts never collide without
# it). Single-worker treats it as recommended, not required.
_CROSS_TOPOLOGY = "cross_worker"


def _aiter_parity_keys() -> tuple[str, ...]:
    """The env keys that MUST match across arms for an honest delta.

    The AITER kernel selection (``AITERConfig().AITER_ENV_VARS``) plus the
    ``VLLM_USE_AITER`` master switch ``arms.aiter_env()`` adds. If any of these differ
    between arms, AITER — not ROMY — could explain (B - A)."""
    keys = set(AITERConfig().AITER_ENV_VARS.keys())
    keys.add("VLLM_USE_AITER")
    return tuple(sorted(keys))


# --------------------------------------------------------------------------- #
# Result types (CONTRACT.md §6.1)                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValidityCheck:
    """One gate's verdict. ``required`` checks gate quotability; ``evidence`` is the
    machine-readable proof copied into the raw log (§9 ``validity.checks``)."""

    name: str            # "apc_on" | "aiter_parity" | "seed_pinned" | "c_control_zero" |
    #                      "shared_prefix_single" | "vram_source_honest" | "n_requests_sufficient"
    passed: bool
    required: bool       # if True and not passed -> run is NOT quotable
    detail: str          # human-readable evidence string (grep match, value, actionable fix)
    evidence: dict       # machine-readable evidence (e.g. {"enable_prefix_caching": True})


@dataclass(frozen=True)
class ValidityReport:
    """Fold of every applicable check. ``quotable`` iff all required checks passed."""

    checks: list[ValidityCheck] = field(default_factory=list)
    quotable: bool = False
    summary: str = ""


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #
# vLLM logs the engine config on one line; ``enable_prefix_caching=<True|False>`` lives
# inside it (see logs_moe_run/vllm_*_crash.log, the core.py:93 init line). ANSI colour
# codes and a leading PID banner precede it, so a substring regex over the whole line is
# the robust read. Word boundary on the left avoids matching a longer flag name.
_APC_RE = re.compile(r"(?<![A-Za-z_])enable_prefix_caching\s*=\s*(True|False)")


def _failed(name: str, *, required: bool, detail: str, evidence: dict) -> ValidityCheck:
    return ValidityCheck(name=name, passed=False, required=required, detail=detail, evidence=evidence)


def _passed(name: str, *, required: bool, detail: str, evidence: dict) -> ValidityCheck:
    return ValidityCheck(name=name, passed=True, required=required, detail=detail, evidence=evidence)


# --------------------------------------------------------------------------- #
# §6.2 public gates                                                            #
# --------------------------------------------------------------------------- #
def check_apc_on(server_log_path: str) -> ValidityCheck:
    """§6(a): prove APC was actually ON. Grep the vLLM startup line for
    ``enable_prefix_caching=<True|False>``.

    The LAST occurrence in the log wins (the engine logs its final resolved config once;
    if a line ever repeats, the most recent reflects what actually ran). ``required=True``;
    ``passed`` iff the parsed value is ``True``. A missing file / missing flag is a LOUD
    fail with the exact path so the operator knows to re-launch with ``--enable-prefix-caching``
    and capture the server log."""
    name = "apc_on"
    path = Path(server_log_path)
    if not path.is_file():
        return _failed(
            name,
            required=True,
            detail=(
                f"server log not found at {server_log_path!r}: cannot prove APC was ON. "
                "Re-run the arm capturing the vLLM stdout/stderr to a file and pass its "
                "path here (the squeeze runner already tees to logs/gate0/<arm>_server.log)."
            ),
            evidence={"enable_prefix_caching": None, "server_log_path": server_log_path, "found": False},
        )

    try:
        text = path.read_text(errors="replace")
    except OSError as exc:  # unreadable file — surface it, do not silently pass.
        return _failed(
            name,
            required=True,
            detail=f"could not read server log {server_log_path!r}: {exc!r}",
            evidence={"enable_prefix_caching": None, "server_log_path": server_log_path, "found": False},
        )

    matches = list(_APC_RE.finditer(text))
    if not matches:
        return _failed(
            name,
            required=True,
            detail=(
                f"no 'enable_prefix_caching=<bool>' line in {server_log_path!r}. The vLLM "
                "engine-init line (core.py) was not captured — the server may have crashed "
                "before init, or only partial logs were saved. Capture the FULL startup log."
            ),
            evidence={"enable_prefix_caching": None, "server_log_path": server_log_path, "found": False},
        )

    last = matches[-1]
    enabled = last.group(1) == "True"
    # Reconstruct the matched line for the human-readable evidence string.
    line_start = text.rfind("\n", 0, last.start()) + 1
    line_end = text.find("\n", last.end())
    line = text[line_start: line_end if line_end != -1 else len(text)].strip()
    evidence = {
        "enable_prefix_caching": enabled,
        "server_log_path": server_log_path,
        "found": True,
        "line": line[:500],  # bound the stored evidence — the init line is huge.
    }
    if enabled:
        return _passed(
            name,
            required=True,
            detail=f"enable_prefix_caching=True in {server_log_path!r} (APC confound cleared).",
            evidence=evidence,
        )
    return _failed(
        name,
        required=True,
        detail=(
            f"enable_prefix_caching=False in {server_log_path!r}. Arm A is supposed to be the "
            "APC-ON floor; with APC off, delta=(B-A) would falsely credit ROMY with vLLM's "
            "native prefix caching. Re-launch this arm WITH --enable-prefix-caching."
        ),
        evidence=evidence,
    )


def check_aiter_parity(arm_launches: list["ArmLaunch"]) -> ValidityCheck:
    """Every arm must carry the SAME AITER env subset (else AITER, not ROMY, could explain
    the delta). ``required=True``. Compares only the AITER parity keys (see
    :func:`_aiter_parity_keys`); other env differences (e.g. B's LMCache vars) are expected
    and ignored here."""
    name = "aiter_parity"
    if not arm_launches:
        return _failed(
            name,
            required=True,
            detail="no arm launches supplied: cannot verify AITER parity across A/B/C.",
            evidence={"arms": []},
        )

    keys = _aiter_parity_keys()
    # Per-arm AITER subset (missing key -> sentinel so it shows up as a mismatch, not silently).
    per_arm: dict[str, dict[str, str]] = {}
    aiter_applied: dict[str, bool] = {}
    for launch in arm_launches:
        env = getattr(launch, "env", {}) or {}
        arm_id = getattr(launch, "arm", "?")
        per_arm[arm_id] = {k: env.get(k, "<unset>") for k in keys}
        aiter_applied[arm_id] = bool(getattr(launch, "aiter_applied", False))

    reference_arm = next(iter(per_arm))
    reference = per_arm[reference_arm]
    mismatches = {arm: subset for arm, subset in per_arm.items() if subset != reference}
    flags_consistent = len(set(aiter_applied.values())) <= 1

    evidence = {
        "keys": list(keys),
        "per_arm": per_arm,
        "aiter_applied": aiter_applied,
        "reference_arm": reference_arm,
    }
    if mismatches or not flags_consistent:
        return _failed(
            name,
            required=True,
            detail=(
                "AITER env differs across arms — AITER (not ROMY) could explain (B-A). "
                f"Reference arm {reference_arm!r} vs mismatching arms {sorted(mismatches)}; "
                f"aiter_applied flags={aiter_applied}. Rebuild all arms via arms.aiter_env() so "
                "the AITER subset is byte-identical."
            ),
            evidence=evidence,
        )
    return _passed(
        name,
        required=True,
        detail=f"AITER env identical across {sorted(per_arm)} (aiter_applied={aiter_applied[reference_arm]}).",
        evidence=evidence,
    )


def check_seed_pinned(arm_launches: list["ArmLaunch"]) -> ValidityCheck:
    """Every arm env must pin ``PYTHONHASHSEED=0`` (mandatory for cross-worker salt
    collision; vLLM keys prefix blocks by ``hash(cache_salt, token_ids, ...)`` and Python's
    per-process hash randomisation would make those keys disagree across workers).

    ``required=True`` if ANY arm is cross-worker topology; recommended (``required=False``)
    when all arms are single-worker. The canonical value is taken from the existing
    ``vllm_launch_config.worker_env()`` so this gate stays in lock-step with the launcher."""
    name = "seed_pinned"
    expected = worker_env().get("PYTHONHASHSEED", PYTHONHASHSEED_REQUIRED)
    if not arm_launches:
        return _failed(
            name,
            required=True,
            detail="no arm launches supplied: cannot verify PYTHONHASHSEED is pinned.",
            evidence={"expected": expected, "arms": []},
        )

    any_cross = any(
        getattr(launch, "topology", "") == _CROSS_TOPOLOGY for launch in arm_launches
    )
    required = any_cross

    per_arm: dict[str, str | None] = {}
    offenders: list[str] = []
    for launch in arm_launches:
        arm_id = getattr(launch, "arm", "?")
        seed = (getattr(launch, "env", {}) or {}).get("PYTHONHASHSEED")
        per_arm[arm_id] = seed
        if seed != expected:
            offenders.append(arm_id)

    evidence = {
        "expected": expected,
        "per_arm": per_arm,
        "any_cross_worker": any_cross,
    }
    if offenders:
        scope = "MANDATORY (cross-worker present)" if required else "recommended (single-worker only)"
        return _failed(
            name,
            required=required,
            detail=(
                f"PYTHONHASHSEED != {expected!r} in arms {offenders} — {scope}. Without a pinned "
                "seed, prefix-block hashes differ per worker and shared salts never collide "
                "cross-worker. Apply worker_env() to every arm subprocess."
            ),
            evidence=evidence,
        )
    return _passed(
        name,
        required=required,
        detail=f"PYTHONHASHSEED={expected!r} pinned on every arm {sorted(per_arm)}.",
        evidence=evidence,
    )


def check_c_control(prefix_metrics_c: "PrefixMetrics", *, max_hit_rate: float = DEFAULT_C_MAX_HIT_RATE) -> ValidityCheck:
    """Arm C (isolated per-request salts) must give ~0% cross-agent hit_rate. This is what
    proves the harness measures *sharing* and not just same-prompt APC: if C also hits, then
    a positive B is not attributable to ROMY's shared salts.

    ``required=True``; ``passed`` iff ``prefix_metrics_c.hit_rate <= max_hit_rate``. A missing
    metric or an upstream ``/metrics`` error fails LOUD (we cannot certify the control)."""
    name = "c_control_zero"
    if prefix_metrics_c is None:
        return _failed(
            name,
            required=True,
            detail=(
                "arm C prefix metrics missing: cannot certify the negative control. Run arm C "
                "and pass its PrefixMetrics so the harness can prove it measures sharing (~0% hit)."
            ),
            evidence={"hit_rate": None, "max_hit_rate": max_hit_rate, "found": False},
        )

    err = getattr(prefix_metrics_c, "error", None)
    if err:
        return _failed(
            name,
            required=True,
            detail=(
                f"arm C /metrics read errored ({err!r}): cannot certify the negative control. "
                "Fix the /metrics endpoint and re-run arm C."
            ),
            evidence={"hit_rate": None, "max_hit_rate": max_hit_rate, "error": str(err)},
        )

    hit_rate = getattr(prefix_metrics_c, "hit_rate", None)
    queries = getattr(prefix_metrics_c, "queries_delta", None)
    evidence = {
        "hit_rate": hit_rate,
        "max_hit_rate": max_hit_rate,
        "queries_delta": queries,
        "hits_delta": getattr(prefix_metrics_c, "hits_delta", None),
    }
    if hit_rate is None:
        return _failed(
            name,
            required=True,
            detail="arm C hit_rate is None: PrefixMetrics did not produce a rate. Cannot certify control.",
            evidence=evidence,
        )
    # A control that sent no queries proves nothing — flag it rather than passing on a vacuous 0.0.
    if queries is not None and queries <= 0:
        return _failed(
            name,
            required=True,
            detail=(
                "arm C recorded 0 prefix-cache queries: the control window saw no traffic, so a "
                "0.0 hit_rate is vacuous. Verify arm C actually drove the workload."
            ),
            evidence=evidence,
        )
    if hit_rate <= max_hit_rate:
        return _passed(
            name,
            required=True,
            detail=f"arm C hit_rate={hit_rate} <= {max_hit_rate} — harness measures sharing (control clean).",
            evidence=evidence,
        )
    return _failed(
        name,
        required=True,
        detail=(
            f"arm C hit_rate={hit_rate} > {max_hit_rate}: the negative control is HITTING. The harness "
            "is measuring same-prompt APC, not cross-agent sharing — a positive B would be uninterpretable. "
            "Check that arm C truly uses ISOLATED per-request salts (arms.salt_for_request ARM_C)."
        ),
        evidence=evidence,
    )


def check_shared_prefix_single(reuse: "ReuseStats") -> ValidityCheck:
    """The workload's prefix (excluding tails) must collapse to exactly ONE distinct prefix.
    If ``reuse.n_distinct_prefixes > 1`` the requests do not actually share a byte-identical
    prefix, so B ≈ A trivially and the gate is void. ``required=True``."""
    name = "shared_prefix_single"
    if reuse is None:
        return _failed(
            name,
            required=True,
            detail="reuse stats missing: cannot verify the workload shares a single prefix.",
            evidence={"n_distinct_prefixes": None, "found": False},
        )

    n_distinct = getattr(reuse, "n_distinct_prefixes", None)
    evidence = {
        "n_distinct_prefixes": n_distinct,
        "canonical_prefix_hash": getattr(reuse, "canonical_prefix_hash", None),
        "shared_prefix_fraction": getattr(reuse, "shared_prefix_fraction", None),
        "n_requests": getattr(reuse, "n_requests", None),
    }
    if n_distinct is None:
        return _failed(
            name,
            required=True,
            detail="reuse.n_distinct_prefixes is None: workload.measure_reuse did not count prefixes.",
            evidence=evidence,
        )
    if n_distinct == 1:
        return _passed(
            name,
            required=True,
            detail="workload collapses to 1 distinct prefix — shared-prefix assumption holds.",
            evidence=evidence,
        )
    return _failed(
        name,
        required=True,
        detail=(
            f"workload has {n_distinct} distinct prefixes (expected 1). Requests are NOT sharing a "
            "byte-identical prefix, so B would equal A trivially and the gate is void. Fix request "
            "assembly so PrefixNormalizer yields ONE canonical prefix across all agents/tails."
        ),
        evidence=evidence,
    )


def check_vram_source_honest(hbm: "HBMReading") -> ValidityCheck:
    """The HBM reading's ``vram_source`` must be a real backend
    (pyrsmi / drm_sysfs / cuda_nvml / cuda_nvidia_smi). The 192 GB default,
    ``cuda_unavailable``, ``unknown`` and ``dry`` are NOT honest and invalidate any quoted
    VRAM number (AUDIT #2). ``required=True`` for any quoted VRAM number."""
    name = "vram_source_honest"
    if hbm is None:
        return _failed(
            name,
            required=True,
            detail="HBM reading missing: cannot certify the VRAM source. No VRAM number is quotable.",
            evidence={"vram_source": None, "found": False, "honest_sources": sorted(HONEST_VRAM_SOURCES)},
        )

    source = getattr(hbm, "vram_source", None)
    # metrics.HBMReading already carries its own `valid` flag (False for the dishonest set);
    # we recompute from the source so this gate is self-contained and never trusts a stale flag.
    evidence = {
        "vram_source": source,
        "reading_valid_flag": getattr(hbm, "valid", None),
        "second_source": getattr(hbm, "second_source", None),
        "honest_sources": sorted(HONEST_VRAM_SOURCES),
    }
    if source in HONEST_VRAM_SOURCES:
        return _passed(
            name,
            required=True,
            detail=f"vram_source={source!r} is a real backend — VRAM numbers are quotable.",
            evidence=evidence,
        )
    # Everything not in the honest set is dishonest; name the AUDIT #2 trap explicitly.
    hint = (
        " This is the AUDIT #2 hardcoded 192 GB fallback — the total was NEVER read."
        if source == "amd_default_192gb"
        else ""
    )
    return _failed(
        name,
        required=True,
        detail=(
            f"vram_source={source!r} is NOT honest (honest set: {sorted(HONEST_VRAM_SOURCES)})."
            f"{hint} Discard this reading; no VRAM number may be quoted. Ensure VRAMMonitor reaches a "
            "real backend (pyrsmi on MI300X) before measuring."
        ),
        evidence=evidence,
    )


def check_n_requests(spec: "WorkloadSpec", *, min_requests: int = DEFAULT_MIN_REQUESTS) -> ValidityCheck:
    """Protocol §6(c): N must be large enough for a tight CI (explicitly NOT ~28).
    ``min_requests`` is a floor, not a target — the report still shows the achieved CI width.
    ``required=True``."""
    name = "n_requests_sufficient"
    if spec is None:
        return _failed(
            name,
            required=True,
            detail="workload spec missing: cannot verify N is sufficient for a tight CI.",
            evidence={"n_requests": None, "min_requests": min_requests, "found": False},
        )

    n_requests = getattr(spec, "n_requests", None)
    evidence = {"n_requests": n_requests, "min_requests": min_requests}
    if n_requests is None:
        return _failed(
            name,
            required=True,
            detail="spec.n_requests is None: cannot verify N is sufficient.",
            evidence=evidence,
        )
    if n_requests >= min_requests:
        return _passed(
            name,
            required=True,
            detail=f"n_requests={n_requests} >= floor {min_requests} — CI can be tight.",
            evidence=evidence,
        )
    return _failed(
        name,
        required=True,
        detail=(
            f"n_requests={n_requests} < floor {min_requests}: too few requests for a tight CI "
            "(protocol §6(c) explicitly forbids ~28). Increase --n-requests. The report headline "
            "is invalid below this floor."
        ),
        evidence=evidence,
    )


# --------------------------------------------------------------------------- #
# Fold                                                                         #
# --------------------------------------------------------------------------- #
def run_all(
    *,
    arm_launches: list["ArmLaunch"],
    reuse: "ReuseStats",
    spec: "WorkloadSpec",
    apc_log_paths: dict[str, str] | None = None,
    prefix_metrics_c: "PrefixMetrics" | None = None,
    hbm: "HBMReading" | None = None,
    min_requests: int = DEFAULT_MIN_REQUESTS,
    c_max_hit_rate: float = DEFAULT_C_MAX_HIT_RATE,
) -> ValidityReport:
    """Run every applicable check and fold into a :class:`ValidityReport`.

    ``apc_log_paths`` maps arm id -> server log path; :func:`check_apc_on` runs once per
    supplied log (typically arm A is the load-bearing one, but every arm should be APC-ON).
    Checks whose inputs are absent are SKIPPED with an explicit, non-passing note rather than
    silently dropped — the operator sees what could not be verified.

    ``quotable`` is ``all(required -> passed)`` over the checks that actually ran. A required
    check that was skipped (its input missing) appears as a failed (not passed) entry, so a
    run missing a required input is correctly NOT quotable."""
    checks: list[ValidityCheck] = []

    # (a) APC ON — one check per supplied server log.
    if apc_log_paths:
        for arm_id, log_path in apc_log_paths.items():
            base = check_apc_on(log_path)
            checks.append(
                ValidityCheck(
                    name=f"apc_on[{arm_id}]",
                    passed=base.passed,
                    required=base.required,
                    detail=base.detail,
                    evidence={**base.evidence, "arm": arm_id},
                )
            )
    else:
        checks.append(
            _failed(
                "apc_on",
                required=True,
                detail="no APC server logs supplied: APC-ON could not be verified for any arm.",
                evidence={"enable_prefix_caching": None, "found": False},
            )
        )

    # Confound guards across arms.
    checks.append(check_aiter_parity(arm_launches))
    checks.append(check_seed_pinned(arm_launches))

    # (b) reuse / negative control.
    checks.append(check_shared_prefix_single(reuse))
    if prefix_metrics_c is not None:
        checks.append(check_c_control(prefix_metrics_c, max_hit_rate=c_max_hit_rate))
    else:
        checks.append(
            _failed(
                "c_control_zero",
                required=True,
                detail="arm C prefix metrics not supplied: negative control could not be verified.",
                evidence={"hit_rate": None, "found": False},
            )
        )

    # (c) N sufficient.
    checks.append(check_n_requests(spec, min_requests=min_requests))

    # (d) VRAM honest — only when a VRAM number is in play (an HBM reading was captured).
    if hbm is not None:
        checks.append(check_vram_source_honest(hbm))
    # If no HBM reading was supplied we do NOT synthesize a failure: a throughput-only run
    # quotes no VRAM number, so VRAM honesty is not a required gate for it. The report's
    # primary-metric selection (analyze.py) decides whether a VRAM number is quoted at all.

    quotable = all(c.passed for c in checks if c.required)
    failed_required = [c.name for c in checks if c.required and not c.passed]
    if quotable:
        summary = f"all {sum(1 for c in checks if c.required)} required checks passed"
    else:
        summary = "NOT quotable — failed required checks: " + ", ".join(failed_required)

    return ValidityReport(checks=checks, quotable=quotable, summary=summary)


# --------------------------------------------------------------------------- #
# Tiny self-check (no GPU, no network) — run `python3 scripts/gate0/validity.py`.
# Exercises the parsers/gates against in-memory stand-ins so the plumbing is verifiable
# on any box. NOT a measurement; prints PASS/FAIL of the gates' own logic only.
# --------------------------------------------------------------------------- #
def _self_check() -> int:  # pragma: no cover — diagnostic harness, not the gate itself.
    import tempfile
    from types import SimpleNamespace

    failures = 0

    def expect(cond: bool, msg: str) -> None:
        nonlocal failures
        status = "PASS" if cond else "FAIL"
        if not cond:
            failures += 1
        print(f"  [{status}] {msg}")

    print("validity.py self-check (plumbing only, no numbers):")

    # APC ON: synthesize a vLLM-style init line for True and False.
    apc_line_true = (
        "\x1b[0;36m(EngineCore_DP0 pid=98)\x1b[0;0m INFO 05-26 [core.py:93] Initializing ... "
        "seed=0, served_model_name=qwen3-32b, enable_prefix_caching=True, enable_chunked_prefill=True ..."
    )
    apc_line_false = apc_line_true.replace("enable_prefix_caching=True", "enable_prefix_caching=False")
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f_true:
        f_true.write(apc_line_true + "\n")
        path_true = f_true.name
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f_false:
        f_false.write(apc_line_false + "\n")
        path_false = f_false.name
    expect(check_apc_on(path_true).passed, "check_apc_on True -> passed")
    expect(not check_apc_on(path_false).passed, "check_apc_on False -> failed")
    expect(not check_apc_on("/nonexistent/server.log").passed, "check_apc_on missing file -> failed")

    # AITER parity: identical AITER subset across arms passes; a diff fails.
    aiter = {**AITERConfig().AITER_ENV_VARS, "VLLM_USE_AITER": "1"}
    seed = {"PYTHONHASHSEED": "0"}
    arm_a = SimpleNamespace(arm="A", topology="single_worker", env={**aiter, **seed}, aiter_applied=True)
    arm_b = SimpleNamespace(arm="B", topology="single_worker", env={**aiter, **seed}, aiter_applied=True)
    arm_c = SimpleNamespace(arm="C", topology="single_worker", env={**aiter, **seed}, aiter_applied=True)
    expect(check_aiter_parity([arm_a, arm_b, arm_c]).passed, "check_aiter_parity identical -> passed")
    bad = SimpleNamespace(arm="B", topology="single_worker", env={**seed}, aiter_applied=False)
    expect(not check_aiter_parity([arm_a, bad]).passed, "check_aiter_parity diff -> failed")

    # Seed: pinned passes; missing on cross-worker fails (required), on single-worker recommended.
    expect(check_seed_pinned([arm_a, arm_b, arm_c]).passed, "check_seed_pinned pinned -> passed")
    no_seed_cross = SimpleNamespace(arm="A", topology="cross_worker", env={**aiter}, aiter_applied=True)
    chk = check_seed_pinned([no_seed_cross])
    expect((not chk.passed) and chk.required, "check_seed_pinned cross missing -> failed+required")
    no_seed_single = SimpleNamespace(arm="A", topology="single_worker", env={**aiter}, aiter_applied=True)
    chk = check_seed_pinned([no_seed_single])
    expect((not chk.passed) and (not chk.required), "check_seed_pinned single missing -> failed+recommended")

    # C control: ~0% passes, hitting fails, vacuous-0 (no queries) fails.
    pm_clean = SimpleNamespace(hit_rate=0.0, queries_delta=300.0, hits_delta=0.0, error=None)
    pm_hot = SimpleNamespace(hit_rate=0.42, queries_delta=300.0, hits_delta=126.0, error=None)
    pm_vacuous = SimpleNamespace(hit_rate=0.0, queries_delta=0.0, hits_delta=0.0, error=None)
    expect(check_c_control(pm_clean).passed, "check_c_control ~0% -> passed")
    expect(not check_c_control(pm_hot).passed, "check_c_control hitting -> failed")
    expect(not check_c_control(pm_vacuous).passed, "check_c_control no-queries -> failed")

    # Shared prefix single.
    expect(check_shared_prefix_single(SimpleNamespace(n_distinct_prefixes=1, canonical_prefix_hash="ab",
                                                       shared_prefix_fraction=1.0, n_requests=320)).passed,
           "check_shared_prefix_single ==1 -> passed")
    expect(not check_shared_prefix_single(SimpleNamespace(n_distinct_prefixes=3, canonical_prefix_hash="ab",
                                                          shared_prefix_fraction=0.4, n_requests=320)).passed,
           "check_shared_prefix_single >1 -> failed")

    # VRAM honesty.
    expect(check_vram_source_honest(SimpleNamespace(vram_source="pyrsmi", valid=True, second_source="rocm-smi")).passed,
           "check_vram_source_honest pyrsmi -> passed")
    expect(not check_vram_source_honest(SimpleNamespace(vram_source="amd_default_192gb", valid=False,
                                                        second_source=None)).passed,
           "check_vram_source_honest 192gb default -> failed")

    # N sufficient.
    expect(check_n_requests(SimpleNamespace(n_requests=320)).passed, "check_n_requests 320 -> passed")
    expect(not check_n_requests(SimpleNamespace(n_requests=28)).passed, "check_n_requests 28 -> failed")

    # Fold: a fully-clean run is quotable.
    report = run_all(
        arm_launches=[arm_a, arm_b, arm_c],
        reuse=SimpleNamespace(n_distinct_prefixes=1, canonical_prefix_hash="ab",
                              shared_prefix_fraction=1.0, n_requests=320),
        spec=SimpleNamespace(n_requests=320),
        apc_log_paths={"A": path_true, "B": path_true, "C": path_true},
        prefix_metrics_c=pm_clean,
        hbm=SimpleNamespace(vram_source="pyrsmi", valid=True, second_source="rocm-smi"),
    )
    expect(report.quotable, f"run_all clean -> quotable ({report.summary})")
    # Fold: a run missing arm C metrics is NOT quotable.
    report_bad = run_all(
        arm_launches=[arm_a, arm_b, arm_c],
        reuse=SimpleNamespace(n_distinct_prefixes=1, canonical_prefix_hash="ab",
                              shared_prefix_fraction=1.0, n_requests=320),
        spec=SimpleNamespace(n_requests=320),
        apc_log_paths={"A": path_true},
        prefix_metrics_c=None,
        hbm=None,
    )
    expect(not report_bad.quotable, f"run_all missing-C -> NOT quotable ({report_bad.summary})")

    Path(path_true).unlink(missing_ok=True)
    Path(path_false).unlink(missing_ok=True)

    print(f"\nself-check: {'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
