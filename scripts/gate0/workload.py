#!/usr/bin/env python3
"""GATE #0 — canonical N=5 shared-prefix workload + MEASURED reuse rate.

This module owns the workload definition for the A/B/C decision harness and the
*measured* prefix-reuse rate of that workload. It is pure: NO GPU, NO network,
NO vLLM / lmcache / torch imports (CONTRACT §1).

The workload is the realistic case ROMY claims to optimize: a pipeline of N=5
agents (retriever, reranker, summarizer, critic, responder) that each receive a
byte-identical shared system/context prefix, differing only in a per-request
``tail`` task line. Every request the harness sends is assembled through
:class:`apohara_context_forge.normalization.prefix_normalizer.PrefixNormalizer`
so the prefix is byte-identical across agents (the precondition for vLLM APC and
ROMY's cross-agent KV-block sharing to even be measurable).

Honesty (CONTRACT §1): the reuse rate is COMPUTED from the actually-built
requests via prefix hashing — never a fabricated literal. ``approx_prefix_tokens``
is a transparent ``chars/4`` heuristic carrying a ``note`` that says so; no
tokenizer is wired here, so we do not pretend to have one.

Reuse map (CONTRACT §2): this module wraps, never reinvents —
  * ``agents.demo_agents.AGENT_CONFIGS`` — canonical N=5 roles.
  * ``PrefixNormalizer`` — byte-identical prefix assembly + canonical hashing.
  * the inline ``SHARED_PREFIX`` / ``TAILS`` skeleton from
    ``scripts.mi300x_measure`` — the long multi-block briefing + variable tails.
  * ``apohara_context_forge.safety.jcr_gate.JUDGE_ROLES`` — drives ``is_judge``.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

# CONTRACT §1: identical import shape to the existing probes.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.demo_agents import AGENT_CONFIGS
from apohara_context_forge.normalization.prefix_normalizer import PrefixNormalizer
from apohara_context_forge.safety.jcr_gate import JUDGE_ROLES

# Reuse the long shared briefing + the variable task lines from the existing
# measurement probe instead of re-authoring them. mi300x_measure.SHARED_PREFIX
# is intentionally long (a 4x-repeated briefing) so the KV cache it produces
# spans many PagedAttention blocks — the regime where sharing is measurable.
from scripts.mi300x_measure import SHARED_PREFIX as _MEASURE_SHARED_PREFIX
from scripts.mi300x_measure import TAILS as _MEASURE_TAILS


# --------------------------------------------------------------------------- #
# Dataclasses (CONTRACT §3.1 — names are binding, other agents import them).   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgentSpec:
    """One agent in the N=5 pipeline. Mirrors agents.demo_agents.AGENT_CONFIGS."""

    agent_id: str            # "retriever" | "reranker" | "summarizer" | "critic" | "responder"
    role_prompt: str         # agent-specific role line (NOT the shared system prefix)
    is_judge: bool           # True for judge-class roles (drives INV-15 via the gate)


@dataclass(frozen=True)
class WorkloadRequest:
    """A single request the harness will send. Prefix is byte-identical across all."""

    request_id: str          # unique, e.g. "retriever:0007"
    agent_id: str
    prompt: str              # PrefixNormalizer-assembled: [system][SEP][role][SEP][tail]
    tail: str                # the variable per-request task line
    max_tokens: int


@dataclass(frozen=True)
class WorkloadSpec:
    """The full workload definition + its declared conditions (enter the report)."""

    name: str                       # e.g. "sprint5_5agent"
    model: str                      # served model name (full-attention dense, e.g. Qwen3-32B)
    canonical_system_prompt: str    # the shared byte-identical prefix (long, multi-block)
    agents: tuple[AgentSpec, ...]   # the N=5 specs
    n_requests: int                 # TOTAL requests across the run (LARGE; protocol forbids ~28)
    concurrency: int                # fixed concurrency for the throughput/footprint arms
    max_tokens: int
    seed: int = 0                   # request-order / tail-cycling seed for reproducibility


@dataclass(frozen=True)
class ReuseStats:
    """MEASURED prefix reuse of the workload (a condition, never assumed)."""

    canonical_prefix_chars: int          # len(canonical_system_prompt)
    canonical_prefix_hash: str           # PrefixNormalizer.get_canonical_hash()
    n_requests: int
    n_distinct_prefixes: int             # distinct normalized prefixes actually built
    shared_prefix_fraction: float        # share of requests on the dominant prefix (0..1)
    approx_prefix_tokens: int | None     # chars/4 heuristic if no tokenizer; None if unknown
    note: str                            # how it was computed (heuristic vs real tokenizer)


# --------------------------------------------------------------------------- #
# Module constants.                                                            #
# --------------------------------------------------------------------------- #
# Default variable task lines (one per agent role, in pipeline order). Shared
# with the mi300x_measure spirit; re-exported so harness/tests reference one set.
DEFAULT_TAILS: tuple[str, ...] = tuple(_MEASURE_TAILS)

# The canonical shared system/context prefix. We reuse the long multi-block
# briefing the existing probe already validated on a real model (it spans many
# 16-token blocks once 4x-repeated). Stripped so its hash is stable regardless
# of trailing whitespace (PrefixNormalizer strips too).
DEFAULT_CANONICAL_SYSTEM_PROMPT: str = _MEASURE_SHARED_PREFIX.strip()

# Default served model: a full-attention dense / GQA model where KV-sharing is
# MOST favorable (protocol §5). Overridable per run via load_workload(model=...).
DEFAULT_MODEL: str = "qwen3-32b"

# chars/4 is the transparent token-count heuristic used when no tokenizer is
# wired. Surfaced via ReuseStats.note so no one mistakes it for a real count.
_CHARS_PER_TOKEN_HEURISTIC = 4
_HEURISTIC_NOTE = "char/4 heuristic (no tokenizer wired)"


# --------------------------------------------------------------------------- #
# Internal helpers.                                                            #
# --------------------------------------------------------------------------- #
def _shared_prefix_of(prompt: str, canonical_system_prompt: str) -> str | None:
    """Return the canonical SYSTEM prefix actually present at the head of ``prompt``.

    ``build_requests`` assembles ``[system][SEP][role][SEP][tail]``. Across the
    N=5 pipeline the per-agent ``role`` and the per-request ``tail`` legitimately
    DIFFER — only the leading ``[system]`` block is byte-identical, and that block
    is exactly what vLLM APC (and ROMY's cross-agent sharing) keys its shared KV
    prefix on. So the "shared prefix" whose distinctness we measure is the
    canonical system block, NOT "everything before the tail" (that would include
    the role line and could never collapse to 1 for a real multi-role pipeline).

    Returns the leading canonical slice if the prompt starts with it (the valid
    case), else ``None`` so :func:`measure_reuse` counts it as a distinct/broken
    prefix instead of silently masking a normalization break.
    """
    if prompt.startswith(canonical_system_prompt):
        return prompt[: len(canonical_system_prompt)]
    return None


def _normalizer_for(spec: WorkloadSpec) -> PrefixNormalizer:
    """Build the PrefixNormalizer that enforces the byte-identical prefix."""
    return PrefixNormalizer(canonical_system_prompt=spec.canonical_system_prompt)


def _load_yaml_spec(
    path: Path,
    *,
    model: str,
    n_requests: int,
    concurrency: int,
    max_tokens: int,
    seed: int,
) -> WorkloadSpec:
    """Adapt an existing pipeline config YAML into a WorkloadSpec.

    The YAML (e.g. ``configs/sprint5_5agent.yaml``) does NOT exist in the repo
    today (CONTRACT §3.2); this path exists so a workload author can point the
    harness at one without code changes. We adapt defensively: any field the
    YAML omits falls back to the default derived from ``default_agents()`` /
    ``DEFAULT_CANONICAL_SYSTEM_PROMPT``, so a partial config still yields a valid
    shared-prefix workload.

    Recognised keys (all optional):
      ``name``                     -> WorkloadSpec.name
      ``model``                    -> overrides the model arg only if arg is falsy
      ``canonical_system_prompt``  -> the shared prefix (else DEFAULT)
      ``system_prompt``            -> alias for canonical_system_prompt
      ``agents``: list of {id|agent_id, role|role_prompt} -> AgentSpec tuple
    """
    import yaml  # local import: keeps the no-YAML default path import-light

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"workload YAML {path} must be a mapping, got {type(raw)!r}")

    canonical = (
        raw.get("canonical_system_prompt")
        or raw.get("system_prompt")
        or DEFAULT_CANONICAL_SYSTEM_PROMPT
    ).strip()

    yaml_agents = raw.get("agents")
    if yaml_agents:
        agents = tuple(
            AgentSpec(
                agent_id=str(a.get("id") or a.get("agent_id")),
                role_prompt=str(a.get("role") or a.get("role_prompt") or ""),
                is_judge=(str(a.get("id") or a.get("agent_id")).lower() in JUDGE_ROLES),
            )
            for a in yaml_agents
        )
    else:
        agents = default_agents()

    return WorkloadSpec(
        name=str(raw.get("name") or path.stem),
        model=model or str(raw.get("model") or DEFAULT_MODEL),
        canonical_system_prompt=canonical,
        agents=agents,
        n_requests=n_requests,
        concurrency=concurrency,
        max_tokens=max_tokens,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Public API (CONTRACT §3.2).                                                  #
# --------------------------------------------------------------------------- #
def default_agents() -> tuple[AgentSpec, ...]:
    """The N=5 specs derived from agents.demo_agents.AGENT_CONFIGS.

    ``is_judge`` is decided by membership in ``jcr_gate.JUDGE_ROLES``
    (``{"critic", "judge"}`` today) — so the N=5 demo pipeline marks ONLY
    ``critic`` as a judge. We do NOT hardcode ``responder`` as a judge; the gate
    owns INV-15 ownership and JUDGE_ROLES is the single source of truth.
    """
    return tuple(
        AgentSpec(
            agent_id=cfg["id"],
            role_prompt=cfg["role"],
            is_judge=cfg["id"].lower() in JUDGE_ROLES,
        )
        for cfg in AGENT_CONFIGS
    )


def load_workload(
    path: str | None = None,
    *,
    model: str,
    n_requests: int,
    concurrency: int,
    max_tokens: int = 64,
    seed: int = 0,
) -> WorkloadSpec:
    """Build the canonical WorkloadSpec.

    If ``path`` is given AND exists, load/adapt it as YAML (see ``_load_yaml_spec``;
    note ``configs/sprint5_5agent.yaml`` does NOT exist today — authoring it is a
    workload-author decision, not this module's job). If ``path`` is None (or does
    not exist), derive the spec from :func:`default_agents` plus the long
    ``DEFAULT_CANONICAL_SYSTEM_PROMPT`` multi-block briefing.

    The numeric run knobs (``n_requests``, ``concurrency``, ``max_tokens``,
    ``seed``) ALWAYS come from the caller — never from the YAML — so the harness
    CLI is the single source of truth for run size (protocol §6: N must be large,
    not ~28).
    """
    if path:
        p = Path(path)
        if p.exists():
            return _load_yaml_spec(
                p,
                model=model,
                n_requests=n_requests,
                concurrency=concurrency,
                max_tokens=max_tokens,
                seed=seed,
            )
        # path given but missing: fall through to the derived default rather than
        # raising, so a stale --workload arg degrades gracefully to the canonical
        # workload (the harness still records spec.name for traceability).

    return WorkloadSpec(
        name="sprint5_5agent",
        model=model or DEFAULT_MODEL,
        canonical_system_prompt=DEFAULT_CANONICAL_SYSTEM_PROMPT,
        agents=default_agents(),
        n_requests=n_requests,
        concurrency=concurrency,
        max_tokens=max_tokens,
        seed=seed,
    )


def build_requests(spec: WorkloadSpec) -> list[WorkloadRequest]:
    """Materialize spec.n_requests requests.

    Every prompt is assembled via :class:`PrefixNormalizer` seeded with
    ``spec.canonical_system_prompt`` so EVERY request shares a byte-identical
    system prefix. The per-request ``tail`` cycles deterministically over
    :data:`DEFAULT_TAILS`, and ``agent_id`` cycles over ``spec.agents`` — both
    keyed off ``spec.seed`` so the request list is fully reproducible. The SAME
    list is replayed across arms A/B/C; only the cache_salt differs (arms.py
    decides that, never this module).

    Determinism: we do NOT use ``random`` — the cycling is a pure function of the
    request index, the seed, and the fixed agent/tail tuples, so two runs with the
    same ``spec`` produce byte-identical request lists on any machine.
    """
    if spec.n_requests <= 0:
        return []

    normalizer = _normalizer_for(spec)
    agents = spec.agents
    tails = DEFAULT_TAILS
    n_agents = len(agents)
    n_tails = len(tails)
    if n_agents == 0 or n_tails == 0:
        raise ValueError("workload needs at least one agent and one tail")

    # The seed offsets the deterministic cycle start. Using distinct multipliers
    # for the agent and tail cycles (1 vs +seed) avoids agent/tail phase-locking
    # for non-zero seeds while staying fully deterministic.
    per_agent = max(1, spec.n_requests // n_agents)  # zero-padding width helper
    pad = max(4, len(str(per_agent)))

    requests: list[WorkloadRequest] = []
    # Per-agent monotonically increasing index for stable, unique request ids.
    agent_seq: Counter[str] = Counter()
    for i in range(spec.n_requests):
        agent = agents[(i + spec.seed) % n_agents]
        tail = tails[(i + spec.seed) % n_tails]
        seq = agent_seq[agent.agent_id]
        agent_seq[agent.agent_id] += 1
        request_id = f"{agent.agent_id}:{seq:0{pad}d}"

        # PrefixNormalizer enforces [system][SEP][role][SEP][user] with the system
        # part byte-identical across every request. The tail is the user_prompt.
        prompt = normalizer.normalize(
            agent_id=agent.agent_id,
            user_prompt=tail,
            agent_role_prompt=agent.role_prompt,
        )
        requests.append(
            WorkloadRequest(
                request_id=request_id,
                agent_id=agent.agent_id,
                prompt=prompt,
                tail=tail,
                max_tokens=spec.max_tokens,
            )
        )
    return requests


def measure_reuse(spec: WorkloadSpec, requests: list[WorkloadRequest]) -> ReuseStats:
    """Compute the REAL reuse rate of the built requests.

    The shared "prefix" of a request is its canonical SYSTEM block — the
    byte-identical ``[system]`` head that every agent's prompt starts with. The
    per-agent ``role`` line and per-request ``tail`` legitimately differ across
    the N=5 pipeline, so they are NOT part of the shared prefix; vLLM APC (and
    ROMY's cross-agent sharing) keys its reusable KV blocks on exactly this
    leading system run. For a valid workload that block MUST collapse to ONE
    distinct value across all requests (CONTRACT §3 invariant;
    ``validity.check_shared_prefix_single`` enforces ``n_distinct_prefixes == 1``).
    We measure this by hashing the leading canonical slice actually present in
    each built prompt and counting distinct hashes — nothing is assumed; a prompt
    that does NOT start with the canonical block is counted as its own distinct
    (broken) prefix so a normalization regression surfaces instead of hiding.

    ``shared_prefix_fraction`` is the share of requests on the single most common
    prefix (1.0 when the workload is correct). ``approx_prefix_tokens`` is a
    transparent ``chars/4`` heuristic over the canonical prefix, with a ``note``
    saying so; no tokenizer is wired, so we never claim a real token count.
    """
    normalizer = _normalizer_for(spec)
    canonical_chars = len(spec.canonical_system_prompt)
    canonical_hash = normalizer.get_canonical_hash()

    n = len(requests)
    if n == 0:
        return ReuseStats(
            canonical_prefix_chars=canonical_chars,
            canonical_prefix_hash=canonical_hash,
            n_requests=0,
            n_distinct_prefixes=0,
            shared_prefix_fraction=0.0,
            approx_prefix_tokens=_approx_tokens(canonical_chars),
            note=_HEURISTIC_NOTE,
        )

    # Hash the canonical SYSTEM prefix actually present at the head of each
    # request. Distinct hashes == distinct shared prefixes actually built. A
    # prompt missing the canonical head hashes a unique "broken:<request_id>"
    # marker so it cannot silently merge with the valid bucket.
    def _key(req: WorkloadRequest) -> str:
        head = _shared_prefix_of(req.prompt, spec.canonical_system_prompt)
        if head is None:
            return "broken:" + req.request_id
        return hashlib.sha256(head.encode("utf-8")).hexdigest()

    prefix_hashes = Counter(_key(req) for req in requests)
    n_distinct = len(prefix_hashes)
    dominant_count = prefix_hashes.most_common(1)[0][1]
    shared_fraction = dominant_count / n

    return ReuseStats(
        canonical_prefix_chars=canonical_chars,
        canonical_prefix_hash=canonical_hash,
        n_requests=n,
        n_distinct_prefixes=n_distinct,
        shared_prefix_fraction=round(shared_fraction, 4),
        approx_prefix_tokens=_approx_tokens(canonical_chars),
        note=_HEURISTIC_NOTE,
    )


def _approx_tokens(chars: int) -> int | None:
    """chars/4 token heuristic. None only if the prefix is empty (unknown)."""
    if chars <= 0:
        return None
    return chars // _CHARS_PER_TOKEN_HEURISTIC


# --------------------------------------------------------------------------- #
# Self-check: `python3 scripts/gate0/workload.py` prints the measured reuse    #
# of the default workload. Numbers are COMPUTED, never hardcoded.              #
# --------------------------------------------------------------------------- #
def _selfcheck(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Build + measure the default GATE #0 workload.")
    ap.add_argument("--workload", default=None, help="optional YAML path (may not exist yet)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-requests", type=int, default=320)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    spec = load_workload(
        args.workload,
        model=args.model,
        n_requests=args.n_requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    requests = build_requests(spec)
    reuse = measure_reuse(spec, requests)

    summary = {
        "name": spec.name,
        "model": spec.model,
        "n_agents": len(spec.agents),
        "agents": [a.agent_id for a in spec.agents],
        "judges": [a.agent_id for a in spec.agents if a.is_judge],
        "n_requests_built": len(requests),
        "reuse": {
            "canonical_prefix_chars": reuse.canonical_prefix_chars,
            "canonical_prefix_hash": reuse.canonical_prefix_hash[:16],
            "n_distinct_prefixes": reuse.n_distinct_prefixes,
            "shared_prefix_fraction": reuse.shared_prefix_fraction,
            "approx_prefix_tokens": reuse.approx_prefix_tokens,
            "note": reuse.note,
        },
        "shared_prefix_valid": reuse.n_distinct_prefixes == 1,
        "sample_request_ids": [r.request_id for r in requests[:6]],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if reuse.n_distinct_prefixes == 1 else 1


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
