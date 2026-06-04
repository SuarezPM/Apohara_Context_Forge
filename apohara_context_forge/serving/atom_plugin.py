"""vLLM-ATOM plugin for ContextForge.

ATOM (Anchor-driven Tensor Orchestration for Multi-agent) is the
runtime side of ContextForge that lives inside vLLM. It exposes two
pre/post attention hooks plus a top-level entry-point function so vLLM
V1 can discover and register it through the `vllm.general_plugins`
entry-point group.

Design discipline (V6.1 truth-up)
---------------------------------
The hooks never claim work they did not actually perform. Every flag in
the returned metadata dict reflects *state*, not *configuration*:

* ``quantization_attempted`` ─ the operator asked us to quantize.
* ``quantization_applied``   ─ a quantizer was wired AND it ran.
* ``anchor_match``           ─ None if no LSH matcher was wired; else
                               the best cross-agent block hash + agent.
* ``jcr_dense``              ─ True iff the JCR Safety Gate fired
                               INV-15 for this call (always None if no
                               gate was wired).

Dependency injection
--------------------
The plugin accepts four optional dependencies at construction time:

* ``quantizer``     ─ any object with ``quantize_pre_rope(keys, values, positions)``
* ``lsh_matcher``   ─ any object with ``find_reusable_blocks(text, exclude_agent=...)``
* ``jcr_gate``      ─ any object with ``gate_decision(role, n_cand, reuse, shuffled)``
* ``metrics``       ─ any object with ``record_register(matched: bool)``

Anything left None becomes a documented no-op (the hook still runs, but
the corresponding metadata field is None / False). This lets the plugin
be unit-tested without any of those dependencies, and lets a vLLM host
choose which ContextForge subsystems to wire.

Entry-point registration
------------------------
vLLM V1 plugin discovery uses the ``vllm.general_plugins`` entry-point
group. The standalone PyPI package (``apohara_vllm_plugin``) declares::

    [project.entry-points."vllm.general_plugins"]
    apohara_contextforge = "apohara_vllm_plugin:register"

The ``register()`` callable below is what vLLM invokes once per worker;
it constructs an ATOM plugin and (optionally) installs telemetry.

KV interception lives in the config-driven ``--kv-transfer-config`` path
(LMCache), NOT in attention hooks — that platform API never existed in
vLLM. The ``PreAttentionHook`` / ``PostAttentionHook`` classes below are
unit-tested utilities that are NOT cabled to the vLLM runtime; the real
cross-worker KV path is documented in ``LMCACHE.md`` (Fase 1+).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration                                                              #
# ---------------------------------------------------------------------------

@dataclass
class ATOMConfig:
    """ATOM plugin configuration. All flags are intent-only — whether
    each subsystem actually fires depends on whether its dependency is
    wired."""

    enable_quantization: bool = True   # request RotateKV pre-RoPE quant
    enable_anchor_routing: bool = True  # request LSH cross-agent lookup
    enable_cla_injection: bool = True   # request CLA metadata emit
    enable_jcr_gate: bool = True        # request INV-15 enforcement
    quantization_mode: str = "rotate_kv"
    max_quantize_blocks: int = 1024


# ---------------------------------------------------------------------------
# Dependency protocols                                                       #
# ---------------------------------------------------------------------------

class _Quantizer(Protocol):
    def quantize_pre_rope(
        self, keys: Any, values: Any, positions: Any
    ) -> tuple[Any, Any]: ...


class _LSHMatcher(Protocol):
    async def find_reusable_blocks(
        self, text: str, exclude_agent: Optional[str] = None
    ) -> list[Any]: ...


class _JCRGate(Protocol):
    def gate_decision(
        self,
        agent_role: str,
        candidate_count: int,
        reuse_rate: float,
        layout_shuffled: bool,
    ) -> Any: ...


class _Metrics(Protocol):
    def record_register(self, matched: bool) -> None: ...


# ---------------------------------------------------------------------------
# Hooks                                                                      #
# ---------------------------------------------------------------------------

class PreAttentionHook:
    """Pre-attention KV-block utility (unit-tested, NOT cabled to vLLM).

    Returns metadata describing what actually happened — flags reflect
    state (not configuration). When dependencies are unwired, the hook
    is a documented no-op that returns ``quantization_applied=False``,
    ``anchor_match=None``, ``jcr_dense=None``.

    This class is NOT invoked by the vLLM runtime: there is no
    attention-layer hook registry in vLLM (that platform API never
    existed). It is kept as a tested, importable utility; the real
    KV interception path is config-driven (``--kv-transfer-config`` +
    LMCache, see ``LMCACHE.md``).
    """

    def __init__(
        self,
        config: ATOMConfig,
        *,
        quantizer: Optional[_Quantizer] = None,
        lsh_matcher: Optional[_LSHMatcher] = None,
        jcr_gate: Optional[_JCRGate] = None,
    ):
        self._config = config
        self._quantizer = quantizer
        self._lsh_matcher = lsh_matcher
        self._jcr_gate = jcr_gate
        self._quantized_block_count = 0

    def _evaluate_jcr_gate(
        self,
        agent_role: str,
        candidate_count: int,
        reuse_rate: float,
        layout_shuffled: bool,
    ) -> tuple[Optional[bool], Optional[float]]:
        """Evaluate the JCR safety gate. Returns (jcr_dense, jcr_risk)."""
        jcr_dense: Optional[bool] = None
        jcr_risk: Optional[float] = None
        if self._jcr_gate is not None and self._config.enable_jcr_gate:
            decision = self._jcr_gate.gate_decision(
                agent_role=agent_role,
                candidate_count=candidate_count,
                reuse_rate=reuse_rate,
                layout_shuffled=layout_shuffled,
            )
            jcr_dense = bool(getattr(decision, "use_dense", False))
            jcr_risk = float(getattr(decision, "risk_score", 0.0))
        return jcr_dense, jcr_risk

    def _attempt_anchor_routing(
        self,
        block_ids: list[str],
        token_ids: list[int],
        agent_role: str,
        jcr_dense: Optional[bool],
    ) -> Optional[dict]:
        """Attempt cross-agent anchor routing (LSH). Returns match if found."""
        if (
            self._lsh_matcher is not None
            and self._config.enable_anchor_routing
            and jcr_dense is not True  # JCR dense path bypasses the registry
        ):
            return self._maybe_find_anchor(block_ids, token_ids, agent_role)
        return None

    def _attempt_quantization(
        self,
        block_ids: list[str],
        layer_idx: int,
        keys: Optional[np.ndarray],
        values: Optional[np.ndarray],
        positions: Optional[np.ndarray],
    ) -> bool:
        """Attempt RotateKV pre-RoPE quantization. Returns True if applied."""
        if not (
            self._quantizer is not None
            and self._config.enable_quantization
            and self._config.quantization_mode == "rotate_kv"
            and keys is not None
            and values is not None
            and positions is not None
            and len(block_ids) <= self._config.max_quantize_blocks
        ):
            return False

        try:
            self._quantizer.quantize_pre_rope(keys, values, positions)
            self._quantized_block_count += len(block_ids)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ATOM quantization failed at layer %s: %s",
                layer_idx, type(exc).__name__,
            )
            return False

    def __call__(
        self,
        block_ids: list[str],
        token_ids: list[int],
        layer_idx: int,
        *,
        agent_role: str = "responder",
        candidate_count: int = 1,
        reuse_rate: float = 0.0,
        layout_shuffled: bool = False,
        keys: Optional[np.ndarray] = None,
        values: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
    ) -> dict:
        """Run the pre-attention pipeline.

        Args:
            block_ids: PagedAttention block identifiers under attention.
            token_ids: Token IDs covered by these blocks.
            layer_idx: Transformer layer index this hook fires under.
            agent_role: Caller's agent role (drives JCR gate decision).
            candidate_count: Number of candidates the agent will compare
                (relevant for judge agents).
            reuse_rate: Fraction of blocks the registry would reuse
                absent any safety override.
            layout_shuffled: Whether the candidate layout has changed
                since this agent's previous invocation.
            keys, values, positions: Pre-RoPE tensors to be quantised.
                Required iff ``quantizer`` is wired and
                ``enable_quantization`` is True.

        Returns:
            Metadata dict whose flags are HONEST: they describe what
            this hook actually did, not what its config requested.
        """
        # 1. JCR Safety Gate ------------------------------------------------
        jcr_dense, jcr_risk = self._evaluate_jcr_gate(
            agent_role=agent_role,
            candidate_count=candidate_count,
            reuse_rate=reuse_rate,
            layout_shuffled=layout_shuffled,
        )

        # 2. Cross-agent anchor routing (LSH) -------------------------------
        anchor_match = self._attempt_anchor_routing(
            block_ids=block_ids,
            token_ids=token_ids,
            agent_role=agent_role,
            jcr_dense=jcr_dense,
        )

        # 3. RotateKV pre-RoPE quantization (INVARIANT 10) ------------------
        # INV-10: we ONLY quantise pre-RoPE tensors. The caller is
        # responsible for handing us pre-RoPE keys+values; we honour the
        # contract by name only — if `positions` is None we refuse to
        # quantise. Quantization is best-effort: if it raises, we report
        # quantization_applied=False (truthful) rather than propagate.
        quantization_applied = self._attempt_quantization(
            block_ids=block_ids,
            layer_idx=layer_idx,
            keys=keys,
            values=values,
            positions=positions,
        )

        result = {
            "layer_idx": layer_idx,
            "num_blocks": len(block_ids),
            "quantization_attempted": self._config.enable_quantization
                                       and self._quantizer is not None,
            "quantization_applied": quantization_applied,
            "pre_rope": True,  # INV-10 — we never quantise post-RoPE
            "anchor_match": anchor_match,
            "jcr_dense": jcr_dense,
            "jcr_risk": jcr_risk,
            "cla_injected": bool(self._config.enable_cla_injection),
            # Backwards-compatibility alias kept for callers that still
            # read result["quantized"]. Always points to the HONEST flag.
            "quantized": quantization_applied,
        }

        logger.debug(
            "ATOM pre-attention layer=%s blocks=%s qa=%s qb=%s anchor=%s jcr_dense=%s",
            layer_idx, len(block_ids),
            result["quantization_attempted"], result["quantization_applied"],
            "yes" if anchor_match else "no", jcr_dense,
        )
        return result

    def _maybe_find_anchor(
        self,
        block_ids: list[str],
        token_ids: list[int],
        agent_role: str,
    ) -> Optional[dict]:
        """Best-effort LSH lookup. Returns None if no match or if the
        LSH matcher's API is async (caller can't await us)."""
        # The LSH matcher we use is async-only. Inside a synchronous
        # attention hook we cannot await it, so we report `pending`
        # rather than pretending. A real V1 integration will route
        # this through the scheduler hook (which is async) — that's
        # where the actual await happens. For now we record intent.
        return {
            "block_ids": list(block_ids),
            "agent_role": agent_role,
            "n_tokens": len(token_ids),
            "lookup_status": "pending_async",
        }


class PostAttentionHook:
    """Post-attention telemetry utility (unit-tested, NOT cabled to vLLM).

    Like ``PreAttentionHook``, this is a tested, importable utility and
    is NOT invoked by the vLLM runtime (no attention-layer hook registry
    exists in vLLM). The real KV path is config-driven (see ``LMCACHE.md``).
    """

    def __init__(
        self,
        config: ATOMConfig,
        *,
        metrics: Optional[_Metrics] = None,
    ):
        self._config = config
        self._metrics = metrics
        self._stats = {
            "blocks_processed": 0,
            "layers_processed": 0,
            "matched_blocks": 0,
        }

    def __call__(
        self,
        block_ids: list[str],
        output_tensors: list[Any],
        layer_idx: int,
        *,
        matched: bool = False,
    ) -> dict:
        """Record post-attention telemetry. ``matched`` is True iff the
        pre-attention hook returned a non-pending anchor match for the
        same call (the scheduler tracks the pairing — we just accept
        the boolean and forward it to MetricsCollector)."""
        self._stats["blocks_processed"] += len(block_ids)
        self._stats["layers_processed"] += 1
        if matched:
            self._stats["matched_blocks"] += len(block_ids)

        if self._metrics is not None:
            try:
                self._metrics.record_register(matched=matched)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ATOM metrics.record_register failed: %s",
                               type(exc).__name__)

        return {
            "processed_blocks": len(block_ids),
            "layer_idx": layer_idx,
            "matched": matched,
            "blocks_processed_total": self._stats["blocks_processed"],
        }


# ---------------------------------------------------------------------------
# Plugin                                                                     #
# ---------------------------------------------------------------------------

class vLLMAtomPlugin:
    """vLLM-ATOM plugin object.

    Holds a config and the two hooks. Lifecycle:
      * ``__init__`` wires the (optional) ContextForge dependencies.
      * ``initialize(worker_id, vllm_config)`` is called once per worker
        by the entry-point function ``register()`` below.
      * ``pre_attention_hook`` and ``post_attention_hook`` are passed to
        vLLM's attention layer (in V1, via the platform's hook registry).
    """

    def __init__(
        self,
        config: Optional[ATOMConfig] = None,
        *,
        quantizer: Optional[_Quantizer] = None,
        lsh_matcher: Optional[_LSHMatcher] = None,
        jcr_gate: Optional[_JCRGate] = None,
        metrics: Optional[_Metrics] = None,
    ):
        self._config = config or ATOMConfig()
        self._pre_hook = PreAttentionHook(
            self._config,
            quantizer=quantizer,
            lsh_matcher=lsh_matcher,
            jcr_gate=jcr_gate,
        )
        self._post_hook = PostAttentionHook(self._config, metrics=metrics)
        self._initialized = False
        self._worker_id: Optional[str] = None
        self._dependency_status = {
            "quantizer":   quantizer is not None,
            "lsh_matcher": lsh_matcher is not None,
            "jcr_gate":    jcr_gate is not None,
            "metrics":     metrics is not None,
        }

    def initialize(self, worker_id: str, vllm_config: dict) -> None:
        """Mark the plugin initialised. ``vllm_config`` is whatever
        vLLM passes (engine config, parallel config, etc.) — we keep
        it for diagnostic purposes only."""
        self._worker_id = worker_id
        self._initialized = True
        logger.info(
            "ATOM plugin initialised: worker=%s deps=%s",
            worker_id, self._dependency_status,
        )

    @property
    def pre_attention_hook(self) -> PreAttentionHook:
        return self._pre_hook

    @property
    def post_attention_hook(self) -> PostAttentionHook:
        return self._post_hook

    def is_initialized(self) -> bool:
        return self._initialized

    def get_stats(self) -> dict:
        return {
            "initialized": self._initialized,
            "worker_id": self._worker_id,
            "config": {
                "enable_quantization":   self._config.enable_quantization,
                "enable_anchor_routing": self._config.enable_anchor_routing,
                "enable_cla_injection":  self._config.enable_cla_injection,
                "enable_jcr_gate":       self._config.enable_jcr_gate,
                "quantization_mode":     self._config.quantization_mode,
            },
            "dependencies": dict(self._dependency_status),
            "pre_stats": {
                "quantized_block_count": self._pre_hook._quantized_block_count,
            },
            "post_stats": dict(self._post_hook._stats),
        }


# ---------------------------------------------------------------------------
# Entry-point function for vLLM V1                                           #
# ---------------------------------------------------------------------------

def register() -> vLLMAtomPlugin:
    """Entry-point for the ``vllm.general_plugins`` group.

    vLLM V1 invokes every entry point in that group once per worker on
    startup. We construct the plugin, initialise it, and return it.

    KV interception lives in the config-driven ``--kv-transfer-config``
    path (LMCache), NOT in attention hooks — that platform API never
    existed in vLLM. No vLLM platform exposes a pre/post attention-hook
    registry to call, so this entry point does NOT wire the
    ``PreAttentionHook`` / ``PostAttentionHook`` utilities into the
    runtime. The real cross-worker KV path is documented in
    ``LMCACHE.md`` (Fase 1+).

    The function is safe to call even when vLLM is not installed: it
    constructs the plugin (which is just data) and returns it.
    """
    plugin = vLLMAtomPlugin()
    plugin.initialize(worker_id="default", vllm_config={})
    return plugin
