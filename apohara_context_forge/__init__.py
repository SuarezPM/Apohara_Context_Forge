"""ContextForge - Shared context compiler for multi-agent LLM systems on AMD MI300X.

The top-level package is now LIGHTWEIGHT. Heavy-dep modules (vllm, gradio,
sentence-transformers, faiss, torch, plotly) are NOT imported at top level;
they live in their own submodules and are loaded lazily on first use. This
makes `import apohara_context_forge` (and `import apohara_context_forge.safety`)
safe in slim installs (no [serve] extra required).

For backward compatibility, the most common names are re-exported via
`__getattr__` (PEP 562) so existing code like
`from apohara_context_forge import FAISSContextIndex` still works — it just
triggers the heavy import on first access.

Install contract:
- `pip install apohara-context-forge` — slim, safety + core only
- `pip install apohara-context-forge[serve]` — full (torch, vllm, etc.)

This change is the Phase 3 (real) split per apohara-probanza's
APOHARA_CONSOLIDATION_BRIEF.md. See docs/upstream/context-forge-slim-pr.md
in that monorepo for the PR description.
"""
from __future__ import annotations

__version__ = "6.2.0"

# Safety kernel is the only thing eagerly imported — it's the slim-install
# value proposition and is stdlib + z3 only.
from apohara_context_forge.safety import (  # noqa: E402, F401
    JCRDecision,
    JCRSafetyGate,
    JUDGE_ROLES,
    DEFAULT_JCR_THRESHOLD,
)

# Lazy re-exports for backward compatibility. PEP 562 module-level __getattr__.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    # name -> (submodule, attr_name)
    "ContextRegistry": ("apohara_context_forge.registry.context_registry", "ContextRegistry"),
    "SharedContextResult": ("apohara_context_forge.registry.context_registry", "SharedContextResult"),
    "RegisteredAgent": ("apohara_context_forge.registry.context_registry", "RegisteredAgent"),
    "PipelineConfig": ("apohara_context_forge.pipeline_config", "PipelineConfig"),
    "TokenCounter": ("apohara_context_forge.token_counter", "TokenCounter"),
    "count_tokens": ("apohara_context_forge.token_counter", "count_tokens"),
    "encode_tokens": ("apohara_context_forge.token_counter", "encode_tokens"),
    "compute_kv_gb": ("apohara_context_forge.token_counter", "compute_kv_gb"),
    "VRAMMonitor": ("apohara_context_forge.metrics.vram_monitor", "VRAMMonitor"),
    "get_monitor": ("apohara_context_forge.metrics.vram_monitor", "get_monitor"),
    "get_vram_pressure": ("apohara_context_forge.metrics.vram_monitor", "get_vram_pressure"),
    "LSHTokenMatcher": ("apohara_context_forge.dedup.lsh_engine", "LSHTokenMatcher"),
    "TokenBlockMatch": ("apohara_context_forge.dedup.lsh_engine", "TokenBlockMatch"),
    "FAISSContextIndex": ("apohara_context_forge.dedup.faiss_index", "FAISSContextIndex"),
    "FAISSMatch": ("apohara_context_forge.dedup.faiss_index", "FAISSMatch"),
    "VRAMAwareCache": ("apohara_context_forge.registry.vram_aware_cache", "VRAMAwareCache"),
    "EvictionMode": ("apohara_context_forge.registry.vram_aware_cache", "EvictionMode"),
}


def __getattr__(name: str):  # PEP 562
    if name in _LAZY_EXPORTS:
        import importlib
        submodule_path, attr_name = _LAZY_EXPORTS[name]
        try:
            submodule = importlib.import_module(submodule_path)
        except ImportError as exc:
            raise ImportError(
                f"apohara_context_forge.{name} requires the [serve] extra "
                f"(install with `pip install apohara-context-forge[serve]`). "
                f"Underlying error: {exc}"
            ) from exc
        value = getattr(submodule, attr_name)
        globals()[name] = value  # cache
        return value
    raise AttributeError(f"module 'apohara_context_forge' has no attribute {name!r}")


__all__ = [
    # Always available (slim)
    "JCRDecision",
    "JCRSafetyGate",
    "JUDGE_ROLES",
    "DEFAULT_JCR_THRESHOLD",
    # Lazy (require [serve] extra at access time)
    "ContextRegistry",
    "SharedContextResult",
    "RegisteredAgent",
    "PipelineConfig",
    "TokenCounter",
    "count_tokens",
    "encode_tokens",
    "compute_kv_gb",
    "VRAMMonitor",
    "get_monitor",
    "get_vram_pressure",
    "LSHTokenMatcher",
    "TokenBlockMatch",
    "FAISSContextIndex",
    "FAISSMatch",
    "VRAMAwareCache",
    "EvictionMode",
]
