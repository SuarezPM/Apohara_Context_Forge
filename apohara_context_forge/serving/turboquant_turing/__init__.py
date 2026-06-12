"""Python wrapper for the in-tree `turboquant-turing` Rust crate.

This module is the public Python entry point. The actual
implementation lives in the `turboquant_turing` wheel
(maturin-built, see ``apohara_context_forge/serving/turboquant_turing/build.sh``).
When the wheel is not built in the active venv, the imports
below raise ``ImportError`` and the in-tree ``turboquant_kv``
shim surfaces a clear "maturin develop" error message.

Sprint 2 / AUDIT #320a: this file is now a thin re-export
shim. The previous placeholder string docstring was load-bearing
as an "honest not-built" signal; the live module replaces it
with a deferred import that re-exports the wheel's
``fwht_inplace``, ``dequant_per_block``, ``encode_kv_py`` and
``decode_kv_py`` symbols. The ``_rust_available`` check in
``apohara_context_forge/serving/turboquant_kv.py`` (live
``importlib.util.find_spec``) is the source of truth for the
"wheel built" / "wheel not built" answer; this file's
``__getattr__`` defers to it via a try/except that raises
``ImportError`` when the wheel is not importable.
"""
from __future__ import annotations

import importlib.util
from typing import Any


def _wheel_is_built() -> bool:
    """Live check: is the maturin-built ``turboquant_turing``
    wheel importable in the current process?

    The check uses ``importlib.util.find_spec`` (the standard-
    library import-finder) so it is fast and has no side effects
    on the import system. Returns True when the wheel is
    importable, False otherwise.
    """
    return importlib.util.find_spec("turboquant_turing") is not None


def __getattr__(name: str) -> Any:
    """PEP 562 module-level ``__getattr__`` — re-export the
    wheel's symbols lazily, raising ``ImportError`` when the
    wheel is not built.

    The deferred import keeps the rest of the in-tree code
    import-safe: callers that need the wheel (``fwht_inplace``,
    ``dequant_per_block``) get a clear ``ImportError`` pointing
    at the maturin develop step, while callers that don't (the
    pure-Python numpy path) are unaffected.
    """
    if not _wheel_is_built():
        raise ImportError(
            "turboquant-turing Rust crate is not built. "
            "Run `cd apohara_context_forge/serving/turboquant_turing && "
            "bash build.sh` to build it."
        )
    import turboquant_turing as _wheel
    return getattr(_wheel, name)
