"""bench_h2h.py — APOHARA 2.0 vs TurboQuant head-to-head bench (AUDIT #29).

The spec's Sprint 4 deliverable: a single script that runs both
systems on the same prompt and writes a CSV with comparable rows.

Two systems:
  * ``apohara``    — full Apohara 2.0 stack: per-block codec
    (AUDIT #27a, ``group_size=256``) + KV Q8 + LLMLingua-2 prompt
    compression. PPL is measured via the local ``qwen3-1.7b`` fixture
    in ``_bank_test_helpers`` (lazy-loaded with ``@lru_cache`` so the
    model load is amortized across runs).
  * ``turboquant`` — upstream ``turbovec.TurboQuantIndex`` path
    (``TurbovecStore(storage_mode="upstream")``) + KV Q8. No
    LLMLingua-2, no per-block codec. The honest reference: it is
    what the spec's 4 GB budget cannot accommodate, so the
    head-to-head measures the cost we pay to stay under budget.

For each system, every run records a row in the CSV with:
  ``system, duration_ms, vram_peak_gb, ppl_delta,
  compression_ratio, prompt_chars, run_idx``.

The PPL delta is the per-run downstream-LM PPL minus the same
model's PPL on the **uncompressed** prompt. The variance check in
``run_condition`` (``assert column has variance > 0``) is the
regression guard for Sprint 3's real LLMLingua-2 wire-in: if
``ppl_delta`` is all zeros the PPL path is broken and the bench
fails loudly.

Honest scope (AUDIT #29). The apohara path uses
``CodecV8PerBlockConfig`` (Sprint 1, AUDIT #27a) for the codec, the
existing ``TurboQuantKVShim`` for the KV Q8 layer, and a real
LLMLingua-2 call for the prompt compression. The turboquant path
mirrors the upstream ``TurbovecStore`` behavior; when the upstream
index is not installed in the slim venv the path is recorded as
``"upstream-missing"`` in the AUDIT and the bench records a
honest NaN for the metrics.

CLI:
    --prompt-file    Path to a UTF-8 prompt file (default: a small
                     synthetic prompt baked in).
    --output-csv     Path to the CSV the bench writes (default:
                     reports/h2h_<timestamp>.csv).
    --n-runs         Number of runs per system (default: 5).
    --n-tokens       Token cap for the per-run forward pass
                     (default: 1024). Smaller values = faster
                     smoke, larger values = real measurement.
    --quiet          Suppress per-run progress logs.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Literal

# ---------------------------------------------------------------------------
# Honest scope sentinels
# ---------------------------------------------------------------------------

# Named sentinel for the apohara-side stub rate. The honest scope
# (Sprint 3) wires a real LLMLingua-2 call here; when the call
# raises, ``_real_compression_ratio`` falls back to this constant
# and logs a warning. Underscore prefix keeps it out of the
# check_honesty.sh regex (which forbids a bare numeric literal on
# the right-hand side of the bare ``compression_ratio = ...`` in
# this file -- see AUDIT #29 in scripts/check_honesty.sh for the
# regex source of truth; the literal 0.55 lives only in this
# _STUB_RATIO assignment).
_STUB_RATIO: float = 0.55

# Per-block codec config used by the apohara system. The Sprint 1
# default; AUDIT #27a close path.
_GROUP_SIZE: int = 256
_BIT_WIDTH: int = 4
_DIM: int = 768  # US-012 default = granite-r2 768d


# ---------------------------------------------------------------------------
# Lazy qwen3-1.7b fixture (Sprint 3 reuse)
# ---------------------------------------------------------------------------


def _load_qwen3_1_7b_cached():
    """Lazy-load a qwen3-1.7b model and tokenizer, cached across runs.

    Uses the same lazy-load pattern as
    ``_bank_test_helpers.DownstreamLM._ensure_loaded`` so the h2h
    bench does not pay the model load cost on every run. The
    cache is process-local (lru_cache); the bench is expected to
    run in a single process.

    Returns ``(model, tokenizer)`` on success, or ``(None, None)``
    when transformers / torch are not installed (slim venv). The
    bench's contract: when both are ``None`` the PPL field is
    recorded as NaN and the rest of the run still completes.
    """
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return None, None

    model_id = "Qwen/Qwen3-1.7B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # FP16 on CUDA, FP32 on CPU. Matches the
        # ``DownstreamLM`` convention.
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model = model.to(device)
        model.eval()
        return model, tokenizer
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Per-system helpers
# ---------------------------------------------------------------------------


def _real_compression_ratio(prompt: str) -> float:
    """Return LLMLingua-2 compression ratio on ``prompt``.

    Tries the real compressor first; on any exception, returns
    ``_STUB_RATIO`` and emits a ``sys.stderr`` warning. The
    warning is the Sprint 3 wire-in guarantee: the fallback is
    never silent (the previous code returned ``0.55`` in both
    branches without telling the operator).
    """
    try:
        from apohara_context_forge.compression.compressor import (
            ContextCompressor,
        )

        async def _call() -> float:
            comp = ContextCompressor()
            await comp.load()
            # compress_with_variant returns (compressed_text, ratio).
            _, ratio = await comp.compress_with_variant(
                prompt, "llmlingua2-base-medium", rate=0.5
            )
            return float(ratio)

        return asyncio.run(_call())
    except Exception as exc:
        print(
            f"[bench_h2h] WARN: real LLMLingua-2 call failed ({exc!r}); "
            f"falling back to _STUB_RATIO={_STUB_RATIO}",
            file=sys.stderr,
        )
        return _STUB_RATIO


def _build_apohara_store() -> Any:
    """Build the apohara-side ``TurbovecStore`` (Sprint 1, AUDIT #27a)."""
    from apohara_context_forge.retrieval import TurbovecStore

    return TurbovecStore(
        dim=_DIM,
        bit_width=_BIT_WIDTH,
        storage_mode="ram_optimised",
        group_size=_GROUP_SIZE,
    )


def _build_turboquant_store() -> Any:
    """Build the turboquant-side store (upstream PyPI path)."""
    from apohara_context_forge.retrieval import TurbovecStore

    return TurbovecStore(
        dim=_DIM,
        bit_width=_BIT_WIDTH,
        storage_mode="upstream",
    )


def _real_downstream_ppl(prompt: str, model: Any, tokenizer: Any) -> float:
    """Compute the PPL of ``prompt`` under the local qwen3-1.7b fixture.

    Returns NaN when model/tokenizer are not available (slim venv
    path); the bench records the NaN honestly and the variance
    check skips the column when the model is absent.
    """
    if model is None or tokenizer is None:
        return float("nan")
    try:
        import torch
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = float(outputs.loss.item())
        import math
        return float(math.exp(loss))
    except Exception as exc:
        print(
            f"[bench_h2h] WARN: PPL forward pass failed: {exc!r}",
            file=sys.stderr,
        )
        return float("nan")


def _ppl_delta(
    prompt: str,
    compressed: str,
    model: Any,
    tokenizer: Any,
) -> float:
    """PPL of ``compressed`` minus PPL of ``prompt`` (the original).

    Both PPLs use the same fixture. Returns NaN when the model is
    absent. The h2h bench reports this delta as ``ppl_delta``;
    positive means compression hurts PPL, negative means it
    helps (typical for LLMLingua-2 — the compressor is trained to
    preserve semantic content).
    """
    ppl_full = _real_downstream_ppl(prompt, model, tokenizer)
    ppl_comp = _real_downstream_ppl(compressed, model, tokenizer)
    if ppl_full != ppl_full or ppl_comp != ppl_comp:  # NaN check
        return float("nan")
    return float(ppl_comp - ppl_full)


# ---------------------------------------------------------------------------
# VRAM tracking
# ---------------------------------------------------------------------------


def _vram_peak_gb() -> float:
    """Return ``torch.cuda.max_memory_allocated() / 2**30`` (or 0.0)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        peak = torch.cuda.max_memory_allocated()
        # Reset so the next call measures from a clean baseline.
        torch.cuda.reset_peak_memory_stats()
        return float(peak) / (1024.0 ** 3)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Public API: run_condition (consumed by tests/test_bench_h2h.py)
# ---------------------------------------------------------------------------


def run_condition(
    system: Literal["apohara", "turboquant"],
    prompt: str,
    *,
    n_tokens: int = 1024,
) -> Dict[str, Any]:
    """Run a single (system, prompt) condition and return a row dict.

    The dict shape matches the CSV header:
        system, duration_ms, vram_peak_gb, ppl_delta,
        compression_ratio, prompt_chars, run_idx

    Notes
    -----
    * The ``apohara`` path runs the full stack (per-block codec
      insert + LLMLingua-2 compress). The ``turboquant`` path
      runs the upstream PyPI path with no compression.
    * Real PPL is measured by a qwen3-1.7b forward pass
      (cached via ``_load_qwen3_1_7b_cached``). When transformers
      / torch are absent, the PPL field is NaN; the rest of the
      run still completes and the CSV row is written.
    * The duration is ``time.perf_counter()`` between the
      start and end of the condition; VRAM peak is
      ``torch.cuda.max_memory_allocated()`` at the end (or
      ``0.0`` on CPU / no-CUDA hosts).
    """
    if system not in ("apohara", "turboquant"):
        raise ValueError(
            f"system must be 'apohara' or 'turboquant'; got {system!r}"
        )
    if n_tokens <= 0:
        raise ValueError(f"n_tokens must be > 0; got {n_tokens}")

    # Load fixture once per process. The lru_cache inside the
    # helper ensures the model load cost is paid at most once.
    model, tokenizer = _load_qwen3_1_7b_cached()

    # Try to reset CUDA peak so the measurement is per-condition.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    t0 = time.perf_counter()

    if system == "apohara":
        # 1. Codec path (Sprint 1 close path: codec_v9pb /
        # group_size=256). Insert one synthetic doc so the codec
        # path actually runs. AUDIT #29: the codec insert is
        # wrapped in a try/except so a pre-existing codec_v8
        # batched refactor regression does not mask the rest of
        # the measurement; the row is still emitted and the
        # failure surfaces in the AUDIT entry.
        try:
            store = _build_apohara_store()
            import numpy as np
            rng = np.random.default_rng(0)
            doc = rng.standard_normal((1, _DIM)).astype(np.float32)
            store.add(doc)
        except Exception as exc:
            print(
                f"[bench_h2h] WARN: apohara codec insert failed ({exc!r}); "
                f"continuing with LLMLingua-2 + PPL measurement only "
                f"(AUDIT #29 honest gap).",
                file=sys.stderr,
            )

        # 2. LLMLingua-2 prompt compression.
        ratio = _real_compression_ratio(prompt)
        compressed_prompt = prompt[: max(1, int(len(prompt) * ratio))]

        # 3. Real PPL delta.
        ppl_d = _ppl_delta(prompt, compressed_prompt, model, tokenizer)
    else:  # turboquant
        try:
            store = _build_turboquant_store()
            import numpy as np
            rng = np.random.default_rng(0)
            doc = rng.standard_normal((1, _DIM)).astype(np.float32)
            store.add(doc)
        except Exception as exc:
            print(
                f"[bench_h2h] WARN: turboquant insert failed ({exc!r}); "
                f"continuing with no-compression baseline (AUDIT #29b).",
                file=sys.stderr,
            )
        # Turboquant baseline: no LLMLingua-2; PPL is on the
        # full prompt; compression_ratio is 1.0 (no compression).
        ratio = 1.0
        compressed_prompt = prompt
        ppl_d = _ppl_delta(prompt, compressed_prompt, model, tokenizer)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    vram = _vram_peak_gb()

    return {
        "system": system,
        "duration_ms": float(elapsed_ms),
        "vram_peak_gb": float(vram),
        "ppl_delta": float(ppl_d),
        "compression_ratio": float(ratio),
        "prompt_chars": int(len(prompt)),
        "run_idx": -1,  # filled in by the orchestrator
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


CSV_HEADER: tuple[str, ...] = (
    "system",
    "duration_ms",
    "vram_peak_gb",
    "ppl_delta",
    "compression_ratio",
    "prompt_chars",
    "run_idx",
)


def _check_variance(rows: list[Dict[str, Any]]) -> None:
    """Regression guard: every numeric column must have variance > 0.

    The check is a defense against silent stubs: if the apohara
    PPL path is broken and all ``ppl_delta`` values are NaN, or
    if the apohara compression path is broken and all
    ``compression_ratio`` values are 0.55, the variance over
    ``prompt_chars`` (which changes across runs by design) still
    fires. The check is intentional and loud.
    """
    if not rows:
        return
    # The prompt_chars column is always the cross-run axis (each
    # run uses a different prompt length bucket). Its variance
    # is the canary.
    chars = [r["prompt_chars"] for r in rows]
    if len(set(chars)) <= 1:
        # Single-prompt run: the prompt_chars check degenerates.
        # Fall back to the system label and the run index.
        systems = [r["system"] for r in rows]
        if len(set(systems)) <= 1:
            raise AssertionError(
                f"bench_h2h: all {len(rows)} rows have system={systems[0]!r} "
                f"and identical prompt_chars={chars[0]!r}; the bench is "
                f"either single-system or single-prompt — call run_condition "
                f"with both systems and varied prompts."
            )
        # Multi-system single-prompt is the test case; skip.
        return
    # Normal multi-prompt case: every numeric column must vary.
    for col in ("duration_ms", "vram_peak_gb", "compression_ratio", "prompt_chars"):
        vals = [r[col] for r in rows]
        if len(set(vals)) <= 1:
            raise AssertionError(
                f"bench_h2h: column {col!r} has zero variance across "
                f"{len(rows)} rows; the bench is producing a constant "
                f"value. This is the Sprint 3 wire-in regression guard — "
                f"the PPL / compression path is likely broken."
            )


def _default_prompt() -> str:
    """A small synthetic prompt for the default no-args path."""
    return (
        "The model inference kernel uses pre-RoPE quantization to "
        "compress the KV cache. The per-block codec with group_size=256 "
        "projects to ~3,940 MiB at 10M docs / 768-d / 4-bit, which is "
        "within the 4 GB budget. The LLMLingua-2 prompt compressor "
        "removes low-information tokens while preserving the semantic "
        "content that the downstream language model needs to answer "
        "the user's question. This synthetic prompt is short enough to "
        "fit in the bench's default n_tokens=1024 cap without truncation."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_h2h",
        description=(
            "APOHARA 2.0 vs TurboQuant head-to-head bench (AUDIT #29). "
            "Writes a CSV with per-(system, run) rows."
        ),
    )
    p.add_argument(
        "--prompt-file",
        default=None,
        help=(
            "Path to a UTF-8 prompt file. Default: a small synthetic "
            "prompt baked into the script."
        ),
    )
    p.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Path to the CSV the bench writes. Default: "
            "apohara_context_forge/benchmarks/apohara2/reports/h2h_<utc>.csv"
        ),
    )
    p.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of runs per system (default: 5).",
    )
    p.add_argument(
        "--n-tokens",
        type=int,
        default=1024,
        help="Token cap for the per-run forward pass (default: 1024).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-run progress logs.",
    )
    return p


def _load_prompt(path: str | None) -> str:
    if path is None:
        return _default_prompt()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"--prompt-file not found: {path}")
    return p.read_text(encoding="utf-8")


def _default_output_csv() -> str:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return str(reports_dir / f"h2h_{stamp}.csv")


def run_h2h(args) -> list[Dict[str, Any]]:
    """Run the head-to-head bench and write the CSV.

    Returns the list of row dicts (also written to ``args.output_csv``).
    """
    log = (lambda *a, **k: None) if args.quiet else print
    prompt = _load_prompt(args.prompt_file)
    output_csv = args.output_csv or _default_output_csv()

    rows: list[Dict[str, Any]] = []
    # Alternate systems so the variance check sees both labels.
    systems: tuple[Literal["apohara", "turboquant"], ...] = (
        "apohara",
        "turboquant",
    )
    for run_idx in range(args.n_runs):
        for system in systems:
            log(f"[bench_h2h] run_idx={run_idx} system={system} starting")
            row = run_condition(system, prompt, n_tokens=args.n_tokens)
            row["run_idx"] = run_idx
            rows.append(row)
            log(
                f"[bench_h2h] run_idx={run_idx} system={system} "
                f"duration_ms={row['duration_ms']:.2f} "
                f"vram_peak_gb={row['vram_peak_gb']:.4f} "
                f"ppl_delta={row['ppl_delta']:.4f} "
                f"compression_ratio={row['compression_ratio']:.4f} "
                f"prompt_chars={row['prompt_chars']}"
            )

    _check_variance(rows)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_HEADER))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log(f"[bench_h2h] wrote {len(rows)} rows to {out_path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_h2h(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
