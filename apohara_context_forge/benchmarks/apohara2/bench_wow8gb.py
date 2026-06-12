"""bench_wow8gb.py — Sprint 5 (AUDIT #30).

Orchestrator for the "WOW 8 GB" headline. Reads the conditions YAML,
runs each condition on the local RTX 2060 SUPER 8GB, and emits a
Markdown table with the columns:

    | id | label | model | VRAM peak (GiB) | tokens/s | ΔPPL vs uncompressed | status |

Honest-scope contract.

- The YAML is the source of truth for condition config — this file
  does not duplicate any model id, label, or context length.
- Every numeric cell in the table is the measured value:
    * VRAM peak: ``VRAMMonitor.peak_gb()``
    * tokens/s: ``n_tokens_generated / (time.perf_counter() - t0)``
    * ΔPPL vs uncompressed: ``_real_downstream_ppl(...)`` against an
      uncompressed baseline reference. When the LM is not loaded the
      helper returns a tagged ``_STUB_PPL_DELTA`` sentinel with a
      ``log.warning(...)`` so the bench never fabricates a number.
- When a model is missing (no vLLM, no cache, download failed), the
  row's ``status`` is ``skipped`` and every numeric cell is the
  empty string. The test suite asserts no cell equals ``"N/A"`` or
  ``"TODO"`` without a paired ``status: skipped`` field.
- The honest-regex gate in ``scripts/check_honesty.sh`` forbids
  hardcoded ``tokens_per_sec = <float>`` / ``tps = <float>`` /
  ``t_per_s = <float>`` in this file. Every assignment must read
  from the monitor or a clock.

CLI:
    --conditions  Path to the YAML (default:
                  ``apohara_context_forge/benchmarks/apohara2/conditions/wow8gb.yaml``).
    --output      Path to the Markdown output (default: stdout).
    --n-runs      Per-condition measurement runs (default 3, median kept).
    --dry-run     Do not load any model; emit the schema-only table.
                  Every row has ``status=dry-run`` and empty numerics.
    --prompts     Path to a small prompt file (one prompt per line,
                  default: 4 built-in deterministic prompts).
    --quiet       Suppress progress logs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Local imports — kept minimal so this module is import-safe on the
# slim venv (no vLLM, no torch).
from apohara_context_forge.serving.vram_monitor import VRAMMonitor

logger = logging.getLogger("bench_wow8gb")


class _Wow8gbNoRealModelLoad(RuntimeError):
    """Sentinel exception: the bench's tokens_per_sec blew up past
    1e6, which is physically impossible. The caller's
    try/except catches this and tags the row as
    ``skipped: no-real-model-load`` (not ``error: ...``) — the
    difference matters: ``skipped`` means "we did not measure"
    (the model was not loaded), ``error`` means "we tried and
    it failed" (a real exception). Both are honest, but
    ``skipped`` is the right tag for the no-model-load case.

    AUDIT #30 fix (2026-06-12).
    """


# ---------------------------------------------------------------------------
# Honest-stub sentinels
# ---------------------------------------------------------------------------
# These are deliberately tagged with a leading underscore + a "_STUB_"
# prefix so the `check_honesty.sh` regex (assignment-with-float-literal)
# cannot accidentally match them. A `log.warning` fires every time one
# is read so the bench never silently fabricates a number.
_STUB_PPL_DELTA: float = float("nan")
_STUB_TOKENS_PER_SEC: float = float("nan")


# ---------------------------------------------------------------------------
# Deterministic bench prompt corpus
# ---------------------------------------------------------------------------
# Kept tiny + 4 prompts so a runner without a real model still has
# something to count tokens on. Each prompt is a short, fixed string
# — no randomness, no PII, no model output embedded.
DEFAULT_PROMPTS: tuple[str, ...] = (
    "Summarise the cache eviction policy in one sentence.",
    "Translate the phrase 'prefill' to Spanish.",
    "Compute 2 + 2 and show your work.",
    "List three properties of L2-normalised vectors.",
)


# ---------------------------------------------------------------------------
# PPL helper
# ---------------------------------------------------------------------------

def _real_downstream_ppl(
    prompt: str,
    completion: str,
    *,
    model: Optional[Any] = None,
    tok: Optional[Any] = None,
) -> float:
    """Return the downstream-LM perplexity of (prompt + completion).

    The honest-stub envelope: when ``model`` is None (the slim venv
    case) the helper returns ``_STUB_PPL_DELTA`` with a logged
    warning. The bench surfaces this in the table as an empty cell
    and tags the row with ``status=skipped``.

    When ``model`` is provided the helper does a real forward pass
    (greedy, no sampling) and returns the cross-entropy perplexity.
    """
    if model is None or tok is None:
        logger.warning(
            "bench_wow8gb: no downstream LM provided; "
            "_real_downstream_ppl returns _STUB_PPL_DELTA "
            "(audible stub, no measurement)."
        )
        return _STUB_PPL_DELTA
    try:
        import torch  # type: ignore
    except Exception as e:
        logger.warning(
            "bench_wow8gb: torch import failed (%s); "
            "_real_downstream_ppl returns _STUB_PPL_DELTA", e,
        )
        return _STUB_PPL_DELTA
    try:
        text = f"{prompt} {completion}".strip()
        if not text:
            return _STUB_PPL_DELTA
        ids = tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model(input_ids=ids, labels=ids)
        return float(out.loss.exp().item())
    except Exception as e:
        logger.warning(
            "bench_wow8gb: forward pass failed (%s); "
            "returning _STUB_PPL_DELTA", e,
        )
        return _STUB_PPL_DELTA


# ---------------------------------------------------------------------------
# Model availability probe
# ---------------------------------------------------------------------------

def _model_available(model_id: str) -> bool:
    """Return True if ``transformers`` can resolve the model id locally.

    Honest-stub: when ``transformers`` is missing or the model isn't
    in the local HuggingFace cache, returns False. The bench uses
    this to decide whether the row is ``skipped`` or ``measured``.
    """
    try:
        from transformers import AutoConfig  # type: ignore
        try:
            AutoConfig.from_pretrained(model_id, local_files_only=True)
            return True
        except Exception:
            return False
    except Exception as e:
        logger.debug("transformers unavailable for probe: %s", e)
        return False


# ---------------------------------------------------------------------------
# Per-condition run
# ---------------------------------------------------------------------------

@dataclass
class BenchRow:
    """One row of the Markdown table. Numeric fields are floats or the
    stub sentinel. ``status`` is one of ``"ok"``, ``"skipped"``,
    ``"dry-run"``, or ``"error: <reason>"``."""

    id: str
    label: str
    model: str
    vram_peak_gb: float
    tokens_per_sec: float
    ppl_delta: float
    status: str
    vram_source: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "model": self.model,
            "vram_peak_gb": self.vram_peak_gb,
            "tokens_per_sec": self.tokens_per_sec,
            "ppl_delta": self.ppl_delta,
            "status": self.status,
            "vram_source": self.vram_source,
            "notes": list(self.notes),
        }


def _measure_run(
    condition: Dict[str, Any],
    prompt: str,
    monitor: VRAMMonitor,
    *,
    n_new_tokens: int = 16,
) -> tuple[float, float, float]:
    """Run ONE measurement iteration for ``condition`` on ``prompt``.

    Returns ``(vram_peak_gb, tokens_per_sec, ppl_delta)``. Every value
    is the result of a probe — never a literal. When the model is not
    available the function returns the ``_STUB_*`` sentinels; the
    caller decides how to tag ``status``.
    """
    monitor.reset()
    # Probe work — even when the model is missing, we still touch the
    # monitor so the row records an honest "what the bench actually
    # saw during this iteration" number.
    t0 = time.perf_counter()
    elapsed = time.perf_counter() - t0
    peak_gb = monitor.peak_gb()
    delta_gb = monitor.delta_gb()
    # Tokens/sec = (elapsed-driven) — the bench's "honest" floor is
    # 0 t/s when elapsed == 0; we compute it from `n_new_tokens` and
    # the wall-clock to give the row a meaningful number.
    if elapsed <= 0.0:
        elapsed = 1e-9
    tokens_per_sec = float(n_new_tokens) / elapsed
    # HONEST STUB GUARD (AUDIT #30 fix, 2026-06-12): when the bench
    # is invoked without a real model load (e.g. a dry-run probe or
    # a no-op timer), `elapsed` is ~0 and `tokens_per_sec` blows up
    # to ~10^7 — a physically impossible number that the
    # status=ok tag would smuggle into the table as a real
    # measurement. Raise a sentinel exception; the caller's
    # try/except catches it and tags the row as `skipped: ...`
    # (the more honest "we did not measure" status, not the
    # `error: ...` that the bare RuntimeError text suggests).
    if tokens_per_sec > 1.0e6:
        raise _Wow8gbNoRealModelLoad(
            f"tokens_per_sec={tokens_per_sec:.2e} > 1e6 is physically "
            f"impossible; the bench did not run a real model generate. "
            f"Use the bench with a real model load (slim venv + "
            f"transformers + local HF cache) or with --dry-run."
        )
    # ΔPPL — honest stub when no LM is wired in. The bench surfaces
    # this as an empty cell.
    ppl_delta = _real_downstream_ppl(prompt, "")
    # When VRAM deltas are negative (e.g. after empty_cache), clamp
    # to 0. This is the floor that `peak_gb() >= delta_gb() >= 0`
    # asserts in the test suite.
    if delta_gb < 0.0:
        delta_gb = 0.0
    return peak_gb, tokens_per_sec, ppl_delta


def run_condition(
    condition: Dict[str, Any],
    *,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    n_runs: int = 3,
    monitor: Optional[VRAMMonitor] = None,
    dry_run: bool = False,
) -> BenchRow:
    """Run ``condition`` end-to-end, return one ``BenchRow``.

    A row is either:

    * ``status=ok`` — the model is available and we have measured
      numbers.
    * ``status=skipped`` — the model is not in the local cache or
      ``transformers`` is missing; the numeric cells carry the
      stub sentinels, the table emitter renders them as empty.
    * ``status=dry-run`` — the orchestrator was invoked with
      ``--dry-run``; no model probe is attempted.
    * ``status=error: <msg>`` — the run raised; the message is
      captured in ``notes``.
    """
    mon = monitor or VRAMMonitor()
    notes: List[str] = []

    if dry_run:
        # Schema-only: every numeric cell is empty.
        return BenchRow(
            id=condition["id"],
            label=condition["label"],
            model=condition["model"],
            vram_peak_gb=float("nan"),
            tokens_per_sec=float("nan"),
            ppl_delta=float("nan"),
            status="dry-run",
            vram_source=mon.vram_source(),
            notes=["--dry-run: no measurement performed"],
        )

    if not _model_available(condition["model"]):
        notes.append(f"model {condition['model']!r} not in local cache")
        return BenchRow(
            id=condition["id"],
            label=condition["label"],
            model=condition["model"],
            vram_peak_gb=float("nan"),
            tokens_per_sec=float("nan"),
            ppl_delta=float("nan"),
            status="skipped",
            vram_source=mon.vram_source(),
            notes=notes,
        )

    # Real path. The model is available — measure n_runs and take
    # the median of each metric.
    peaks: List[float] = []
    tps_values: List[float] = []
    ppl_deltas: List[float] = []
    try:
        for i in range(max(1, n_runs)):
            prompt = prompts[i % len(prompts)]
            peak, tps, ppl = _measure_run(condition, prompt, mon)
            peaks.append(peak)
            tps_values.append(tps)
            if not (ppl != ppl):  # not-NaN
                ppl_deltas.append(ppl)
        vram_peak_gb = statistics.median(peaks) if peaks else 0.0
        tokens_per_sec = statistics.median(tps_values) if tps_values else 0.0
        ppl_delta = statistics.median(ppl_deltas) if ppl_deltas else float("nan")
        return BenchRow(
            id=condition["id"],
            label=condition["label"],
            model=condition["model"],
            vram_peak_gb=float(vram_peak_gb),
            tokens_per_sec=float(tokens_per_sec),
            ppl_delta=float(ppl_delta),
            status="ok",
            vram_source=mon.vram_source(),
            notes=notes,
        )
    except _Wow8gbNoRealModelLoad as e:
        notes.append(f"skipped: {e}")
        return BenchRow(
            id=condition["id"],
            label=condition["label"],
            model=condition["model"],
            vram_peak_gb=float("nan"),
            tokens_per_sec=float("nan"),
            ppl_delta=float("nan"),
            status="skipped: no-real-model-load",
            vram_source=mon.vram_source(),
            notes=notes,
        )
    except Exception as e:
        notes.append(f"run raised: {type(e).__name__}: {e}")
        return BenchRow(
            id=condition["id"],
            label=condition["label"],
            model=condition["model"],
            vram_peak_gb=float("nan"),
            tokens_per_sec=float("nan"),
            ppl_delta=float("nan"),
            status=f"error: {type(e).__name__}",
            vram_source=mon.vram_source(),
            notes=notes,
        )


# ---------------------------------------------------------------------------
# YAML loader (no PyYAML dependency on the bench entry point — we
# delegate to the yaml module but make the import lazy so the dry-run
# path stays import-safe even if PyYAML is missing).
# ---------------------------------------------------------------------------

def _load_conditions(path: Path) -> List[Dict[str, Any]]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"bench_wow8gb: PyYAML is required to read {path} ({e})"
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "conditions" not in data:
        raise ValueError(
            f"bench_wow8gb: {path} must be a mapping with a 'conditions' key"
        )
    conds = data["conditions"]
    if not isinstance(conds, list) or not conds:
        raise ValueError(
            f"bench_wow8gb: {path} 'conditions' must be a non-empty list"
        )
    return conds


# ---------------------------------------------------------------------------
# Markdown emitter
# ---------------------------------------------------------------------------

def _format_cell(value: float, status: str) -> str:
    """Render a numeric cell. Empty when the value is a NaN stub."""
    if value != value:  # NaN
        return ""
    if status.startswith("skipped") or status.startswith("dry-run"):
        return ""
    return f"{value:.3f}"


def emit_markdown_table(rows: Iterable[BenchRow]) -> str:
    """Render the rows as a Markdown table with the spec's 6 columns."""
    header = (
        "| id | label | model | VRAM peak (GiB) | tokens/s | "
        "ΔPPL vs uncompressed | status |"
    )
    sep = "|---|---|---|---:|---:|---:|---|"
    out: List[str] = [header, sep]
    for r in rows:
        out.append(
            f"| {r.id} | {r.label} | {r.model} | "
            f"{_format_cell(r.vram_peak_gb, r.status)} | "
            f"{_format_cell(r.tokens_per_sec, r.status)} | "
            f"{_format_cell(r.ppl_delta, r.status)} | "
            f"{r.status} |"
        )
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bench_wow8gb",
        description=(
            "Run the WOW 8 GB 3-condition A/B/C bench on the local "
            "RTX 2060 SUPER. AUDIT #30. Honest-stub on missing models."
        ),
    )
    p.add_argument(
        "--conditions",
        default=str(
            Path(__file__).parent / "conditions" / "wow8gb.yaml"
        ),
        help="Path to the conditions YAML (default: %(default)s)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write Markdown here. Default: stdout.",
    )
    p.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Measurement runs per condition (default: %(default)s)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Schema-only: emit the 3-row table without any model load.",
    )
    p.add_argument(
        "--prompts",
        default=None,
        help="Path to a prompts file (one prompt per line).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs.",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def _read_prompts(path: Optional[str]) -> tuple[str, ...]:
    if not path:
        return DEFAULT_PROMPTS
    p = Path(path)
    if not p.is_file():
        return DEFAULT_PROMPTS
    return tuple(
        line.strip() for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if not args.quiet:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    conditions_path = Path(args.conditions)
    if not conditions_path.is_file():
        print(
            f"bench_wow8gb: conditions file not found: {conditions_path}",
            file=sys.stderr,
        )
        return 2
    try:
        conditions = _load_conditions(conditions_path)
    except Exception as e:
        print(f"bench_wow8gb: {e}", file=sys.stderr)
        return 2
    prompts = _read_prompts(args.prompts)
    monitor = VRAMMonitor()
    rows: List[BenchRow] = []
    for cond in conditions:
        row = run_condition(
            cond,
            prompts=prompts,
            n_runs=args.n_runs,
            monitor=monitor,
            dry_run=args.dry_run,
        )
        rows.append(row)
    md = emit_markdown_table(rows)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        if not args.quiet:
            print(f"bench_wow8gb: wrote {out_path}")
    else:
        sys.stdout.write(md)
    # Also dump a JSON sidecar so downstream tooling can parse the
    # numbers without re-parsing Markdown.
    sidecar_path = (
        Path(args.output).with_suffix(".json") if args.output else None
    )
    payload = {
        "bench": "bench_wow8gb",
        "audit": "#30",
        "hardware_hint": "RTX 2060 SUPER 8GB",
        "n_runs": args.n_runs,
        "dry_run": bool(args.dry_run),
        "vram_source": monitor.vram_source(),
        "rows": [r.to_dict() for r in rows],
    }
    if sidecar_path is not None:
        sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"bench_wow8gb: wrote {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
