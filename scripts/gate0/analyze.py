#!/usr/bin/env python3
"""GATE #0 — analysis + report (pure, log-only).

This module is the *decision* stage of the GATE #0 harness. It is intentionally
DECOUPLED from the live path (no vLLM, no GPU, no network): it reads the raw-log
JSON written by ``harness.py`` (schema in CONTRACT.md §9), re-aggregates
confidence intervals from the retained per-request samples, computes the single
decisive number ``delta_pct = (B - A) / A * 100`` for the chosen primary metric,
applies the *preregistered* cut, and emits the Spanish markdown report from the
template (CONTRACT.md §12).

The gate rule lives here, encoded once:

  * ``delta_pct < 5``        -> ABANDON  (mechanical sharing not worth it)
  * ``5 <= delta_pct <= 15`` -> GREY_ZONE
  * ``delta_pct > 15``       -> INVEST   (Fase 2)
  * primary metric null/not-isolable on A or B, OR validity not quotable
                              -> INDECISIVE

Honesty contract (CONTRACT.md §1): this module NEVER invents a number. Every
value comes from the raw log or is ``None`` with a reason. If an arm is missing
from the log it is marked pending, never imputed. ``(B - C)`` is reported only as
a harness-validity sanity row — never as the decision.

Reuse: confidence intervals are computed via ``metrics.confidence_interval`` when
the sibling module is importable; otherwise a stdlib-only fallback with the same
``CI`` shape keeps ``analyze.py`` runnable in CI on a box with no GPU deps.

Apache-2.0 — Apohara ContextForge.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, is_dataclass, asdict
from pathlib import Path
from typing import Any

# CONTRACT §1: pin the repo root the same way the existing probes do, so the
# `metrics` sibling and `scripts.*` resolve identically regardless of cwd.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# CI type + aggregator: prefer the canonical metrics implementation; fall back
# to a stdlib-only twin so this module is independently runnable in dry/CI.
# Both expose the SAME field layout (CONTRACT §5.1) so downstream code is blind
# to which one is in use.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import wiring, exercised at runtime
    from scripts.gate0.metrics import CI, confidence_interval  # type: ignore
    _HAVE_METRICS = True
except ImportError:
    _HAVE_METRICS = False

    @dataclass
    class CI:  # type: ignore[no-redef]
        """A mean with a confidence interval and the n it summarizes.

        Field-compatible with ``metrics.CI`` (CONTRACT §5.1) so the report
        renderer never needs to know which implementation produced it.
        """

        mean: float | None
        lo: float | None
        hi: float | None
        n: int
        method: str
        confidence: float = 0.95

    def confidence_interval(  # type: ignore[no-redef]
        samples: list[float],
        *,
        confidence: float = 0.95,
        method: str = "bootstrap",
        n_boot: int = 2000,
        seed: int = 0,
    ) -> "CI":
        """Mean +- CI over ``samples``. Pure stdlib twin of metrics.confidence_interval.

        n < 2 -> CI(mean=mean_or_None, lo=None, hi=None, n=n, method='none(n<2)').
        NEVER invents a spread when n is too small.
        """
        clean = [float(s) for s in samples if s is not None and not _is_nan(s)]
        n = len(clean)
        if n == 0:
            return CI(mean=None, lo=None, hi=None, n=0, method="none(n<2)", confidence=confidence)
        mean = sum(clean) / n
        if n < 2:
            return CI(mean=mean, lo=None, hi=None, n=n, method="none(n<2)", confidence=confidence)
        if method == "bootstrap":
            lo, hi = _bootstrap_ci(clean, confidence=confidence, n_boot=n_boot, seed=seed)
            return CI(mean=mean, lo=lo, hi=hi, n=n, method="bootstrap", confidence=confidence)
        # method == "t": normal-ish small-n interval via the sample std error.
        var = sum((x - mean) ** 2 for x in clean) / (n - 1)
        se = math.sqrt(var) / math.sqrt(n)
        z = _z_for(confidence)
        return CI(mean=mean, lo=mean - z * se, hi=mean + z * se, n=n, method="t", confidence=confidence)


def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except TypeError:
        return False


def _z_for(confidence: float) -> float:
    """Two-sided normal critical value for a handful of common confidences.

    Used only by the stdlib 't' fallback; the real metrics module owns the
    precise implementation. We avoid scipy to stay import-clean.
    """
    table = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    return table.get(round(confidence, 2), 1.9600)


def _bootstrap_ci(
    samples: list[float], *, confidence: float, n_boot: int, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Stdlib `random` only (no numpy)."""
    import random

    rng = random.Random(seed)
    n = len(samples)
    means: list[float] = []
    for _ in range(n_boot):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    alpha = 1.0 - confidence
    lo_idx = max(0, int(math.floor((alpha / 2.0) * n_boot)))
    hi_idx = min(n_boot - 1, int(math.ceil((1.0 - alpha / 2.0) * n_boot)) - 1)
    return means[lo_idx], means[hi_idx]


# ---------------------------------------------------------------------------
# Preregistered cut (CONTRACT §0 / protocol §2). These thresholds DO NOT move
# after seeing data. They live here as named constants so the rationale strings
# and the decision are derived from a single source.
# ---------------------------------------------------------------------------
CUT_ABANDON_BELOW_PCT = 5.0
CUT_INVEST_ABOVE_PCT = 15.0

CUT_ABANDON = "ABANDON"
CUT_GREY_ZONE = "GREY_ZONE"
CUT_INVEST = "INVEST"
CUT_INDECISIVE = "INDECISIVE"

# Arm keys (mirror arms.ARM_A/B/C; duplicated as literals to keep analyze.py
# decoupled from the live-path modules per CONTRACT §13).
ARM_A = "A"
ARM_B = "B"
ARM_C = "C"

# Default template / output locations (CONTRACT §10, §12).
DEFAULT_TEMPLATE = "docs/research/_internal/GATE-0-report-TEMPLATE.md"

# Metrics whose larger value is BETTER (so a positive (B-A) delta is a *win*).
# KV footprint and TTFT are inverted: lower is better, so the gate inverts sign
# when judging them. hit_rate is contextual (secondary) and not gate-decisive.
_HIGHER_IS_BETTER = {"decode_tok_s", "total_tok_s", "hit_rate"}
_LOWER_IS_BETTER = {"kv_used_gb", "gpu_cache_usage_perc", "mean_ttft_s", "p50_ttft_s", "p95_ttft_s"}

# The primary-metric selectors the CLI exposes (CONTRACT §10).
PRIMARY_KV = "kv_footprint"
PRIMARY_KV_UTIL = "kv_cache_util"
PRIMARY_THROUGHPUT = "throughput"

# Mapping: --primary selector -> the concrete metric field decided on.
#
# kv_cache_util (gpu_cache_usage_perc) is the KV primary that SURVIVES vLLM's
# pre-allocation: vLLM reserves the whole KV pool at startup (gpu_memory_utilization),
# so device-wide HBM (kv_used_gb) is ~constant regardless of real KV use and the
# B-A delta dilutes to ~0. The /metrics gauge gpu_cache_usage_perc is the fraction
# of that pre-allocated pool actually in use, so a lower value on B at the SAME
# workload is exactly the sharing win — visible even under pre-allocation. It needs
# no VRAM-monitor source (it comes from vLLM /metrics, not rocm-smi), so the
# vram_source-honesty gate does not apply to it (see arm_metric). Lower is better.
_PRIMARY_FIELD = {
    PRIMARY_KV: "kv_used_gb",
    PRIMARY_KV_UTIL: "gpu_cache_usage_perc",
    PRIMARY_THROUGHPUT: "decode_tok_s",
}


# ---------------------------------------------------------------------------
# Public dataclasses (CONTRACT §10 "analyze.py public API"). GateRunResult is
# typed loosely as the parsed raw-log dict: analyze.py consumes the §9 JSON
# directly and never depends on harness.py's in-memory dataclasses.
# ---------------------------------------------------------------------------
GateRunResult = dict  # parsed §9 raw-log JSON; kept as dict for decoupling.


@dataclass(frozen=True)
class ArmMetric:
    """One metric for one arm, with its CI, the condition it was taken under,
    and whether it is valid for the report."""

    arm: str
    metric: str
    value: float | None
    ci: "CI"
    condition: dict
    valid: bool


@dataclass(frozen=True)
class Verdict:
    """The decision object. ``cut`` is the preregistered verdict; the gate is
    decided on ``delta_pct`` = (B-A)/A*100 of the primary metric, NEVER (B-C)."""

    primary_metric: str
    a: "ArmMetric"
    b: "ArmMetric"
    c: "ArmMetric"
    delta_b_minus_a: float | None
    delta_pct: float | None
    delta_ci_pct: "CI" | None
    cut: str
    rationale: str
    quotable: bool


# ---------------------------------------------------------------------------
# Raw-log loading
# ---------------------------------------------------------------------------
def load_raw(path: str) -> GateRunResult:
    """Load and lightly validate a §9 raw-log JSON file.

    Returns the parsed dict as-is. Raises ``ValueError`` if the file is not the
    GATE #0 schema (so a wrong --in fails loudly instead of producing a bogus
    verdict). We do NOT mutate or fill any numeric field here.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"raw log not found: {path}")
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "arms" not in data:
        raise ValueError(
            f"{path}: not a GATE #0 raw log (missing 'arms'); refusing to analyze."
        )
    return data


# ---------------------------------------------------------------------------
# Per-arm metric extraction (numbers come from the log only)
# ---------------------------------------------------------------------------
def _arm_block(run: GateRunResult, arm: str) -> dict | None:
    arms = run.get("arms") or {}
    block = arms.get(arm)
    return block if isinstance(block, dict) and block else None


def _vram_source_honest(vram_source: str | None) -> bool:
    """CONTRACT §1 / §6: a VRAM reading is only valid for the report if its
    source is a real backend. The dishonest set is fixed by the protocol."""
    dishonest = {"amd_default_192gb", "cuda_unavailable", "unknown", "dry"}
    return bool(vram_source) and vram_source not in dishonest


def _hbm_from_block(block: dict) -> dict | None:
    kv = block.get("kv_footprint") or {}
    hbm = kv.get("hbm")
    return hbm if isinstance(hbm, dict) else None


def _condition_for(run: GateRunResult, block: dict) -> dict:
    """Attach the §8 condition block to a metric. The run-level conditions are
    the authority; per-arm topology overrides it if present (cross vs single)."""
    cond = dict(run.get("conditions") or {})
    topo = block.get("topology")
    if topo:
        cond.setdefault("topology", topo)
    return cond


def _value_and_samples(block: dict, metric: str) -> tuple[float | None, list[float]]:
    """Pull a metric's point value and (when it exists) its per-request sample
    list out of an arm block. Returns ``(None, [])`` when the field is absent
    or null — NEVER a substitute number.

    Sample provenance per metric:
      * kv_used_gb         -> kv_footprint.kv_used_gb (scalar; no per-req samples)
      * decode_tok_s       -> throughput.decode_tok_s (aggregate; no samples)
      * total_tok_s        -> throughput.total_tok_s  (aggregate; no samples)
      * mean_ttft_s        -> throughput.mean_ttft_s, samples = throughput.ttft_samples_s
      * p50_ttft_s/p95     -> throughput.<field>,     samples = throughput.ttft_samples_s
      * hit_rate           -> prefix_metrics.hit_rate (scalar)
    """
    if metric == "kv_used_gb":
        kv = block.get("kv_footprint") or {}
        return _as_float(kv.get("kv_used_gb")), []
    if metric == "gpu_cache_usage_perc":
        # KV-pool utilization fraction from vLLM /metrics — the pre-allocation-robust
        # KV proxy. Scalar (no per-request samples); sourced from /metrics, not the
        # VRAM monitor, so arm_metric does NOT apply the vram_source-honesty gate.
        kv = block.get("kv_footprint") or {}
        return _as_float(kv.get("gpu_cache_usage_perc")), []
    if metric in {"decode_tok_s", "total_tok_s"}:
        tp = block.get("throughput") or {}
        return _as_float(tp.get(metric)), []
    if metric in {"mean_ttft_s", "p50_ttft_s", "p95_ttft_s"}:
        tp = block.get("throughput") or {}
        samples = [
            float(s)
            for s in (tp.get("ttft_samples_s") or [])
            if isinstance(s, (int, float)) and not _is_nan(s)
        ]
        return _as_float(tp.get(metric)), samples
    if metric == "hit_rate":
        pm = block.get("prefix_metrics") or {}
        return _as_float(pm.get("hit_rate")), []
    return None, []


def _as_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool):  # guard: bools are ints in Python
        return None
    if isinstance(x, (int, float)):
        f = float(x)
        return None if _is_nan(f) else f
    return None


def arm_metric(run: GateRunResult, arm: str, metric: str) -> ArmMetric:
    """Extract one ``ArmMetric`` from the raw log.

    Validity rules (CONTRACT §1, §5, §9):
      * A missing arm -> value=None, valid=False (pending; never imputed).
      * A null/absent metric value -> value=None, valid=False.
      * For ``kv_used_gb`` the underlying HBM ``vram_source`` must be honest AND
        the arm's ``measured`` flag true, else valid=False (dry/192GB never
        enters the report).
      * For other metrics, valid iff the arm is ``measured`` and the value is a
        real number.
    The CI is recomputed from ``ttft_samples_s`` when available (TTFT metrics);
    scalar metrics carry a degenerate CI(method='none(n<2)') around their point.
    """
    block = _arm_block(run, arm)
    if block is None:
        # Arm absent from the log -> pending. No number, no condition guess.
        empty_ci = CI(mean=None, lo=None, hi=None, n=0, method="none(n<2)")
        return ArmMetric(
            arm=arm,
            metric=metric,
            value=None,
            ci=empty_ci,
            condition={},
            valid=False,
        )

    measured = bool(block.get("measured", False))
    condition = _condition_for(run, block)
    value, samples = _value_and_samples(block, metric)

    # CI: from samples when present, else a degenerate point CI.
    if samples:
        ci = confidence_interval(samples)
    elif value is not None:
        ci = CI(mean=value, lo=None, hi=None, n=1, method="none(n<2)")
    else:
        ci = CI(mean=None, lo=None, hi=None, n=0, method="none(n<2)")

    valid = measured and value is not None
    if metric == "kv_used_gb":
        hbm = _hbm_from_block(block)
        vram_source = hbm.get("vram_source") if hbm else None
        hbm_valid = bool(hbm) and bool(hbm.get("valid", False))
        valid = valid and hbm_valid and _vram_source_honest(vram_source)

    return ArmMetric(
        arm=arm,
        metric=metric,
        value=value,
        ci=ci,
        condition=condition,
        valid=valid,
    )


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def _is_quotable(run: GateRunResult) -> bool:
    validity = run.get("validity") or {}
    return bool(validity.get("quotable", False))


def _classify(delta_pct: float, metric: str) -> tuple[str, str]:
    """Apply the preregistered cut to a signed *improvement* percentage.

    ``delta_pct`` here is already oriented so that POSITIVE == improvement
    (the caller flips sign for lower-is-better metrics). The thresholds are the
    immutable protocol values.
    """
    if delta_pct > CUT_INVEST_ABOVE_PCT:
        return (
            CUT_INVEST,
            f"mejora incremental {delta_pct:.2f}% > {CUT_INVEST_ABOVE_PCT:.0f}% "
            "(sostener y reproducir antes de Fase 2)",
        )
    if delta_pct < CUT_ABANDON_BELOW_PCT:
        return (
            CUT_ABANDON,
            f"mejora incremental {delta_pct:.2f}% < {CUT_ABANDON_BELOW_PCT:.0f}% "
            "(ROMY queda como salting honesto INV-15, no memory optimizer)",
        )
    return (
        CUT_GREY_ZONE,
        f"mejora incremental {delta_pct:.2f}% en zona gris "
        f"[{CUT_ABANDON_BELOW_PCT:.0f}%, {CUT_INVEST_ABOVE_PCT:.0f}%]",
    )


def compute_verdict(run: GateRunResult, *, primary_metric: str) -> Verdict:
    """Compute (B-A) ± CI for the chosen primary metric and apply the cut.

    ``primary_metric`` accepts either a CLI selector ('kv_footprint',
    'throughput') or a concrete field name ('kv_used_gb', 'decode_tok_s', ...).

    INDECISIVE when:
      * validity.quotable is False, OR
      * the primary metric is null / not-isolable / invalid on A or B.
    The decision is ALWAYS on delta = (B-A); C is carried for the report's
    sanity row only and never enters the cut.
    """
    metric = _PRIMARY_FIELD.get(primary_metric, primary_metric)

    a = arm_metric(run, ARM_A, metric)
    b = arm_metric(run, ARM_B, metric)
    c = arm_metric(run, ARM_C, metric)

    quotable = _is_quotable(run)

    # Guard 1: validity gate. Per CONTRACT §10 / protocol §6 a non-quotable run
    # is INDECISIVE no matter what the numbers look like.
    if not quotable:
        return Verdict(
            primary_metric=metric,
            a=a,
            b=b,
            c=c,
            delta_b_minus_a=None,
            delta_pct=None,
            delta_ci_pct=None,
            cut=CUT_INDECISIVE,
            rationale=(
                "INDECISIVE: la corrida no es citable (validity.quotable=False); "
                "no se computa delta hasta que pasen los gates requeridos."
            ),
            quotable=False,
        )

    # Guard 2: the primary metric must be a real, isolable number on BOTH A and
    # B. A null KV footprint (device-wide HBM not isolating KV) -> not isolable.
    if not (a.valid and b.valid) or a.value is None or b.value is None:
        missing = []
        if not a.valid or a.value is None:
            missing.append("A")
        if not b.valid or b.value is None:
            missing.append("B")
        joined = "/".join(missing) if missing else "A/B"
        return Verdict(
            primary_metric=metric,
            a=a,
            b=b,
            c=c,
            delta_b_minus_a=None,
            delta_pct=None,
            delta_ci_pct=None,
            cut=CUT_INDECISIVE,
            rationale=(
                f"INDECISIVE: la métrica primaria '{metric}' no es aislable/válida "
                f"en el/los brazo(s) {joined} (p.ej. HBM device-wide no aísla KV, "
                "o fuente VRAM no honesta). No se decide sobre un número ausente."
            ),
            quotable=True,
        )

    # Guard 3: division by zero on A would make the percentage meaningless.
    if a.value == 0:
        return Verdict(
            primary_metric=metric,
            a=a,
            b=b,
            c=c,
            delta_b_minus_a=(b.value - a.value),
            delta_pct=None,
            delta_ci_pct=None,
            cut=CUT_INDECISIVE,
            rationale=(
                f"INDECISIVE: el baseline A '{metric}' es 0; el porcentaje "
                "(B-A)/A no está definido."
            ),
            quotable=True,
        )

    delta_abs = b.value - a.value
    raw_pct = delta_abs / a.value * 100.0

    # Orient the percentage so POSITIVE == improvement. For lower-is-better
    # metrics (KV GB, TTFT) a reduction (B<A) is the win, so flip the sign.
    if metric in _LOWER_IS_BETTER:
        improvement_pct = -raw_pct
    else:
        improvement_pct = raw_pct

    cut, rationale = _classify(improvement_pct, metric)

    # CI on the percentage delta (only when both arms carry per-request samples,
    # i.e. TTFT). For scalar primaries (KV GB, decode tok/s aggregate) there is
    # no per-request spread to bootstrap, so delta_ci_pct stays None — honest.
    delta_ci = _delta_ci_pct_from_log(run, metric)
    if delta_ci is not None and metric in _LOWER_IS_BETTER:
        # Re-orient the CI to the improvement convention for the report.
        delta_ci = CI(
            mean=-delta_ci.mean if delta_ci.mean is not None else None,
            lo=-delta_ci.hi if delta_ci.hi is not None else None,
            hi=-delta_ci.lo if delta_ci.lo is not None else None,
            n=delta_ci.n,
            method=delta_ci.method,
            confidence=delta_ci.confidence,
        )

    return Verdict(
        primary_metric=metric,
        a=a,
        b=b,
        c=c,
        delta_b_minus_a=delta_abs,
        delta_pct=improvement_pct,
        delta_ci_pct=delta_ci,
        cut=cut,
        rationale=rationale,
        quotable=True,
    )


def _delta_ci_pct_from_log(run: GateRunResult, metric: str) -> "CI" | None:
    """Bootstrap a CI on the raw (B-A)/A*100 percentage when both A and B expose
    per-request samples for ``metric``. Sign is the *raw* (not improvement-
    oriented) percentage; the caller flips it for lower-is-better metrics."""
    if metric not in {"mean_ttft_s", "p50_ttft_s", "p95_ttft_s"}:
        return None
    a_block = _arm_block(run, ARM_A)
    b_block = _arm_block(run, ARM_B)
    if not a_block or not b_block:
        return None
    _, a_samples = _value_and_samples(a_block, metric)
    _, b_samples = _value_and_samples(b_block, metric)
    if len(a_samples) < 2 or len(b_samples) < 2:
        return None
    import random

    rng = random.Random(0)
    n_boot = 2000
    deltas: list[float] = []
    na, nb = len(a_samples), len(b_samples)
    for _ in range(n_boot):
        ma = sum(a_samples[rng.randrange(na)] for _ in range(na)) / na
        mb = sum(b_samples[rng.randrange(nb)] for _ in range(nb)) / nb
        if ma == 0:
            continue
        deltas.append((mb - ma) / ma * 100.0)
    if len(deltas) < 2:
        return None
    deltas.sort()
    lo = deltas[int(math.floor(0.025 * len(deltas)))]
    hi = deltas[min(len(deltas) - 1, int(math.ceil(0.975 * len(deltas))) - 1)]
    mean = sum(deltas) / len(deltas)
    return CI(mean=mean, lo=lo, hi=hi, n=min(na, nb), method="bootstrap")


# ---------------------------------------------------------------------------
# Report rendering (Spanish prose; numbers strictly from the log)
#
# The renderer fills the canonical template (CONTRACT §12,
# docs/research/_internal/GATE-0-report-TEMPLATE.md). That template uses a
# DOTTED placeholder vocabulary (e.g. {{kv.A.value}}, {{verdict.cut}},
# {{conditions.model}}). This module builds exactly that key set; any value not
# present in the log renders as the pending marker, never a fabricated number.
# ---------------------------------------------------------------------------
PENDING = "_pendiente_"
NOT_COMPUTABLE = "_no computable_"


def _fmt_num(value: float | None, *, decimals: int) -> str:
    """Render a number to the schema's decimals, or the pending marker."""
    if value is None:
        return PENDING
    return f"{value:.{decimals}f}"


def _decimals_for(metric: str) -> int:
    # CONTRACT §1: GB to 3 decimals, tok/s to 1, TTFT seconds to 4.
    if metric == "kv_used_gb":
        return 3
    if metric in {"decode_tok_s", "total_tok_s"}:
        return 1
    if metric in {"mean_ttft_s", "p50_ttft_s", "p95_ttft_s"}:
        return 4
    if metric == "hit_rate":
        return 4
    if metric == "gpu_cache_usage_perc":
        return 4
    return 3


def _scalar(value: Any) -> str:
    """Render any log scalar verbatim, or the pending marker when absent.
    Numbers come straight from the log; we do not reformat ints/bools so the
    report mirrors the raw schema exactly."""
    if value is None:
        return PENDING
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(str(x) for x in value) if value else PENDING
    return str(value)


def _ci_end(value: float | None, *, decimals: int) -> str:
    """One CI bound (lo/hi). Pending when n<2 (no honest spread)."""
    return _fmt_num(value, decimals=decimals)


def _passed_mark(value: Any) -> str:
    if value is None:
        return PENDING
    return "✅" if value else "❌"


def _required_mark(value: Any) -> str:
    if value is None:
        return PENDING
    return "sí" if value else "no"


def _verdict_es(cut: str) -> str:
    return {
        CUT_ABANDON: "ABANDONAR",
        CUT_GREY_ZONE: "ZONA GRIS",
        CUT_INVEST: "INVERTIR (Fase 2)",
        CUT_INDECISIVE: "INDECISO",
    }.get(cut, cut)


def _decision_es(cut: str) -> str:
    return {
        CUT_ABANDON: (
            "ABANDONAR el sharing mecánico. ROMY se reduce a salting honesto "
            "INV-15 en el serving (aislamiento de jueces), no a un optimizador de "
            "memoria. Foco 100% en compresión de tokens + safety O(1) (Plan B, §8 del "
            "protocolo). → Fase 1 del PLAN-DEFINITIVO."
        ),
        CUT_GREY_ZONE: (
            "ZONA GRIS: mantener ROMY solo si el costo de mantenimiento es bajo y no "
            "compite con upstream que lo cierra (vLLM #26201). Re-evaluar contra el "
            "roadmap vLLM/SGLang antes de invertir. → Fase 1 del PLAN-DEFINITIVO."
        ),
        CUT_INVEST: (
            "INVERTIR: ROMY tiene diferencial. Confirmar reproducibilidad y "
            "sostenimiento del delta, luego pasar a Fase 2. → Fase 1 del "
            "PLAN-DEFINITIVO."
        ),
        CUT_INDECISIVE: (
            "INDECISO: no hay número citable para decidir. Resolver la causa de "
            "no-citabilidad (gates de validez requeridos, o aislar la métrica "
            "primaria) y re-ejecutar antes de cualquier veredicto. El gate permanece "
            "abierto; no avanza a Fase 1."
        ),
    }.get(cut, "")


# Canonical → log-key resolvers, keyed by the placeholder's leading token. Each
# entry returns the {{dotted.key}} -> rendered-string pairs for that section.
def _arm_kv_keys(run: GateRunResult, arm: str) -> dict[str, str]:
    """{{kv.<arm>.*}} — PRIMARY 1 KV footprint per arm (kv_used_gb + the
    pre-allocation-robust gpu_cache_usage_perc proxy)."""
    m = arm_metric(run, arm, "kv_used_gb")
    util = arm_metric(run, arm, "gpu_cache_usage_perc")
    block = _arm_block(run, arm) or {}
    kv = block.get("kv_footprint") or {}
    hbm = _hbm_from_block(block) or {}
    dec = _decimals_for("kv_used_gb")
    udec = _decimals_for("gpu_cache_usage_perc")
    return {
        f"kv.{arm}.value": _fmt_num(m.value, decimals=dec),
        f"kv.{arm}.ci.lo": _ci_end(m.ci.lo, decimals=dec),
        f"kv.{arm}.ci.hi": _ci_end(m.ci.hi, decimals=dec),
        f"kv.{arm}.ci.n": _scalar(m.ci.n),
        f"kv.{arm}.ci.method": _scalar(m.ci.method),
        f"kv.{arm}.method": _scalar(kv.get("method")),
        f"kv.{arm}.vram_source": _scalar(hbm.get("vram_source")),
        f"kv.{arm}.valid": _scalar(m.valid),
        f"kv.{arm}.cache_util": _fmt_num(util.value, decimals=udec),
        f"kv.{arm}.cache_util_valid": _scalar(util.valid),
    }


def _arm_tps_keys(run: GateRunResult, arm: str) -> dict[str, str]:
    """{{tps.<arm>.*}} — PRIMARY 2 throughput per arm."""
    m = arm_metric(run, arm, "decode_tok_s")
    block = _arm_block(run, arm) or {}
    tp = block.get("throughput") or {}
    dec = _decimals_for("decode_tok_s")
    return {
        f"tps.{arm}.value": _fmt_num(m.value, decimals=dec),
        f"tps.{arm}.ci.lo": _ci_end(m.ci.lo, decimals=dec),
        f"tps.{arm}.ci.hi": _ci_end(m.ci.hi, decimals=dec),
        f"tps.{arm}.ci.n": _scalar(m.ci.n),
        f"tps.{arm}.ci.method": _scalar(m.ci.method),
        f"tps.{arm}.n_requests": _scalar(tp.get("n_requests")),
        f"tps.{arm}.valid": _scalar(m.valid),
    }


def _arm_ttft_keys(run: GateRunResult, arm: str) -> dict[str, str]:
    """{{ttft.<arm>.*}} — SECONDARY 3 TTFT per arm (mean ± CI, p50, p95)."""
    mean = arm_metric(run, arm, "mean_ttft_s")
    block = _arm_block(run, arm) or {}
    tp = block.get("throughput") or {}
    dec = _decimals_for("mean_ttft_s")
    return {
        f"ttft.{arm}.mean": _fmt_num(mean.value, decimals=dec),
        f"ttft.{arm}.ci.lo": _ci_end(mean.ci.lo, decimals=dec),
        f"ttft.{arm}.ci.hi": _ci_end(mean.ci.hi, decimals=dec),
        f"ttft.{arm}.p50": _fmt_num(_as_float(tp.get("p50_ttft_s")), decimals=dec),
        f"ttft.{arm}.p95": _fmt_num(_as_float(tp.get("p95_ttft_s")), decimals=dec),
        f"ttft.{arm}.n_requests": _scalar(tp.get("n_requests")),
    }


def _arm_prefix_keys(run: GateRunResult, arm: str) -> dict[str, str]:
    """{{prefix.<arm>.*}} — SECONDARY 4 prefix-cache counters + INV-15 fires."""
    block = _arm_block(run, arm) or {}
    pm = block.get("prefix_metrics") or {}
    dec = _decimals_for("hit_rate")
    return {
        f"prefix.{arm}.hit_rate": _fmt_num(_as_float(pm.get("hit_rate")), decimals=dec),
        f"prefix.{arm}.queries_delta": _scalar(pm.get("queries_delta")),
        f"prefix.{arm}.hits_delta": _scalar(pm.get("hits_delta")),
        f"prefix.{arm}.external_hits_delta": _scalar(pm.get("external_hits_delta")),
        f"prefix.{arm}.external_kv_tokens_delta": _scalar(pm.get("external_kv_tokens_delta")),
        f"prefix.{arm}.inv15_fires": _scalar(block.get("inv15_fires")),
    }


def _conditions_keys(conditions: dict) -> dict[str, str]:
    """{{conditions.*}} — the §8 condition block, key by key (pending if absent)."""
    fields = [
        "model", "hardware_label", "vram_source", "second_source", "topology",
        "n_agents", "n_requests", "concurrency", "max_tokens", "approx_prefix_tokens",
        "shared_prefix_fraction", "block_size", "kv_cache_dtype", "max_model_len",
        "gpu_memory_utilization", "aiter_applied", "pythonhashseed",
    ]
    return {f"conditions.{f}": _scalar(conditions.get(f)) for f in fields}


def _reuse_keys(reuse: dict) -> dict[str, str]:
    return {
        "reuse.canonical_prefix_chars": _scalar(reuse.get("canonical_prefix_chars")),
        "reuse.n_distinct_prefixes": _scalar(reuse.get("n_distinct_prefixes")),
        "reuse.shared_prefix_fraction": _scalar(reuse.get("shared_prefix_fraction")),
        "reuse.approx_prefix_tokens": _scalar(reuse.get("approx_prefix_tokens")),
        "reuse.note": _scalar(reuse.get("note")),
    }


def _workload_keys(workload: dict) -> dict[str, str]:
    return {
        "workload.name": _scalar(workload.get("name")),
        "workload.canonical_prefix_hash": _scalar(workload.get("canonical_prefix_hash")),
        "workload.agents": _scalar(workload.get("agents")),
    }


def _validity_keys(run: GateRunResult) -> dict[str, str]:
    """{{validity.<check>.*}} + {{validity.summary}} +
    {{validity.c_control_zero.max_hit_rate}}. Checks the template names
    explicitly; a check absent from the log renders pending."""
    validity = run.get("validity") or {}
    checks = validity.get("checks") or []
    by_name = {c.get("name"): c for c in checks if isinstance(c, dict)}
    names = [
        "apc_on", "c_control_zero", "aiter_parity", "seed_pinned",
        "shared_prefix_single", "vram_source_honest", "n_requests_sufficient",
    ]
    out: dict[str, str] = {}
    for name in names:
        ch = by_name.get(name) or {}
        out[f"validity.{name}.passed"] = _passed_mark(ch.get("passed"))
        out[f"validity.{name}.required"] = _required_mark(ch.get("required"))
        out[f"validity.{name}.detail"] = _scalar(ch.get("detail")).replace("|", "\\|")
    out["validity.summary"] = _scalar(validity.get("summary"))
    # The C-control threshold lives on the check's evidence when present.
    cc = by_name.get("c_control_zero") or {}
    max_hr = (cc.get("evidence") or {}).get("max_hit_rate")
    out["validity.c_control_zero.max_hit_rate"] = _scalar(max_hr)
    return out


def _verdict_keys(verdict: Verdict, run: GateRunResult) -> dict[str, str]:
    """{{verdict.*}}, {{sanity.b_minus_c}}, and the decisive delta keys."""
    dec = _decimals_for(verdict.primary_metric)
    a, b, c = verdict.a, verdict.b, verdict.c

    # (B - C) sanity row: reference only, NEVER the decision (CONTRACT §0).
    if b.value is not None and c.value is not None:
        b_minus_c = f"{(b.value - c.value):+.{dec}f}"
    else:
        b_minus_c = PENDING

    if verdict.delta_pct is None:
        delta_pct = NOT_COMPUTABLE
    else:
        delta_pct = f"{verdict.delta_pct:+.2f}"

    dci = verdict.delta_ci_pct
    return {
        "verdict.a.value": _fmt_num(a.value, decimals=dec),
        "verdict.a.ci.lo": _ci_end(a.ci.lo, decimals=dec),
        "verdict.a.ci.hi": _ci_end(a.ci.hi, decimals=dec),
        "verdict.b.value": _fmt_num(b.value, decimals=dec),
        "verdict.b.ci.lo": _ci_end(b.ci.lo, decimals=dec),
        "verdict.b.ci.hi": _ci_end(b.ci.hi, decimals=dec),
        "verdict.c.hit_rate": _fmt_num(
            arm_metric(run, ARM_C, "hit_rate").value, decimals=_decimals_for("hit_rate")
        ),
        "verdict.delta_b_minus_a": (
            f"{verdict.delta_b_minus_a:+.{dec}f}"
            if verdict.delta_b_minus_a is not None
            else NOT_COMPUTABLE
        ),
        "verdict.delta_pct": delta_pct,
        "verdict.delta_ci_pct.lo": _fmt_num(dci.lo if dci else None, decimals=2),
        "verdict.delta_ci_pct.hi": _fmt_num(dci.hi if dci else None, decimals=2),
        "verdict.delta_ci_pct.n": _scalar(dci.n if dci else None),
        "verdict.delta_ci_pct.method": _scalar(dci.method if dci else None),
        "verdict.cut": verdict.cut,
        "verdict.rationale": verdict.rationale,
        "verdict.quotable": _scalar(verdict.quotable),
        "sanity.b_minus_c": b_minus_c,
    }


def _fill(template: str, mapping: dict[str, str]) -> str:
    """Replace {{key}} occurrences. Longest keys first so dotted keys
    (e.g. verdict.delta_ci_pct.lo) are replaced before any prefix collision."""
    out = template
    for key in sorted(mapping, key=len, reverse=True):
        out = out.replace("{{" + key + "}}", mapping[key])
    return out


def render_report_md(verdict: Verdict, run: GateRunResult, template_path: str) -> str:
    """Fill the GATE-0 report template (CONTRACT §12) with the A/B/C table, the
    (B-A) delta, the verdict vs the cut, and the decision. Spanish prose; numbers
    come from the log only — a value absent from the log renders as the pending
    marker, never a fabricated number.

    The template path is required and must exist: it is the deliverable skeleton
    and the binding placeholder contract. If it is missing we fail loudly rather
    than emit a divergent document.
    """
    tpl_file = Path(template_path)
    if not tpl_file.is_file():
        raise FileNotFoundError(
            f"report template not found: {template_path} "
            "(expected docs/research/_internal/GATE-0-report-TEMPLATE.md, CONTRACT §12)"
        )
    template = tpl_file.read_text(encoding="utf-8")

    workload = run.get("workload") or {}
    conditions = run.get("conditions") or {}
    reuse = run.get("reuse") or {}

    mapping: dict[str, str] = {}

    # Per-arm metric blocks.
    for arm in (ARM_A, ARM_B, ARM_C):
        mapping.update(_arm_kv_keys(run, arm))
        mapping.update(_arm_tps_keys(run, arm))
        mapping.update(_arm_ttft_keys(run, arm))
        mapping.update(_arm_prefix_keys(run, arm))
        block = _arm_block(run, arm) or {}
        mapping[f"server_log.{arm}"] = _scalar(block.get("server_log_path"))

    # Conditions / reuse / workload blocks.
    mapping.update(_conditions_keys(conditions))
    mapping.update(_reuse_keys(reuse))
    mapping.update(_workload_keys(workload))

    # Validity + verdict.
    mapping.update(_validity_keys(run))
    mapping.update(_verdict_keys(verdict, run))

    # Top-level scalars.
    measured = bool(run.get("measured", False))
    raw_paths = run.get("__source_path__") or PENDING
    mapping.update(
        {
            "report_status": (
                "citable" if (verdict.quotable and measured) else "NO citable (dry/validez)"
            ),
            "primary_metric": verdict.primary_metric,
            "timestamp_utc": _scalar(run.get("timestamp_utc")),
            "schema_version": _scalar(run.get("schema_version")),
            "measured": _scalar(measured),
            "raw_log_paths": str(raw_paths),
            "verdict_json_path": str(run.get("__verdict_json_path__") or PENDING),
            "decision_text": _decision_es(verdict.cut),
        }
    )
    return _fill(template, mapping)


# ---------------------------------------------------------------------------
# Verdict serialization (for --out-json)
# ---------------------------------------------------------------------------
def _ci_to_dict(ci: "CI" | None) -> dict | None:
    if ci is None:
        return None
    if is_dataclass(ci):
        return asdict(ci)
    return {
        "mean": getattr(ci, "mean", None),
        "lo": getattr(ci, "lo", None),
        "hi": getattr(ci, "hi", None),
        "n": getattr(ci, "n", 0),
        "method": getattr(ci, "method", "unknown"),
        "confidence": getattr(ci, "confidence", 0.95),
    }


def _arm_metric_to_dict(m: ArmMetric) -> dict:
    return {
        "arm": m.arm,
        "metric": m.metric,
        "value": m.value,
        "ci": _ci_to_dict(m.ci),
        "condition": m.condition,
        "valid": m.valid,
    }


def verdict_to_dict(verdict: Verdict) -> dict:
    return {
        "primary_metric": verdict.primary_metric,
        "a": _arm_metric_to_dict(verdict.a),
        "b": _arm_metric_to_dict(verdict.b),
        "c": _arm_metric_to_dict(verdict.c),
        "delta_b_minus_a": verdict.delta_b_minus_a,
        "delta_pct": verdict.delta_pct,
        "delta_ci_pct": _ci_to_dict(verdict.delta_ci_pct),
        "cut": verdict.cut,
        "rationale": verdict.rationale,
        "quotable": verdict.quotable,
    }


# ---------------------------------------------------------------------------
# CLI (CONTRACT §10)
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gate0-analyze",
        description=(
            "GATE #0 analysis: read raw A/B/C logs, build the per-metric table "
            "(value ± CI + conditions), compute (B-A) delta, apply the "
            "preregistered cut, and emit the Spanish markdown report. Never "
            "invents numbers; a missing arm is marked pending."
        ),
    )
    parser.add_argument(
        "--in",
        dest="inputs",
        nargs="+",
        required=True,
        metavar="RAW_LOG",
        help="one or more §9 raw-log JSON files (later files override earlier "
        "arms on key collisions, e.g. a cross-worker B over a single-worker run).",
    )
    parser.add_argument(
        "--primary",
        choices=[PRIMARY_KV, PRIMARY_KV_UTIL, PRIMARY_THROUGHPUT],
        default=PRIMARY_KV,
        help="primary metric the cut is decided on (default: kv_footprint). On vLLM "
        "prefer 'kv_cache_util' (gpu_cache_usage_perc delta): device-wide kv_footprint "
        "dilutes to ~0 under vLLM's pre-allocated KV pool, while the utilization gauge "
        "still shows ROMY's sharing. 'throughput' decides on decode tok/s.",
    )
    parser.add_argument(
        "--out-md",
        dest="out_md",
        default=None,
        help="path to write the rendered markdown report (default: stdout).",
    )
    parser.add_argument(
        "--out-json",
        dest="out_json",
        default=None,
        help="path to write the machine-readable verdict JSON.",
    )
    parser.add_argument(
        "--template",
        dest="template",
        default=DEFAULT_TEMPLATE,
        help=f"report template path (default: {DEFAULT_TEMPLATE}).",
    )
    args = parser.parse_args(argv)

    # Merge multiple raw logs: union of arms (later wins on collision), keeping
    # the first log's run-level metadata as the base. This lets an operator
    # combine a single-worker run with a cross-worker B without re-running.
    runs = []
    for path in args.inputs:
        try:
            run = load_raw(path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        run["__source_path__"] = path
        runs.append(run)

    merged = _merge_runs(runs)
    # Surface the verdict-json destination so the report's {{verdict_json_path}}
    # placeholder resolves (set before rendering; written below).
    if args.out_json:
        merged["__verdict_json_path__"] = args.out_json

    verdict = compute_verdict(merged, primary_metric=args.primary)
    try:
        report = render_report_md(verdict, merged, args.template)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"wrote report: {out}", file=sys.stderr)
    else:
        print(report)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(verdict_to_dict(verdict), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote verdict: {out}", file=sys.stderr)

    # Always surface the headline on stderr so the operator sees it even when
    # the report goes to a file.
    print(
        f"VERDICT: {verdict.cut} "
        f"(delta_pct={'n/a' if verdict.delta_pct is None else f'{verdict.delta_pct:+.2f}%'}) "
        f"— {verdict.rationale}",
        file=sys.stderr,
    )
    return 0


def _merge_runs(runs: list[GateRunResult]) -> GateRunResult:
    """Combine multiple raw logs into one for analysis. Base = first run; later
    runs' arms override on key collision (e.g. cross-worker B over single B).
    Numeric fields are never combined arithmetically — only whole arm blocks are
    swapped, preserving each metric's source and condition."""
    if not runs:
        raise ValueError("no raw logs to merge")
    base = dict(runs[0])
    base_arms = dict(base.get("arms") or {})
    sources = [r.get("__source_path__") for r in runs if r.get("__source_path__")]
    for r in runs[1:]:
        for arm_key, arm_block in (r.get("arms") or {}).items():
            if arm_block:  # only override with a non-empty arm block
                base_arms[arm_key] = arm_block
        # A non-quotable later run must not silently upgrade quotability.
        later_validity = r.get("validity")
        if later_validity and not later_validity.get("quotable", False):
            base["validity"] = later_validity
    base["arms"] = base_arms
    base["__source_path__"] = " + ".join(sources) if sources else base.get("__source_path__")
    return base


if __name__ == "__main__":
    raise SystemExit(main())
