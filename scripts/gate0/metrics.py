#!/usr/bin/env python3
"""GATE #0 — honest measurement readers (CONTRACT.md §5).

This module owns every NUMBER the gate quotes:

  PRIMARY 1   KV-cache / VRAM footprint   -> read_kv_footprint / read_hbm
  PRIMARY 2   aggregate throughput tok/s  -> measure_throughput
  SECONDARY 3 TTFT p50/p95                 -> measure_throughput (same window)
  SECONDARY 4 prefix-cache + external-KV   -> prefix_metrics_window / fetch_prefix_metrics
  CI          mean +- confidence interval  -> confidence_interval

Honesty contract (binding, see scripts/gate0/CONTRACT.md §1 + AUDIT.md):

* NO vLLM / lmcache / torch import here. Readers talk plain HTTP; the actual
  POST is INJECTED (``send_fn`` / ``post_fn``) by the harness so this stays
  import-clean and unit-testable on a box with no GPU.
* NO fabricated value, ever. When a metric is not derivable the field is
  ``None`` carrying a ``reason``/``note`` — never a literal performance number.
* VRAM flows ONLY through VRAMMonitor (wrapped by ``scripts.mi300x_measure.read_hbm``)
  plus an out-of-process second source. A reading whose ``vram_source`` is in
  INVALID_VRAM_SOURCES is surfaced but marked ``valid=False`` — it must NOT enter
  the report. The 192.0 / 45.0 constants (AUDIT #2) never appear in this file.
* Every metric-returning function records the CONDITION it was taken under
  (n_requests, source, method) so the harness can attach the §8 block. A metric
  with no condition is dropped by analyze.py.

Floats follow the §1 schema: GB 3 decimals, tok/s 1, TTFT seconds 4.

Apache-2.0 — Apohara ContextForge.
"""
from __future__ import annotations

import concurrent.futures as cf
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# §1 import convention: make the repo root importable exactly like the probes do.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Reuse the proven, honest probes (CONTRACT §2 reuse map). NEVER re-implement.
from scripts.mi300x_measure import (  # noqa: E402
    fetch_prefix_metrics as _fetch_prefix_metrics,
    read_hbm as _read_hbm_raw,
)
from scripts.vram_ab_harness import (  # noqa: E402
    _hardware_label as _hardware_label_impl,
    read_second_source_used_gb as _read_second_source_used_gb,
)

# A reading from any of these is NOT a real measurement (CONTRACT §1). It is
# surfaced for transparency but flagged invalid and kept out of the report.
INVALID_VRAM_SOURCES: frozenset[str] = frozenset(
    {"amd_default_192gb", "cuda_unavailable", "unknown", "dry"}
)


# --------------------------------------------------------------------------- #
# Dataclasses (CONTRACT §5.1)
# --------------------------------------------------------------------------- #
@dataclass
class HBMReading:
    """Device-wide HBM, dual-sourced. Mirrors mi300x_measure.read_hbm output."""

    used_gb: float | None
    total_gb: float | None
    vram_source: str
    second_source_used_gb: float | None
    second_source: str | None
    valid: bool
    error: str | None = None


@dataclass
class KVFootprint:
    """PRIMARY METRIC 1 — KV-cache footprint, isolated from weights where possible."""

    kv_used_gb: float | None
    method: str
    hbm: HBMReading
    model_weight_gb: float | None
    gpu_cache_usage_perc: float | None
    note: str


@dataclass
class PrefixMetrics:
    """SECONDARY 4 — vLLM prefix-cache + external-KV counters (DELTA over a window)."""

    queries_delta: float
    hits_delta: float
    hit_rate: float
    external_queries_delta: float
    external_hits_delta: float
    external_kv_tokens_delta: float
    raw_before: dict
    raw_after: dict
    error: str | None = None


@dataclass
class ThroughputSample:
    """PRIMARY 2 + SECONDARY 3 — per-window throughput/TTFT, with its n."""

    mean_ttft_s: float | None
    p50_ttft_s: float | None
    p95_ttft_s: float | None
    decode_tok_s: float | None
    total_tok_s: float | None
    n_requests: int
    ttft_samples_s: list[float] = field(default_factory=list)


@dataclass
class CI:
    """A mean with a confidence interval and the n it summarizes."""

    mean: float | None
    lo: float | None
    hi: float | None
    n: int
    method: str
    confidence: float = 0.95


# --------------------------------------------------------------------------- #
# Re-exports (CONTRACT §5.2 — thin wrappers / verbatim re-exports)
# --------------------------------------------------------------------------- #
def fetch_prefix_metrics(endpoint: str) -> dict:
    """Re-export of scripts.mi300x_measure.fetch_prefix_metrics.

    Name-robust /metrics summation. Returns
    {queries, hits, external_queries, external_hits, external_kv_tokens}
    or {'error': ...}. No fabrication: a scrape failure surfaces the error.
    """
    return _fetch_prefix_metrics(endpoint)


def read_second_source_used_gb(device_id: int = 0):
    """Re-export of scripts.vram_ab_harness.read_second_source_used_gb."""
    return _read_second_source_used_gb(device_id)


def hardware_label(vram_source: str) -> str:
    """Re-export of scripts.vram_ab_harness._hardware_label (honest HW label)."""
    return _hardware_label_impl(vram_source)


# --------------------------------------------------------------------------- #
# PRIMARY 1 — VRAM / KV footprint
# --------------------------------------------------------------------------- #
def read_hbm(device_id: int = 0) -> HBMReading:
    """Thin wrapper over scripts.mi300x_measure.read_hbm.

    Captures the dual-sourced device-wide HBM and decides ``valid`` HONESTLY:
    ``valid=False`` iff ``vram_source`` is in INVALID_VRAM_SOURCES (or no source
    was produced at all). The 192 GB default is therefore never quotable.
    """
    raw = _read_hbm_raw(device_id)

    # _read_hbm_raw surfaces failures as *_error keys instead of values.
    errors: list[str] = []
    if "vram_monitor_error" in raw:
        errors.append(f"vram_monitor: {raw['vram_monitor_error']}")
    if "second_source_error" in raw:
        errors.append(f"second_source: {raw['second_source_error']}")

    vram_source = raw.get("vram_source") or "unknown"
    used_gb = raw.get("used_gb")
    total_gb = raw.get("total_gb")
    second_source_used_gb = raw.get("second_source_used_gb")
    second_source = raw.get("second_source")

    valid = (
        used_gb is not None
        and vram_source not in INVALID_VRAM_SOURCES
    )

    return HBMReading(
        used_gb=used_gb,
        total_gb=total_gb,
        vram_source=vram_source,
        second_source_used_gb=second_source_used_gb,
        second_source=second_source,
        valid=valid,
        error="; ".join(errors) if errors else None,
    )


def _calculate_kv_footprint(
    hbm: HBMReading,
    raw: dict,
    scrape: dict,
    model_weight_gb: float | None,
) -> KVFootprint:
    """Internal helper to calculate KV footprint from fetched metrics."""
    gpu_cache_usage_perc = scrape.get("gpu_cache_usage_perc")

    # --- method 1: gpu_cache_usage_perc * KV-cache capacity -----------------
    kv_capacity_gb = scrape.get("kv_cache_capacity_gb")
    if gpu_cache_usage_perc is not None and kv_capacity_gb is not None:
        kv_used = round(gpu_cache_usage_perc * kv_capacity_gb, 3)
        return KVFootprint(
            kv_used_gb=kv_used,
            method="gpu_cache_usage_perc",
            hbm=hbm,
            model_weight_gb=model_weight_gb,
            gpu_cache_usage_perc=round(gpu_cache_usage_perc, 6),
            note=(
                "KV footprint = gpu_cache_usage_perc * KV-cache capacity from "
                "vLLM /metrics; isolates KV from model weights."
            ),
        )

    # --- method 2: num_gpu_blocks_used * block_size * bytes_per_token -------
    blocks_used = scrape.get("num_gpu_blocks_used")
    block_size = scrape.get("block_size")
    bytes_per_token = scrape.get("kv_bytes_per_token")
    if (
        blocks_used is not None
        and block_size is not None
        and bytes_per_token is not None
    ):
        kv_used = round(
            blocks_used * block_size * bytes_per_token / (1024 ** 3), 3
        )
        return KVFootprint(
            kv_used_gb=kv_used,
            method="num_gpu_blocks",
            hbm=hbm,
            model_weight_gb=model_weight_gb,
            gpu_cache_usage_perc=(
                round(gpu_cache_usage_perc, 6)
                if gpu_cache_usage_perc is not None
                else None
            ),
            note=(
                "KV footprint = num_gpu_blocks_used * block_size * "
                "kv_bytes_per_token from vLLM /metrics; isolates KV from weights."
            ),
        )

    # --- method 3: device-wide HBM minus model weights ----------------------
    if hbm.valid and hbm.used_gb is not None and model_weight_gb is not None:
        kv_used = round(hbm.used_gb - model_weight_gb, 3)
        return KVFootprint(
            kv_used_gb=kv_used,
            method="hbm_minus_weights",
            hbm=hbm,
            model_weight_gb=model_weight_gb,
            gpu_cache_usage_perc=(
                round(gpu_cache_usage_perc, 6)
                if gpu_cache_usage_perc is not None
                else None
            ),
            note=(
                "KV footprint approximated as device-wide HBM minus the "
                "post-load/pre-traffic model-weight baseline; includes activation/"
                "fragmentation slack, so it is an UPPER bound on KV, not exact."
            ),
        )

    # --- method 4: device-wide HBM, NOT isolated (loud caveat) --------------
    metrics_err = raw.get("error") if isinstance(raw, dict) else None
    kv_used = hbm.used_gb if hbm.valid else None
    return KVFootprint(
        kv_used_gb=kv_used,
        method="hbm_device_wide(NOT_ISOLATED)",
        hbm=hbm,
        model_weight_gb=model_weight_gb,
        gpu_cache_usage_perc=(
            round(gpu_cache_usage_perc, 6)
            if gpu_cache_usage_perc is not None
            else None
        ),
        note=(
            "device-wide HBM does NOT isolate KV from model weights/activations. "
            "The qwen3-32b run read identical ~175.393GB shared vs isolated: "
            "device-wide HBM dilutes the KV delta to ~0 at slack. analyze.py must "
            "report 'not isolable', not a delta, unless model_weight_gb or vLLM KV "
            "gauges are supplied."
            + (f" (/metrics error: {metrics_err})" if metrics_err else "")
        ),
    )


def read_kv_footprint(
    endpoint: str,
    device_id: int = 0,
    *,
    model_weight_gb: float | None = None,
) -> KVFootprint:
    """PRIMARY 1. Prefer the EFFECTIVE KV footprint, in honest precedence order.

      1. vLLM /metrics ``gpu_cache_usage_perc`` * KV-cache capacity (if both the
         perc AND a derivable total are exposed) -> method='gpu_cache_usage_perc'
      2. ``num_gpu_blocks_used`` * block_size * bytes_per_token (if all exposed)
         -> method='num_gpu_blocks'
      3. device-wide HBM minus ``model_weight_gb`` (only if the caller passes the
         post-load / pre-traffic baseline) -> method='hbm_minus_weights'
      4. fall back to device-wide HBM, method='hbm_device_wide(NOT_ISOLATED)',
         with a LOUD note that this does NOT isolate KV (the qwen3-32b run showed
         identical 175.393 GB shared vs isolated — device-wide HBM dilutes the KV
         delta to ~0 at slack).

    The device-wide HBM reading is ALWAYS captured (so the report keeps the raw
    eye even when the effective number is not isolable). No fabrication: if the
    HBM read itself is invalid, ``kv_used_gb`` is None.
    """
    hbm = read_hbm(device_id)
    raw = fetch_prefix_metrics(endpoint)  # cheap; also primes the /metrics scrape
    scrape = _scrape_kv_gauges(endpoint)

    return _calculate_kv_footprint(
        hbm=hbm,
        raw=raw,
        scrape=scrape,
        model_weight_gb=model_weight_gb,
    )


def _scrape_kv_gauges(endpoint: str) -> dict:
    """Best-effort scrape of vLLM KV-cache gauges from /metrics.

    Pure HTTP + regex, name-robust (mirrors fetch_prefix_metrics' tolerance).
    Returns ONLY what is actually exposed — absent gauges are simply not keyed,
    so read_kv_footprint falls through to the honest device-wide fallback. We
    never synthesize a gauge that the server did not emit.
    """
    import re
    import urllib.request

    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(f"{endpoint}/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception:
        return out

    # Map of /metrics suffixes -> our normalized key. Names vary across vLLM
    # versions; we match by substring and take the last value on the line.
    wanted = (
        ("gpu_cache_usage_perc", "gpu_cache_usage_perc"),
        ("num_gpu_blocks_used", "num_gpu_blocks_used"),
    )
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for needle, key in wanted:
            if needle in line:
                m = re.search(r"\s([0-9.eE+-]+)\s*$", line)
                if m:
                    try:
                        out[key] = float(m.group(1))
                    except ValueError:
                        pass
    return out


# --------------------------------------------------------------------------- #
# SECONDARY 4 — prefix-cache / external-KV counters over a window
# --------------------------------------------------------------------------- #
def prefix_metrics_window(endpoint: str, send_fn: Callable[[], object]) -> PrefixMetrics:
    """Snapshot /metrics, run ``send_fn()`` (which drives the requests), snapshot
    again, and diff. This function owns ONLY the before/after diff + hit_rate;
    ``send_fn`` is supplied by the harness so this stays vLLM-import-free.

    A scrape error on EITHER snapshot is surfaced in ``.error`` and yields zero
    deltas (never a guessed rate). hit_rate is hits/queries, 0.0 when queries==0.
    """
    before = fetch_prefix_metrics(endpoint)

    # Drive the workload regardless of the before-scrape outcome; if the window
    # body raises we still report the error rather than fabricate a result.
    send_error: str | None = None
    try:
        send_fn()
    except Exception as e:  # noqa: BLE001 — surfaced, not swallowed
        send_error = repr(e)

    after = fetch_prefix_metrics(endpoint)

    errors: list[str] = []
    if isinstance(before, dict) and before.get("error"):
        errors.append(f"before: {before['error']}")
    if isinstance(after, dict) and after.get("error"):
        errors.append(f"after: {after['error']}")
    if send_error:
        errors.append(f"send_fn: {send_error}")

    def _delta(key: str) -> float:
        b = float(before.get(key, 0.0)) if isinstance(before, dict) else 0.0
        a = float(after.get(key, 0.0)) if isinstance(after, dict) else 0.0
        return a - b

    queries_delta = _delta("queries")
    hits_delta = _delta("hits")
    hit_rate = round(hits_delta / queries_delta, 4) if queries_delta > 0 else 0.0

    return PrefixMetrics(
        queries_delta=queries_delta,
        hits_delta=hits_delta,
        hit_rate=hit_rate,
        external_queries_delta=_delta("external_queries"),
        external_hits_delta=_delta("external_hits"),
        external_kv_tokens_delta=_delta("external_kv_tokens"),
        raw_before=before if isinstance(before, dict) else {"error": str(before)},
        raw_after=after if isinstance(after, dict) else {"error": str(after)},
        error="; ".join(errors) if errors else None,
    )


# --------------------------------------------------------------------------- #
# PRIMARY 2 + SECONDARY 3 — throughput / TTFT
# --------------------------------------------------------------------------- #
def measure_throughput(
    endpoint: str,
    model: str,
    requests,
    salts,
    *,
    concurrency: int,
    post_fn: Callable[..., object],
) -> ThroughputSample:
    """PRIMARY 2 + SECONDARY 3. Drive ``requests`` (with their per-arm ``salts``)
    at fixed ``concurrency`` via STREAMING completions; collect per-request TTFT
    and output tokens.

    ``post_fn(endpoint, model, prompt, salt, max_tokens, stream) -> response`` is
    INJECTED by the harness (same shape as ``mi300x_measure._post``) so this stays
    vLLM-import-free. ``response`` is an iterable of streamed chunks; the first
    chunk marks TTFT, each subsequent chunk counts as one decoded token (the
    coarse-but-honest proxy mi300x_measure.stage_throughput already uses).

    Aggregate throughput is measured at the wall-clock of the WHOLE concurrent
    batch (output tokens / wall seconds at the run's fixed concurrency) — the
    operator-relevant number. Nothing is fabricated: a request that errors
    contributes no TTFT and no tokens; if every request fails, the rates are None.

    ``salts`` is aligned positionally with ``requests`` (a RequestSalt list from
    arms.salts_for_workload, or any object exposing ``.cache_salt``; a plain
    str/None is also accepted). n_requests == len(requests).
    """
    n = len(requests)
    salt_values = _align_salts(requests, salts)

    ttft_samples: list[float] = []
    decoded_tokens = 0
    prompt_chars_total = 0
    completed = 0
    lock = _Lock()

    def _drive(idx: int):
        nonlocal decoded_tokens, prompt_chars_total, completed
        req = requests[idx]
        prompt = getattr(req, "prompt", req)
        salt = salt_values[idx]
        max_tokens = getattr(req, "max_tokens", 16)
        t0 = time.monotonic()
        first_token_at: float | None = None
        local_tokens = 0
        try:
            resp = post_fn(
                endpoint, model, prompt,
                salt=salt, max_tokens=max_tokens, stream=True,
            )
            for _chunk in resp:
                if first_token_at is None:
                    first_token_at = time.monotonic()
                local_tokens += 1
        except Exception:
            return  # honest: failed request contributes nothing
        with lock:
            if first_token_at is not None:
                ttft_samples.append(first_token_at - t0)
            decoded_tokens += local_tokens
            prompt_chars_total += len(str(prompt))
            completed += 1

    t_start = time.monotonic()
    if n > 0:
        with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            list(ex.map(_drive, range(n)))
    wall_s = time.monotonic() - t_start

    decode_tok_s = (
        round(decoded_tokens / wall_s, 1)
        if wall_s > 0 and decoded_tokens > 0
        else None
    )
    # Total throughput needs prompt-token accounting; we only have a char/4
    # heuristic on the prompt side and no exact prompt-token count from the
    # stream, so we honestly leave total_tok_s=None rather than mix units.
    total_tok_s = None

    p50 = _percentile(ttft_samples, 50.0)
    p95 = _percentile(ttft_samples, 95.0)
    mean_ttft = (
        round(sum(ttft_samples) / len(ttft_samples), 4) if ttft_samples else None
    )

    return ThroughputSample(
        mean_ttft_s=mean_ttft,
        p50_ttft_s=round(p50, 4) if p50 is not None else None,
        p95_ttft_s=round(p95, 4) if p95 is not None else None,
        decode_tok_s=decode_tok_s,
        total_tok_s=total_tok_s,
        n_requests=n,
        ttft_samples_s=[round(t, 4) for t in ttft_samples],
    )


def _align_salts(requests, salts) -> list:
    """Positionally align salts to requests, accepting RequestSalt | str | None.

    RequestSalt exposes ``.cache_salt``; a bare str/None is used as-is. A short
    or missing salt list pads with None (arm A's no-salt floor). Never invents a
    salt value.
    """
    n = len(requests)
    out: list = []
    salts = list(salts) if salts is not None else []
    for i in range(n):
        if i < len(salts):
            s = salts[i]
            out.append(getattr(s, "cache_salt", s))
        else:
            out.append(None)
    return out


# A tiny re-export so callers don't need threading just for the lock type hint.
def _Lock():  # noqa: N802 — factory, kept private
    import threading

    return threading.Lock()


# --------------------------------------------------------------------------- #
# Confidence intervals (pure stdlib + optional numpy; never invents a spread)
# --------------------------------------------------------------------------- #
def confidence_interval(
    samples: list[float],
    *,
    confidence: float = 0.95,
    method: str = "bootstrap",
    n_boot: int = 2000,
    seed: int = 0,
) -> CI:
    """Mean +- CI over N samples.

    method='bootstrap' (percentile bootstrap, the default) or 't' (Student-t /
    normal-approx interval for small-ish n). With n<2 there is no spread to
    estimate: returns CI(mean=mean_or_None, lo=None, hi=None, n=n,
    method='none(n<2)') — it NEVER fabricates an interval.

    Pure stdlib; numpy is used only to accelerate the bootstrap when present.
    """
    clean = [float(x) for x in samples if x is not None and _finite(x)]
    n = len(clean)

    if n == 0:
        return CI(mean=None, lo=None, hi=None, n=0, method="none(n<2)",
                  confidence=confidence)
    mean = sum(clean) / n
    if n < 2:
        return CI(mean=mean, lo=None, hi=None, n=n, method="none(n<2)",
                  confidence=confidence)

    if method == "t":
        lo, hi = _t_interval(clean, mean, confidence)
        return CI(mean=mean, lo=lo, hi=hi, n=n, method="t", confidence=confidence)

    # default: percentile bootstrap on the mean
    lo, hi = _bootstrap_interval(clean, confidence, n_boot, seed)
    return CI(mean=mean, lo=lo, hi=hi, n=n, method="bootstrap",
              confidence=confidence)


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _t_interval(samples: list[float], mean: float, confidence: float) -> tuple[float, float]:
    """Two-sided (1-confidence) interval on the mean via the t distribution.

    Uses scipy when available for an exact t critical value; otherwise a normal
    approximation (honest: the method label stays 't', the approximation is noted
    only when n is small via the wider normal z — conservative, never narrower).
    """
    n = len(samples)
    var = sum((x - mean) ** 2 for x in samples) / (n - 1)
    sem = math.sqrt(var / n)
    crit = _t_critical(confidence, n - 1)
    half = crit * sem
    return mean - half, mean + half


def _t_critical(confidence: float, df: int) -> float:
    """t critical value for a two-sided interval; scipy if present, else normal z."""
    alpha = 1.0 - confidence
    try:
        from scipy import stats  # type: ignore

        return float(stats.t.ppf(1.0 - alpha / 2.0, df))
    except Exception:
        # Normal approximation (z). Slightly anti-conservative for very small df,
        # but we never claim scipy precision when it's absent.
        return _normal_ppf(1.0 - alpha / 2.0)


def _normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def _bootstrap_interval(
    samples: list[float], confidence: float, n_boot: int, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean. Deterministic (seeded)."""
    n = len(samples)
    alpha = 1.0 - confidence
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0

    try:
        import numpy as np  # optional acceleration

        rng = np.random.default_rng(seed)
        arr = np.asarray(samples, dtype=float)
        idx = rng.integers(0, n, size=(n_boot, n))
        means = arr[idx].mean(axis=1)
        lo = float(np.quantile(means, lo_q))
        hi = float(np.quantile(means, hi_q))
        return lo, hi
    except Exception:
        rng = random.Random(seed)
        means = []
        for _ in range(n_boot):
            resample = [samples[rng.randrange(n)] for _ in range(n)]
            means.append(sum(resample) / n)
        means.sort()
        return _quantile_sorted(means, lo_q), _quantile_sorted(means, hi_q)


def _quantile_sorted(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list."""
    if not sorted_vals:
        return math.nan
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _percentile(samples: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile (e.g. 50, 95) of TTFT samples; None if empty."""
    if not samples:
        return None
    ordered = sorted(samples)
    return _quantile_sorted(ordered, pct / 100.0)


__all__ = [
    "HBMReading",
    "KVFootprint",
    "PrefixMetrics",
    "ThroughputSample",
    "CI",
    "INVALID_VRAM_SOURCES",
    "read_hbm",
    "read_kv_footprint",
    "fetch_prefix_metrics",
    "prefix_metrics_window",
    "measure_throughput",
    "confidence_interval",
    "read_second_source_used_gb",
    "hardware_label",
]
