"""GATE #0 — unit tests for the kv_cache_util primary metric (gpu_cache_usage_perc delta).

These exercise analyze.compute_verdict against synthetic raw-log dicts ONLY (no GPU, no
vLLM, no network). They lock in the pre-allocation-robust KV metric: vLLM reserves the
whole KV pool at startup, so device-wide kv_used_gb dilutes to ~0; gpu_cache_usage_perc
(fraction of the pool in use) still shows ROMY's sharing. Lower is better, so a B below A
at the same workload is the win and the cut is applied to the sign-flipped improvement.

No fabricated production numbers: every value here is an explicit synthetic fixture.
"""
from __future__ import annotations

from scripts.gate0.analyze import (
    CUT_ABANDON,
    CUT_GREY_ZONE,
    CUT_INDECISIVE,
    CUT_INVEST,
    PRIMARY_KV_UTIL,
    compute_verdict,
)


def _run(util_a, util_b, *, util_c=0.79, quotable=True, measured=True):
    """Synthetic §9-shaped raw log with only the fields compute_verdict reads for the
    kv_cache_util primary. util_* is gpu_cache_usage_perc (fraction 0..1); None => absent."""

    def _arm(util):
        kv = {} if util is None else {"gpu_cache_usage_perc": util, "method": "synthetic"}
        return {"measured": measured, "kv_footprint": kv, "topology": "single_worker"}

    return {
        "schema_version": "test",
        "measured": measured,
        "conditions": {"model": "Qwen3-32B", "topology": "single_worker"},
        "validity": {"quotable": quotable, "checks": []},
        "arms": {"A": _arm(util_a), "B": _arm(util_b), "C": _arm(util_c)},
    }


def _verdict(util_a, util_b, **kw):
    return compute_verdict(_run(util_a, util_b, **kw), primary_metric=PRIMARY_KV_UTIL)


def test_metric_resolves_to_gpu_cache_usage_perc():
    v = _verdict(0.80, 0.60)
    assert v.primary_metric == "gpu_cache_usage_perc"


def test_lower_util_on_b_is_a_positive_improvement():
    # B uses less of the pool than A -> ROMY shares -> improvement is POSITIVE.
    v = _verdict(0.80, 0.60)
    assert v.delta_b_minus_a < 0  # raw (B-A) is negative (B lower)
    assert v.delta_pct is not None and v.delta_pct > 0  # oriented: positive == better


def test_big_reduction_invests():
    # (0.60-0.80)/0.80 = -25% raw -> +25% improvement -> > 15% -> INVEST.
    v = _verdict(0.80, 0.60)
    assert v.cut == CUT_INVEST
    assert round(v.delta_pct, 1) == 25.0


def test_grey_zone():
    # (0.72-0.80)/0.80 = -10% raw -> +10% improvement -> in [5,15] -> GREY_ZONE.
    v = _verdict(0.80, 0.72)
    assert v.cut == CUT_GREY_ZONE


def test_tiny_reduction_abandons():
    # (0.78-0.80)/0.80 = -2.5% raw -> +2.5% improvement -> < 5% -> ABANDON.
    v = _verdict(0.80, 0.78)
    assert v.cut == CUT_ABANDON


def test_b_worse_than_a_abandons():
    # ROMY makes it worse (B uses MORE pool): improvement negative -> < 5% -> ABANDON.
    v = _verdict(0.80, 0.90)
    assert v.cut == CUT_ABANDON
    assert v.delta_pct < 0


def test_missing_b_is_indecisive():
    v = _verdict(0.80, None)
    assert v.cut == CUT_INDECISIVE
    assert v.delta_pct is None


def test_not_quotable_is_indecisive():
    v = _verdict(0.80, 0.60, quotable=False)
    assert v.cut == CUT_INDECISIVE


def test_dry_unmeasured_is_indecisive():
    # measured=False arms are not valid -> INDECISIVE even if values are present.
    v = _verdict(0.80, 0.60, measured=False)
    assert v.cut == CUT_INDECISIVE
