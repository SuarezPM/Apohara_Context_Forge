#!/usr/bin/env bash
# CI honesty guard — fails the build if a hardcoded performance number
# sneaks back into demo/. The V6.0 benchmark shipped with
# `duration_ms=250.0` (and four siblings) hardcoded; V6.1 wired real
# time.perf_counter() calls. This script keeps it that way.
#
# Run: ./scripts/check_honesty.sh
# Exit: 0 if clean, 1 if a regression is detected.
#
# What we forbid:
#   * `duration_ms = <number>`  in demo/*.py — must come from a real
#     time.perf_counter() pair.
#   * `vram_peak_gb = <number>` in demo/*.py — must come from VRAMMonitor.
#   * The Chinese-character rocm-smi flag.
#
# Allowlist:
#   * demo/benchmark.py keeps its per-scenario `vram_peak_gb` floats —
#     they were never claimed to be measured. We only forbid duration_ms
#     in that file.
#   * A `duration_ms=0` sentinel returned on the exception path is not a
#     measurement, and is excluded.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

violations=0

echo "🕵  honesty guard — scanning demo/ and apohara_context_forge/"
echo

# 1. Hardcoded duration_ms in demo/benchmark*.py (excluding the
#    sentinel `duration_ms=0` exception path).
echo "▸ hardcoded duration_ms in benchmark scenarios"
if grep -nE "duration_ms\s*=\s*[1-9][0-9]*\.?[0-9]*" \
       demo/benchmark.py 2>/dev/null \
   | grep -v -E "scenario_id=.*duration_ms=0" \
   | grep -v "duration_ms=0," ; then
    violations=$((violations + 1))
    echo "  ❌ regression: at least one duration_ms is a hardcoded literal."
    echo "     Wrap the scenario body in t = time.perf_counter() instead."
    echo
fi

# 2. Hardcoded vram_peak_gb in V5+ benchmarks.
echo "▸ hardcoded vram_peak_gb (V5+ benchmarks only)"
if grep -nE "vram_peak_gb\s*=\s*[0-9]+\.?[0-9]*" \
       demo/benchmark.py 2>/dev/null \
   | grep -v "scenario_id=.*vram_peak_gb=0," ; then
    : # V5 scenarios still report indicative VRAM peaks until we wire
      # VRAMMonitor into the benchmark harness. This is documented
      # truth-up-debt; we warn but do not fail.
    echo "  ⚠  V5 scenarios still report indicative vram_peak_gb literals."
    echo "    Tracked under V6.1+ work: wire VRAMMonitor into the harness."
fi

# 3. The Chinese-character rocm-smi flag. This must never come back.
#    We only flag lines where 占用率 appears inside an active subprocess
#    invocation — lines that mention the historical flag in a
#    docstring or comment (the truth-up note) are explicitly allowed.
echo "▸ rocm-smi flag with Chinese characters (in active subprocess calls)"
if grep -rn --include="*.py" -E "(subprocess\.|Popen|\.run\()[^#]*占用率" \
       apohara_context_forge demo agents 2>/dev/null; then
    violations=$((violations + 1))
    echo "  ❌ regression: --showgpu占用率 detected inside an active call."
    echo "     Use --showuse / --showmemuse instead."
    echo
fi

# 4. The fabricated draft_prob_estimate formula must only appear inside
#    a function that documents the legacy fallback (visible via
#    'INV-12 (target distribution preservation) is NOT guaranteed' nearby).
echo "▸ draft_prob_estimate must only live in the legacy fallback branch"
if grep -n "draft_prob_estimate" apohara_context_forge/decoding/speculative_coordinator.py >/dev/null 2>&1; then
    # File contains the symbol — verify the warning string is present
    # alongside it.
    if ! grep -q "INV-12.*NOT guaranteed" apohara_context_forge/decoding/speculative_coordinator.py; then
        violations=$((violations + 1))
        echo "  ❌ regression: draft_prob_estimate present without the"
        echo "     'INV-12 NOT guaranteed' warning. The fabricated q_i path"
        echo "     must be opt-in and audible."
        echo
    fi
fi

# 5. The "return 45.0, 192.0" hardcoded VRAM tuple. Killed in V6.1.
echo "▸ hardcoded VRAM fallback tuple in metrics/collector.py"
if grep -n "return 45.0, 192.0" apohara_context_forge/metrics/collector.py >/dev/null 2>&1; then
    violations=$((violations + 1))
    echo "  ❌ regression: metrics/collector.py contains 'return 45.0, 192.0'."
    echo "     Use VRAMMonitor instead — see commit e0362d7."
    echo
fi

# 6. AUDIT #30 (Sprint 5) — forbid hardcoded tokens/s or VRAM literals
#    in the wow8gb bench. Every value in that file must come from
#    VRAMMonitor or time.perf_counter(). Allowlist is not needed
#    because every assignment in the bench reads from a probe.
echo "▸ hardcoded tokens/s literal in bench_wow8gb.py (AUDIT #30)"
if grep -nE "(tokens_per_sec|tps|t_per_s)\s*=\s*[0-9]+\.[0-9]+\b" \
       apohara_context_forge/benchmarks/apohara2/bench_wow8gb.py 2>/dev/null; then
    violations=$((violations + 1))
    echo "  ❌ regression: hardcoded tokens/sec literal in wow8gb bench."
    echo "     Read from VRAMMonitor + time.perf_counter() instead."
    echo
fi

# 7. AUDIT #320b (Track A1) — forbid hardcoded `speedup = <float>`
#    literals in the Rust-vs-numpy bench. Every speedup value must
#    be computed at runtime from time.perf_counter() deltas.
#    The regex matches `speedup = N.NN` style assignment; named
#    constants (e.g. `_STUB_SPEEDUP = 0.0` sentinels) are allowed
#    by virtue of the leading underscore, which the regex
#    does not require.
echo "▸ hardcoded speedup literal in bench_rust_speedup.py (AUDIT #320b)"
if grep -nE "speedup\s*=\s*[0-9]+\.[0-9]+\b" \
       apohara_context_forge/benchmarks/apohara2/bench_rust_speedup.py 2>/dev/null; then
    violations=$((violations + 1))
    echo "  ❌ hardcoded speedup literal in Rust speedup bench."
    echo "     Compute at runtime from time.perf_counter() deltas."
    echo
fi

# 8. AUDIT #29 (Sprint 4) — forbid hardcoded compression_ratio=0.55
#    in the h2h bench as a non-named assignment. The regex matches
#    `compression_ratio = 0.55` (assignment) but NOT
#    `_STUB_RATIO = 0.55` (named constant with leading underscore —
#    that is the Sprint 3 honest-gap sentinel and is allowed).
echo "▸ hardcoded compression_ratio=0.55 in h2h bench (AUDIT #29)"
if grep -nE "compression_ratio\s*=\s*0\.55" \
       apohara_context_forge/benchmarks/apohara2/bench_h2h.py \
       apohara_context_forge/benchmarks/apohara2/bench_e2e.py 2>/dev/null; then
    violations=$((violations + 1))
    echo "  ❌ hardcoded compression_ratio=0.55 detected in h2h bench"
    echo
fi

echo
if [ "$violations" -eq 0 ]; then
    echo "✅ honesty guard PASS — no regressions detected."
    exit 0
fi
echo "❌ honesty guard FAIL — $violations regression(s) above."
echo "   See AUDIT.md for the full list of V6.0 overclaims that V6.1 fixed."
exit 1
