"""bench_rust_speedup.py — measure the Rust PyO3 path against the numpy fallback.

AUDIT #320b (Track A1). Compares two operations that ship in
``apohara_context_forge/serving/turboquant_turing/``:

* ``fwht_inplace(x)`` — Walsh-Hadamard on a flat ``&[f32]``. The Rust
  kernel runs in one PyO3 call. The numpy fallback is the existing
  ``_fwht_butterfly_numpy`` in
  ``apohara_context_forge/quantization/fwht.py:120-149``.
* ``dequant_per_block(codes, scales, zps, group_size)`` — per-block
  INT4 dequant with a ``(scale, zp)`` per group_size packed bytes.
  The Rust kernel returns a flat ``&[f32]``. The numpy fallback is a
  hand-rolled loop that mirrors the same math.

Output: ``reports/rust_speedup_2026_06_12.csv`` (or whatever the
caller passes via ``--output``) with the 7-tuple schema:

    op, n, dim, rust_ms, numpy_ms, speedup, source

The speedup is the simple ratio ``numpy_ms / rust_ms`` (positive when
Rust is faster, negative when Rust is slower — never expected in
practice but logged for completeness). ``source`` is the repo
commit the bench was run against.

Honesty discipline: no literal speedup values in the script (the
honesty gate forbids that pattern). All numbers come from
``time.perf_counter()``. If the Rust wheel is not importable, the
bench reports ``rust_ms = NaN`` and the numpy row is the only data;
the speedup column is left as a tagged sentinel (None).

Verification gate:

* AUDIT #320b entry in ``AUDIT.md`` cites the produced CSV file
  path + the median speedup per operation.
* If the median speedup is <2x on at least one operation, AUDIT
  #320a stays at YELLOW with the framing "Rust path shipped,
  numpy fallback preferred at this size". If >=2x on both, AUDIT
  #320a flips to GREEN with the measured numbers.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np


# --------------------------------------------------------------------------
# numpy fallbacks (mirror the Rust surface; not the same code path)
# --------------------------------------------------------------------------


def _fwht_butterfly_numpy_inplace(buf: np.ndarray) -> None:
    """Self-inverse Walsh-Hadamard in-place on a flat float32 buffer.

    Length must be a power of two. Mirrors the Rust kernel bit-for-bit
    in terms of the output values (the Rust kernel is also
    un-normalized — calling ``fwht(fwht(x))`` yields ``d * x``).

    Implementation is the same as
    ``apohara_context_forge.quantization.fwht._fwht_butterfly_numpy``
    but inlined here so the bench is a self-contained script (no
    cross-package import graph for the timing path).
    """
    n = buf.shape[0]
    if n & (n - 1) != 0:
        raise ValueError(f"FWHT length must be a power of two; got {n}")
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = buf[j]
                b = buf[j + h]
                buf[j] = a + b
                buf[j + h] = a - b
        h *= 2


def _dequant_per_block_numpy(
    codes: np.ndarray, scales: np.ndarray, zps: np.ndarray, group_size: int
) -> np.ndarray:
    """Per-block INT4 dequant (lo/hi nibble) with one (scale_lo,
    scale_hi, zp_lo, zp_hi) per ``group_size`` packed bytes. Mirrors
    the Rust kernel signature verbatim:

      codes:  (n_blocks * group_size,) uint8
      scales: (n_blocks * 2,) float32  -- (scale_lo, scale_hi) per block
      zps:    (n_blocks * 2,) float32  -- (zp_lo, zp_hi) per block
    Output: (codes.shape[0] * 2,) float32  -- lo/hi interleave
    """
    n_bytes = codes.shape[0]
    n_blocks = n_bytes // group_size
    lo = (codes & 0x0F).astype(np.float32)
    hi = ((codes >> 4) & 0x0F).astype(np.float32)
    # Broadcast (scale_lo, scale_hi) and (zp_lo, zp_hi) per block.
    # scales layout: [scale_lo_0, scale_hi_0, scale_lo_1, scale_hi_1, ...]
    # which can be reshaped to (n_blocks, 2) and unstacked into
    # (scale_lo, scale_hi) of shape (n_blocks,).
    scales_2d = scales.reshape(n_blocks, 2)
    zps_2d = zps.reshape(n_blocks, 2)
    scale_lo = scales_2d[:, 0]  # shape (n_blocks,)
    scale_hi = scales_2d[:, 1]
    zp_lo = zps_2d[:, 0]
    zp_hi = zps_2d[:, 1]
    # lo / hi have shape (n_bytes,). Reshape to (n_blocks, group_size)
    # to broadcast the per-block (scale, zp) over the group_size axis.
    lo_2d = lo.reshape(n_blocks, group_size)
    hi_2d = hi.reshape(n_blocks, group_size)
    deq_lo = (lo_2d - zp_lo[:, None]) * scale_lo[:, None]
    deq_hi = (hi_2d - zp_hi[:, None]) * scale_hi[:, None]
    stacked = np.stack([deq_lo, deq_hi], axis=-1)  # (n_blocks, group_size, 2)
    return stacked.reshape(n_bytes * 2)


# --------------------------------------------------------------------------
# timing harness
# --------------------------------------------------------------------------


def _time_callable(
    fn: Callable[[], None], warmup: int, iters: int
) -> float:
    """Median wall-clock seconds over ``iters`` runs, after ``warmup``
    warm-up calls. ``fn`` is the unit-of-work closure; we time only
    the closure, not the import overhead or the numpy array
    allocation (the caller pre-allocates everything).
    """
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3  # ms


def _rust_available() -> bool:
    return importlib.util.find_spec("turboquant_turing") is not None


def _bench_fwht(n: int, warmup: int, iters: int) -> tuple[float, float, str]:
    """Returns (rust_ms, numpy_ms, status). status is one of
    'rust+numpy', 'numpy-only'."""
    x_rust = np.arange(n, dtype=np.float32)
    x_numpy = x_rust.copy()
    rust_ms: float = float("nan")
    status = "numpy-only"

    if _rust_available():
        import turboquant_turing as t  # type: ignore[import-not-found]

        def _run_rust() -> None:
            buf = x_rust.copy()  # fresh copy each iter (the kernel is in-place)
            t.fwht_inplace(buf)

        rust_ms = _time_callable(_run_rust, warmup, iters)
        status = "rust+numpy"

    def _run_numpy() -> None:
        buf = x_numpy.copy()
        _fwht_butterfly_numpy_inplace(buf)

    numpy_ms = _time_callable(_run_numpy, warmup, iters)
    return rust_ms, numpy_ms, status


def _bench_dequant(
    n_packed: int, group_size: int, warmup: int, iters: int
) -> tuple[float, float, str]:
    """Returns (rust_ms, numpy_ms, status). Apples-to-apples bench:
    both paths use the same 1-D ``(n_blocks,)`` layout for
    codes / scales / zps, and the same ``group_size`` argument.
    The Rust ``dequant_per_block`` is 1-D per its PyO3 signature;
    the numpy fallback mirrors that.
    """
    rng = np.random.default_rng(0)
    n_blocks = max(1, n_packed // group_size)
    n_bytes = n_blocks * group_size  # ensure divisibility
    codes = rng.integers(0, 256, size=n_bytes, dtype=np.uint8)
    scales = rng.random(n_blocks * 2, dtype=np.float32)
    zps = rng.integers(0, 16, size=n_blocks * 2, dtype=np.int32).astype(np.float32)
    rust_ms: float = float("nan")
    status = "numpy-only"

    if _rust_available():
        import turboquant_turing as t  # type: ignore[import-not-found]

        def _run_rust() -> None:
            t.dequant_per_block(codes, scales, zps, group_size)

        rust_ms = _time_callable(_run_rust, warmup, iters)
        status = "rust+numpy"

    def _run_numpy() -> None:
        _dequant_per_block_numpy(codes, scales, zps, group_size)

    numpy_ms = _time_callable(_run_numpy, warmup, iters)
    return rust_ms, numpy_ms, status


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _git_head_short() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/rust_speedup_2026_06_12.csv"),
        help="CSV output path (default: reports/rust_speedup_2026_06_12.csv)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="warm-up iterations (default 10)"
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="measured iterations (default 100)"
    )
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    head = _git_head_short()

    # Test matrix: (op, n, dim_or_group_size)
    # For dequant: n is n_blocks (each block is 1 packed byte, group_size=1
    # is the apples-to-apples form per the Rust PyO3 signature).
    cases: list[tuple[str, int, int]] = [
        ("fwht", 1024, 1024),
        ("fwht", 8192, 8192),
        ("fwht", 65536, 65536),
        ("dequant", 1024, 1),
        ("dequant", 8192, 1),
        ("dequant", 65536, 1),
    ]

    rows: list[dict[str, str]] = []
    for op, n, dim in cases:
        if op == "fwht":
            rust_ms, numpy_ms, status = _bench_fwht(n, args.warmup, args.iters)
        else:
            rust_ms, numpy_ms, status = _bench_dequant(
                n, dim, args.warmup, args.iters
            )
        speedup: float | None
        if rust_ms == rust_ms and numpy_ms > 0:  # NaN-safe
            speedup = numpy_ms / rust_ms
        else:
            speedup = None
        rows.append(
            {
                "op": op,
                "n": str(n),
                "dim": str(dim),
                "rust_ms": f"{rust_ms:.6f}" if rust_ms == rust_ms else "NaN",
                "numpy_ms": f"{numpy_ms:.6f}",
                "speedup": f"{speedup:.4f}" if speedup is not None else "NaN",
                "source": f"git={head};status={status}",
            }
        )

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["op", "n", "dim", "rust_ms", "numpy_ms", "speedup", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Console summary
    print(f"\nRust speedup bench — head={head} (AUDIT #320b)")
    print(f"Rust available: {_rust_available()}")
    print(f"Output: {args.output}")
    print(
        f"{'op':<10} {'n':>8} {'dim':>8} {'rust_ms':>12} {'numpy_ms':>12} {'speedup':>10}"
    )
    for r in rows:
        print(
            f"{r['op']:<10} {r['n']:>8} {r['dim']:>8} {r['rust_ms']:>12} "
            f"{r['numpy_ms']:>12} {r['speedup']:>10}"
        )

    # Honest summary for the AUDIT entry
    fwht_speedups = [
        float(r["speedup"]) for r in rows if r["op"] == "fwht" and r["speedup"] != "NaN"
    ]
    dequant_speedups = [
        float(r["speedup"]) for r in rows if r["op"] == "dequant" and r["speedup"] != "NaN"
    ]
    if fwht_speedups:
        fwht_med = statistics.median(fwht_speedups)
        print(f"\nFWHT median speedup (Rust/numpy): {fwht_med:.2f}x")
    if dequant_speedups:
        dq_med = statistics.median(dequant_speedups)
        print(f"Dequant median speedup (Rust/numpy): {dq_med:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
