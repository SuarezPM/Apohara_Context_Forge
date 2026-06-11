"""bench_kv.py — TurboQuant-KV benchmark (US-002 stub).

US-002 placeholder. Real implementation lands in US-006 (Phase 4, Step 4.6).
The numeric targets are VRAM reduction >=2.5x and EM degradation <=1% on
HotpotQA-200.

HARDWARE PIVOT (per `.omc/plans/apohara-2-0.md` Phase 4):
  The TurboQuant-KV path requires Ampere+ (CC 8.0+). RTX 2060 SUPER
  (CC 7.5) runs the 3 non-KV layers locally; the KV layer pivots to
  H100 or MI300X. The stub prints the pivot banner placeholder so the
  Phase 4 implementation wires the same string.

Usage (Phase 0):
    python -m apohara_context_forge.benchmarks.apohara2.bench_kv --help

Usage (Phase 4 target):
    python -m apohara_context_forge.benchmarks.apohara2.bench_kv \\
        --model qwen3-32b --ctx 32k --kv-bit 4 --hardware h100
"""

from __future__ import annotations

import argparse

PIVOT_BANNER = (
    "TurboQuant-KV path requires Ampere+; running on H100/MI300X"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_kv",
        description=(
            "[US-002 stub] TurboQuant-KV VRAM + EM benchmark. "
            "Real implementation in US-006 (Phase 4). "
            f"Note: {PIVOT_BANNER}."
        ),
    )
    p.add_argument("--model", default="qwen3-32b",
                   help="Model identifier (default: qwen3-32b)")
    p.add_argument("--ctx", type=int, default=32768,
                   help="Context length in tokens (default: 32k)")
    p.add_argument("--kv-bit", type=int, default=4, choices=[4, 8],
                   help="KV-cache bit width (default: 4)")
    p.add_argument("--hardware", default="h100",
                   choices=["h100", "mi300x", "rtx2060s"],
                   help=(
                       "Target hardware. RTX 2060S is a documentation marker "
                       "for the pivot; the KV path itself runs on Ampere+."
                   ))
    p.add_argument("--seeds", default="0..4",
                   help="Seed range (default: 0..4)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(
        f"{PIVOT_BANNER}. US-002 stub. Real bench_kv implementation "
        "lands in US-006 (Phase 4, Step 4.6 — VRAM >=2.5x reduction, "
        "EM degradation <=1% on HotpotQA-200)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
