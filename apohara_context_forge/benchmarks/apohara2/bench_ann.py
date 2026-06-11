"""bench_ann.py — Turbovec ANN index benchmark (US-002 stub).

US-002 placeholder. Real implementation lands in US-004 (Phase 2, Step 2.4).
The numeric target is recall parity with FAISS-IVF on HotpotQA-200 and RAM
<=4GB for 10M docs at 4-bit, 768-d (the two Phase 2 acceptance numbers from
`.omc/plans/apohara-2-0.md` Section 2).

Usage (Phase 0):
    python -m apohara_context_forge.benchmarks.apohara2.bench_ann --help

Usage (Phase 2 target):
    python -m apohara_context_forge.benchmarks.apohara2.bench_ann \\
        --docs 10M --bits 4 --queries 1000 --seed 42
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_ann",
        description=(
            "[US-002 stub] Turbovec ANN recall + RAM benchmark. "
            "Real implementation in US-004 (Phase 2)."
        ),
    )
    p.add_argument("--docs", type=int, default=10_000_000,
                   help="Number of documents to index (default: 10M)")
    p.add_argument("--bits", type=int, default=4, choices=[4, 8],
                   help="Scalar-quantization bit width (default: 4)")
    p.add_argument("--queries", type=int, default=1000,
                   help="Number of query vectors (default: 1000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(
        "US-002 stub. Real bench_ann implementation lands in US-004 "
        "(Phase 2, Step 2.4 — recall parity with FAISS-IVF on HotpotQA-200, "
        "RAM <=4GB for 10M docs at 4-bit, 768-d)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
