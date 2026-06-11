"""bench_compress.py — LLMLingua-2 PPL-delta benchmark (US-002 stub).

US-002 placeholder. Real implementation lands in US-005 (Phase 3, Step 3.4).
The numeric target is PPL delta <=5% per variant (short/medium/long) on a
LongBench subset, judged by the M3 LLM-as-judge (greedy decoding).

The `--router` flag is wired in the stub already per
`.omc/plans/apohara-2-0.md` Finding 3 (default `pinned` for the spec's
512/2048 bin policy; `learned` is the off-by-default logistic-regression
router that emits an AUDIT #23 entry if its edges deviate >10%).

Usage (Phase 0):
    python -m apohara_context_forge.benchmarks.apohara2.bench_compress --help

Usage (Phase 3 target):
    python -m apohara_context_forge.benchmarks.apohara2.bench_compress \\
        --task longbench_subset --variant all --seeds 0..4 --judge m3
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_compress",
        description=(
            "[US-002 stub] LLMLingua-2 PPL-delta per variant. "
            "Real implementation in US-005 (Phase 3)."
        ),
    )
    p.add_argument("--task", default="longbench_subset",
                   help="Benchmark task (default: longbench_subset)")
    p.add_argument("--variant", default="all",
                   choices=["all", "base-short", "base-medium", "long"],
                   help="LLMLingua-2 variant (default: all)")
    p.add_argument("--seeds", default="0..4",
                   help="Seed range (default: 0..4)")
    p.add_argument("--judge", default="m3",
                   help="LLM-as-judge identifier (default: m3)")
    p.add_argument("--router", default="pinned",
                   choices=["pinned", "learned"],
                   help=(
                       "Variant selection policy. 'pinned' uses the spec's "
                       "512/2048 bin policy (default); 'learned' runs the "
                       "off-by-default logistic-regression router."
                   ))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(
        "US-002 stub. Real bench_compress implementation lands in US-005 "
        "(Phase 3, Step 3.4 — PPL delta <=5% per variant on LongBench, "
        "judged by greedy-decoding M3)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
