"""bench_e2e.py — Full Apohara 2.0 bank test (US-002 stub).

US-002 placeholder. Real implementation lands in US-008 (Phase 6, Step 6.1).
The 5 tasks × 5 seeds suite is the spec's local-bank-test verification
gate. Each task asserts p<0.05 (Holm-Bonferroni corrected) vs. the
uncompressed baseline.

The 5 pinned tasks (per `.omc/specs/deep-interview-apohara-2-0.md`):
  1. HotpotQA            (multi-hop QA)
  2. NaturalQuestions    (open-domain QA)
  3. GSM8K               (math reasoning)
  4. BBH                 (BIG-Bench Hard)
  5. summarization       (LongBench summarization subset)

Statistical pre-registration lives at
`docs/research/reconcile/apohara2-prereg.md` (Step 0.4) — see that
file for the Holm-Bonferroni step-down statement, alpha=0.05, MDE,
and the CITABLE vs A VERIFICAR ledger.

Usage (Phase 0):
    python -m apohara_context_forge.benchmarks.apohara2.bench_e2e --help

Usage (Phase 6 target):
    python -m apohara_context_forge.benchmarks.apohara2.bench_e2e \\
        --tasks hotpotqa,naturalquestions,gsm8k,bbh,summarization \\
        --seeds 0..4 --hardware rtx2060s
"""

from __future__ import annotations

import argparse

PINNED_TASKS = (
    "hotpotqa",
    "naturalquestions",
    "gsm8k",
    "bbh",
    "summarization",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_e2e",
        description=(
            "[US-002 stub] Full bank test (5 tasks x 5 seeds). "
            "Real implementation in US-008 (Phase 6)."
        ),
    )
    p.add_argument("--tasks", default=",".join(PINNED_TASKS),
                   help=(
                       f"Comma-separated task list. Default: {','.join(PINNED_TASKS)}"
                   ))
    p.add_argument("--seeds", default="0..4",
                   help="Seed range (default: 0..4)")
    p.add_argument("--hardware", default="rtx2060s",
                   choices=["rtx2060s", "h100", "mi300x"],
                   help="Target hardware (default: rtx2060s)")
    p.add_argument("--correction", default="holm-bonferroni",
                   choices=["holm-bonferroni", "bonferroni", "none"],
                   help=(
                       "Multiple-comparison correction for the 5-task p-value "
                       "set (default: holm-bonferroni, pre-registered)."
                   ))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The 5-task list is pinned at module level; the bench fails loudly if
    # the caller passes anything else.
    requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if tuple(requested) != PINNED_TASKS:
        raise NotImplementedError(
            f"US-002 stub. bench_e2e runs the 5 pinned tasks {PINNED_TASKS}; "
            f"got {requested}. Custom task subsets land in US-008 (Phase 6)."
        )
    raise NotImplementedError(
        "US-002 stub. Real bench_e2e implementation lands in US-008 "
        "(Phase 6, Step 6.1 — paired t-test vs. baseline, p<0.05 per task "
        "with Holm-Bonferroni step-down correction)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
