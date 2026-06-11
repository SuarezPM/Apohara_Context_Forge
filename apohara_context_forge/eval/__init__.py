"""Apohara 2.0 evaluation helpers (Phase 3, US-005).

Public surface:
  - `M3Judge` (`m3_judge.M3Judge`): greedy-decoding M3 LLM-as-judge
    client. Real HTTP call lands when the M3 provider is wired; for
    US-005 the `judge()` call is a deterministic stub that returns
    `score=0.0` and a `raw` string that echoes the first 100 chars
    of the prompt.
  - `fit_router` (`router.fit_router`): the off-by-default learned
    router for LLMLingua-2 variant selection. Returns the pinned
    bin edges by default, so the audit emit hook is silent until
    someone wires the real logistic-regression fit.

The eval package is consumed by the
`apohara_context_forge.benchmarks.apohara2.bench_compress` runner
(US-005 Step 3.4). It is NOT wired into the production compression
path; the production path uses the pinned bin policy in
`apohara_context_forge.compression.compressor.VARIANTS`.
"""

from apohara_context_forge.eval.m3_judge import (
    JudgeResult,
    M3Judge,
    M3_TEMPERATURE,
    M3_TOP_P,
    M3_TOP_K,
    M3_VERSION,
)
from apohara_context_forge.eval.router import (
    DEVIATION_THRESHOLD,
    PINNED_BIN_EDGES,
    RouterResult,
    fit_router,
)

__all__ = [
    "JudgeResult",
    "M3Judge",
    "M3_TEMPERATURE",
    "M3_TOP_P",
    "M3_TOP_K",
    "M3_VERSION",
    "DEVIATION_THRESHOLD",
    "PINNED_BIN_EDGES",
    "RouterResult",
    "fit_router",
]
