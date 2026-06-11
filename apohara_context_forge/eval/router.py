"""Learned router for LLMLingua-2 variant selection.

Off by default. The pinned bin policy (short <=512, medium <=2K, long
>2K) is the production default. The learned router is an opt-in
alternative that fits a logistic regression on prompt features and
emits an AUDIT entry if its learned bin edges deviate >10% from the
pinned policy.

US-005 (Phase 3, Step 3.3). The current `fit_router` is an honest
stub: it returns the pinned edges, so `emits_audit` is False by
default. The real logistic-regression fit lands when the bench wires
the `--router learned` path end-to-end (deferred; not a US-005
deliverable, but the seam is here).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Pinned bin policy: short max, medium max. `long` is everything above
# medium and is not constrained by the deviation check (a logistic
# regression over 2 bins implicitly pins the long max to +inf).
PINNED_BIN_EDGES: tuple[int, ...] = (512, 2048)

# 10% deviation from the pinned policy triggers the audit emit hook.
# Bench calls consume the `emits_audit` field of `RouterResult`.
DEVIATION_THRESHOLD: float = 0.10


@dataclass(frozen=True)
class RouterResult:
    """Result of a `fit_router` call.

    Fields:
        learned_short_max: int — upper bound (inclusive) of the
            learned short bin.
        learned_medium_max: int — upper bound (inclusive) of the
            learned medium bin.
        learned_long_max: int — upper bound of the learned long bin.
            In practice this is +inf or 10**9 (the surrogate used in
            `compressor.VARIANTS`).
        pinned_short_max: int — the spec's pinned short max.
        pinned_medium_max: int — the spec's pinned medium max.
        pinned_long_max: int — the spec's pinned long max surrogate.
        deviation_pct: float — max |learned - pinned| / pinned across
            the short and medium edges, expressed as a fraction (0.0
            = no deviation; 0.10 = the 10% threshold).
        emits_audit: bool — True if `deviation_pct > DEVIATION_THRESHOLD`.
    """
    learned_short_max: int
    learned_medium_max: int
    learned_long_max: int
    pinned_short_max: int
    pinned_medium_max: int
    pinned_long_max: int
    deviation_pct: float
    emits_audit: bool


def _long_surrogate() -> int:
    """The 'long' bin upper bound, kept in one place."""
    return 10**9


def fit_router(
    features: np.ndarray,
    labels: np.ndarray,
) -> RouterResult:
    """Fit a logistic regression on (prompt_features, bin_label) and
    return the learned bin edges.

    Args:
        features: shape `(n_samples, n_features)`. The first column
            is expected to be the prompt word count.
        labels: shape `(n_samples,)` integer in {0, 1, 2} mapping
            to {short, medium, long}. Not consumed by the stub.

    Returns:
        `RouterResult` with `emits_audit=False` by default (the stub
        returns the pinned edges). The real implementation will:
          1. Fit a multinomial logistic regression on `(features, labels)`.
          2. Sweep `word_count` over a log-spaced grid, find the
             word count where the predicted short/medium/long
             posterior crosses 0.5 — that is the learned bin edge.
          3. Compare to `PINNED_BIN_EDGES`. If max |Δ|/pinned >
             `DEVIATION_THRESHOLD`, set `emits_audit=True` so the
             bench logs an AUDIT follow-up.

    The honest scope: this function returns the pinned edges
    unconditionally until the real fit is wired. Documented in the
    module docstring and in AUDIT #24.
    """
    # Stub: ignore features/labels and return the pinned policy. The
    # real implementation lands when the --router learned path is
    # wired in bench_compress.py (US-005 follow-up).
    pinned_short = PINNED_BIN_EDGES[0]
    pinned_medium = PINNED_BIN_EDGES[1]
    pinned_long = _long_surrogate()
    logger.debug(
        "fit_router stub: returning pinned edges (%d, %d, %d).",
        pinned_short,
        pinned_medium,
        pinned_long,
    )
    return RouterResult(
        learned_short_max=pinned_short,
        learned_medium_max=pinned_medium,
        learned_long_max=pinned_long,
        pinned_short_max=pinned_short,
        pinned_medium_max=pinned_medium,
        pinned_long_max=pinned_long,
        deviation_pct=0.0,
        emits_audit=False,
    )
