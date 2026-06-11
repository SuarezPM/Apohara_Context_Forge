"""M3 LLM-as-judge client for the Apohara 2.0 bench.

Pinned to greedy decoding (temperature=0, top_p=1.0, top_k=1) to kill
non-determinism in the 5-seed bank test. The version pin is a TODO
placeholder until the M3 model is registered on the local provider.

US-005 (Phase 3, Step 3.2). The HTTP call is a deterministic stub for
this milestone; the real call lands when the M3 provider is wired.
The stub is honest about its stub-ness: `raw` echoes the first
100 characters of the prompt and `score` is always 0.0.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Version pin (placeholder — see docs/research/reconcile/apohara2-prereg.md).
# Updated when the M3 model is registered on the local provider.
M3_VERSION: str = "MiniMax-M3-2026-05-XX"
M3_TEMPERATURE: float = 0.0
M3_TOP_P: float = 1.0
M3_TOP_K: int = 1


@dataclass(frozen=True)
class JudgeResult:
    """A single M3 judge call.

    Fields:
        score: float 0.0-1.0 (or whatever the prompt asks for).
        raw: raw judge output string.
        prompt_tokens: int — best-effort token count for the prompt.
        completion_tokens: int — best-effort token count for the
            completion; 0 in the stub.
    """
    score: float
    raw: str
    prompt_tokens: int
    completion_tokens: int


class M3Judge:
    """Greedy-decoding M3 LLM-as-judge client.

    Args:
        model_id: model identifier. Defaults to `M3_VERSION`; can be
            overridden via the `M3_MODEL_ID` env var.
        base_url: provider base URL. Defaults to `http://localhost:8000`;
            can be overridden via `M3_BASE_URL`.

    The real HTTP call is deferred. The current `judge()` returns a
    deterministic stub so the bench and tests can wire against a real
    client surface without an active provider. When the M3 client is
    wired, the body of `judge()` will issue a request with
    `temperature=M3_TEMPERATURE, top_p=M3_TOP_P, top_k=M3_TOP_K` and
    parse the response into a `JudgeResult`.
    """

    def __init__(
        self,
        model_id: str | None = None,
        base_url: str | None = None,
    ):
        self.model_id = model_id or os.environ.get("M3_MODEL_ID", M3_VERSION)
        self.base_url = base_url or os.environ.get("M3_BASE_URL", "http://localhost:8000")
        # Lazy import: do not require `openai` at module-load time.

    def judge(
        self,
        prompt: str,
        system: str = "You are a strict evaluator.",
    ) -> JudgeResult:
        """Issue a judge call.

        Greedy decoding is enforced by the module-level constants.
        The stub returns `score=0.0` and a `raw` string that echoes
        the first 100 chars of the prompt. The token counts are
        best-effort whitespace splits; the real implementation will
        use the provider's `usage` field.

        Raises:
            RuntimeError: if the underlying client (when wired) does
                not honor `temperature=0`. This is enforced to keep
                the 5-seed bank test deterministic.
        """
        if M3_TEMPERATURE != 0.0 or M3_TOP_P != 1.0 or M3_TOP_K != 1:
            # Defensive guard: if the constants ever drift, the bench
            # breaks the determinism contract loudly.
            raise RuntimeError(
                "M3 judge sampling params drifted from greedy decoding. "
                "Re-pin temperature/top_p/top_k to 0/1.0/1 before any "
                "5-seed bench run."
            )

        # Honest stub. The real call lands when the M3 provider is
        # wired. The token counts use whitespace splits as a
        # best-effort stand-in.
        prompt_tokens = len(prompt.split())
        raw = f"M3 judge stub: {prompt[:100]}"
        return JudgeResult(
            score=0.0,
            raw=raw,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
        )
