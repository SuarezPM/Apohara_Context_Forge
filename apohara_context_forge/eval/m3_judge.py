"""M3 LLM-as-judge client for the Apohara 2.0 bench.

Pinned to greedy decoding (temperature=0, top_p=1.0, top_k=1) to kill
non-determinism in the 5-seed bank test. Wire-up to a real M3 endpoint
via the M3_BASE_URL env var. When the endpoint is unreachable, the
judge returns a JudgeResult with score=None and raw='<error: ...>'
so the bench's deterministic local judge can take over.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Version pin (placeholder — see docs/research/reconcile/apohara2-prereg.md).
M3_VERSION: str = "MiniMax-M3-2026-05-XX"
M3_TEMPERATURE: float = 0.0
M3_TOP_P: float = 1.0
M3_TOP_K: int = 1

# Default endpoint (Pablo's local M3 serve). Overridable via env var.
M3_DEFAULT_BASE_URL: str = "http://localhost:8000"

@dataclass(frozen=True)
class JudgeResult:
    score: Optional[float]          # 0.0-1.0 (or whatever the prompt asks for); None on error
    raw: str                        # raw judge output, or '<error: M3 unreachable>' on failure
    prompt_tokens: int              # 0 on error
    completion_tokens: int          # 0 on error
    degraded: bool = False           # True if the call fell back to the deterministic envelope


class M3Judge:
    def __init__(self, model_id: Optional[str] = None, base_url: Optional[str] = None,
                 timeout_sec: float = 30.0):
        self.model_id = model_id or os.environ.get("M3_MODEL_ID", M3_VERSION)
        self.base_url = base_url or os.environ.get("M3_BASE_URL", M3_DEFAULT_BASE_URL)
        self.timeout_sec = timeout_sec
        # Lazy import: do not require `httpx` at module-load time (slim venv may not have it).
        self._httpx = None

    def _get_httpx(self):
        if self._httpx is None:
            import httpx
            self._httpx = httpx
        return self._httpx

    def judge(self, prompt: str, system: str = "You are a strict evaluator.") -> JudgeResult:
        """Call M3 via the OpenAI-compatible /v1/chat/completions endpoint.
        Returns a JudgeResult. If M3 is unreachable, returns score=None and
        raw='<error: M3 unreachable: ...>' with degraded=True (the bench's
        deterministic local judge takes over).
        """
        # Greedy decoding pins: these are non-negotiable.
        body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": M3_TEMPERATURE,
            "top_p": M3_TOP_P,
            "top_k": M3_TOP_K,
        }
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        try:
            httpx = self._get_httpx()
            resp = httpx.post(url, json=body, timeout=self.timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            # The score is whatever the M3 prompt asks for. We don't
            # parse it here — the bench does. We just pass the raw content
            # and the usage tokens. The M3 prompt template in
            # bench_compress.py asks for a float in [0, 1] on the first
            # line.
            score = self._parse_score(content)
            return JudgeResult(
                score=score,
                raw=content,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                degraded=False,
            )
        except Exception as e:
            logger.warning(f"M3 judge unreachable at {url}: {e}")
            return JudgeResult(
                score=None,
                raw=f"<error: M3 unreachable: {type(e).__name__}: {str(e)[:100]}>",
                prompt_tokens=0,
                completion_tokens=0,
                degraded=True,
            )

    @staticmethod
    def _parse_score(content: str) -> Optional[float]:
        """Parse the M3 judge's first-line score. The prompt template
        asks for a float in [0, 1] on the first line. Returns None if
        parsing fails (the bench falls back to a deterministic score).
        """
        if not content:
            return None
        first_line = content.strip().splitlines()[0].strip()
        try:
            return float(first_line)
        except (ValueError, IndexError):
            return None
