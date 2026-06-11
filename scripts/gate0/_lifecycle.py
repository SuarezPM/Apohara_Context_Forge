#!/usr/bin/env python3
"""GATE #0 — shared live-server lifecycle helpers (subprocess vllm).

This module factors out the process/HTTP plumbing that BOTH ``harness.py`` (the
single-worker-per-arm runner) and ``cross_worker.py`` (the real two-worker
store->retrieve runner) need, so neither duplicates it. It owns ONLY:

  * the resolved ``vllm`` launcher (explicit > venv-local > PATH);
  * launch / health-wait / teardown of a vLLM subprocess;
  * the single completion POST (mirrors ``scripts.mi300x_measure._post``);
  * the endpoint string + the per-arm default ports.

vLLM is NEVER imported in Python here — the launcher only ever shells out to the
``vllm`` CLI as a subprocess, exactly like ``local_cross_worker_smoke.py`` and
``scripts/mi300x_squeeze_all.sh:run_model`` already do.

Honesty discipline (CONTRACT §1): this module fabricates no number. It launches,
waits, posts, and tears down; the readers in ``metrics.py`` own every measured value.

Apache-2.0 — Apohara ContextForge.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

# CONTRACT §1: REPO on sys.path so sibling gate0 modules + apohara_context_forge.*
# resolve the same way the existing probes do, regardless of the invoking cwd.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.gate0.arms import ARM_A, ARM_B, ARM_C, ArmLaunch  # noqa: E402

# Health-wait / sampling knobs (operator overridable via env; never affect numbers).
HEALTH_TIMEOUT_S = 900.0       # vLLM load of a 32B dense model is large
HEALTH_POLL_S = 5.0
SETTLE_S = 2.0                 # let HBM settle after warmup / before sampling
TEARDOWN_TIMEOUT_S = 60.0


def log(msg: str) -> None:
    print(f"[gate0] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# HTTP — same shape as mi300x_measure._post, injected into the metrics readers so
# those stay vLLM-import-free (they never construct a request body themselves).
# --------------------------------------------------------------------------- #
def post(endpoint, model, prompt, *, salt=None, max_tokens=16, stream=False):
    """POST one completion. Mirrors scripts.mi300x_measure._post byte-for-byte so the
    metrics readers can drive traffic without importing vLLM or knowing the wire format."""
    # vLLM advertises the model under --served-model-name (basename of the HF repo id).
    # When the harness is invoked with the full HF id (e.g. "Qwen/Qwen3-32B") the server
    # registers it under "Qwen3-32B" — and /v1/completions resolves the body `model` field
    # against that registry. Passing the full id here 404s. Mirror arms._served_name so
    # the request always uses the same name the server is serving.
    served_name = model.rsplit("/", 1)[-1]
    body = {"model": served_name, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
    if salt is not None:
        body["cache_salt"] = salt
    if stream:
        body["stream"] = True
    req = urllib.request.Request(
        f"{endpoint}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=180)


# --------------------------------------------------------------------------- #
# Live server lifecycle — subprocess vllm, health-wait, teardown. Pattern lifted
# from mi300x_squeeze_all.run_model + local_cross_worker_smoke.{start_worker,wait_ready}.
# --------------------------------------------------------------------------- #
def vllm_bin(explicit: Optional[str]) -> str:
    """Resolve the vllm launcher: explicit > venv-local > PATH (smoke-script pattern).

    IMPORTANT (deployment): the planned MI300X setup runs vLLM INSIDE the ``rocm/vllm``
    container (ROCm + AITER prebuilt), NOT on the host — there is no host ``vllm`` on PATH.
    The launcher only ever invokes ``[<bin>, "serve", <model>, ...]`` (see ``launch_server``),
    so the operator MUST pass ``--vllm-bin`` pointing at a launcher that forwards
    ``serve ...`` into the container (a tiny ``docker run --network host ... rocm/vllm vllm``
    wrapper that appends its args). Without it, a bare host without vLLM raises
    FileNotFoundError -> ``wait_health`` returns False -> the arm is recorded UNMEASURED while
    the GPU still bills. ``--mode dry`` never reaches this path, so this surfaces ONLY live;
    the runbook ships the wrapper (§2.2) and pins ``--vllm-bin`` on every live command (§3).
    """
    if explicit:
        return explicit
    cand = Path(sys.executable).parent / "vllm"
    return str(cand) if cand.exists() else "vllm"


def launch_server(
    launch: ArmLaunch,
    *,
    vllm_bin: str,
    log_path: str,
    extra_env: Optional[dict[str, str]] = None,
) -> tuple[subprocess.Popen, str]:
    """Start one vLLM server for an arm. Returns (process, log_path).

    ``launch.serve_args`` already starts with ``["serve", model, ...]`` (arms.py owns
    the exact arg list); we prepend the resolved ``vllm`` binary. ``launch.env`` carries
    the AITER vars + PYTHONHASHSEED=0 (+ LMCache worker_env for cross-worker B); we apply
    it on top of the inherited environment so HF cache / ROCm device vars survive.
    """
    args = [vllm_bin, *launch.serve_args]
    env = os.environ.copy()
    env.update(launch.env)
    if extra_env:
        env.update(extra_env)
    log(f"launch arm {launch.arm}/{launch.topology}: {' '.join(args)}")
    lf = open(log_path, "w")
    proc = subprocess.Popen(args, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return proc, log_path


def wait_health(proc: subprocess.Popen, endpoint: str, *, timeout_s: float = HEALTH_TIMEOUT_S) -> bool:
    """Poll ``/health`` until 200 or the process dies (local_cross_worker_smoke pattern)."""
    deadline = time.monotonic() + timeout_s
    url = f"{endpoint}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log(f"server EXITED early (code {proc.returncode}) before /health")
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(HEALTH_POLL_S)
    return False


def teardown(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL — the squeeze runner removes the container; we kill the proc."""
    proc.terminate()
    try:
        proc.wait(timeout=TEARDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def endpoint(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def default_ports() -> dict[str, int]:
    """One port per arm so a stray server can't collide on the next arm's launch."""
    return {ARM_A: 8000, ARM_B: 8001, ARM_C: 8002}
