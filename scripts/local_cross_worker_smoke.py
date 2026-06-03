#!/usr/bin/env python3
"""Local cross-worker KV-reuse smoke via LMCache + Redis (single GPU, sequential).

Proves the F3 plumbing on REAL hardware with a SHARED LMCache/Redis backend:
worker-2, starting with an EMPTY local prefix cache, reuses the prefix KV that
worker-1 already stored in Redis — instead of recomputing it from scratch. This
is the cross-process KV offload that the MI300X run will later measure at scale.

A single 8 GB card cannot host two vLLM servers at once, so we go SEQUENTIAL:

  1. launch worker-1 with LMCache (kv_both) -> Redis, replay the shared-prefix
     workload so worker-1 STORES the prefix KV chunks into Redis;
  2. kill worker-1 (its LOCAL prefix cache dies with it; Redis keeps the chunks);
  3. launch worker-2 against the SAME Redis with an EMPTY local cache, replay the
     SAME workload, and check that worker-2 RETRIEVES the prefix KV from Redis
     (LMCache remote hits) rather than recomputing it.

It uses the REAL production builders — this is the production caller that
``vllm_launch_config.build_vllm_serve_args``/``worker_env`` previously lacked:

  apohara_context_forge.serving.vllm_launch_config.build_vllm_serve_args(...)
  apohara_context_forge.serving.vllm_launch_config.worker_env(...)
  apohara_context_forge.serving.vllm_launch_config.build_kv_transfer_config(...)

NOT a benchmark. Single-GPU, tiny model: it validates the cross-process REUSE
path (store on one process, retrieve on another via Redis). The headline GB
number is gated to MI300X (>=2 workers, real model). RTX 2060 == plumbing proof.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3-1.7B")
REDIS_URL = os.environ.get("SMOKE_REDIS_URL", "redis://localhost:6379")
BLOCK_SIZE = 16
STARTUP_TIMEOUT_S = float(os.environ.get("SMOKE_STARTUP_TIMEOUT", "900"))
N = int(os.environ.get("SMOKE_N", "6"))

# Long byte-identical shared prefix (same spirit as the single-worker smoke):
# it must span many 16-token blocks for cross-worker chunk reuse to register.
SHARED_PREFIX = (
    "You are one agent in a multi-agent retrieval pipeline operated by Apohara "
    "ContextForge. Every agent receives this identical briefing verbatim and only "
    "the final task line differs. Briefing: the system serves heterogeneous "
    "technical documents about GPU inference, paged attention, key-value cache "
    "management, prefix caching, attention-tensor quantization, and cross-worker "
    "key-value offload to a shared store. Ground every statement in the briefing, "
    "never invent sources, prefer measured evidence over projections, and keep "
    "answers terse. This shared prefix is intentionally long so the key-value "
    "cache it produces spans many paged-attention blocks; offloading those blocks "
    "to a shared backend so a second worker can retrieve instead of recompute is "
    "the entire point of cross-worker key-value reuse. Treat the briefing as fixed.\n\n"
)
TAILS = [
    "Task: list the 3 most relevant documents. Terse.",
    "Task: rerank the 5 documents by relevance. Ranking only.",
    "Task: summarize the briefing in two sentences.",
    "Task: verify one claim. ACCEPT or REJECT plus one sentence.",
    "Task: write a 3-sentence grounded answer.",
    "Task: name the single most important constraint. One line.",
]


def log(m: str) -> None:
    print(f"[xworker] {m}", flush=True)


def vllm_bin() -> str:
    cand = Path(sys.executable).parent / "vllm"
    return str(cand) if cand.exists() else "vllm"


def write_lmcache_config() -> str:
    """Write a local LMCache config pointing at the local Redis. Returns path.

    Delegates to the shared, parameterized writer in ``vllm_launch_config`` so the
    smoke and the GATE #0 cross-worker harness emit byte-identical config files.
    """
    from apohara_context_forge.serving.vllm_launch_config import write_lmcache_config as _write

    p = _write("/tmp/lmcache_local.yaml", remote_url=REDIS_URL, chunk_size=BLOCK_SIZE)
    log(f"wrote {p}:\n{Path(p).read_text().strip()}")
    return p


def server_args(port: int) -> list[str]:
    """Production args from vllm_launch_config + the 8GB-card flags."""
    from apohara_context_forge.serving.vllm_launch_config import build_vllm_serve_args

    base = build_vllm_serve_args(MODEL, block_size=BLOCK_SIZE, chunk_size=BLOCK_SIZE)
    return [vllm_bin(), *base,
            "--port", str(port),
            "--enable-prefix-caching",
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.82",
            "--max-num-seqs", "16",
            "--enforce-eager"]


def server_env(lmcache_cfg: str) -> dict:
    """os.environ + the REAL worker_env() + local-card / Turing knobs."""
    from apohara_context_forge.serving.vllm_launch_config import worker_env

    env = os.environ.copy()
    env.update(worker_env())  # PYTHONHASHSEED=0, LMCACHE_USE_EXPERIMENTAL=True
    env["LMCACHE_CONFIG_FILE"] = lmcache_cfg
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"     # Turing sm_75
    # NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True with a KV
    # connector — vLLM rejects the combo: the VMM allocator can remap KV virtual
    # addresses, invalidating registered KV memory the connector pins. max_num_seqs=16
    # keeps the sampler warmup small enough that we don't need it here.
    return env


def start_worker(name: str, port: int, env: dict) -> tuple[subprocess.Popen, str]:
    args = server_args(port)
    log(f"launching {name}: {' '.join(args)}")
    logpath = f"/tmp/xworker_{name}.log"
    lf = open(logpath, "w")
    proc = subprocess.Popen(args, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return proc, logpath


def wait_ready(proc: subprocess.Popen, port: int) -> bool:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log(f"worker EXITED early (code {proc.returncode})")
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def replay(port: int, salt: str) -> int:
    """Send N sequential shared-prefix requests; return count of HTTP 200s."""
    ok = 0
    for i in range(N):
        body = {"model": MODEL, "prompt": SHARED_PREFIX + TAILS[i % len(TAILS)],
                "max_tokens": 8, "temperature": 0.0, "cache_salt": salt}
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                ok += 1 if r.status == 200 else 0
        except Exception as e:
            log(f"request {i} failed: {e}")
        time.sleep(0.3)
    time.sleep(1.0)
    return ok


def lmcache_stats_from_log(logpath: str) -> dict:
    """Best-effort parse of LMCache store/retrieve activity from a worker log.

    LMCache logs lines like 'Stored X tokens' / 'Retrieved Y tokens' / 'hit'.
    We grep counts rather than assume an exact format (LMCache log strings move
    between versions); we report what we actually found, honestly.
    """
    try:
        text = Path(logpath).read_text(errors="ignore")
    except Exception:
        return {}
    low = text.lower()
    return {
        "log_mentions_retrieve": low.count("retriev"),
        "log_mentions_store": low.count("store") + low.count("stored"),
        "log_mentions_hit": low.count("hit"),
        # capture any 'N tokens' hit/retrieve numbers for eyeballing
        "token_numbers": re.findall(r"(?:retriev\w*|hit)[^\n]*?(\d+)\s*tokens", low)[:10],
    }


def prefix_metrics(port: int) -> dict:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if "prefix_cache" in line or "kv_transfer" in line or "external" in line:
            m = re.search(r"^(\S+).*\s([0-9.eE+-]+)\s*$", line)
            if m:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def main() -> int:
    from apohara_context_forge.serving.prefix_salt_planner import PrefixSaltPlanner
    import hashlib

    planner = PrefixSaltPlanner()
    anchor = hashlib.sha256(SHARED_PREFIX.encode()).hexdigest()[:16]
    salt = planner.shared_salt(anchor, "xworker")  # collaborators share this salt
    log(f"shared cache_salt = {salt!r}")

    lmcache_cfg = write_lmcache_config()
    result = {"model": MODEL, "redis_url": REDIS_URL,
              "hardware": "RTX 2060 SUPER 8GB (Turing) — PLUMBING PROOF, not cited",
              "kv_transfer_config": None}
    try:
        from apohara_context_forge.serving.vllm_launch_config import build_kv_transfer_config
        result["kv_transfer_config"] = build_kv_transfer_config(BLOCK_SIZE, BLOCK_SIZE)
    except Exception as e:
        result["kv_transfer_config_error"] = repr(e)

    env = server_env(lmcache_cfg)

    # ---- Worker-1: populate Redis -------------------------------------------
    p1, log1 = start_worker("w1", 8021, env)
    try:
        if not wait_ready(p1, 8021):
            result["status"] = "w1_failed"
            result["w1_log_tail"] = Path(log1).read_text(errors="ignore")[-1800:]
            print("\n=== XWORKER RESULT ===\n" + json.dumps(result, indent=2))
            return 1
        log("worker-1 READY — populating Redis")
        result["w1_requests_ok"] = replay(8021, salt)
        result["w1_prefix_metrics"] = prefix_metrics(8021)
    finally:
        p1.terminate()
        try: p1.wait(timeout=30)
        except subprocess.TimeoutExpired: p1.kill()
    result["w1_lmcache_log"] = lmcache_stats_from_log(log1)
    time.sleep(3)  # let Redis settle

    # ---- Worker-2: empty local cache, must reuse from Redis -----------------
    p2, log2 = start_worker("w2", 8022, env)
    try:
        if not wait_ready(p2, 8022):
            result["status"] = "w2_failed"
            result["w2_log_tail"] = Path(log2).read_text(errors="ignore")[-1800:]
            print("\n=== XWORKER RESULT ===\n" + json.dumps(result, indent=2))
            return 1
        log("worker-2 READY (empty local cache) — replaying; expect Redis retrieves")
        result["w2_requests_ok"] = replay(8022, salt)
        result["w2_prefix_metrics"] = prefix_metrics(8022)
    finally:
        p2.terminate()
        try: p2.wait(timeout=30)
        except subprocess.TimeoutExpired: p2.kill()
    result["w2_lmcache_log"] = lmcache_stats_from_log(log2)

    # ---- Verdict ------------------------------------------------------------
    w2 = result.get("w2_lmcache_log", {})
    retrieved = w2.get("log_mentions_retrieve", 0) > 0 or bool(w2.get("token_numbers"))
    result["verdict"] = {
        "cross_worker_reuse_observed": bool(retrieved),
        "interpretation": (
            "worker-2 (empty local cache) retrieved prefix KV from the shared "
            "Redis/LMCache backend that worker-1 stored — cross-process reuse works."
            if retrieved else
            "No clear retrieve signal in worker-2 logs. Inspect /tmp/xworker_w2.log "
            "and the prefix_metrics; LMCache may need a different remote_serde or "
            "the prefix may not have been offloaded. Plumbing ran; reuse unproven."
        ),
    }
    result["status"] = "ok"
    print("\n=== XWORKER RESULT ===\n" + json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
