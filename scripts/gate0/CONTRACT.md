# GATE #0 Harness — Interface Contract (BINDING)

> Status: **frozen for parallel build**. This file is the single source of truth for the
> module boundaries of the GATE #0 decision harness. Eight agents build against it in
> parallel; if a signature here is wrong, fix it HERE first (one PR), then code to it.
> Do **not** silently diverge — divergence is what makes parallel builds collide.

Read first: `docs/research/_internal/GATE-0-protocol.md` (the preregistered protocol).
This contract operationalizes that protocol. Where they disagree, the protocol wins and
this file is a bug.

---

## 0. The question this harness answers (one sentence)

Does ROMY's cross-agent KV-block sharing add an **incremental** win over native vLLM APC
(already ON) on MI300X, under a realistic N=5 shared-prefix multi-agent workload, large
enough to decide the preregistered cut?

**The decisive number is `delta = (B − A)`**, with `A = APC ON, no ROMY salt`. Never
`(B − C)`, never `(B − no-cache)`. `(B − C)` only proves the harness *can* see sharing.

Preregistered cut (does NOT move after seeing data):
- `delta < 5%` → **ABANDON** mechanical sharing (ROMY stays as honest INV-15 salting only).
- `5% ≤ delta ≤ 15%` → **GREY ZONE**.
- `delta > 15%` sustained + reproducible → **INVEST** (Fase 2).

---

## 1. Global conventions (every module obeys)

- **Language:** code + comments in English. Runbook/report prose in Spanish.
- **Imports:** `REPO = Path(__file__).resolve().parents[2]` then `sys.path.insert(0, str(REPO))`
  at the top of each runnable module so `apohara_context_forge.*`, `agents.*`, and
  `scripts.*` import the same way the existing probes do.
- **No vLLM / lmcache / torch imports** in `workload.py`, `arms.py`, `metrics.py` *parsers*,
  `validity.py`, or `analyze.py`. Only `harness.py`'s live launcher path may shell out to the
  `vllm` CLI (as a subprocess, like `local_cross_worker_smoke.py` and `mi300x_squeeze_all.sh`
  already do). vLLM is never `import`ed in Python.
- **Honesty trip-wires (hard fail in review):**
  - No literal performance number assigned to a field that names a measurement
    (`*_gb`, `*_tok_s`, `*_ttft*`, `hit_rate`, `delta*`). Numbers come from a reader or are
    `None` with a `reason`.
  - VRAM ONLY through `metrics.read_kv_footprint` / `metrics.read_hbm` (which wrap
    `VRAMMonitor` + second source). Never the `192.0` / `45.0` constants. A reading whose
    `vram_source ∈ {"amd_default_192gb", "cuda_unavailable", "unknown", "dry"}` is **invalid
    for the report** — surfaced, but `valid=False`.
  - Nothing this harness writes may trip `scripts/check_honesty.sh` (it scans `demo/`,
    `apohara_context_forge/`, `agents/`; we live in `scripts/gate0/`, but keep the same
    discipline so a future scan extension stays green).
- **Determinism for sharing:** every worker env MUST carry `PYTHONHASHSEED=0`
  (`worker_env()` already pins it). The harness asserts this; a missing seed invalidates any
  cross-worker arm.
- **Floats:** round GB to 3 decimals, tok/s to 1, TTFT seconds to 4 — matching the existing
  `logs/mi300x_squeeze/qwen3-32b_measure.json` schema.
- **Mock vs numbers:** "mock" (dry) mode exists ONLY to exercise plumbing/CI on a box with no
  GPU. Dry results carry `measured=False` and NEVER enter the report. (Protocol §7: mock the
  HARNESS, not the numbers.)

### 1.1 Shared enums / sentinels (defined in `arms.py`, imported everywhere)

```python
# arms.py
ARM_A = "A"   # baseline: APC ON, no ROMY salt
ARM_B = "B"   # ROMY: APC ON + shared cache_salt (+ LMCache in cross-worker)
ARM_C = "C"   # negative control: isolated per-request salts
ARMS = (ARM_A, ARM_B, ARM_C)

TOPOLOGY_SINGLE = "single_worker"
TOPOLOGY_CROSS  = "cross_worker"   # >=2 workers + LMCache + Redis
```

---

## 2. Reuse map — what each module wraps (do NOT reinvent)

| New module | Reuses (verbatim or thin-wrap) | Why |
|---|---|---|
| `workload.py` | `agents.demo_agents.AGENT_CONFIGS`; `apohara_context_forge.normalization.prefix_normalizer.PrefixNormalizer`; the inline `SHARED_PREFIX`/`TAILS` skeleton from `scripts/mi300x_measure.py` | Canonical N=5 roles + byte-identical prefix assembly already exist. |
| `arms.py` | `apohara_context_forge.serving.prefix_salt_planner.PrefixSaltPlanner` (`.plan/.shared_salt/.isolated_salt`); `apohara_context_forge.serving.vllm_launch_config` (`build_vllm_serve_args`, `build_kv_transfer_config`, `build_kv_transfer_config_json`, `worker_env`, `DEFAULT_BLOCK_SIZE`); `apohara_context_forge.serving.aiter_config.AITERConfig`; `apohara_context_forge.safety.jcr_gate.JCRSafetyGate` (indirectly, via planner) | Salt decisions (B/C), LMCache launch config, AITER parity, INV-15 ownership all live here already. |
| `metrics.py` | `scripts.mi300x_measure.fetch_prefix_metrics`, `read_hbm`; `scripts.vram_ab_harness.read_second_source_used_gb`, `_read_rocm_smi_used_gb`, `_read_nvidia_smi_used_gb`, `_hardware_label`; `apohara_context_forge.metrics.vram_monitor.VRAMMonitor` | Honest /metrics parser, dual-source HBM, honest hardware label already proven on MI300X. |
| `validity.py` | `apohara_context_forge.serving.vllm_launch_config.worker_env`; `AITERConfig.status`; `metrics.read_prefix_metrics`; greps server logs like `logs_moe_run/vllm_*_crash.log` (`enable_prefix_caching=<bool>`) | APC-ON proof, AITER parity, C-control sanity, seed check. |
| `harness.py` | launch pattern from `scripts/mi300x_squeeze_all.sh:run_model` + `scripts/local_cross_worker_smoke.py` (subprocess vllm, health-wait, teardown); `arms.build_*`; `workload.*`; `metrics.*`; `validity.*` | One runner that orchestrates A/B/C lifecycle + sampling. |
| `analyze.py` | reads raw logs only; uses `metrics.confidence_interval` for re-aggregation | Pure: computes (B−A) ± CI, applies cut, emits the §9 table. |
| `GATE-0-runbook.md` | structure of `scripts/mi300x_runbook.md`; orchestration of `mi300x_squeeze_all.sh` | Operator one-pager (Spanish), A/B/C edition. |
| `GATE-0-report-TEMPLATE.md` | schema from `logs/mi300x_squeeze/qwen3-32b_measure.json`; protocol §9 | The deliverable skeleton (Spanish). |

**Explicitly DO NOT use** for numbers: `benchmarks/run_benchmark.py`, `demo/benchmark.py`
(both simulate — AUDIT #2 violations). `apohara_context_forge.metrics.collector.py` synthetic
path (`return 0.0, 192.0`). `lmcache_bridge.LMCacheConnectorV1` (dead trap code). The
`romy_plugin` attention hooks (not wired to runtime; config-driven only).

---

## 3. `scripts/gate0/workload.py`

Owns the canonical workload and the **measured** reuse rate. No GPU, no network.

### 3.1 Dataclasses

```python
@dataclass(frozen=True)
class AgentSpec:
    """One agent in the N=5 pipeline. Mirrors agents.demo_agents.AGENT_CONFIGS."""
    agent_id: str            # "retriever" | "reranker" | "summarizer" | "critic" | "responder"
    role_prompt: str         # agent-specific role line (NOT the shared system prefix)
    is_judge: bool           # True for critic/responder-class (drives INV-15 via the gate)

@dataclass(frozen=True)
class WorkloadRequest:
    """A single request the harness will send. Prefix is byte-identical across all."""
    request_id: str          # unique, e.g. "retriever:0007"
    agent_id: str
    prompt: str              # PrefixNormalizer-assembled: [system][SEP][role][SEP][tail]
    tail: str                # the variable per-request task line
    max_tokens: int

@dataclass(frozen=True)
class WorkloadSpec:
    """The full workload definition + its declared conditions (enter the report)."""
    name: str                       # e.g. "sprint5_5agent"
    model: str                      # served model name (full-attention dense, e.g. Qwen3-32B)
    canonical_system_prompt: str    # the shared byte-identical prefix (long, multi-block)
    agents: tuple[AgentSpec, ...]   # the N=5 specs
    n_requests: int                 # TOTAL requests across the run (LARGE; protocol forbids ~28)
    concurrency: int                # fixed concurrency for the throughput/footprint arms
    max_tokens: int
    seed: int = 0                   # request-order / tail-cycling seed for reproducibility

@dataclass(frozen=True)
class ReuseStats:
    """MEASURED prefix reuse of the workload (a condition, never assumed)."""
    canonical_prefix_chars: int          # len(canonical_system_prompt)
    canonical_prefix_hash: str           # PrefixNormalizer.get_canonical_hash()
    n_requests: int
    n_distinct_prefixes: int             # distinct normalized prefixes actually built
    shared_prefix_fraction: float        # share of requests on the dominant prefix (0..1)
    approx_prefix_tokens: int | None     # chars/4 heuristic if no tokenizer; None if unknown
    note: str                            # how it was computed (heuristic vs real tokenizer)
```

### 3.2 Public functions

```python
def default_agents() -> tuple[AgentSpec, ...]:
    """The N=5 specs derived from agents.demo_agents.AGENT_CONFIGS.
    is_judge = (agent_id in jcr_gate.JUDGE_ROLES) — i.e. {"critic"} today; do NOT hardcode
    "responder" as judge unless JUDGE_ROLES says so."""

def load_workload(path: str | None = None,
                  *, model: str,
                  n_requests: int,
                  concurrency: int,
                  max_tokens: int = 64,
                  seed: int = 0) -> WorkloadSpec:
    """Build the canonical WorkloadSpec.
    If `path` is given and exists, load YAML (configs/sprint5_5agent.yaml — note: this file does
    NOT exist today; creating it is a workload-author decision, NOT this contract's job).
    If `path` is None, derive from default_agents() + a long canonical_system_prompt
    (the multi-block briefing; reuse the spirit of mi300x_measure.SHARED_PREFIX but make it
    long enough to span many 16-token blocks)."""

def build_requests(spec: WorkloadSpec) -> list[WorkloadRequest]:
    """Materialize spec.n_requests requests. Prefix assembled via PrefixNormalizer
    (canonical_system_prompt=spec.canonical_system_prompt) so EVERY request shares a
    byte-identical system prefix. Tails cycle deterministically over a fixed tail set
    seeded by spec.seed. agent_id cycles over spec.agents. The SAME request list is
    replayed across arms A/B/C — only the cache_salt differs (arms.py decides that)."""

def measure_reuse(spec: WorkloadSpec, requests: list[WorkloadRequest]) -> ReuseStats:
    """Compute the REAL reuse rate of the built requests. Uses PrefixNormalizer hashing to
    count distinct prefixes; approx_prefix_tokens via len/4 with note='char/4 heuristic'
    unless a tokenizer is wired (then note names it). NEVER returns a fabricated rate."""

# Default tail set (variable task lines); shared with mi300x_measure spirit.
DEFAULT_TAILS: tuple[str, ...]
```

**Invariant:** `build_requests` returns prefixes that are byte-identical up to `tail`.
`measure_reuse(...).n_distinct_prefixes` over the *prefix* (excluding tail) MUST be 1 for a
valid shared-prefix workload; if >1 the workload is broken and `validity.py` will flag it.

---

## 4. `scripts/gate0/arms.py`

Owns the A/B/C definitions: launch args, env, and the per-request `cache_salt` for each arm.
Pure (no GPU/network). This is where the protocol's delta definition is enforced at the
salt/flag level.

### 4.1 Dataclasses

```python
@dataclass(frozen=True)
class ArmLaunch:
    """Everything needed to launch ONE vLLM server for an arm/topology."""
    arm: str                       # ARM_A | ARM_B | ARM_C
    topology: str                  # TOPOLOGY_SINGLE | TOPOLOGY_CROSS
    serve_args: list[str]          # full `vllm serve ...` arg list (excluding the "vllm" binary)
    env: dict[str, str]            # env to apply to the subprocess (includes PYTHONHASHSEED=0)
    enable_prefix_caching: bool    # MUST be True for A and B and C (the gate is vs APC-ON)
    uses_lmcache: bool             # True only for B in TOPOLOGY_CROSS
    block_size: int                # DEFAULT_BLOCK_SIZE
    aiter_applied: bool            # AITERConfig parity flag (same across arms)
    note: str

@dataclass(frozen=True)
class RequestSalt:
    """The cache_salt a given arm assigns to a given request."""
    request_id: str
    arm: str
    cache_salt: str | None         # None ONLY if an arm intentionally sends no salt
    shared: bool                   # True if this salt is reused across agents (B shared path)
    reason: str                    # mirrors SaltPlan.reason / "arm A: APC-native, no ROMY salt"
```

### 4.2 Public functions

```python
def build_arm_launch(arm: str,
                     *, model: str,
                     topology: str = TOPOLOGY_SINGLE,
                     block_size: int = DEFAULT_BLOCK_SIZE,
                     max_model_len: int = 16384,
                     gpu_memory_utilization: float = 0.90,
                     kv_cache_dtype: str = "auto",
                     port: int = 8000,
                     extra_args: list[str] | None = None) -> ArmLaunch:
    """Build the launch for one arm.

    ALL THREE arms launch with --enable-prefix-caching (APC ON). This is the heart of the
    protocol's honest comparison: A is APC-ON-without-ROMY, NOT --no-enable-prefix-caching.
    (The legacy vram_ab_harness 'OFF = --no-enable-prefix-caching' path is the confound §6
    forbids; this harness REDEFINES A. Do not copy that flag.)

    - A (single): vllm serve <model> --enable-prefix-caching --block-size N ...  (no kv-transfer)
    - B (single): identical launch to A (the salt is what differs at request time, not the flags)
    - C (single): identical launch to A
    - B (cross):  A's args + --kv-transfer-config <build_kv_transfer_config_json(...)>
                  and env += worker_env() (LMCACHE_USE_EXPERIMENTAL=True, PYTHONHASHSEED=0)
    - A/C (cross): A's args, NO --kv-transfer-config (APC-only across workers; the cross-worker
                   baseline). env still includes PYTHONHASHSEED=0.

    serve_args MUST start exactly like scripts/mi300x_squeeze_all.sh:run_model:
      ["serve", model, "--served-model-name", <served>, "--port", str(port),
       "--enable-prefix-caching", "--kv-cache-dtype", kv_cache_dtype,
       "--max-model-len", str(max_model_len),
       "--gpu-memory-utilization", str(gpu_memory_utilization), "--trust-remote-code", ...]
    For cross B, append --kv-transfer-config via build_kv_transfer_config_json(block_size).
    env starts from AITERConfig().AITER_ENV_VARS + {"VLLM_USE_AITER":"1"} + worker_env();
    aiter_applied reflects whether AITER vars were merged (MUST be identical across arms)."""

def salt_for_request(arm: str,
                     req: "WorkloadRequest",
                     *, anchor_hash: str,
                     cla_group: str = "gate0",
                     planner: "PrefixSaltPlanner" | None = None,
                     candidate_count: int = 5,
                     reuse_rate: float = 0.0) -> RequestSalt:
    """Decide the cache_salt for (arm, request). Single place that encodes the arm semantics:

      ARM_A: cache_salt=None, shared=False, reason='arm A: APC-native prefix sharing, no ROMY salt'.
             (APC still shares byte-identical prefixes natively WITHOUT a salt — that is the floor.)
      ARM_B: planner.plan(agent_role=req.agent_id, anchor_hash=anchor_hash, cla_group=cla_group,
             request_id=req.request_id, candidate_count=candidate_count, reuse_rate=reuse_rate).
             Non-judge -> shared_salt (shared=True). Judge tripping INV-15 -> isolated_salt
             (shared=False) — INV-15 is owned by JCRSafetyGate inside the planner, never re-decided here.
      ARM_C: planner.isolated_salt(anchor_hash, req.request_id) for EVERY request (shared=False,
             reason='arm C: isolated salt per request — negative control, expect ~0% cross hit')."""

def salts_for_workload(arm: str,
                       requests: list["WorkloadRequest"],
                       *, anchor_hash: str,
                       cla_group: str = "gate0",
                       reuse_rate: float = 0.0) -> list[RequestSalt]:
    """salt_for_request mapped over the request list, sharing one PrefixSaltPlanner instance
    (so the gate_log accumulates and validity.py can read INV-15 fires)."""

def aiter_env() -> dict[str, str]:
    """The AITER env that MUST be identical across all arms (AITERConfig().AITER_ENV_VARS +
    VLLM_USE_AITER=1). Returned as a plain dict; harness applies it per subprocess."""
```

**Invariant (the whole gate depends on it):** A, B, C share *identical* `serve_args` except B's
cross-worker `--kv-transfer-config`. Identical model, block size, kv dtype, max-model-len,
gpu-util, and AITER env. The ONLY intended difference between A and B single-worker is the
per-request `cache_salt`. If launches differ otherwise, `validity.py` fails the run.

---

## 5. `scripts/gate0/metrics.py`

Owns honest measurement: KV footprint (primary 1), throughput/TTFT (primary 2 + secondary 3),
prefix-cache counters (secondary 4), and confidence intervals. Readers talk HTTP to a live
endpoint; parsers are pure. No fabricated value ever — `None` + `reason` instead.

### 5.1 Dataclasses

```python
@dataclass
class HBMReading:
    """Device-wide HBM, dual-sourced. Mirrors mi300x_measure.read_hbm output exactly."""
    used_gb: float | None
    total_gb: float | None
    vram_source: str                 # VRAMMonitor.get_vram_source()
    second_source_used_gb: float | None
    second_source: str | None        # "rocm-smi" | "nvidia-smi" | None
    valid: bool                      # False if vram_source in the invalid set (see §1)
    error: str | None = None

@dataclass
class KVFootprint:
    """PRIMARY METRIC 1 — KV-cache footprint, isolated from model weights where possible."""
    kv_used_gb: float | None         # effective KV bytes (see below); None if not derivable
    method: str                      # "gpu_cache_usage_perc" | "num_gpu_blocks" | "hbm_minus_weights" | "hbm_device_wide(NOT_ISOLATED)"
    hbm: HBMReading                  # the raw device-wide reading (always captured)
    model_weight_gb: float | None    # post-load/pre-traffic baseline, if measured
    gpu_cache_usage_perc: float | None  # from vLLM /metrics if exposed
    note: str                        # honest caveat: device-wide HBM does NOT isolate KV

@dataclass
class PrefixMetrics:
    """SECONDARY 4 — vLLM prefix-cache + external-KV counters (DELTA over a window)."""
    queries_delta: float
    hits_delta: float
    hit_rate: float                  # hits/queries, 0.0 if queries==0
    external_queries_delta: float
    external_hits_delta: float       # cross-worker external_prefix_cache_hits
    external_kv_tokens_delta: float  # prompt_tokens_by_source external_kv_transfer
    raw_before: dict                 # fetch_prefix_metrics() snapshot before the window
    raw_after: dict                  # ... and after
    error: str | None = None

@dataclass
class ThroughputSample:
    """PRIMARY 2 + SECONDARY 3 — per-window throughput/TTFT, with the n it came from."""
    mean_ttft_s: float | None
    p50_ttft_s: float | None
    p95_ttft_s: float | None
    decode_tok_s: float | None       # aggregate output tok/s at the run's fixed concurrency
    total_tok_s: float | None        # prompt+output tok/s if derivable, else None
    n_requests: int
    ttft_samples_s: list[float]      # raw per-request TTFTs (for CI re-aggregation in analyze)

@dataclass
class CI:
    """A mean with a confidence interval and the n it summarizes."""
    mean: float | None
    lo: float | None
    hi: float | None
    n: int
    method: str                      # "bootstrap" | "t" | "none(n<2)"
    confidence: float = 0.95
```

### 5.2 Public functions

```python
def read_hbm(device_id: int = 0) -> HBMReading:
    """Thin wrapper over scripts.mi300x_measure.read_hbm; sets valid=False when
    vram_source in {"amd_default_192gb","cuda_unavailable","unknown","dry"}."""

def read_kv_footprint(endpoint: str, device_id: int = 0,
                      *, model_weight_gb: float | None = None) -> KVFootprint:
    """PRIMARY 1. Prefer the EFFECTIVE KV footprint, in order:
      1. vLLM /metrics gpu_cache_usage_perc * total_kv_bytes (if exposed) -> method='gpu_cache_usage_perc'
      2. num_gpu_blocks_used * block_size * bytes_per_token (if exposed) -> method='num_gpu_blocks'
      3. device-wide HBM minus model_weight_gb (if caller passes the post-load baseline)
         -> method='hbm_minus_weights'
      4. fall back to device-wide HBM, method='hbm_device_wide(NOT_ISOLATED)', with a loud note
         that this does NOT isolate KV (the qwen3-32b run showed identical 175.393GB shared vs
         isolated — device-wide HBM dilutes the KV delta to ~0 at slack)."""

def fetch_prefix_metrics(endpoint: str) -> dict:
    """Re-export of scripts.mi300x_measure.fetch_prefix_metrics (name-robust /metrics summation).
    Returns {queries, hits, external_queries, external_hits, external_kv_tokens} or {'error':...}."""

def prefix_metrics_window(endpoint: str, send_fn) -> PrefixMetrics:
    """Snapshot /metrics, call send_fn() (which drives the requests), snapshot again, diff.
    send_fn is supplied by the harness; this function owns ONLY the before/after diff + hit_rate."""

def measure_throughput(endpoint: str, model: str, requests, salts,
                       *, concurrency: int, post_fn) -> ThroughputSample:
    """PRIMARY 2 + SECONDARY 3. Drive `requests` (with their per-arm `salts`) at fixed
    `concurrency` via streaming completions; collect per-request TTFT and total output tokens.
    post_fn(endpoint, model, prompt, salt, max_tokens, stream) -> response is injected so this
    stays vLLM-import-free (same shape as mi300x_measure._post). p50/p95 computed from
    ttft_samples_s. n_requests == len(requests)."""

def confidence_interval(samples: list[float], *, confidence: float = 0.95,
                        method: str = "bootstrap", n_boot: int = 2000,
                        seed: int = 0) -> CI:
    """Mean +- CI. method='bootstrap' (percentile, default) or 't' (normal-ish small n).
    n<2 -> CI(mean=mean_or_None, lo=None, hi=None, n=n, method='none(n<2)'). Pure stdlib +
    optional numpy; NEVER invents a spread when n is too small."""

def read_second_source_used_gb(device_id: int = 0):  # re-export
    """Re-export of scripts.vram_ab_harness.read_second_source_used_gb."""

def hardware_label(vram_source: str) -> str:  # re-export of scripts.vram_ab_harness._hardware_label
    ...
```

**Invariant:** every metric-returning function records the **condition** it was taken under by
returning enough context (n_requests, source, method) that `harness.py` can attach the full
condition block. A metric with no condition is dropped by `analyze.py`.

---

## 6. `scripts/gate0/validity.py`

Owns the gates that, per protocol §6, make-or-break the experiment. Each gate returns a
`ValidityCheck`; the run is only quotable if all `required` checks pass.

### 6.1 Dataclasses

```python
@dataclass(frozen=True)
class ValidityCheck:
    name: str            # "apc_on" | "aiter_parity" | "seed_pinned" | "c_control_zero" |
                         # "shared_prefix_single" | "vram_source_honest" | "n_requests_sufficient"
    passed: bool
    required: bool       # if True and not passed -> run is NOT quotable
    detail: str          # evidence string (grep match, value, etc.)
    evidence: dict       # machine-readable evidence (e.g. {"enable_prefix_caching": True})

@dataclass(frozen=True)
class ValidityReport:
    checks: list[ValidityCheck]
    quotable: bool       # all(required -> passed)
    summary: str
```

### 6.2 Public functions

```python
def check_apc_on(server_log_path: str) -> ValidityCheck:
    """§6(a) confound APC. Grep the vLLM startup line for enable_prefix_caching=<True|False>.
    Pattern proven in logs_moe_run/vllm_*_crash.log (core.py init line). required=True;
    passed iff =True. evidence={'enable_prefix_caching': bool, 'line': str}."""

def check_aiter_parity(arm_launches: list["ArmLaunch"]) -> ValidityCheck:
    """All arms must carry IDENTICAL AITER env (else AITER confounds the delta). required=True.
    Compares the AITER subset of each ArmLaunch.env."""

def check_seed_pinned(arm_launches: list["ArmLaunch"]) -> ValidityCheck:
    """Every arm env has PYTHONHASHSEED=0 (mandatory for cross-worker salt collision).
    required=True for cross-worker; recommended otherwise."""

def check_c_control(prefix_metrics_c: "PrefixMetrics", *, max_hit_rate: float = 0.05) -> ValidityCheck:
    """Arm C must give ~0% cross-agent hit_rate (it proves the harness measures SHARING, not
    just same-prompt APC). required=True; passed iff prefix_metrics_c.hit_rate <= max_hit_rate."""

def check_shared_prefix_single(reuse: "ReuseStats") -> ValidityCheck:
    """The workload's prefix (excluding tails) must collapse to ONE distinct prefix
    (reuse.n_distinct_prefixes == 1 over prefixes). Else B≈A trivially and the gate is void.
    required=True."""

def check_vram_source_honest(hbm: "HBMReading") -> ValidityCheck:
    """vram_source must be a real backend (pyrsmi/drm_sysfs/cuda_nvml/cuda_nvidia_smi).
    amd_default_192gb / cuda_unavailable / unknown / dry -> not honest. required=True for any
    quoted VRAM number."""

def check_n_requests(spec: "WorkloadSpec", *, min_requests: int = 200) -> ValidityCheck:
    """Protocol §6(c): N must be large enough for tight CIs (explicitly NOT ~28). required=True.
    min_requests is a floor, not a target; the report still shows the achieved CI width."""

def run_all(*, arm_launches, reuse, spec,
            apc_log_paths: dict[str, str] | None = None,
            prefix_metrics_c: "PrefixMetrics" | None = None,
            hbm: "HBMReading" | None = None) -> ValidityReport:
    """Run every applicable check and fold into a ValidityReport. apc_log_paths maps arm->logpath."""
```

---

## 7. `scripts/gate0/harness.py`

The runner. Owns vLLM server lifecycle (subprocess), arm sequencing, sampling, and writing the
**raw log** (schema in §9). It is the ONLY module allowed to launch processes / hit the network
broadly. Modes: `dry` (plumbing, `measured=False`, CI/numbers absent) and `live` (real, gated).

### 7.1 Dataclasses

```python
@dataclass
class ArmResult:
    """Everything measured for ONE arm in ONE topology."""
    arm: str
    topology: str
    kv_footprint: "KVFootprint"
    throughput: "ThroughputSample"
    prefix_metrics: "PrefixMetrics"
    model_weight_gb: float | None      # post-load/pre-traffic HBM baseline (for KV isolation)
    server_log_path: str | None
    inv15_fires: int                   # judge requests that took the isolated path (from salts)
    measured: bool                     # False in dry mode

@dataclass
class GateRunResult:
    """The whole run: conditions + per-arm results + validity. Serialized to the raw log."""
    schema_version: str                # == gate0.__version__
    timestamp_utc: str
    topology: str
    workload: "WorkloadSpec"           # serialized via dataclasses.asdict
    reuse: "ReuseStats"
    conditions: dict                   # see §8 condition block
    arms: dict[str, "ArmResult"]       # keyed by "A"/"B"/"C"
    validity: "ValidityReport"
    measured: bool
```

### 7.2 Public functions

```python
def run_gate(spec: "WorkloadSpec",
             *, topology: str = TOPOLOGY_SINGLE,
             mode: str = "dry",
             device_id: int = 0,
             ports: dict[str, int] | None = None,
             vllm_bin: str | None = None,
             redis_url: str | None = None,
             out_path: str | None = None) -> GateRunResult:
    """Top-level entry. For each arm in ARMS:
        1. build ArmLaunch via arms.build_arm_launch(...)
        2. (live) launch vLLM subprocess with env, wait /health (pattern from
           mi300x_squeeze_all.run_model / local_cross_worker_smoke.wait_ready), capture server log
        3. measure model_weight_gb (post-load, pre-traffic) via metrics.read_hbm
        4. build per-arm salts via arms.salts_for_workload(...)
        5. drive the SAME workload requests, sampling throughput + prefix-metrics window +
           KV footprint via metrics.*
        6. (live) tear the server down
       Then run validity.run_all(...) and assemble GateRunResult. In dry mode no server is
       launched: arms/salts/validity-of-config are still exercised; numeric fields are None /
       measured=False. Writes the raw log to out_path (or logs/gate0/<name>_<topology>.json)."""

def _post(endpoint, model, prompt, *, salt=None, max_tokens=16, stream=False):
    """Same shape as mi300x_measure._post; injected into metrics.measure_throughput /
    prefix_metrics_window so those stay import-clean."""

def main(argv: list[str] | None = None) -> int:
    """CLI — see §10."""
```

**Server lifecycle rule:** A, B, C single-worker run **sequentially** on one card (launch →
measure → teardown), exactly like the squeeze runner, so HBM readings are not cross-contaminated.
Cross-worker B uses the sequential store→retrieve pattern of `local_cross_worker_smoke.py`
(worker-1 stores to Redis, dies; worker-2 retrieves), with an APC-only A/C cross baseline.

---

## 8. Condition block (attached to EVERY quoted metric)

`conditions` dict in `GateRunResult` and echoed per metric in the report. No metric without it.

```json
{
  "model": "qwen3-32b",
  "hardware_label": "AMD/ROCm",
  "vram_source": "pyrsmi",
  "second_source": "rocm-smi",
  "topology": "single_worker",
  "n_agents": 5,
  "n_requests": 320,
  "concurrency": 32,
  "max_tokens": 64,
  "approx_prefix_tokens": 512,
  "shared_prefix_fraction": 1.0,
  "block_size": 16,
  "kv_cache_dtype": "auto",
  "max_model_len": 16384,
  "gpu_memory_utilization": 0.90,
  "aiter_applied": true,
  "pythonhashseed": "0"
}
```

---

## 9. Raw-log JSON schema (`logs/gate0/<workload>_<topology>.json`)

Superset of `logs/mi300x_squeeze/qwen3-32b_measure.json` so existing tooling/eyes transfer.
`analyze.py` consumes exactly this. Numbers are `null` (not 0, not a guess) when unmeasured.

```json
{
  "schema_version": "0.1.0",
  "timestamp_utc": "2026-06-01T12:00:00Z",
  "topology": "single_worker",
  "conditions": { "...": "see §8" },
  "workload": {
    "name": "sprint5_5agent",
    "model": "qwen3-32b",
    "canonical_prefix_hash": "…16+ hex…",
    "n_requests": 320,
    "concurrency": 32,
    "agents": ["retriever","reranker","summarizer","critic","responder"]
  },
  "reuse": {
    "canonical_prefix_chars": 2048,
    "canonical_prefix_hash": "…",
    "n_requests": 320,
    "n_distinct_prefixes": 1,
    "shared_prefix_fraction": 1.0,
    "approx_prefix_tokens": 512,
    "note": "char/4 heuristic"
  },
  "arms": {
    "A": {
      "arm": "A", "topology": "single_worker", "measured": true,
      "model_weight_gb": null,
      "server_log_path": "logs/gate0/A_server.log",
      "inv15_fires": 0,
      "kv_footprint": {
        "kv_used_gb": null,
        "method": "hbm_device_wide(NOT_ISOLATED)",
        "model_weight_gb": null,
        "gpu_cache_usage_perc": null,
        "note": "device-wide HBM does NOT isolate KV; …",
        "hbm": {
          "used_gb": 175.256, "total_gb": 191.688, "vram_source": "pyrsmi",
          "second_source_used_gb": 175.256, "second_source": "rocm-smi", "valid": true
        }
      },
      "throughput": {
        "mean_ttft_s": 0.0581, "p50_ttft_s": 0.055, "p95_ttft_s": 0.090,
        "decode_tok_s": 107.5, "total_tok_s": null, "n_requests": 320,
        "ttft_samples_s": [0.05, 0.06, "…"]
      },
      "prefix_metrics": {
        "queries_delta": 3229.0, "hits_delta": 2736.0, "hit_rate": 0.8473,
        "external_queries_delta": 0.0, "external_hits_delta": 0.0,
        "external_kv_tokens_delta": 0.0
      }
    },
    "B": { "…": "same shape as A" },
    "C": { "…": "same shape as A; expect hit_rate ~0.0" }
  },
  "validity": {
    "quotable": true,
    "checks": [
      {"name":"apc_on","passed":true,"required":true,"detail":"enable_prefix_caching=True","evidence":{"enable_prefix_caching":true}},
      {"name":"c_control_zero","passed":true,"required":true,"detail":"C hit_rate=0.0","evidence":{"hit_rate":0.0}}
    ],
    "summary": "all required checks passed"
  },
  "measured": true
}
```

Notes:
- `kv_used_gb` is the PRIMARY-1 number when isolable; otherwise `null` and the report quotes the
  HBM caveat. `analyze.py` must NOT compute a delta on a `null` metric — it reports "not isolable".
- `ttft_samples_s` is retained so `analyze.py` can recompute CIs without re-running.

---

## 10. CLI

### `harness.py`
```
python3 scripts/gate0/harness.py \
  --mode {dry,live} \
  --model <served-model-name> \
  --workload <path-or-empty> \
  --topology {single_worker,cross_worker} \
  --n-requests <int>            # LARGE; protocol forbids ~28 \
  --concurrency <int> \
  --max-tokens <int> \
  --device-id <int> \
  --redis-url <url>             # cross_worker only \
  --port-a <int> --port-b <int> --port-c <int> \
  --vllm-bin <path> \
  --out <logs/gate0/...json>
```
- `--mode dry` runs on any box (no GPU): exercises workload/arms/salts/validity-of-config,
  writes a `measured:false` log. `--mode live` is gated (GPU + vllm).

### `analyze.py`
```
python3 scripts/gate0/analyze.py \
  --in <raw-log.json> [<more raw logs ...>] \
  --primary {kv_footprint,throughput} \
  --out-md <docs/research/_internal/GATE-0-report-<date>.md> \
  --out-json <logs/gate0/verdict.json>
```

### `analyze.py` public API
```python
@dataclass(frozen=True)
class ArmMetric:
    arm: str
    metric: str           # "kv_used_gb" | "decode_tok_s" | "mean_ttft_s" | "hit_rate" | ...
    value: float | None
    ci: "CI"
    condition: dict       # the §8 block
    valid: bool

@dataclass(frozen=True)
class Verdict:
    primary_metric: str
    a: "ArmMetric"
    b: "ArmMetric"
    c: "ArmMetric"
    delta_b_minus_a: float | None       # absolute
    delta_pct: float | None             # (B-A)/A * 100 — the gate number
    delta_ci_pct: "CI" | None           # CI on the percentage delta (bootstrap over samples)
    cut: str                            # "ABANDON" | "GREY_ZONE" | "INVEST" | "INDECISIVE"
    rationale: str
    quotable: bool                      # mirrors validity; INDECISIVE if not quotable

def load_raw(path: str) -> "GateRunResult": ...
def arm_metric(run: "GateRunResult", arm: str, metric: str) -> "ArmMetric": ...
def compute_verdict(run: "GateRunResult", *, primary_metric: str) -> "Verdict":
    """delta_pct = (B-A)/A*100. Apply the preregistered cut:
       <5 ABANDON, 5..15 GREY_ZONE, >15 INVEST. If primary metric is null/not-isolable on A or B,
       or validity.quotable is False -> cut='INDECISIVE' with rationale. NEVER decide on (B-C)."""
def render_report_md(verdict: "Verdict", run: "GateRunResult", template_path: str) -> str:
    """Fill GATE-0-report-TEMPLATE.md with the A/B/C table (value ± CI + condition), the (B-A)
    delta, the verdict vs the cut, and the decision. Spanish prose; numbers from the log only."""
def main(argv: list[str] | None = None) -> int: ...
```

**Gate rule encoded once, here:** the verdict is computed on `delta_pct = (B−A)/A·100` for the
chosen primary metric, with its CI; `(B−C)` is reported only as the harness-validity sanity row,
never as the decision.

---

## 11. `docs/research/_internal/GATE-0-runbook.md` (Spanish, gitignored)

Operator one-pager, A/B/C edition. Sections (prose in Spanish):
1. **Pre-requisitos locales (mock):** `--mode dry` corre sin GPU; valida workload + arms + salts +
   validez-de-config + esquema de log. Comando exacto. Cero números.
2. **Pre-requisitos MI300X (live):** preflight (rocm-smi, imagen `rocm/vllm`, `pyrsmi`, deps host),
   Redis docker para cross-worker, `PYTHONHASHSEED=0`. Reusa el patrón `preflight()` del squeeze.
3. **Orden de ejecución:** single-worker A→B→C secuencial (un server por vez), luego cross-worker
   (worker-1 store → muere → worker-2 retrieve). Comando `harness.py` por topología.
4. **Verificación de validez en vivo:** confirmar `enable_prefix_caching=True` en el server log de
   CADA brazo (grep), parity AITER, seed, C≈0%.
5. **Análisis y veredicto:** `analyze.py --primary kv_footprint` y `--primary throughput`; leer
   `cut`; commitear logs crudos.
6. **Disciplina de honestidad:** descartar lecturas con `vram_source` no-honesto; nunca el 192GB;
   toda métrica con su condición; CI obligatorio.

## 12. `docs/research/_internal/GATE-0-report-TEMPLATE.md` (Spanish, gitignored)

The §9 deliverable skeleton. Placeholders `{{...}}` filled by `analyze.render_report_md`:
- Encabezado: pregunta, criterio de corte preregistrado (copiado, inmutable), fecha.
- **Tabla A/B/C por métrica primaria** (KV footprint GB y throughput tok/s): columnas
  `valor ± CI` y `condiciones` (bloque §8). Fila secundaria TTFT p50/p95 y hit_rate.
- **Delta (B−A)** absoluto y porcentual con su CI. Fila de control: hit_rate de C (~0).
- **Veredicto** contra el corte (`ABANDON` / `GREY_ZONE` / `INVEST` / `INDECISIVE`) + racional.
- **Decisión** y enlace a Fase 1 del PLAN-DEFINITIVO.
- **Validez:** tabla de `ValidityCheck` (apc_on, c_control_zero, aiter_parity, seed, etc.).
- **Logs crudos:** rutas commiteadas. Nota AUDIT: ninguna cifra sin fuente.

---

## 13. Build ownership (8 agents, minimal overlap)

| # | Agent builds | Depends on (contract only — build against signatures, not impls) |
|---|---|---|
| 1 | `workload.py` | none (uses existing repo modules) |
| 2 | `arms.py` | §1.1 enums; existing serving modules |
| 3 | `metrics.py` | existing probes; defines `CI`, `KVFootprint`, etc. |
| 4 | `validity.py` | `metrics.PrefixMetrics/HBMReading`, `arms.ArmLaunch`, `workload.ReuseStats` |
| 5 | `harness.py` | ALL of 1–4 (integration); build last among the .py runners |
| 6 | `analyze.py` | §9 schema + `metrics.CI` only (decoupled from live path) |
| 7 | `GATE-0-runbook.md` | §10 CLI + §11 |
| 8 | `GATE-0-report-TEMPLATE.md` | §9 schema + §12 |

Cross-module contact points are the dataclasses in §3–§7. Agents import types by the names here;
if a name must change, change it in this file in a single edit first.
