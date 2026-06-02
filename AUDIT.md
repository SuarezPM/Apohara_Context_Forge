# ContextForge — V6.0 Honest Audit

> **Status:** Living document. Maintained alongside the codebase.
> Every overclaim shipped in V6.0 is listed here with file:line evidence
> and a tracked fix in V6.1 ("Truth-Up Release"). New mechanisms must
> declare which of the four states below they live in *before* they
> show up in a benchmark.

Every research / systems project ships with a gap between *claims in the
README* and *what the code actually computes*. ContextForge is no exception.
The V6.0 release and the published paper (DOI
[10.5281/zenodo.20114594](https://doi.org/10.5281/zenodo.20114594))
captured the V6.0 state. This document is the public accountability layer:
it lists, with file:line evidence, the things that **look measured** but
are actually **synthesized**, and tracks each one through to a fix.

The document also lists the parts that are **production-grade**, so the
reader knows where the codebase carries its own weight.

---

## The four states

| State          | Meaning |
|----------------|---------|
| 🟢 PRODUCTION   | Real implementation. Computes its claimed value from real inputs. Tests cover real behavior. |
| 🟡 HONEST STUB  | Clearly marked as stub / fallback in docstring or runtime warning. Returns plausible defaults without claiming they are measured. |
| 🟠 PARTIAL      | Real algorithm but with synthetic inputs or hardcoded constants where the claim implies measurement. |
| 🔴 OPTIMISTIC   | The README / paper / benchmark implies "live" or "measured" but the code is actually mocked / hardcoded. |

---

## V6.0 confirmed overclaims (sorted by severity)

### 1. 🔴 Speculative coordinator: fabricated draft probability

- **Claim** *(README §benchmark, paper §1)*: "Speculative acceptance rate ≥ 0.875"; INV-12 (target output distribution preserved by speculation).
- **Reality** *(`apohara_context_forge/decoding/speculative_coordinator.py:261`)*:
  ```python
  draft_prob_estimate = max(0.4, 1.0 - 0.4 * self.config.acceptance_threshold)
  ratio = min(1.0, p_i / draft_prob_estimate)
  ```
  The draft probability `q_i` is **not from the draft model** — it is
  fabricated from a config knob. With `acceptance_threshold=0.9` the
  estimate is 0.64; any target probability above 0.64 gives `ratio=1.0`
  (deterministic accept). INV-12 (distribution-preservation guarantee
  from Leviathan et al. 2023) is **mathematically broken** under this
  formula.
- **Severity:** High. Reviewers reading the paper section on speculative
  decoding will spot this in five minutes.
- **V6.1 fix:** Either expose real draft logprobs across the agent
  boundary and use the real `min(1, p/q)` (preferred), or rename
  `verify_and_commit` to `verify_and_commit_stub`, document it as a
  placeholder, and drop the INV-12 claim from the README and paper §3.

### 2. 🔴 VRAM telemetry: corrupted rocm-smi flag, hardcoded fallback

- **Claim** *(README, paper §4.4)*: live MI300X VRAM monitoring via rocm-smi.
- **Reality** *(`apohara_context_forge/metrics/collector.py:50`)*:
  ```python
  result = subprocess.run(
      ["/opt/rocm/bin/rocm-smi", "--showgpu占用率", "--json"],
      ...
  )
  ```
  The flag contains Chinese characters ("占用率" = "usage rate") — almost
  certainly an LLM-generated mistranslation that stitched English and
  Chinese tokens. **This subprocess call fails on every ROCm install in
  existence.** The function then falls through to line 66:
  ```python
  return 45.0, 192.0
  ```
  Every VRAM number that flows through `MetricsCollector.snapshot()` is
  the hardcoded pair `(45.0 GB, 192.0 GB)`. The dashboard, `/health`,
  and `MetricsSnapshot.vram_source="rocm-smi"` all report fake values.
- **Severity:** High. The dashboard is the single most-visible artifact;
  it's also the one that ships fake numbers most frequently.
- **V6.1 fix:** Replace the flag with `--showuse --showmemuse --json` (or
  whichever valid combination), parse the real JSON keys, and delete the
  hardcoded fallback in favor of `apohara_context_forge/metrics/vram_monitor.py`
  (which already implements the honest pyrsmi → /sys/class/drm path).

### 3. 🔴 S-11 queueing controller: 299% real deviation reported as 0%

- **Claim** *(paper Table 2, S-11 benchmark)*: "QueueingController λ_critical deviation **0.00%**, target < 10%, PASS".
- **Reality** *(`demo/benchmark_v5.py:567-575`)*:
  ```python
  if not is_stable:
      ...
  else:
      # No failure observed — use highest rate as proxy
      observed_lambda_critical = arrival_rates[-1]
      predicted_lambda_critical = controller.compute_stability_state(...).lambda_critical
      deviation_pct = 0.0
  ```
  When the system never goes unstable (which the seeded toy load
  guarantees), the code **sets deviation_pct to 0 unconditionally**.
  The actual values in the published JSON (`demo/benchmark_v5_results.json`):
  ```
  lambda_critical_observed:  2.5
  lambda_critical_predicted: 9.99
  reported deviation_pct:    0.0
  real deviation_pct:        299.76%
  ```
  The controller's math is sound; the benchmark logic launders a 299%
  prediction error into a 0% PASS.
- **Severity:** High. This is the headline metric of S-11.
- **V6.1 fix:** When no instability is observed, report
  `|predicted - max(arrival_rates)| / max(arrival_rates) * 100`. Expect a
  large number under the current toy load — that is *honest signal* that
  we need an adversarial scenario (higher rates, smaller blocks) to stress
  the model, *not* a worse implementation of the model.

### 4. 🔴 Benchmark scenarios S-11..S-15: hardcoded duration_ms

- **Claim** *(paper Table 1)*: per-scenario latency and throughput.
- **Reality** *(`demo/benchmark_v5.py:580, 656, 730, 794, 855`)*:
  ```python
  duration_ms=250.0  # S-11
  duration_ms=150.0  # S-12
  duration_ms=100.0  # S-13
  duration_ms=120.0  # S-14
  duration_ms=  5.0  # S-15
  ```
  The reported `throughput_tps` is then `tokens_processed / (duration_ms
  / 1000)` — pure arithmetic, no actual timing. The work inside each
  scenario completes in microseconds; the "real MI300X durations" in
  paper Table 1 are constants.
- **Severity:** Medium-High. The PASS badges are tautologies, but any
  reviewer running `git grep "duration_ms\s*=\s*[0-9]"` finds it.
- **V6.1 fix:** Wrap each scenario body in `time.perf_counter()` and use
  the measured duration. Same change for `throughput_tps`.

### 5. 🟠 S-12 visual encoder: no encoder is ever called

- **Claim** *(README, paper)*: "5× encoder call reduction" via
  cross-agent VisualKVCache sharing.
- **Reality** *(`demo/benchmark_v5.py:644, 681`)*:
  ```python
  encoder_calls_baseline = 5      # hardcoded
  encoder_calls_actual   = 1      # hardcoded
  reduction              = 5 / 1  # = 5×
  ```
  No vision model is invoked anywhere. The scenario is `store()` once
  plus `lookup()` four times on a numpy random tensor. The cache, hash,
  and store mechanics are real; the "5×" is arithmetic.
- **Severity:** Medium. The VisualKVCache module is real; the headline
  is staged.
- **V6.1 fix:** Either integrate a small CLIP / SigLIP encoder (real
  call, measured wall time), or replace the headline with the legitimate
  one: "cache lookup latency vs. encoder-call latency = O(µs) vs O(ms)
  on the same hardware". Drop the "5×" claim unless we measure it.

### 6. 🟠→🟡→🟢 RotateKV: FWHT rotation fully wired in V7.0.0-alpha.2

- **Claim** *(README, paper §2 mechanism #5)*: "Pre-RoPE INT4 grouped-head
  rotation, 3.97× VRAM reduction".
- **Original V6.0 reality** *(`apohara_context_forge/quantization/rotate_kv.py:215-247`)*:
  `use_fwht` flag read but never applied — only channel reordering + INT4 quant.
- **V7.0.0-alpha.1:** Real orthonormal FWHT shipped as standalone
  module at `apohara_context_forge/quantization/fwht.py` (112 LOC, 8/8 tests).
  Module itself **🟢**, but `quantize_pre_rope()` still didn't call it → 🟡.
- **V7.0.0-alpha.2:** Wire-up landed at
  `apohara_context_forge/quantization/rotate_kv.py:24` (import) +
  lines 162-166 (conditional `fwht(key_states)` + `fwht(value_states)` when
  `cfg.use_fwht=True`, applied after channel reordering and before sink
  separation). INV-10 (pre_rope=True) preserved — verified by
  `tests/test_rotate_kv_fwht_integration.py::test_fwht_preserves_inv10`.
  All 18 tests across the FWHT + RotateKV stack pass (8 FWHT + 5 integration
  + 5 RotateKV).
- **Status:** **🟢 PRODUCTION** — FWHT really executes when configured.

### 7. 🟠 S-15 JCR gate: cherry-picked sweep cases

- **Claim** *(paper §5.2, abstract)*: "0 INV-15 violations across the
  full sweep".
- **Reality** *(`demo/benchmark_v5.py:826-872`)*: the "sweep" is **5
  hand-picked Critic cases plus 4 non-judge cases**, all chosen so the
  invariant holds by construction. The gate module itself
  (`apohara_context_forge/safety/jcr_gate.py`) is honest and well-tested;
  it's the *framing* of S-15 as "empirical evidence" that overreaches.
- **Severity:** Low-Medium. The mechanism is novel and real; the result
  is closer to a unit test than an empirical sweep.
- **V6.1 fix:** Generate the sweep procedurally over the full Cartesian
  product of `(role ∈ {critic, judge, retriever, …}) × (candidates ∈ [1..10])
  × (reuse ∈ [0.1..1.0]) × (shuffle ∈ {0,1})`. Report both fire-rate and
  the *closed-form check* that the gate matches the spec on all points.
  Frame as "exhaustive contract check" rather than "empirical violation rate".

### 8. 🟠→🟢 `tests/test_pipeline.py` — pre-existing regression FIXED in V7.0.0-alpha.2

- **Discovered:** 2026-05-12 (V7.0.0-alpha.1 verification)
- **Root cause:** Commit `466cc3d` ("fix: test_mcp_server 12 failures
  resolved") introduced `_passthrough_decision` in
  `apohara_context_forge/mcp/server.py` which hardcodes `original_tokens=0`
  in the 503-fallback response when the coordinator is unavailable.
  `test_mcp_server.py:307` LOCKS IN this server contract — so the server
  cannot be changed. The fix belongs in the CLIENT.
- **V7.0.0-alpha.2 fix:** `agents/base_agent.py:46-50` — when
  `call_contextforge_optimize` receives `original_tokens=0` on a
  non-empty context (the coordinator_unavailable passthrough),
  fall back to local `len(context.split())` count. Server contract
  preserved (12 mcp tests still pass); client metrics restored.
- **Verification:** `tests/test_pipeline.py` 6/6 PASS (was 4/6).
  Full regression: 359 passed / 25 skipped / 0 failed.
- **Status:** **🟢 RESOLVED.**
- **2026-05-25 (rc.2 branch — root cause beneath these band-aids):**
  `CompressionCoordinator.decide()` was newing up its own `ContextRegistry`
  (ignoring the injected one) and calling a non-existent `find_similar()` →
  `AttributeError` → the MCP `/optimize` endpoint was *always* the 503
  passthrough in production. This is *why* the `original_tokens=0` /
  `base_agent` fallbacks were load-bearing. Fixed: restored DI + a 4-branch
  strategy in `decide()` (closes the 11 `tests/test_coordinator.py` failures);
  added `ContextRegistry.find_similar` + a `PrefixDedup` default for `.dedup`.
  **Verification:** M1 (contract) — the 11 `tests/test_coordinator.py`
  failures are closed. M2 (production `find_similar`) — verified end-to-end by
  installing `faiss-cpu==1.14.2` into the dev venv: both integration tests
  (`tests/test_find_similar.py`, `tests/test_coordinator_integration.py`) pass
  against a real FAISS index, confirming `decide()` no longer raises
  `AttributeError` and `/optimize` returns a real decision. Full suite:
  **394 passed / 27 skipped / 0 failed** with faiss present (363/58/0 without).
  Both new tests stay `faiss`-guarded so CI without faiss skips them cleanly.
  The `original_tokens=0` / `base_agent` fallbacks remain as defense-in-depth,
  no longer the sole reason `/optimize` returns. (Note: `faiss-cpu` is not yet
  pinned in `pyproject.toml`/`requirements.txt` — deferred, those files have
  unrelated uncommitted edits.)

### 9. 🟠→🟢 V6.1 INT4 packing/unpacking asymmetry RESOLVED in V7.0.0-alpha.3

- **Discovered:** in V7.0.0-alpha.2 by the FWHT wire-up work (Track 2) during
  round-trip validation of FWHT integration
- **Symptom:** Round-trip `quantize_pre_rope → dequantize_pre_rope` of a
  random KV tensor shows ~6.3 max absolute error — far above the
  theoretical INT4 step bound. Reproduced with `use_fwht=False` too,
  proving the bug is **pre-existing in V6.1**, not introduced by FWHT.
- **Reality** *(`apohara_context_forge/quantization/rotate_kv.py:222-229` and `:287-294`)*:
  `_quantize_block` packs two nibbles into `keys_int4[blk, i, h, d] |= (val << 4)`
  using the SAME `i` index (write side). `_dequantize_block` unpacks both
  `val1` and `val2` from a SINGLE byte at `packed_int4[blk, i, h, d]`
  (read side). The two routines are **asymmetric** — write puts each
  nibble in a different byte position; read expects them in the same
  byte. Hence the codec round-trip is broken.
- **Severity:** Medium. The 3.97× VRAM reduction claim is unaffected
  (compression IS happening), but the *fidelity* of dequantization
  is much worse than INT4 theory says it should be. The integration
  test `tests/test_rotate_kv_fwht_integration.py::test_fwht_roundtrip_through_pipeline`
  uses a 3× slack tolerance against this baseline.
- **V7.0.0-alpha.3 fix:** `_quantize_block` rewritten to pack along
  head_dim (not seq) to match the read side's invariant. Single
  `(scale, zero_point)` per packed byte governs both nibbles. Pre-fix
  max round-trip error: ~6.3; post-fix: 0.0332 (well under 0.07 INT4
  envelope). New `tests/test_rotate_kv_int4_codec.py` (4 tests, all
  PASS) locks in the fix; `tests/test_rotate_kv_fwht_integration.py`
  tolerance tightened from 3× to 1.5× baseline (catches any future
  regression).
- **Status:** **🟢 RESOLVED.**

### 10. 🟠→🟢 K8s operator security hardening RESOLVED in V7.0.0-alpha.3

- **Surfaced by:** V7.0.0-alpha.2 Phase 4 security-reviewer
- **Concerns** (operator/controllers/apoharacontextforgecluster_controller.go):
  - **No SecurityContext** on worker or Redis pods (`runAsNonRoot`,
    `readOnlyRootFilesystem`, drop ALL capabilities are all unset).
    Pods would run as root with all Linux capabilities → node-level
    compromise potential under RCE.
  - **No dedicated ServiceAccount + RBAC manifests** (deferred per
    `operator/config/manager/kustomization.yaml:6` comment).
  - **Redis sidecar runs unauthenticated** (no `--requirepass`); any
    namespace pod can read/write the shared KV cache.
  - **No NetworkPolicy** isolating worker pods or Redis.
  - **Default image is `:latest`** (mutable tag — supply-chain risk).
- **Mitigation in V7.0.0-alpha.2:** `operator/README.md` carries a
  prominent ⚠️ NOT PRODUCTION READY warning listing these 5 items as
  prerequisites. The operator binary is **not** built or
  deployed in V7.0.0-alpha.2 — only the reconcile logic + unit tests +
  integration-test skeleton are shipped. None of these issues are
  exploitable in the current V7.0.0-alpha.2 state because the operator is
  not running anywhere.
- **V7.0.0-alpha.3 delivery:**
  - **SecurityContext** ✅ — both Redis + worker pods get full hardening:
    PodSecurityContext (runAsNonRoot, runAsUser, FSGroup-on-Redis,
    SeccompProfileTypeRuntimeDefault) + per-container SecurityContext
    (AllowPrivilegeEscalation=false, ReadOnlyRootFilesystem=true,
    Capabilities.Drop=ALL). EmptyDir volumes mounted at /data (Redis) and
    /tmp (worker) for the readonly rootfs. 4 new controller tests assert
    each field.
  - **ServiceAccount + namespaced RBAC** ✅ — `operator/config/rbac/`
    ships SA + namespaced Role (no ClusterRole, no wildcards) + RoleBinding +
    leader-election Role/RoleBinding. Phase 4.5 tightened secrets verbs to
    `get;list;watch;create` only (no update/patch/delete since controller
    never writes after first Create).
  - **Redis authentication** ✅ — `reconcileRedisAuthSecret` uses
    `crypto/rand` to generate a 32-char alphanumeric password, stored
    as Secret `<cluster>-redis-auth` with OwnerReference. Redis pod
    consumes via `--requirepass $(REDIS_PASSWORD)` + SecretKeyRef env;
    worker pods get the same SecretKeyRef. Idempotent (no rotation per
    reconcile). 2 new controller tests cover creation + stability.
  - **NetworkPolicy** ✅ — `operator/config/networkpolicy/` ships 4
    manifests: `default_deny_all` (deny ingress+egress by default),
    `worker_to_redis` (allow worker → Redis on 6379 + DNS), `worker_ingress`
    (allow same-namespace → worker:8000), `redis_ingress` (allow
    worker → Redis:6379). Admin-applied; not auto-managed by operator.
  - **Image digest pinning** 🟡 — moved from `:latest` to `:v7.0.0-alpha.3`
    versioned tag + explicit `ImagePullPolicy: IfNotPresent` on both Redis
    and worker containers. Sample CR carries a `# TODO: pin to @sha256:...`
    comment. Full digest pinning is deferred to V7.0.0 final release when
    the production image is published.
  - **Phase 4.5 additional hardening:** `AutomountServiceAccountToken: false`
    on both Redis + worker pods (neither needs K8s API access); leader-election
    Role `delete` verbs removed (controller never deletes leases/configmaps).
- **Tracked open items (not release blockers):**
  - kubebuilder RBAC marker `+kubebuilder:rbac:groups=contextforge.apohara.dev,...,verbs=get;list;watch;create;update;patch;delete` (controller.go:51-56) would regenerate a ClusterRole if `make manifests` is run. The hand-written namespaced role.yaml is currently the source of truth. Follow-up: align markers with intent.
  - `govulncheck ./operator/...` not yet run in CI. `golang.org/x/net@v0.19.0` may have newer patches; recommend `go get golang.org/x/net@latest && go mod tidy` before V7.0.0 final.
- **Status:** **🟢 RESOLVED** (5/5 items closed; image pinning at versioned-tag is alpha-acceptable per security-reviewer; production hardening tracked above as known follow-ups for V7.0.0).

### V7.0.0-alpha.5 — extended deltas (2026-05-12, real MI300X)

| Finding | Severity | Status |
|---------|----------|--------|
| 🚨 **FWHT degrades INT4 quality 200×** under current codec. Measured MSE: use_fwht=False → 1.01e-02; use_fwht=True → 2.01e+00. Paper v2.0 conclusion: use_fwht=False is the recommended config. | High | Follow-up candidate: per-nibble independent scales codec rewrite would reclaim FWHT benefit at cost of ~0.5× storage. |
| 🟡 V6.x #3 `LMCacheConnectorV2` only supports NVIDIA-CUDA LMCache. AMD ROCm fallback (lmcache.non_cuda_equivalents) has a different API. Currently enters honest-fallback on MI300X even with lmcache + redis-server installed. | Medium | Follow-up candidate: adapt connector to non-CUDA backend API. |
| 🟡 FWHT torch path has +700% peak GPU alloc overhead from `.clone()` at each butterfly stage. Throughput 25-33 GB/s vs 3.73 TB/s HBM3 measured. | Medium | Follow-up candidate: in-place strided butterfly to drop overhead to ~+10%. |
| 🟢 HBM3 effective bandwidth measured at **3.73 TB/s = 70.5% of advertised 5.3 TB/s peak** on MI300X VF (SR-IOV slice). Honest paper §3 number. | Info | Promoted in paper v2.0 (replaces "5.3 TB/s peak"). |
| 🟢 Full pytest regression on MI300X+ROCm: **347/358 pass** (~~11 failures in test_coordinator.py are version-mismatch with newer rich/sentence-transformers/numpy 2.2.6~~ — **CORRECTED 2026-05-25:** the 11 `test_coordinator.py` failures were a `ContextMatch` schema/API drift (model required `tokens_saved`; tests used `shared_prefix_tokens`) compounded by a broken `CompressionCoordinator.decide()`, **not** a dependency-version issue. Fixed on the `rc2-foundation` branch — see item #8). FWHT, observability, INT4 codec, rotate_kv all pass on real ROCm. | Info | V6.1 honesty: substrate works on real AMD hardware. |
| 🟢 INT4 codec quality at 3.55× reduction: MSE = 1.01e-02 (use_fwht=False), max abs err 0.33. Pareto-acceptable for KV cache. | Info | Paper v2.0 §5 Pareto table. |
| 🟢 Hardware label honesty: JSON logs now report `rocm-hip:6.2.41133:AMD Instinct MI300X VF`, not just `cuda`. V6.1 discipline applied. | Info | V7.0.0-alpha.5 fix from user catch. |

### V7.0.0-alpha.4 — deltas (2026-05-12, real MI300X)

| Claim | Source | Status post-measurement |
|-------|--------|--------------------|
| **RotateKV pre-RoPE INT4 → 3.97× VRAM reduction** (paper §2 mech #5) | Literature target (RotateKV, IJCAI 2025) | **🟡 NOT measured by Apohara on MI300X.** Real measurement on AMD Instinct MI300X VF (192 GB, gfx942, ROCm 7.2.0, torch 2.5.1+rocm6.2) across 8 shape configs (4K-32K seq × 16-64 heads × 64-256 head_dim): `reduction_factor = 3.55×` essentially constant. Paper v2.0 MUST report 3.55× measured, not 3.97× literature target. |
| **FWHT integration runs on real MI300X** | V7.0.0-alpha.2 + V7.0.0-alpha.3 wire-up | **🟢** — 9/9 tests pass on MI300X in 1.33 s. Log `logs/mi300x_fwht_*.json`. |
| **`reduction_factor` scales with sequence length** | Paper assumption | **🟢 CONFIRMED** — constant 3.55× from seq=4K to seq=32K. Per-block scale/zero_point + sink-fp16 overhead amortizes well. |
| **`reduction_factor` scales with head_dim and num_heads** | Paper assumption | **🟢 CONFIRMED** — same 3.55× across head_dim=64/128/256 and num_heads=16/32/64. |
| **V6.2 adversarial bench needs MI300X** | measurement plan | **🟢→ honest skip.** `demo/benchmark_v62_adversarial.py` is pure NumPy simulation (no torch, no GPU). MI300X execution would have produced identical numbers to laptop, so it was skipped. |

The 0.42× gap between literature target (3.97×) and Apohara's measured
3.55× is the cost of single (scale, zero_point) per packed byte (V7.0.0-alpha.3
AUDIT #9 fix) instead of per-nibble independent scales. The choice was forced
by the read-side byte layout (see #9). Reclaiming the 0.42× would require a
codec rewrite (per-nibble scales, ~2× metadata overhead) — paper v2.0 reports
the trade-off honestly rather than chasing the literature number.

### V7.0.0-alpha.3 — deltas (2026-05-12)

| Track | Change | State |
|-------|--------|-------|
| 1 | `apohara_context_forge/quantization/rotate_kv.py` `_quantize_block` rewritten (pack along head_dim) | #9 🟠 → 🟢 |
| 2 | `operator/controllers/apoharacontextforgecluster_controller.go` Pod + container SecurityContext + image versioned-tag + ImagePullPolicy + AutomountServiceAccountToken=false | #10 SecurityContext + image-pin → 🟢 / 🟡 (digest pin V7.0.0 final) |
| 3 | `operator/config/rbac/` — SA + namespaced Role + RoleBinding + leader-election RBAC (secrets verbs tightened in Phase 4.5) | #10 RBAC → 🟢 |
| 4 | `operator/controllers/...` Redis auth Secret via crypto/rand + `operator/config/networkpolicy/` (4 policies: default-deny + worker-to-redis + worker-ingress + redis-ingress) + `scripts/mi300x_*` for MI300X measurement | #10 Redis-auth → 🟢, #10 NetworkPolicy → 🟢, MI300X prep ✓ |
| Phase 4.5 fixes | mi300x_vram_measurement.py rewritten with honest CPU-NumPy bridge protocol; CRD Phase enum trimmed to actually-emitted values; malformed `manager/kustomization.yaml` fixed | V6.1 discipline honored |

**Honest measurement protocol for `scripts/mi300x_vram_measurement.py`:**
The current `RotateKVQuantizer` is NumPy-only (no torch fast path).
The script now allocates the baseline KV cache as `torch.float16` on
CUDA (real MI300X allocation footprint = `baseline_fp16_bytes`),
copies to NumPy on CPU for the quantize call (canonical
`(batch, seq_len, num_heads, head_dim)` layout), measures
packed-storage footprint = `keys_int4.nbytes + values_int4.nbytes +
scales.nbytes + zero_points.nbytes` = the bytes you'd write to
Redis/LMCache. The `reduction_factor` is honest because both
numerator and denominator are real. A separate `peak_gpu_alloc_bytes`
captures CUDA peak during the round-trip (includes the device↔host
copy — disclosed in the docstring rather than hidden). A future
release can add a torch fast path to RotateKVQuantizer and re-measure
on-GPU peak without the copy; the CPU bridge protocol is the V6.1
discipline applied to compute as well as claims.

### V7.0.0-alpha.2 — deltas (2026-05-12)

| Change | State delta |
|--------|-------------|
| `apohara_context_forge/quantization/rotate_kv.py` — FWHT wired into `quantize_pre_rope()` | #6 🟡 → 🟢 |
| `agents/base_agent.py` — token-count client fallback for `original_tokens=0` server passthrough | #8 🟠 → 🟢 |
| `apohara_context_forge/observability/otlp_exporter.py` + recorders OTLP fan-out + `dashboards/inv15.json` | 🟢 (new) — Track 3 |
| `operator/controllers/apoharacontextforgecluster_controller.go` 40→453 LOC real reconciler + 4 tests | 🟡 (real logic, not deployed) — Track 4 |
| (security-reviewer Phase 4) | NEW: #9 INT4 packing bug (pre-existing) + #10 K8s operator hardening (deferred to V7.0.0-alpha.3) |
| Inline security fixes Phase 4.5 (`raise_for_status()` in base_agent.py, OTLP `insecure=False` default, path canonicalization for `APOHARA_OBSERVABILITY_DIR`) | Security baseline hardened |

### V7.0.0-alpha.1 — deltas added (2026-05-12)

Three new modules entered the audit, all marked at their honest status:

| Module | State | Why |
|--------|-------|-----|
| `apohara_context_forge/quantization/fwht.py` | 🟢 PRODUCTION | Real butterfly recursion, 8/8 tests, orthonormal, fp16 upcast. Standalone — not yet called by `RotateKVQuantizer` (closing #6 from 🟠 to 🟡 above). |
| `apohara_context_forge/observability/{prometheus_exporter,audit_log,recorders}.py` | 🟢 PRODUCTION | Real `prometheus_client` Counter/Gauge + real JSONL audit log. Honest-fallback when `prometheus_client` not installed. Smoke wire-up at `safety/jcr_gate.py:159` (late import, best-effort). 6/6 tests. |
| `operator/` + `charts/apohara-contextforge/` | 🟡 HONEST STUB | CRD + helm chart YAML validate (`bash operator/validate.sh` exits 0). Reconciler logs "reconciled" only — real reconciliation lands in V7.0.0-alpha.2. README declares this status. |

The community-policy track (CONTRIBUTING + DCO + CoC + PR template) is
governance, not a code module, so it does not enter the state table.

---

## What is actually real (don't apologize)

These modules are production-grade and back the substrate of the system:

| Module | What it does, honestly |
|--------|------------------------|
| `safety/jcr_gate.py` | Risk function + threshold + audit log. Deterministic. The INV-15 concept is the most original IP in the repo. |
| `storage/token_dance.py` | Real master-mirror sparse-diff numpy. Reconstructs byte-correct to ~1e-7 (float roundoff). |
| `registry/context_registry.py` + `registry/vram_aware_cache.py` | Real DI, real LSH+FAISS+VRAM-pressure eviction across five modes. |
| `dedup/lsh_engine.py` + `dedup/faiss_index.py` | Real 64-bit SimHash with Hamming distance + real FAISS IndexFlatIP with IVF upgrade path. |
| `scheduling/step_graph.py` + `scheduling/pbkv_predictor.py` | Real DAG with topological compute + real 2nd-order Markov with Laplace smoothing and JSONL persistence. |
| `compression/{coordinator,compressor,budget_manager}.py` | Real LLMLingua-2 wrapper + sensible per-segment compression policies. |
| `agents/*.py` + `mcp/server.py` | Real 5-agent pipeline, real FastAPI lifespan-managed MCP server with Depends-based DI. |
| `metrics/vram_monitor.py` | The *correct* VRAM path (pyrsmi → /sys/class/drm → 192GB default). Just needs to be wired into `MetricsCollector`. |

The substrate of the system — registries, indexes, schedulers, agents,
compressors, server — earns its keep. The lies are concentrated in
**(a) metrics/collector.py**, **(b) demo/benchmark_v5.py V5/V6
scenarios**, and **(c) speculative_coordinator.py:261**.

---

## V6.1 — "Truth-Up Release" (2 weeks, before any new feature)

Ordered by leverage; each item links to its fix above.

| # | Fix | Effort | Risk if skipped |
|---|-----|--------|-----------------|
| 1 | metrics/collector.py rocm-smi flag → real numbers via VRAMMonitor | 1 h | Anyone running on real MI300X sees the lie immediately. |
| 2 | benchmark_v5.py S-11 deviation logic + 5 hardcoded `duration_ms` → real timing | 4 h | Paper Table 1 cannot survive `git grep`. |
| 3 | speculative_coordinator.py:261 — either real `q_i` or downgrade to stub | 1 d | Reputationally the worst because the paper makes a formal-correctness claim about it. |
| 4 | S-15 procedural Cartesian sweep | 4 h | Reframes "0 violations" as "exhaustive contract check" — stronger, not weaker. |
| 5 | S-12 real encoder OR honest reframing | 4 h | The 5× claim is the easiest to disprove. |
| 6 | RotateKV: implement FWHT OR relabel as "follows IJCAI 2025; FWHT pending" | 1 d | Low urgency; can stay 🟠 if labeled. |
| 7 | `AUDIT.md` (this file) committed at root | — | Done. |
| 8 | README hero stat strip cross-references AUDIT.md for the figures | 30 min | Public accountability multiplies the credibility of the rest. |

Total V6.1 effort: **~3.5 dev-days**. Ship as **V6.1 with full
changelog**, including a Zenodo replacement deposit so the DOI tracks
the corrected numbers.

---

## Maintenance discipline (from V6.1 onward)

1. **No new mechanism enters the README mechanism table without an entry in this file** declaring its state (🟢/🟡/🟠/🔴).
2. **No benchmark scenario merges without** (a) real `time.perf_counter()` measurement and (b) a procedurally-generated input set, *not* a hand-curated one.
3. **Every paper-claimed invariant must have a test** that exhaustively verifies it on at least 100 procedurally-generated points, not 5 hand-picked ones.
4. **Every external paper we cite as "implemented"** must have one of: (a) faithful implementation with a passing test against the paper's reference output, OR (b) a "follows X, with delta Y" disclaimer that lists what we actually do differently.
5. **The CI runs `git grep -E "duration_ms\s*=\s*[0-9]"` on `demo/`** and fails if any match — same for `vram_peak_gb\s*=\s*[0-9]`. Hardcoded perf numbers are a build failure.

---

## Open questions deferred to V6.x scoping

These are the questions where the answer determines what we build next.
See the V6.x roadmap discussion for the current direction.

- Is the **speculative coordinator** worth implementing properly, or is
  the right move to remove it entirely (it isn't load-bearing for any
  other mechanism)?
- Is **RotateKV FWHT** worth implementing in Apohara given that the
  paper's authors have released CUDA reference code that we'd be
  duplicating, or do we cite-and-skip?
- Does the **vLLM ATOM plugin (V6.x item #1)** justify a true V1 plugin
  PR upstream to vLLM, or do we publish the standalone Apohara plugin
  on PyPI and let users wire it themselves?

---

## 12. 🟢 7 critical bugs fixed (2026-05-16)

External strategist review (Perplexity Deep Research + an external
reviewer) independently validated seven defects in the codebase that a
first-time reader would surface in minutes. They are now all closed.
Each fix landed as a separate atomic commit on `main`.

| # | Area | File:line | Bug | Commit |
|---|------|-----------|-----|--------|
| 1 | registry | `apohara_context_forge/registry/context_registry.py:330-331` | `tokens_saved = blocks_per_match * block_size * len(valid_matches)` was `len(valid_matches)² × block_size` — a quadratic over-count of every cache-hit savings number reported by `SharedContextResult.total_tokens_saved`. Fixed to drop the redundant `len(valid_matches)` factor. | `0409de4` |
| 2 | mcp/lifespan | `apohara_context_forge/mcp/server.py:57-61` | `ContextRegistry()` was constructed but `.start()` was never invoked, so the VRAM cache background monitor never ran for the life of the FastAPI server. Added `await registry.start()` after construction (guarded by `getattr` so monkeypatched test fakes still pass) and a symmetric `await registry.stop()` in the lifespan finally block. | `ba096d9` + fixup `1f61cc5` |
| 3 | mcp/metrics | `apohara_context_forge/mcp/server.py:253-` | The background `metrics_loop` snapshotted the module-level `metrics = MetricsCollector()` singleton, but every endpoint resolves the collector via `Depends(get_metrics)` → `app.state.metrics`. The loop was logging an empty, never-updated snapshot. Loop now accepts an optional `FastAPI` arg and reads `app_.state.metrics` per iteration. | `8a7d3ad` |
| 4 | agents | `agents/base_agent.py:53-99` | `BaseAgent.call_vllm` measured request-total wall time and labelled it `ttft_ms`. True TTFT requires streaming. Renamed local + docstring to `request_latency_ms` and added an inline comment so any future reader knows what is and isn't measured. The legitimate `ttft_ms` field on `apohara_context_forge.models` and the `contextforge_agent_ttft_ms` Prometheus histogram are unaffected. | `621b4a8` |
| 5 | agents | `agents/base_agent.py:46-58` | When the MCP server returns `original_tokens=0` on the `coordinator_unavailable` passthrough, the fallback was `len(context.split())` (whitespace word count, under-counts for code / multibyte by ~1.3-3x). Routed through `TokenCounter.get().count(context)`, the same Qwen3 tokenizer used by the registry and LSH engine. | `959bc46` |
| 6 | serving | `apohara_context_forge/serving/lmcache_bridge.py:38-` | `LMCacheConnectorV1.on_save_kv_layer` constructed `LMCacheMeta` and emitted a debug log but never called `self._client.put`. README documented V2 as the replacement; V1 stayed in tree and several callers (tests + demo scripts) still imported it. Option B applied: class is now marked deprecated, active-client construction emits `DeprecationWarning`, and the active save path raises `NotImplementedError` so the previously-silent stub surfaces loudly. The inactive (no-client) no-op semantics that the existing tests and demos rely on are preserved. | `9fac9eb` |
| 7 | decoding | `apohara_context_forge/decoding/speculative_coordinator.py:280-291` | The V6.0 `draft_prob_estimate` field was already removed by the V6.1 truth-up (replaced by a proper `draft_logprobs` argument, the Leviathan path). The fallback-path local was still named `estimate`, which made its stub-nature opaque. Renamed to `_stub_draft_prob` with an inline comment pointing back at this section and the V6.0 retraction so any future reader sees the lie immediately. No behaviour change. | `37196eb` |

**Verification:**

```
PYTHONPATH=. python3 -m pytest tests/ -q
# 373 passed, 26 skipped, 6 warnings in 200.43s

bash scripts/check_honesty.sh
# honesty guard PASS — no regressions detected
```

No test was changed to "match the corrected expectation" — all existing
assertions were already consistent with the corrected semantics. The
one test that initially failed after Bug 2
(`test_lifespan_constructs_and_disposes`) was a mock-substitution
collateral: its `_LifeReg` fake omits `start`/`stop`. The fixup commit
(`1f61cc5`) wraps the new `start()` call in `getattr` — same defensive
pattern already used for `clear` and `vllm.aclose` in the lifespan
teardown — and the test passes unchanged.

The 7 fixes total 8 commits (one fixup for Bug 2 to keep the test
suite green without amending the original bug-fix commit). Final
commit:  *(filled in after push)*.

---

## 13. 🟢 INV-15 paper V2.0 preprint draft committed (2026-05-16)

A V2.0 preprint draft of the INV-15 paper was committed to the
`papers/` directory. The draft refines `paper/inv15_paper.pdf` (V2.0.1, May 13,
2026, 12-reference graph, DOI [10.5281/zenodo.20114594](https://doi.org/10.5281/zenodo.20114594))
with three additions specified in the acceptance criteria.

**Files committed:**

| Path | Bytes | Purpose |
|------|-------|---------|
| `papers/inv15_v2.tex` | ~63 KB | V2.0 LaTeX source (1,280+ lines). |
| `papers/inv15_v2.pdf` | ~416 KB, 13 pp | Pre-built PDF via tectonic 0.15+. |
| `papers/references.bib` | ~21 KB, 23 entries | 17 entries inherited from V2.0.1, 6 new for V2.0. |
| `papers/figures/` | 4 PNG | Carried over from V2.0.1 (HBM3 bandwidth, FWHT perf, quant Pareto, reduction-factor). |
| `papers/README.md` | preprint disclaimer + build command + reproducibility table. |

**V2.0 additions (over V2.0.1):**

1. *Adjacent attack surfaces* subsection (§2.4): NDSS 2025 KV-cache
   timing side-channel \cite{kvcacheleak}, KV-Cloak rotation defense
   \cite{kvcloak}, Adversa AI red-team toolchain \cite{adversa}, AMD
   vLLM-ATOM official May 2026 launch \cite{amdvllmatom}.
2. *Sister-stack judge-defense validation* (new §): JailbreakBench
   (Chao et al. NeurIPS 2024 D&B) `93.75% ± 2.7%, 95% CI [86.2%,
   97.3%], n=80` and HarmBench (Mazeika et al. NeurIPS 2024 D&B)
   `77.50% ± 12.6%, 95% CI [62.5%, 87.7%], n=40` from the Apohara
   Aegis sister repository (separate project, same author).
3. *Vendor-Fallback Architecture* (new §): sketches a
   FallbackVendorAdapter that decouples the gate logic from a single
   LLM vendor; outlines a three-tier defense (INV-15 cache invariant
   + KV-Cloak side-channel + vendor fallback).
4. *Appendix A*: reference-implementation pointer to
   `apohara_context_forge/safety/jcr_gate.py` with the coefficient
   mapping between Eq. 1 of the paper and the runtime Python
   constants. Notes the implementation conservatism
   (`_RISK_HIGH_REUSE=0.15` vs theory $\alpha_u=0.1$) and why it
   preserves Theorem 1.

**Honesty discipline applied:**

- Hardware label `rocm-hip:6.2.41133:AMD Instinct MI300X VF` (not `cuda`).
- No 7.8x TTFT claim (per CLAUDE.md §6 and AUDIT.md item 12 bug 4).
- All measurements trace to committed logs (`logs/*.json` in either
  this repo for MI300X numbers, or `apohara-aegis/logs/*.json` for
  JBB / HarmBench numbers).
- Confidence intervals reported with sample sizes; the
  $77.50\% \pm 12.6\%$ HarmBench result is honest about a $0/8$ block
  rate on the copyright sub-category (not a defense surface) rather
  than dropping that category to inflate the overall number.

**Build command:**

```bash
cd papers
tectonic inv15_v2.tex   # 13-page PDF in ~10 s; warnings about
                        # underfull hboxes are cosmetic
```

**Scope disclaimer:** **This is a preprint draft committed to the
repository only.** Real arXiv submission requires the endorsement
chain (2--3 days minimum) and is scheduled for a later milestone. The
version of record for citation today remains the Zenodo deposit
([DOI 10.5281/zenodo.20114594](https://doi.org/10.5281/zenodo.20114594)).

**Status: 🟢 SHIPPED** (acceptance criteria 1--7 satisfied).

---

## 14. 🟡→⬛ 5-agent benchmark side-by-side: scripted, CPU-mock only, never run on GPU (2026-05-16)

A side-by-side benchmark of `vllm --enable-prefix-caching` (baseline) vs
`vllm + apohara-context-forge plugin` on the 5-agent shared-context
workload was scripted but **only ever ran in CPU-mock mode**. The
real-GPU side-by-side measurement was never executed — GPU access was
not available at the time — so **no GPU benchmark numbers exist** for
this workload.

**Honesty disclosure (the technical finding worth keeping):**

- The composed JSON's `hardware` field read literally `"CPU-mock
  fallback"`. No "we ran it on a GPU" claim was ever made.
- HBM in the mock output was **modeled, not measured**, via a documented
  closed-form (Llama-3-8B; 32 layers × GQA-8 KV heads × fp16; mean reuse
  rate from the workload spec). The schema's `honesty_note` field stated
  which fields were real (latency, tokens, JCR — from the workload run)
  vs modeled (HBM — from the closed-form).
- The ~76% HBM-saved figure was a closed-form consequence of the
  workload's mean reuse rate (0.76) **by construction**, not a measured
  result.

**Resolution.** The mock-only benchmark toolchain (orchestrator,
JSON-composer, GIF-replay generator, the `BENCHMARKS.md` placeholder
table, and the associated mock JSON logs) carried no real measurement
and was **removed from the repository** rather than left as a
GPU-deferred placeholder. The durable, GPU-measured benchmark evidence
lives in items #16, #18, and #19 (real MI300X runs).

**Status: ⬛ RETIRED** — the only artifact this entry described was a
CPU-mock benchmark with no measured GPU data; the toolchain was deleted.
Real side-by-side KV-sharing evidence is item #19 (84.7% prefix-cache
hit-rate, measured on MI300X).

---

## 15. 🟢 FORGE-LEDGER: per-decision INV-15 certifier + tamper-evident ledger

Continuous formal-invariant auditing for the JCR gate. Opt-in, default
off (set `APOHARA_FORGE_LEDGER` to enable; certification costs ~ms of Z3
per gate decision).

| Component | File | What it does, honestly |
|-----------|------|------------------------|
| Per-decision certifier | `apohara_context_forge/safety/inv15_certifier.py` | `certify_decision(...)` asks Z3 whether the observed `use_dense` could differ from the mandate at that input point; UNSAT ⇒ they match. Reuses `build_inv15_constraints` from `z3_inv15_proof`. Fails closed on out-of-domain inputs so a pinned-UNSAT case can't become a vacuous false-green. |
| Hash-chained ledger | `apohara_context_forge/observability/ledger.py` | Real SHA-256 chain `entry_hash = sha256(prev_hash + canonical(payload))`, append-only. `verify()` reports the first mis-hashed/malformed/unparseable line. |
| Certified recorder | `apohara_context_forge/observability/recorders.py` | `record_certified_inv15_decision(...)` certifies + appends the cert to the ledger, then does the normal Prometheus/AuditLog/OTLP fan-out. |
| Verify CLI | `apohara_context_forge/observability/ledger_cli.py` | `verify <path>` → exit `0` intact / `2` tampered / `64` usage. |
| Gate wiring | `apohara_context_forge/safety/jcr_gate.py` | `gate_decision()` emits a certified entry only when `APOHARA_FORGE_LEDGER` is set; best-effort (try/except, never raises into the gate path). |

**Scope caveat (no overclaim).** The certifier verifies the **modeled
domain** — the closed-form INV-15 decision logic encoded in
`build_inv15_constraints` — and confirms each observed decision matches
that model. This is the *same* caveat as the general `prove_inv15`
theorem (see `z3_inv15_proof.py` docstring: "valid over the modeled
domain"): it verifies the gate's closed-form logic, **NOT** the LLM's
semantics, the JCR risk-model coefficients themselves, or whether dense
prefill actually improves judge consistency. The ledger guarantees the
*record* of decisions is tamper-evident; it does not vouch for the
correctness of the world outside the model.

**Hardware-validated (2026-05-26, MI300X / ROCm 7.2, torch 2.9.1+rocm6.3).**
Driven over the full 1,210-point input sweep (5 roles × 11 candidate
counts × 11 reuse rates × 2 layouts) with `APOHARA_FORGE_LEDGER=1`, the
production gate produced **1210/1210 INV-15-satisfying** certificates
(Z3 unsat); the hash chain verified (exit 0, 0.24 s) and a one-byte
tamper was caught (exit 2, `broken_at=719`). Within-model claim only
(the scope caveat above still holds). Evidence:
`scripts/mi300x_forge_ledger_proof.py` →
`logs_mi300x_p2/mi300x_p2_forge_ledger.json`.

**Status: 🟢 PRODUCTION** — certifier, ledger, recorder, CLI, and the
env-gated gate wiring all do what they claim. Covered by
`tests/test_inv15_certifier.py`, `tests/test_ledger.py`,
`tests/test_certified_recorder.py`, `tests/test_ledger_cli.py`,
`tests/test_gate_ledger_wiring.py`.

---

## 16. 🔴→🟢 LLMLingua-2 compressor never actually compressed (fixed 2026-05-26)

**The overclaim.** The README listed LLMLingua-2 as an implemented mechanism
("8× memory reduction") and the live demo implied real compression. **The code
never compressed anything.** `ContextCompressor` loaded the LLMLingua-2
token-classifier checkpoint but constructed `PromptCompressor(...)` **without
`use_llmlingua2=True`**, so it ran the LLMLingua-1 perplexity path (which
expects a causal LM) and raised `AttributeError: 'TokenClassifierOutput' has no
attribute past_key_values` on every `compress()`. Any path reaching compression
got the 503 passthrough.

**The fix.** `use_llmlingua2=True` + CPU-default device
(`CONTEXTFORGE_COMPRESSOR_DEVICE`; LLMLingua defaulted to CUDA and crashed on a
GPU-less coordinator host) + input chunking for the 512-token model limit. After
the fix: **2.23× on a probe and 44.4% prompt-token savings end-to-end on live
frontier-MoE inference (MI300X)**. Commits `476df4b`, `5d1e7d9`, `95e1756`.
- **Status: 🟢 PRODUCTION** — compression runs and is measured on real inference.

## 17. 🔴→🟢 README/paper honesty pass + repo cleanup (2026-05-26)

Triggered by the first real end-to-end coordinator test against live frontier
MoE on MI300X.
- **"79.85% live token savings"** was the **local synthetic demo** (263→53
  tokens, local tokenizer, no model loaded), shown as a headline/hardware
  metric. Relabeled as a demo upper-bound; the **real-model figure is ~44%**.
- **"235B fits single-card" / "model under test"** — FP8 (~221 GB) does **not**
  fit 192 GB; only **INT4** fits one card. The INV-15 gate results are
  model-independent (closed-form) and the codec results are synthetic-tensor
  measurements — neither needed a 235B end-to-end run.
- **Cross-agent KV-block sharing (ATOM plugin)** computes reuse decisions but
  does **not** physically share blocks in vLLM yet → the "68% VRAM" projection
  is unbuilt; marked 🔬 in-progress, no VRAM number quoted until measured.
- **Semantic dedup** falls back to pseudo-embeddings (`qwen3-embed` absent) → 🔬.
- **Codec 3.97× → 3.55×** synced in the README mechanism table.
- **Repo cleanup**: removed `hf_spaces/`, stale `papers/` v2 dup, `docs/legacy/`,
  untracked `CLAUDE.md`.
- **New honest evidence**: 3 frontier MoE serve single-card on MI300X;
  FORGE-LEDGER over real inference; NIAH 174K. Paper **v4.2**; companion systems
  paper planned for the MoE evidence.
- **Status: 🟢 RESOLVED** — README + paper v4.2 match runtime reality.

---

## 18. 🔴→🟢 ATOM `register()` pointed at a vLLM hook API that never existed (fixed 2026-05-28)

**The overclaim.** `apohara_context_forge/serving/atom_plugin.py` `register()`
did a late `from vllm.platforms import current_platform` and then probed
`getattr(current_platform, "register_pre_attention_hook"/"register_post_attention_hook", None)`
to "install" the ATOM pre/post attention hooks. **No vLLM platform has ever
exposed such an attention-hook registry** — the getattr always returned None,
so the branch was a permanent no-op dressed up as "kernel-level interception
until the API stabilises." The probe implied a runtime wiring path that does
not and never did exist.

**The fix (Fase 0).**
- `register()` now just constructs `vLLMAtomPlugin()`, calls
  `plugin.initialize(...)`, and returns it. The phantom getattr probe and the
  late `vllm.platforms` import are removed.
- `register()`'s docstring (and the module docstring) now state plainly:
  KV interception lives in the config-driven `--kv-transfer-config` path
  (LMCache), NOT in attention hooks — that platform API never existed in vLLM.
  The real cross-worker KV path is config-driven and documented in
  [`LMCACHE.md`](LMCACHE.md) (Fase 1+).
- `PreAttentionHook` / `PostAttentionHook` are **kept** (19 tests depend on
  them) but their docstrings now say they are unit-tested, importable
  utilities that are **NOT cabled to the vLLM runtime**.

**Verification:**
- `grep -rn "register_pre_attention_hook\|register_post_attention_hook" apohara_context_forge/`
  → **0 matches** (the phantom API is gone from `apohara_context_forge/`; the
  PyPI shim under `pypi/apohara-vllm-plugin/` was cleaned of its lingering
  attention-hook references in the same truth-up pass).
- `tests/test_atom_plugin.py` → **19 passed** (count unchanged; the
  `test_register_returns_initialised_plugin` docstring was re-aimed at the new
  honest reality — no assertions weakened).
- Full suite: **441 passed, 25 skipped** was the post-F0 baseline measured in
  isolation; after F1-F3 landed the total is **487 passed, 25 skipped** (no
  regressions).
- **Status: 🟢 RESOLVED** — `register()` no longer references a nonexistent
  vLLM API; the real KV-interception path is config-driven (Fase 1+).

## 19. 🟢 ATOM F1-F3 validated on hardware + the honest scope (full-attention) (2026-05-29)

**What we built and measured (F1-F3 — the real KV-sharing lever).** ATOM's
serving path — `PrefixSaltPlanner` → byte-identical prefix via
`PrefixNormalizer` → vLLM Automatic Prefix Caching, plus the config-driven
LMCache `--kv-transfer-config` for cross-worker — was validated end-to-end:

- **`cache_salt` drives KV-block sharing, measured on a real MI300X**
  (Qwen3-32B, dense full-attention, `rocm/vllm`): SHARED salt → **84.7 %** vLLM
  prefix-cache hit-rate vs ISOLATED salt → **0.0 %** (judges physically isolated
  via the block hash — INV-15 realised on the serving side). Shared-prefix
  **TTFT 0.058 s vs 0.135 s** distinct (−57 %). Model+KV footprint **175 GB / 192**,
  64 concurrent sustained. Raw: `logs/mi300x_squeeze/qwen3-32b_measure.json`.
- **Cross-worker KV reuse via LMCache+Redis** proven locally (RTX 2060, CUDA):
  worker-2 with an empty local cache pulled prefix KV from Redis that worker-1
  stored — vLLM `external_prefix_cache_hits` **0 → 240**,
  `prompt_tokens_by_source{external_kv_transfer}=240`. Raw:
  `logs/local_cross_worker_result.json`.
- Suite **487 passed, 25 skipped** (+46 over the F0 baseline).

**Honest non-results from the 2026-05-29 MI300X run (NOT reported as wins):**
- `qwen3-32b` token savings read **0 %** — the LLMLingua-2 compressor **did not
  run** in that VM (it failed to load; identical baseline==contextforge token
  counts confirm no compression happened). The **44.4 %** figure stands on its
  own from the 2026-05-26 `logs_moe_run/` run (compressor active). Not
  double-counted.
- `qwen3-32b` NIAH read **0/12** — a *script artifact*, not a recall failure:
  Qwen3 answers in `<think>` mode (the probe truncates before the code is
  emitted) and prompts > the configured `max_model_len` (16384) returned HTTP
  400. The real **NIAH 12/12 → 174K** stands from the 2026-05-26 run. We do not
  cite the 0/12.
- The three Gated-DeltaNet hybrids (Coder-Next, Qwen3.5-122B, Qwen3.6-35B)
  failed to start on the `rocm/vllm:latest` image: its **Transformers does not
  recognize the `qwen3_5_moe` architecture** (today's BLOCKER logs). The
  2026-05-26 evidence separately records Coder-Next serving cleanly on a 0.19.1
  image — so this is an image/environment miss on our side, not a model
  limitation. (We did not pin today's exact vLLM/Transformers version string.)

**The honest scope — why full-attention, and where it stops.** ContextForge has
two independent levers:
1. **Token compression (LLMLingua-2, ~44 %)** — *architecture-agnostic*; shrinks
   the prompt pre-serving and applies to full, sparse, linear and sliding-window
   models alike. The **durable** lever.
2. **KV-block sharing (the 84.7 % above)** — its win scales with KV-cache size,
   so it is **largest on full-attention**, which is the bulk of today's
   *installed* production fleet (Llama 3.x, Qwen2.5/3-dense, Mistral).

We measured the KV lever on full-attention **on purpose**. The honest limit,
stated plainly: the **2026 frontier is moving away from full attention** —
DeepSeek-V4 / GLM-5 (sparse DSA), Qwen3-Next/3.5/3.6 (linear-hybrid), Gemma 4 /
OLMo 3 / MiMo (sliding-window) — *precisely to shrink the KV-cache bottleneck the
sharing lever optimises*. On those architectures the KV win is smaller by
design. ContextForge's KV lever is for the large full-attention fleet that
exists now; its compression lever is for everything. We do **not** claim
KV-sharing relevance on sparse/linear frontier models.

- **Status: 🟢 VALIDATED + SCOPED** — both levers measured on real MI300X
  hardware (44 % tokens, 2026-05-26; 84.7 % KV-sharing, 2026-05-29), full-attention
  scope and frontier limit stated honestly.

## 20. 🟢 ATOM plugin renamed to ROMY (naming collision with AMD ROCm/ATOM) + invalid entry-point group fixed (2026-05-31)

**The naming collision.** We shipped the plugin under the name **ATOM**
(*Anchor-driven Tensor Orchestration for Multi-agent*). AMD's official ROCm
team ships an engine literally called **ATOM** (*AiTer Optimized Model*,
[ROCm/ATOM](https://github.com/ROCm/ATOM)) in **the same domain** — a vLLM
acceleration path for the MI300X. Two "ATOM" plugins for vLLM-on-MI300X is a
recipe for confusion and an implicit (false) association with AMD's project.
Honesty extends to naming: we do not squat a name an upstream vendor already
owns in our exact niche.

**The rename.** ATOM → **ROMY** (*Runtime for Orchestrated Matrix Yields*).
This is a pure identifier/prose rename — no behaviour changed:
- `apohara_context_forge/serving/atom_plugin.py` → `serving/romy_plugin.py`
  (and `tests/test_atom_plugin.py` → `tests/test_romy_plugin.py`).
- `ATOMConfig` → `ROMYConfig`, `vLLMAtomPlugin` → `vLLMRomyPlugin`; the PyPI
  shim re-exports, `__all__`, and docs updated to match.
- No backwards-compat aliases were kept: the `ATOM` name is retired entirely
  to avoid leaving the colliding identifier importable.

**The entry-point fix (real bug, same commit).** `apohara_context_forge/pyproject.toml`
declared the plugin under `[project.entry-points."vllm.plugin"]` — a group that
**does not exist in vLLM**. vLLM V1 discovers plugins through the
`vllm.general_plugins` group (verified against docs.vllm.ai); the PyPI shim
already used the correct group, but the in-tree `contextforge` package would
have registered an entry point vLLM never walks. Fixed:
`vllm.plugin` → `vllm.general_plugins`, and
`contextforge_atom = "...atom_plugin:vLLMAtomPlugin"` →
`contextforge_romy = "contextforge.serving.romy_plugin:vLLMRomyPlugin"`.

**Verification:**
- `rg -i "\batom\b|atom_plugin|atomconfig|vLLMAtomPlugin"` over
  `apohara_context_forge/ tests/ pypi/ deploy/ README.md LMCACHE.md` →
  **0 matches**. The historical entries above (#18, #19) intentionally keep the
  `ATOM` name as it was at the time.
- Full suite: **487 passed, 25 skipped, 0 failed** (unchanged; the renamed
  `tests/test_romy_plugin.py` keeps its 19 tests, no assertions weakened).
- **Pending:** `paper/inv15_paper.tex` + `references.bib` still say "ATOM"; the
  academic artifact (DOI-bearing) is left untouched here and gets a separate
  editorial pass so the rename lands cleanly in the next paper revision.
- **Status: 🟢 RESOLVED** — name no longer collides with AMD's ATOM engine; the
  in-tree entry point now targets a real vLLM plugin group.

**Follow-up (2026-06-02, PyPI prep).** `apohara_context_forge/pyproject.toml`
was **removed entirely**, so the "entry-point fix" above is now moot. On a
closer look that fix was cosmetic: the inner manifest was an orphan. Its
distribution name `contextforge` is already taken on PyPI by an unrelated
project; its declared target `contextforge.serving.romy_plugin` does **not**
resolve (the in-tree package is `apohara_context_forge` — there is no
top-level `contextforge` module), so the entry point would have failed to
load even with the correct group; and its MIT license contradicted the
repo's Apache-2.0. The package was never pip-installed (tests run via
`PYTHONPATH=.`), so the broken entry point was never actually walked. The
real, working vLLM entry point lives in the `pypi/apohara-vllm-plugin` shim
(`apohara_contextforge = "apohara_vllm_plugin:register"`), which is now the
single source of truth. Net: the in-tree entry point is **gone, not fixed**.

---

*Last updated: 2026-06-02 · maintained by the same person who wrote the lies.*
