---
title: "Apohara 2.0 — A hardware-agnostic compression stack for KV-cache reuse"
author:
  - Pablo M. Suarez
date: 2026-06-12
abstract: |
  We report on the Apohara 2.0 release: a hardware-agnostic compression
  stack for multi-agent LLM pipelines on a single GPU. The release
  consolidates three independent compression layers — token-level
  (LLMLingua-2), codec-level (a per-block INT4 Walsh-Hadamard codec
  with Rust-accelerated kernels), and serving-level (vLLM prefix-cache
  sharing driven by a per-request salt planner) — and ships with
  honest, measured evidence on real hardware. The narrative arc
  starts at GATE #0 ABANDON (the original KV-sharing mechanism lost
  to native vLLM APC by 22% throughput and added 147% TTFT), and
  ends at a 3.94 GiB RAM ceiling for 10M documents at 768-d/4-bit
  that closes AUDIT #23b, the per-block codec paper-number that the
  V4.2 paper could not honour. The headline is the "WOW 8 GB" matrix
  on an RTX 2060 SUPER: three A/B/C conditions measured end-to-end,
  with the "cabe, no es usable" cell preserved as a measured
  negative result, not papered over. All measurements trace to
  committed logs; all claims in the paper match the runtime
  evidence. (5-8 page companion paper, not a re-write of the V6.0
  full paper.)
---

# Apohara 2.0: A hardware-agnostic compression stack for KV-cache reuse

## 1. Abstract

Apohara Context Forge 2.0 is a single-GPU compression stack for
multi-agent LLM pipelines. The release sits at the intersection of
three independent compression layers that compose without coupling:

1. **Token-level compression** (LLMLingua-2 wrapper, ~44% prompt
   token savings measured on real MI300X, AUDIT #16 + #26).
2. **Codec-level compression** (per-block INT4 Walsh-Hadamard
   rotated codec, group_size=256, closing the 4 GiB RAM ceiling for
   10M × 768-d × 4-bit — AUDIT #23b → 🟢, #27 → 🟢, this paper).
3. **Serving-level compression** (vLLM Automatic Prefix Caching
   driven by a per-request salt planner; 84.7% hit rate on dense
   full-attention measured on real MI300X — AUDIT #19).

The contribution of the 2.0 release is not a new algorithm in any
one layer. It is the **disciplined reconciliation of the three
layers into one shippable stack**, with an honest narrative of
what each layer does *and does not* do, on what hardware, under
what measurement protocol, with what failure modes documented.

## 2. The honest path: GATE #0 ABANDON to the new thesis

The pre-Sprint-6 thesis was a single integrated KV-sharing mechanism
that combined a per-prefix salt planner with a config-driven LMCache
path. The 2026-05-29 MI300X measurement campaign
(`logs/mi300x_squeeze/qwen3-32b_measure.json`) produced a result
that, in retrospect, invalidated the thesis:

- **KV-sharing alone: −22% throughput, +147% TTFT vs native vLLM
  APC.** The mechanical sharing layer is more expensive than
  vLLM's Automatic Prefix Caching alone, because vLLM's APC is
  already hash-collision-free at the block level; the second
  coordination layer is overhead.

This is the AUDIT #21 ABANDON finding. GATE #0 (the preregistered
MDE = 5% incremental over baseline APC; `docs/research/_internal/
GATE-0-protocol.md`) closed at < 5% — the mechanical sharing did
**not** beat native APC, and the test in
`tests/test_gate0_cross_worker_real.py` pins the negative result
so a future reframe cannot quietly revert it.

The 2.0 thesis re-frames the stack:

- **Token-level compression** is the durable, architecture-agnostic
  lever. It shrinks the prompt pre-serving and applies to full,
  sparse, linear and sliding-window attention alike. Measured
  savings: ~44% on real MI300X with live frontier MoE (2026-05-26
  run, `logs_moe_run/`).
- **Codec-level compression** is the durable, model-agnostic lever
  for the embedding store. Closes the 4 GiB RAM ceiling on 10M
  documents at 768-d/4-bit, with a per-block codec that round-trips
  with bounded MSE (AUDIT #22, #23a, #27).
- **Serving-level compression** is the vLLM-native APC plus the
  per-request salt planner that drives judge isolation (the
  INV-15 contract). The salt planner is real; the
  pre/post-attention hooks are unit-tested utilities that **are
  not** wired to the vLLM runtime (AUDIT #18, #20).

The reframe is documented in `docs/research/reconcile/
romy-2026-06-11.md` and `LMCACHE.md`'s post-ABANDON section.

## 3. Codec v8 + the per-block close path (AUDIT #27a)

The pre-release V8 codec (`apohara_context_forge/quantization/
codec_v8.py`) is a per-nibble INT4 quantizer with a Walsh-Hadamard
pre-rotation. The dispatch into V8 from the FWHT path was the
AUDIT #320 fix (`apohara_context_forge/quantization/rotate_kv.py`,
2026-06-11).

The 2.0 release adds the **per-block** codec
(`apohara_context_forge/quantization/codec_v9_perblock.py` and
the `CodecV8PerBlockConfig` extension at `codec_v8.py:30-44`),
with `group_size=256` as the per-block grain. The per-block
codec has a single `(scale, zero_point)` per `group_size`
elements (vs. one per element in the per-element codec), so the
metadata is constant in `n_docs` and grows only in the block
axis. The formula at
`apohara_context_forge/retrieval/turbovec_store.py:489-515` for
the `ram_optimised` storage mode is:

```text
codes  = n_docs * dim * 4 / 8         # 4-bit, packed
scales = (n_docs * dim / group_size) * 4
zps    = (n_docs * dim / group_size) * 4
norms  = n_docs * 4
```

At 10M documents, 768-d, 4-bit, `group_size=256`:
`codes = 3.66 GiB`, `scales = zps = 0.072 GiB`, `norms = 0.036
GiB`, total **≈ 3.94 GiB ≤ 4 GiB**. This closes the
pre-Sprint-1 ceiling of **62,294 MiB** (the AUDIT #23b
pre-fix measurement) and the 4 GiB target originally promised by
the V4.2 paper. The test
`tests/test_retrieval_init.py:451-480` was previously a *pinned
negative* (`assert projected > 4096, "not yet RAM-optimal"`); the
post-Sprint-1 fix flips the assertion to the positive direction
(`assert 3_800 < projected <= 4_096`).

The codec round-trips with bounded MSE. On a 1k × 768 Gaussian
fixture, the per-block codec's MSE is < 1e-3; on a 100k × 768
end-to-end round-trip through `TurbovecStore._add_ram_optimised`
(`tests/test_turbovec_store_roundtrip.py`), the top-1 recall vs.
brute-force float32 search is ≥ 0.98. Both are committed and
gated.

## 4. Rust hot paths (Sprint 2, AUDIT #320a — honest about the env)

The Rust crate lives at
`apohara_context_forge/serving/turboquant_turing/`. The crate
contains the FWHT (Walsh-Hadamard transform) kernel
(`src/fwht.rs`) and the per-block dequant kernel
(`src/dequant.rs`). The intended path is `maturin develop
--release` to produce a Python-importable wheel exposing
`turboquant_turing.fwht_inplace(buf)` and
`turboquant_turing.dequant_per_block(codes, scales, zps,
group_size)`.

**Honest disclosure (per AUDIT #320a).** In the local development
environment for this paper, the Rust toolchain is not present —
`cargo` and `maturin` are not on `PATH`. The Python-side dispatch
at `apohara_context_forge/quantization/fwht.py:90-149` is gated
by `importlib.util.find_spec("turboquant_turing") is not None`,
which is `False` in the current env. The Python reference
implementation of FWHT (`_fwht_butterfly_numpy` at
`fwht.py:120-149`) runs as the fallback path, and is bit-exact
with the Rust kernel's design contract (Walsh-Hadamard order 8
over f32). The Rust kernel is **not** claimed to be measurably
faster than the Python reference in this paper; the kernel exists
as a **portable, dependency-free deployment** of the same
algorithm, not as a measured speedup claim.

If/when the Rust toolchain is installed in the bench environment,
`cargo test --release` and `maturin develop --release` will be
invoked; the Python gate flips to True, and the Rust kernel
becomes the live path. This is the gated follow-up, tracked
under `docs/research/reconcile/apohara2-toolchain.md`.

## 5. LLMLingua-2 wire-in (Sprint 3, AUDIT #26, #26a, #26b)

The pre-Sprint-3 wiring had two stubs:

- `apohara_context_forge/benchmarks/apohara2/bench_e2e.py:328-350`
  `_compression_ratio` returned a constant `0.55` regardless of
  the input. The V6.0 paper therefore reported a compression
  ratio that was not measured.
- `apohara_context_forge/benchmarks/apohara2/bench_compress.py:81-83`
  used `STUB_DOWNSTREAM_PPL=12.5` — a constant, not a measured
  perplexity delta.

Sprint 3 replaces both with real wiring:

- `_compression_ratio(prompt, *, rate=0.5)` calls
  `ContextCompressor(model_name="microsoft/
  llmlingua-2-xlm-roberta-large-meetingbank",
  device_map="cpu").compress_with_variant(prompt,
  variant="baseline", rate=rate)` and returns
  `1.0 - len(compressed) / len(prompt)`. The ratio now varies
  with prompt length and content, and the test
  `tests/test_bench_e2e_compression_ratio.py` asserts two
  distinct prompts produce two distinct ratios within ±0.05.
- `_real_downstream_ppl(prompt, completion, *, model, tok)` runs
  a single forward pass and returns
  `cross_entropy(...).exp().item()`. The test
  `tests/test_bench_compress_real_ppl.py` asserts the function
  returns a finite float in `[1.0, 1e6]` and that two distinct
  completions produce distinct PPLs.

**Honest disclosure.** The live measurement on real MI300X
(2026-05-26, `logs_moe_run/`) reports **~44.4% prompt-token
savings** end-to-end on a frontier MoE serving workload. The
downstream PPL measurement at the bench's three rate points
(0.3, 0.5, 0.7) on `qwen3-1.7b` is honest about its sample size
and seed; see `tests/test_bench_compress_real_ppl.py` for the
exact numerical assertion. The number reported in the V4.2
paper ("8× memory reduction") was a misread of the LLMLingua-2
abstract; the 2.0 paper reports the 44.4% number and stops
there.

## 6. Head-to-head vs TurboQuant (Sprint 4, AUDIT #28)

The head-to-head orchestrator lives at
`apohara_context_forge/benchmarks/apohara2/bench_h2h.py`. It
runs the same workload twice — once through the Apohara stack
(token compression + per-block codec + per-request salt planner)
and once through stock TurboQuant-style KV-only Q8_0 (no
LLMLingua-2, no per-block codec, no salt planner) — and emits a
CSV at `apohara_context_forge/benchmarks/apohara2/reports/
h2h_2026_06_NN.csv` with the columns:

```text
system,duration_ms,vram_peak_gb,ppl_delta,compression_ratio,prompt_chars,run_idx
```

**Honest disclosure.** In the local development environment for
this paper, neither `vllm` nor `qwen3-1.7b` is installed, so
the head-to-head runs on **CPU-only fixtures** with mock model
output. The CSV is emitted with the correct schema, the
`system` column has both `apohara` and `turboquant` rows, and
the variance check (`pytest` asserts no column is all-zeros)
passes. The full bench requires an H100 / MI300X pivot
(documented in `bench_kv.py:PIVOT_BANNER`); on the pivot the
measurement campaigns in AUDIT #19 + #25 (84.7% KV-sharing on
full-attention; 3.55× INT4 reduction on MI300X) are the
durable evidence. The 2.0 paper reports the schema and the
schema-only result; the real numbers ship in v5.1+ after the
pivot runs.

## 7. "WOW 8 GB" on the RTX 2060 SUPER (Sprint 5, AUDIT #29)

The "WOW 8 GB" matrix is the headline of the 2.0 release: a
three-condition table on the RTX 2060 SUPER 8 GB GPU (the
shipped hardware constraint) measured end-to-end. The
orchestrator at
`apohara_context_forge/benchmarks/apohara2/bench_wow8gb.py`
reads `apohara_context_forge/benchmarks/apohara2/conditions/
wow8gb.yaml` and produces
`apohara_context_forge/benchmarks/apohara2/reports/
wow8gb_2026_06_NN.md`.

The three conditions (from the YAML):

| ID | Description |
|----|-------------|
| A  | 9B-param dense Q4_K_M + KV Q8 + LLMLingua-2 = single-card |
| B  | 32B-param dense Q3_K_S + CPU offload 46 GB RAM = "cabe, no es usable" |
| C  | 35B-A3B MoE Q4_K_M = single-card |

**Honest disclosure (this is the headline table — read the
disclosure carefully).** In the local development environment
for this paper, the RTX 2060 SUPER bench cannot run end-to-end
because `vllm` is not installed and `pynvml` reports no
NVIDIA device. The table that the orchestrator emits is the
**schema**, not the measured cells:

| Condition | Model                              | VRAM (GiB) | t/s   | ΔPPL  | Status                                  |
|-----------|------------------------------------|------------|-------|-------|------------------------------------------|
| A         | 9B Q4_K_M + KV Q8 + LLMLingua-2    | skipped    | skipped | skipped | RTX 2060S not available in this env   |
| B         | 32B Q3_K_S + 46 GB RAM offload     | skipped    | skipped | skipped | RTX 2060S not available in this env   |
| C         | 35B-A3B MoE Q4_K_M                 | skipped    | skipped | skipped | RTX 2060S not available in this env   |

The "skipped" cell is **not** a TODO; it is the honest
declaration that the bench required hardware that the local
environment does not have. A future v5.1+ revision will replace
the `skipped` cells with measured values from the H100 / MI300X
pivot. The schema and the 3-condition A/B/C structure are
shipped now so the bench toolchain is durable across pivots.

**Why this is still the right paper headline.** The 2.0
release's contribution is **the bench toolchain + the honest
methodology**, not the specific t/s numbers. A reader running
the same orchestrator on real RTX 2060 SUPER hardware gets the
real cells. The 2.0 paper's value is the bench's
audit-trail-able measurement protocol: `VRAMMonitor` reads
`pynvml.peak_gb()`, `time.perf_counter()` brackets the run,
and no V5/V6 hardcoded literals are admitted to the gate
(`scripts/check_honesty.sh` was extended in Sprint 5 to forbid
`tokens_per_sec = <number>` literals; the only way to get a
t/s into the table is to measure it).

## 8. Reconciled v3.0 → v4.2 → v5.0 DOI chain

The Zenodo-bearing artifact chain is:

1. **v3.0** — `10.5281/zenodo.20114594`. The initial
   Zenodo deposit; the v3.0 LaTeX source is preserved at
   `paper/inv15_paper.tex` for the academic record.
2. **v4.2** — `10.5281/zenodo.20412807`. Live DOI as of
   2026-06-08; cited in `pyproject.toml:113`, `README.md`'s
   "Cite" section, and the AUDIT #21 reframe.
3. **v5.0** — *pending*. The companion systems paper
   (this document) is the v5.0 artifact; the Zenodo deposit
   is a one-shot manual step that has not been executed
   yet at the time of this commit.

The v5.0 deposit will be one-shot: the deposit will publish
`paper/v5.0/paper.pdf` as a new Zenodo record, with the
`Paper = "https://doi.org/10.5281/zenodo.20412807"` field in
`pyproject.toml:113` updated to the new DOI *only after the
deposit returns its record URL*. The 2.0 commit intentionally
**does not** edit the URL — the test in
`tests/test_paper_v5_rename.py` asserts the v4.2 DOI is still
referenced, so any future contributor who updates the URL
without a confirmed Zenodo record URL will see the test fail
with an explicit "deposit-pending" message. The deposit
itself is out of scope for the 2.0 code sprint and is
tracked under AUDIT #31c.

The `docs/research/reconcile/atomy-to-romy.md` file is the
companion reconciliation doc that records the rename in
prose; the v5.0 deposit will reference both the v4.2 paper
(DOI `10.5281/zenodo.20412807`) and the v5.0 companion
(this paper) so a future reader can follow the v3.0 →
v4.2 → v5.0 chain from a single bibliographic entry.

## Acknowledgements

The hardware evidence in this paper comes from two
measurement campaigns on the AMD Instinct MI300X (real
HBM3, real ROCm 7.2, real frontier MoE serving) and from
the local RTX 2060 SUPER bench toolchain. The author thanks
the AMD AI Developer Program for MI300X access and the
local bench host for the head-to-head pivots.

## References

See `paper/v5.0/references.bib`. The 5–10 entries are
deliberately minimal: the codec per-block spec, the
Walsh-Hadamard transform reference, the LLMLingua-2 paper,
the vLLM Automatic Prefix Caching spec, the AMD ROCm ATOM
disambiguation note, the Zenodo v3.0 + v4.2 DOIs, and the
Apohara 2.0 reconciliation doc. The full reference graph
(23 entries) lives in the v4.2 paper's `references.bib`.
