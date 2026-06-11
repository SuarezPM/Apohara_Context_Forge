# Apohara 2.0 — Statistical Pre-Registration

> **Source of truth:** This file pre-registers the statistical contract for
> the Apohara 2.0 bank test (US-008 / Phase 6) **before** Phase 6 begins.
> Per `.omc/plans/apohara-2-0.md` Section 5 ("Statistical Pre-Registration"),
> no post-hoc changes to alpha, correction, or MDE.

## M3 LLM-as-judge version pin

- **Model:** `MiniMax-M3`
- **Version pin:** `MiniMax-M3-2026-05-XX`
- **TODO:** pin exact version when bench runs (US-005 will replace this
  placeholder with the actual build hash; the pre-registration is
  committed **now** with the placeholder so the contract is locked
  before any bench output exists).

The M3 judge is pinned to **greedy decoding** for determinism:

- `temperature = 0`
- `top_p = 1.0`
- `top_k = 1`

This is asserted in `tests/test_m3_judge.py` (US-005, Phase 3) via a
sampling-params round-trip test.

## Holm-Bonferroni step-down statement

For the 5-task p-value family (paired t-test of per-task EM / PPL delta
vs. uncompressed baseline, 5 seeds 0..4), we apply the **Holm-Bonferroni**
step-down procedure to control the family-wise error rate (FWER) at
alpha = 0.05.

### Procedure (formal)

1. Order the 5 per-task p-values ascending: `p_(1) <= p_(2) <= ... <= p_(5)`.
2. For k = 1..5, reject H_k (task k shows a non-zero effect) at step k iff
   `p_(k) <= alpha / (5 - k + 1)`.
3. Stop rejecting at the first non-rejection; all subsequent H_j are
   retained.

This is strictly more powerful than Bonferroni alone while still
controlling FWER at alpha = 0.05 (Holm 1979). The correction is
applied to the **final** 5×5 p-value set, not to the rolling smokes
(those are 1-seed and need no correction).

### Alpha and MDE

- **alpha = 0.05** (two-sided)
- **MDE (minimum detectable effect):** documented per task in the bench
  output CSV metadata. If a task flips pass/fail between runs, the runner
  escalates to `--seeds 0..9` and reports the per-task effect size + 95% CI
  (US-008 will replace this paragraph with the actual MDE table once the
  fixture scale is fixed).

### Non-normality fallback

If Shapiro-Wilk on the 5 paired deltas returns p<0.05 for a task
(non-normal residuals), that task alone falls back to **Wilcoxon
signed-rank**. The CSV metadata flags which test was used per task so
the pre-registration is auditable after the run.

## The 5 pinned tasks

| # | Task             | Metric       | Type           | Hardware (default) |
|---|------------------|--------------|----------------|--------------------|
| 1 | HotpotQA         | EM (Exact Match) | multi-hop QA   | RTX 2060S          |
| 2 | NaturalQuestions | EM           | open-domain QA | RTX 2060S          |
| 3 | GSM8K            | accuracy     | math reasoning | RTX 2060S          |
| 4 | BBH              | accuracy     | BIG-Bench Hard | RTX 2060S          |
| 5 | summarization    | ROUGE-L      | long-context   | RTX 2060S          |

The 5 tasks are the same list in `bench_e2e.py:PINNED_TASKS` and the
spec's Component D topology. Task length is capped at 8K tokens for
the local RTX 2060S 8GB run; long-context (32K) verification pivots
to H100 / MI300X with the explicit banner from
`benchmarks/apohara2/bench_kv.py:PIVOT_BANNER`.

**Gate:** each of the 5 tasks passes at p<0.05 (Holm-corrected) vs. the
uncompressed baseline.

## CITABLE vs A VERIFICAR

This ledger is the honesty contract for the bank test. The README, paper,
and release notes MUST honor these labels.

### CITABLE (upstream published; we cite, not re-derive from scratch)

- TurboQuant (paper 2504.19874, ICLR 2026) — Lloyd-Max + 1-bit QJL codec.
- FWHT pre-RoPE (paper IJCAI 2025; our INV-10 wrapper at
  `apohara_context_forge/quantization/rotate_kv.py:34`).
- LLMLingua-2 base + LongLLMLingua (Microsoft, upstreams).
- granite-embedding-311m-multilingual-r2 (MTEB-Multilingual 65.2,
  Apache 2.0, drop-in sentence-transformers).
- LMCache vLLM V1 connector (the real path that ROMY's salt planner
  rides on).
- MiniMax M3 (the LLM-as-judge; per global CLAUDE.md, M3 not GPT-4).
- Holm-Bonferroni step-down (Holm 1979, Scand. J. Stat.).

### A VERIFICAR (local measurement pending; placeholder until US-004/5/6/7/8)

- Turbovec recall parity vs. FAISS-IVF on HotpotQA-200 — measured in
  US-004 / Phase 2. **Threshold:** >= parity.
- Turbovec RAM at 10M docs, 4-bit, 768-d — measured in US-004 / Phase 2.
  **Threshold:** <= 4 GB.
- LLMLingua-2 PPL delta per variant on LongBench subset — measured in
  US-005 / Phase 3. **Threshold:** <= 5%.
- TurboQuant-KV VRAM reduction vs. FP16 — measured in US-006 / Phase 4
  on H100/MI300X (pivot banner). **Threshold:** >= 2.5x.
- TurboQuant-KV EM degradation on HotpotQA-200 — measured in US-006 /
  Phase 4. **Threshold:** <= 1%.
- ROMY 0% hit rate between judges under `cache_salt="isolated"` —
  measured in US-007 / Phase 5. **Threshold:** 0%.
- Bank test 5×5 p-values vs. baseline (Holm-corrected, p<0.05) —
  measured in US-008 / Phase 6. **Threshold:** p<0.05 per task.
- Codec MSE parity (FWHT + codec_v8) vs. unrotated baseline — measured
  in US-003 / Phase 1. **Threshold:** <= 1.1x.

## Change log

- 2026-06-11 — US-002 (Phase 0): pre-registration filed with M3
  placeholder version. Real pin lands in US-005 when M3 build hash is
  committed; the alpha, correction, and MDE contract is locked here.
