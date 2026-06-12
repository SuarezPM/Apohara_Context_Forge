# ROMY Safety O(1) — Threat Model

> **Status:** Draft for upstream PR to `vllm-project/vllm`.
> **Cite:** Companion to paper v5.0 (Zenodo, in deposit prep —
> see `paper/v5.0/zenodo-v5-metadata.json`).
> **Author:** Pablo M. Suarez <suarezpm@csnat.unt.edu.ar>
> **Date:** 2026-06-12

## 1. Problem statement

Multi-agent LLM pipelines share a long system prompt and tool
spec across agents (retriever, reranker, summarizer, critic,
responder). vLLM's Automatic Prefix Caching (APC) reuses the
KV-cache for the shared prefix, which is the right default.

The subtle failure mode is **judge contamination**: a critic
agent reads the *candidate* generation as part of its context.
If the critic's KV-cache is reused from a *prior* ranking,
the prior ordering is silently baked into the critic's
attention pattern, and the verdict is biased toward the
old ranking. This is the failure mode the ROMY safety
layer exists to prevent.

## 2. The ROMY contract

**ROMY** is the **isolation contract** that backs INV-15 of
the Apohara / ContextForge safety kernel. The contract has
two parts:

1. **Stable isolation salt per judge.** Every judge pipeline
   request carries a `cache_salt` UUID in its prompt-metadata.
   The salt is mixed into the prefix-cache key derivation, so
   the same logical prefix (system prompt + tool spec) maps
   to *different* KV-cache blocks when the salt differs. Two
   judge requests with different salts therefore land in
   different KV-cache blocks even if their text prefixes
   are byte-identical.

2. **Zero hit rate between judges.** The expected outcome
   is `0.0%` prefix-cache hit rate between judge requests
   (a measured invariant on the MI300X AUDIT #19 run;
   `reports/wow8gb_2026_06_12.md` style benchmark). The
   salt is the only mechanism; the judge itself never
   reaches into the KV-cache to "look up" prior verdicts.

The contract is **O(1) per request** (one extra hash mix in
the cache-key derivation; no scan over the KV-cache, no
"what was the prior ranking" lookup).

## 3. Threat model

### 3.1 Adversary capabilities

The threat model assumes an adversary who can:

- **Inject content** into the shared system prompt or tool
  spec (direct prompt injection).
- **Inject content** into the documents the retriever
  returns (indirect prompt injection).
- **Observe** the public model API and timing.
- **Not** observe the ROMY salt (the salt is a server-side
  per-judge UUID; the adversary does not have access to
  the judge host's `cache_salt` generator).

### 3.2 Threats ROMY addresses

| Threat                                              | Mitigation |
|-----------------------------------------------------|------------|
| **KV-cache contamination of judge from prior ranking** | Each judge gets a unique salt → APC reuses 0 blocks between judges → prior ranking is invisible to the new judge. |
| **Cross-judge information leak via shared prefix**    | Salt mixes into the cache key; byte-identical prefixes land in different blocks. |
| **Deterministic leakage of verdict-order to the LM**   | The salt is opaque to the LM; the LM never sees the cache_key directly. |

### 3.3 Threats ROMY does NOT address

- **Prompt injection in the system prompt or retrieved
  documents**: the salt mixes into the cache *key*; the
  *value* (the cached KV blocks) is still the KV of the
  injected prefix. ROMY is a *cache-isolation* layer, not
  a content-filter. The agent harness must still defend
  against injection (the 2026-06 SoK on judge security
  documents the attack surface; see
  `docs/research/reconcile/romy-2026-06-11.md` for the
  related work and the limit of ROMY).
- **Adversary who can read the salt** (e.g. a co-tenant
  on the same vLLM instance with privilege to read the
  metadata table): the salt is not a secret; it is an
  isolation key. An adversary who can observe the salt
  can still not cause cross-judge contamination because
  the salt is per-judge-pipeline and rotated for every
  new judge request.
- **KV-cache side channels (timing, memory, power)**:
  out of scope; the threat model is logical.

## 4. Formal property (informal Z3 sketch)

The Z3 model lives in
`apohara_context_forge/safety/z3_inv15_proof.py` and is
proved in 10.08 ms (the existing v6.0 audit entry). The
property is: for any two judge requests `J1, J2` with
different `cache_salt` values, the KV-cache blocks
allocated to `J1` and `J2` are disjoint. The model abstracts
the cache as a `(prefix_hash, salt) -> block_id` function
and asserts the bijection under fixed `prefix_hash`. The
Z3 proof was independently validated against the
production JCRSafetyGate (see AUDIT #22 in the v6.0
paper).

## 5. Operational guarantees (measured)

The Sprint 1 / Track A1 ralph run produced these measurements
on the local RTX 2060 SUPER 8 GB with the vLLM 0.20.x
plugin entry-point (commit `dc8add5`):

| Metric                                           | Value     | Source |
|--------------------------------------------------|-----------|--------|
| `cache_salt` wire overhead (HTTP body, no server) | -1.99 µs p50 | AUDIT #19 |
| Judge-vs-judge prefix-cache hit rate             | 0.0%      | AUDIT #19 |
| Z3 INV-15 proof time                             | 10.08 ms  | v6.0 paper |
| JCR Safety Gate p50 latency                      | 146 µs    | v6.0 paper |
| Throughput regression vs native APC              | -21.8%    | GATE #0 (ABANDON) |

The negative throughput number is the GATE #0 ABANDON
result. ROMY is the **isolation contract** that survived the
ABANDON — judges get the safety, the rest of the pipeline
gets native vLLM APC at full speed.

## 6. Open issues for the upstream PR

The PR to `vllm-project/vllm` will need:

1. **A `cache_salt` API surface** in the OpenAI-compatible
   schema (the `metadata` field already exists; the
   convention is `metadata["cache_salt"]: "<uuid>"`).
2. **A cache-key derivation that mixes the salt** into the
   hash. The current vLLM `vllm/v1/core/kv_cache_manager.py`
   hashes `block_hash = hash(prefix_tokens)`. The PR
   changes this to `block_hash = hash((prefix_tokens, salt))`
   when the salt is non-null; the salt-mixed hash is what
   ROMY uses for the KV-cache key.
3. **A test that pins the 0.0% hit rate** between two
   judge requests with different salts and identical
   text prefixes. The test is mechanical: send two
   `/v1/chat/completions` requests with identical `messages`
   but different `metadata.cache_salt`, then assert that
   the `usage.prompt_tokens_details.cached_tokens` is 0
   for both.
4. **A documentation update** in
   `docs/usage/salt.md` (or equivalent) that explains
   the contract and the measured overhead.

The PR is scoped tight: ~200 lines of code change, one
test, one doc update. The actual performance cost is
**O(1) per request** (one extra hash mix), measured at
**-1.99 µs p50 wire overhead** on the bench.

## 7. Companion systems paper reference

This document is the threat model that backs the
**ROMY isolation contract** section of the Apohara 2.0
companion paper v5.0 (`paper/v5.0/paper.md`). The paper
references this file as the formal-model appendix;
the paper itself is the human-readable narrative.

## 8. Honest gap (filed here, not papered over)

The upstream PR to `vllm-project/vllm` is **not yet
opened**. This document is the draft for the PR
description and the threat model. The actual PR
submission is a manual one-shot for Pablo: the agent
cannot open a PR on a foreign repo without his
credentials, and the PR text needs a human pass
before submission. The 2026-06-12 commit landed this
document + the linked AUDIT entry as the durable
artifact; the PR submission is the next step, gated
on Pablo's review of this draft.
