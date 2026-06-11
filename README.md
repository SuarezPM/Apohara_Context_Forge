<p align="center">
  <img src="assets/banner.svg" alt="APOHARA · ContextForge — the first formally-verified safety layer for multi-agent LLM pipelines" width="100%">
</p>

<h1 align="center">APOHARA&nbsp;·&nbsp;ContextForge</h1>

<p align="center">
  <strong>The first formally-verified safety layer for multi-agent LLM pipelines.</strong><br>
  Share the KV-cache to make agents affordable — <em>without</em> silently corrupting your judges.<br>
  <sub>AMD Instinct MI300X-native&nbsp;·&nbsp;machine-checked by Z3&nbsp;·&nbsp;honest by construction.</sub>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.20277875"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20277875-22D3EE?style=for-the-badge&logo=doi&logoColor=0D1117&labelColor=0D1117" alt="DOI"></a>
  <a href="#-proof-not-promises"><img src="https://img.shields.io/badge/AMD%20MI300X-hardware%20validated-ED1C24?style=for-the-badge&logo=amd&logoColor=white&labelColor=0D1117" alt="Hardware-validated on MI300X"></a>
  <a href="paper/inv15_paper.pdf"><img src="https://img.shields.io/badge/INV--15-Z3%20machine--checked-C724B1?style=for-the-badge&logoColor=white&labelColor=0D1117" alt="Z3 proof"></a>
</p>
<p align="center">
  <a href="https://pypi.org/project/apohara-context-forge/"><img src="https://img.shields.io/pypi/v/apohara-context-forge?style=flat-square&logo=pypi&logoColor=white&label=PyPI&labelColor=0D1117&color=39D353" alt="PyPI version"></a>
  <a href="#-verification"><img src="https://img.shields.io/badge/tests-496%20passed-39D353?style=flat-square&labelColor=0D1117" alt="496 tests"></a>
  <a href="AUDIT.md"><img src="https://img.shields.io/badge/we%20publish%20our%20own-audit-FF8A00?style=flat-square&labelColor=0D1117" alt="Public audit"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-39D353?style=flat-square&labelColor=0D1117" alt="License Apache 2.0"></a>
</p>

<p align="center">
  <a href="#-the-judge-is-the-agent-cache-sharing-breaks">Problem</a>&nbsp;·&nbsp;
  <a href="#-proof-not-promises"><b>Proof</b></a>&nbsp;·&nbsp;
  <a href="#-architecture">Architecture</a>&nbsp;·&nbsp;
  <a href="#-quick-start">Quick start</a>&nbsp;·&nbsp;
  <a href="#-who-needs-this">Who needs it</a>&nbsp;·&nbsp;
  <a href="#-honest-by-construction"><b>Honesty</b></a>&nbsp;·&nbsp;
  <a href="#-cite">Cite</a>
</p>

---

## 🎯 The judge is the agent cache-sharing breaks

Serious AI in 2026 is built from **multi-agent pipelines** — retriever → reranker → summarizer → **critic** → responder. Every agent re-reads the same long context, so the obvious way to make them affordable is to **share the KV-cache** across agents.

**That quietly breaks the one agent you trust most — the judge.** When a Critic compares candidates, reused attention from a prior ranking encodes the *old* ordering and biases the new verdict. Accuracy on everything else still looks fine, so the corruption is **invisible** ([Liang et al., 2026](https://arxiv.org/abs/2601.08343)). Teams are left with two bad options: burn GPU re-computing everything, or ship judges that silently lie.

**ContextForge is the layer that proves when reuse is safe** — and serves frontier models a single AMD MI300X can hold but an 80 GB card cannot.

---

## 💡 What you get

| | |
|---|---|
| 🛡️ **A guarantee, not a heuristic** | `INV-15`: judge-class agents fall back to dense prefill when KV-reuse risk crosses threshold — **machine-checked by a Z3 SMT proof**, **zero violations across all 1,210 input points**, backed by a **tamper-evident certified ledger** of every decision. The first formal safety contract for cross-agent KV reuse. |
| ✂️ **Efficiency measured on silicon** | **44 % fewer prompt tokens** on live frontier-MoE inference (LLMLingua-2) · **84.7 % cross-agent prefix-KV reuse** (`cache_salt`, full-attention, MI300X) · **3.55× KV-cache VRAM reduction** at INT4 (RotateKV). |
| 🚀 **The 192 GB memory moat** | Three frontier MoE models served on **one MI300X** — an 80B hybrid-attention MoE and a 235B at INT4 among them — with **needle-in-a-haystack recall to 174K tokens**. Footprints we measured ourselves. |
| 🔍 **Radical transparency** | Every number traces to a committed log on real hardware. We even publish [`AUDIT.md`](AUDIT.md) — our own ledger of past overclaims and their fixes. In a field drowning in inflated benchmarks, that *is* the differentiator. |

---

## 🔬 Proof, not promises

> **1× AMD Instinct MI300X** (192 GB HBM3, ROCm 7.2) · vLLM in Docker, coordinator host-side. Raw artifacts: [`logs_mi300x_p2/`](logs_mi300x_p2/) + [`logs_moe_run/`](logs_moe_run/) · summary in the [evidence report](logs_moe_run/MI300X_MOE_EVIDENCE.md).

### 🛡️ Safety core — formally verified

| Property | Result |
|---|---|
| INV-15 violations across the full **1,210-point** input sweep | **0 / 1,210** |
| Z3 SMT proof of INV-15 (negation `unsat` over the modeled domain) | **PROVED** · 10.08 ms |
| FORGE-LEDGER — hash-chained certified decisions + live tamper test | verify → **exit 0** · tamper → **exit 2** |
| Per-decision Z3 certification latency (p99) | **0.25 ms** · 243 certs/s |

### ✂️ Efficiency — measured, not modeled

| Metric | Result |
|---|---|
| **Prompt compression on live MoE** | **44.4 %** fewer prompt tokens (5,265 → 2,926), 5-agent workload |
| **Cross-agent KV-block sharing** (`cache_salt`) | **84.7 %** prefix-cache hit shared vs **0 %** isolated · **TTFT −57 %** (0.058 s vs 0.135 s) — Qwen3-32B full-attention |
| **INT4 RotateKV** KV-cache reduction | **3.55×**, length-invariant 4K → 262K (`use_fwht=False`) |
| **HBM3** effective bandwidth | **3.79 TB/s** (72 % of peak), STREAM-triad fp16 |

### 🚀 Frontier MoE on a single card

| Model | Params | Precision | One MI300X | Long-context recall |
|---|---|---|---|---|
| **Qwen3-30B-A3B-2507** | 30B / 3B MoE | FP8 | ✅ ~186 GB | **NIAH 12/12 → 174K tok** · 2,667 tok/s |
| **Qwen3-Coder-Next** (hybrid) | 80B / 3B MoE | FP8 | ✅ ~175 GiB | **NIAH 12/12 → 174K tok** · 2,149 tok/s |
| **Qwen3-235B-A22B** | 235B / 22B MoE | INT4 | ✅ ~181 GiB | single-card **+ ~56 GB CPU offload** |

> An 80 GB GPU cannot hold these. A 192 GB MI300X can. That gap is the moat — and these are *our* measured footprints, not a datasheet.

### 🎯 Where ContextForge applies — and where it doesn't

Two independent levers, measured separately and honestly:

- **Token compression (44.4 %)** is *architecture-agnostic* — it shrinks the prompt **before** serving, so it helps full-attention, sparse, linear-hybrid and sliding-window models alike. The **durable** win.
- **Cross-agent KV-block sharing (84.7 %)** scales with KV-cache size, so it is **largest on full-attention** models — Llama 3.x, Qwen2.5/3-dense, Mistral: the bulk of the *installed* production fleet today.

**The honest limit.** The 2026 frontier is moving *away* from full attention — DeepSeek-V4 / GLM-5 (sparse), Qwen3-Next/3.5/3.6 (linear-hybrid), Gemma 4 / OLMo 3 / MiMo (sliding-window) — **precisely to shrink the KV-cache bottleneck the sharing lever optimises.** On those architectures the KV-sharing win is smaller by design, and we don't claim otherwise. The KV lever is for the large full-attention fleet that exists now; the compression lever is for everything. (Scope & raw evidence: [`AUDIT.md` §19](AUDIT.md).)

---

## 🏗️ Architecture

```mermaid
%%{init: {'theme':'dark', 'themeVariables': {'fontFamily':'ui-monospace, monospace'}}}%%
flowchart TB
    subgraph Agents["5-Agent Pipeline"]
        A1[Retriever] & A2[Reranker] & A3[Summarizer] & A4[Critic] & A5[Responder]
    end
    subgraph CF["ContextForge Coordinator · FastAPI + asyncio"]
        REG["Context Registry"]
        COMP["Compression · LLMLingua-2<br/>✅ 44% on live MoE"]
        JCR{"JCR Safety Gate · INV-15<br/>✅ Z3-proved"}
        LEDGER["FORGE-LEDGER<br/>✅ tamper-evident certs"]
        DEDUP["Semantic dedup · LSH+FAISS<br/>🔬 needs qwen3-embed"]
    end
    subgraph Serving["AMD MI300X · ROCm 7.2 · vLLM V1"]
        VLLM["vLLM V1 endpoint"]
        ROMY["ROMY plugin · cross-agent<br/>KV-block sharing · 🔬 in progress"]
    end
    A1 & A2 & A3 & A5 --> REG --> COMP --> VLLM
    A4 --> JCR -->|risk > 0.7| VLLM
    JCR --> LEDGER
    REG -.-> DEDUP
    VLLM -.-> ROMY
    style JCR fill:#39D353,stroke:#0D1117,color:#0D1117
    style LEDGER fill:#39D353,stroke:#0D1117,color:#0D1117
    style COMP fill:#22D3EE,stroke:#0D1117,color:#0D1117
    style ROMY fill:#30363D,stroke:#C724B1,color:#fff,stroke-dasharray:4
    style DEDUP fill:#30363D,stroke:#C724B1,color:#fff,stroke-dasharray:4
```

<sub>✅ validated on MI300X&nbsp;·&nbsp;🔬 in progress — and we tell you which is which.</sub>

---

## 🧩 Every mechanism, graded by what we verified

We refuse to claim a paper's number as our own. Each mechanism is graded by **what runs and what we measured**:

| Mechanism | Source | Status |
|---|---|---|
| **JCR Safety Gate (INV-15)** | [arXiv:2601.08343](https://arxiv.org/abs/2601.08343) | ✅ **Validated + Z3-proved** |
| **RotateKV INT4 codec** | [arXiv:2501.16383](https://arxiv.org/abs/2501.16383) | ✅ **Validated** — 3.55× |
| **LLMLingua-2 compression** | ACL 2024 | ✅ **Validated** — 44 % on live MoE |
| **FORGE-LEDGER** certified audit | this work | ✅ **Validated** on-hardware |
| TokenDance · KVCOMM · KVFlow · PBKV · CLA · VisualKVCache · Queueing | various | 🟡 Implemented + unit-tested (synthetic) |
| **Cross-agent KV-block sharing** (ROMY) | this work | 🔬 **In progress** — the efficiency moat we're building next |
| Semantic dedup on `qwen3-embed` · LMCache ROCm bridge | various | 🔬 In progress |

---

## 🚀 Quick start

```bash
# From PyPI — slim core (safety kernel + INV-15 gate; no torch/vllm):
pip install apohara-context-forge
# …or the full serving stack (vLLM, embeddings, Gradio demo):
pip install apohara-context-forge[serve]

# …or from source (development):
git clone https://github.com/SuarezPM/Apohara_Context_Forge.git
cd Apohara_Context_Forge && pip install -e '.[dev]'  # or: uv sync

PYTHONPATH=. pytest tests/ -q                        # 496 passed · 25 skipped

# Machine-check the INV-15 safety invariant (Z3):
python -m apohara_context_forge.safety.z3_inv15_proof
# → {"status": "PROVED", "elapsed_ms": 10.08, "z3_version": "4.16.0"}

# Verify a certified decision ledger (intact → exit 0, tampered → exit 2):
python -m apohara_context_forge.observability.ledger_cli verify <ledger.jsonl>
```

**Reproduce on MI300X:** [`scripts/forge_p2_run_all.sh`](scripts/forge_p2_run_all.sh) · [`scripts/mi300x_contextforge_e2e.py`](scripts/mi300x_contextforge_e2e.py).

---

## 🏢 Who needs this

You don't ship LLM-as-judge to production on a hunch — and regulators won't let you. ContextForge is built for teams running **multi-agent / judge pipelines on-prem on AMD MI300X** that must **prove** their AI is safe:

- **Banks** (SR 11-7 model risk) · **defense** (DFARS / CMMC) · **healthcare** (HIPAA) · any team under the **EU AI Act**'s high-risk audit obligations — code and data that legally cannot leave the VPC, on hardware that fits frontier MoE single-card.
- **AI-safety & eval teams** whose entire product is a judge pipeline — exactly where the JCR failure mode bites.

The JCR Safety Gate + certified ledger are the **audit-grade, machine-checked answer** to *"prove your judge agent isn't silently wrong."*

---

## 🔍 Honest by construction

Most AI repos inflate. We do the opposite — on purpose, because trust is the product.

[`AUDIT.md`](AUDIT.md) is our **public ledger of every claim we ever overstated**, with `file:line` evidence and its fix; [`scripts/check_honesty.sh`](scripts/check_honesty.sh) runs in CI to catch hardcoded numbers and misleading labels. Recent entries: the codec figure (literature 3.97× → **measured 3.55×**), a compressor bug that left compression non-functional until we fixed it, and the line between a local demo and real-model inference.

**If a number is here, it ran on real silicon and there's a log to prove it. If it isn't built yet, we mark it 🔬.**

---

## ✅ Verification

| Check | Result |
|---|---|
| `PYTHONPATH=. pytest tests/` | **496 passed · 25 skipped · 0 failed** |
| `z3_inv15_proof` | **PROVED** (`unsat` on negation) |
| `ledger_cli verify` (intact / tampered) | exit **0** / **2** |
| Honesty CI guard | **PASS** |

**Invariants enforced:** INV-10…INV-14 + **INV-15 (JCR dense-prefill — Z3-proved).**

---

## 🗺️ Roadmap

**Now — the efficiency moat** (VRAM quoted only once measured): 🔬 physical cross-agent KV-block sharing inside vLLM (**ROMY** plugin) · 🔬 real-embedding semantic dedup (`qwen3-embed`) · 🔬 needle-in-a-haystack under INT4 at 200K.

**Next — safety & audit depth** (the differentiator): adaptive INV-15 thresholds · Z3 extended to INV-10…INV-14 · OTLP compliance export.

**Later — scale:** multi-GPU TokenDance over RCCL · LMCache ROCm build · companion systems paper.

---

## 📚 Cite

> Suarez, P. M. (2026). *INV-15: A Formal Safety Invariant for KV-Cache Reuse in Multi-Agent Judge Pipelines* (APOHARA · ContextForge). Zenodo. https://doi.org/10.5281/zenodo.20277875

```bibtex
@software{contextforge2026,
  author    = {Suarez, Pablo M.},
  title     = {{INV-15: A Formal Safety Invariant for KV-Cache Reuse in Multi-Agent Judge Pipelines}},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.20277875}
}
```

Paper: [`paper/inv15_paper.pdf`](paper/inv15_paper.pdf)&nbsp;·&nbsp;Apache 2.0 ([LICENSE](LICENSE))&nbsp;·&nbsp;Pablo M. Suarez&nbsp;·&nbsp;[`suarezpm@csnat.unt.edu.ar`](mailto:suarezpm@csnat.unt.edu.ar)&nbsp;·&nbsp;[@SuarezPM](https://github.com/SuarezPM)

<p align="center"><sub><strong>APOHARA · ContextForge</strong> — provably-safe multi-agent inference on AMD Instinct MI300X.</sub></p>
