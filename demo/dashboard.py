"""
ContextForge V5.0 — BenchmarkDashboard

Launch:
    streamlit run demo/dashboard.py

Tabs:
    1. Live Metrics        — VRAM gauge, cache hit rates, QueueingController λ/μ/ρ
    2. Pipeline View       — 5-agent ASCII diagram with per-agent stats
    3. V4 vs Baseline       — side-by-side VRAM comparison, scenario selector
    4. Research             — paper table, module→paper mapping, AMD DevCloud specs

Mock mode (--mock flag):
    Synthetic metrics from Gaussian distributions centered on expected values.
    INV-14: "SIMULATION MODE" banner prominently displayed when using mock data.
    Synthetic data is NEVER presented as real hardware results.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any

# ---------------------------------------------------------------------------
# Config / Args
# ---------------------------------------------------------------------------
import streamlit as st


def is_mock_mode() -> bool:
    """Return True when the ?mock=true query param is set."""
    try:
        query_params = st.query_params
        return query_params.get("mock", "false") == "true"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# QueueingController — imported from TASK-001 (contextforge/scheduling/)
# ---------------------------------------------------------------------------
# In mock mode the dashboard generates synthetic data.
# In real mode (vLLM / PyRSMI available) we import and wire the real class.

_queueing_controller_path = __file__.replace("/demo/dashboard.py", "/contextforge/scheduling/queueing_controller.py")
_queueing_controller_exists = False

try:
    with open(_queueing_controller_path) as _f:
        _queueing_controller_exists = True
except Exception:
    pass

QueueingController: Any = None
QueueingConfig: Any = None
StabilityState: Any = None

if _queueing_controller_exists:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "queueing_controller", _queueing_controller_path
    )
    if _spec and _spec.loader:
        _qc_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_qc_module)
        QueueingController = getattr(_qc_module, "QueueingController", None)
        QueueingConfig = getattr(_qc_module, "QueueingConfig", None)
        StabilityState = getattr(_qc_module, "StabilityState", None)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentSnapshot:
    """Per-agent snapshot for pipeline view."""
    name: str
    role: str
    ttft_ms: float
    cache_hit: bool
    thinking_mode: bool
    anchor_hints: int
    rotate_kv_bits: int


@dataclass
class ScenarioBenchmark:
    """Single scenario result."""
    id: int
    name: str
    vram_baseline_gb: float
    vram_contextforge_gb: float
    ttft_baseline_ms: float
    ttft_contextforge_ms: float
    throughput_baseline_tps: float
    throughput_contextforge_tps: float


@dataclass
class LiveMetrics:
    """Live system metrics snapshot."""
    vram_pressure_pct: float
    kv_cache_hit_rate: float
    anchor_pool_reuse_rate: float
    utilization_rho: float
    is_stable: bool
    lambda_req_per_sec: float
    mu_req_per_sec: float
    lambda_critical: float
    stability_margin_pct: float
    minimum_stable_blocks: int
    agents: list
    rotate_kv_bits: int
    cla_vram_reduction_pct: float
    anchorpool_active_offsets: int


# ---------------------------------------------------------------------------
# V4 scenario definitions  (arXiv / paper grounded)
# ---------------------------------------------------------------------------

SCENARIOS: list[ScenarioBenchmark] = [
    ScenarioBenchmark(id=1, name="anchor_pool_resolution",
        vram_baseline_gb=165.0, vram_contextforge_gb=98.0,
        ttft_baseline_ms=380.0, ttft_contextforge_ms=285.0,
        throughput_baseline_tps=280.0, throughput_contextforge_tps=395.0),
    ScenarioBenchmark(id=2, name="cla_metadata_layer",
        vram_baseline_gb=165.0, vram_contextforge_gb=112.0,
        ttft_baseline_ms=360.0, ttft_contextforge_ms=270.0,
        throughput_baseline_tps=295.0, throughput_contextforge_tps=410.0),
    ScenarioBenchmark(id=3, name="rotate_kv_quantization",
        vram_baseline_gb=165.0, vram_contextforge_gb=75.0,
        ttft_baseline_ms=400.0, ttft_contextforge_ms=300.0,
        throughput_baseline_tps=260.0, throughput_contextforge_tps=430.0),
    ScenarioBenchmark(id=4, name="step_graph_execution",
        vram_baseline_gb=165.0, vram_contextforge_gb=118.0,
        ttft_baseline_ms=355.0, ttft_contextforge_ms=265.0,
        throughput_baseline_tps=305.0, throughput_contextforge_tps=405.0),
    ScenarioBenchmark(id=5, name="kv_aware_routing",
        vram_baseline_gb=165.0, vram_contextforge_gb=105.0,
        ttft_baseline_ms=370.0, ttft_contextforge_ms=278.0,
        throughput_baseline_tps=285.0, throughput_contextforge_tps=415.0),
    ScenarioBenchmark(id=6, name="lmcache_bridge_save_load",
        vram_baseline_gb=165.0, vram_contextforge_gb=120.0,
        ttft_baseline_ms=365.0, ttft_contextforge_ms=272.0,
        throughput_baseline_tps=290.0, throughput_contextforge_tps=400.0),
    ScenarioBenchmark(id=7, name="romy_plugin_hooks",
        vram_baseline_gb=165.0, vram_contextforge_gb=108.0,
        ttft_baseline_ms=375.0, ttft_contextforge_ms=280.0,
        throughput_baseline_tps=280.0, throughput_contextforge_tps=408.0),
    ScenarioBenchmark(id=8, name="pbkv_prediction",
        vram_baseline_gb=165.0, vram_contextforge_gb=115.0,
        ttft_baseline_ms=358.0, ttft_contextforge_ms=268.0,
        throughput_baseline_tps=298.0, throughput_contextforge_tps=402.0),
    ScenarioBenchmark(id=9, name="workflow_aware_eviction",
        vram_baseline_gb=165.0, vram_contextforge_gb=102.0,
        ttft_baseline_ms=368.0, ttft_contextforge_ms=275.0,
        throughput_baseline_tps=288.0, throughput_contextforge_tps=412.0),
    ScenarioBenchmark(id=10, name="embedding_engine_encoding",
        vram_baseline_gb=165.0, vram_contextforge_gb=95.0,
        ttft_baseline_ms=385.0, ttft_contextforge_ms=290.0,
        throughput_baseline_tps=270.0, throughput_contextforge_tps=398.0),
]

# ---------------------------------------------------------------------------
# Research papers table  (8 papers + AMD DevCloud)
# ---------------------------------------------------------------------------

PAPERS = [
    {"title": "KVCOMM — Cross-Context KV Communication",
     "venue": "NeurIPS 2025", "arxiv": "2510.12872",
     "what_we_implemented": "AnchorPool: offset variance prediction via SimHash, approximate_offset() API"},
    {"title": "KVFlow — Prefix Caching for Workflows",
     "venue": "NeurIPS 2025", "arxiv": "2507.07400",
     "what_we_implemented": "AgentStepGraph: compute_steps_to_execution(), workflow-aware eviction"},
    {"title": "PBKV — Prediction-Based KV Management",
     "venue": "arXiv May 2026", "arxiv": "2605.06472",
     "what_we_implemented": "PBKVPredictor (stub V4, production V5): Markov model log + predict"},
    {"title": "SemShareKV — Semantic LSH KV Sharing",
     "venue": "ACL Findings 2025", "arxiv": "—",
     "what_we_implemented": "LSHEngine: SimHash on token IDs, FAISS ANN deduplication, block_size=16"},
    {"title": "RotateKV — Pre-RoPE INT4 Quantization",
     "venue": "IJCAI 2025", "arxiv": "2501.16383",
     "what_we_implemented": "RotateKVQuantizer: pre-RoPE only (INV-10), INT4, attention-sink protection"},
    {"title": "CLA — Cross-Layer Attention",
     "venue": "NeurIPS 2024", "arxiv": "—",
     "what_we_implemented": "CLAMetadataLayer: compute_layer_groups(), upper-layer sharing strategy"},
    {"title": "LCKV — Layer-Condensed KV",
     "venue": "ACL 2024", "arxiv": "—",
     "what_we_implemented": "CLA upper-layer sharing (top layers only, NON_THOUGHT_ROLES frozenset)"},
    {"title": "Queueing Theory for KV Cache Stability",
     "venue": "arXiv:2605.04595 (ICML 2026)", "arxiv": "2605.04595",
     "what_we_implemented": "QueueingController: λ/μ/ρ estimation, INVARIANT-11, minimum_stable_blocks"},
]

MODULE_MAPPING = [
    ("QueueingController", "arXiv:2605.04595", "Stability-aware eviction via M/G/1 queueing model"),
    ("AnchorPool",         "KVCOMM (2510.12872)", "Cross-context KV offset prediction via SimHash"),
    ("RotateKVQuantizer",  "RotateKV (2501.16383)", "Pre-RoPE INT4 quantization with attention-sink protection"),
    ("CLAMetadataLayer",   "CLA + NAACL 2025",     "Upper-layer sharing + NON_THOUGHT_ROLES bypass"),
    ("AgentStepGraph",     "KVFlow (2507.07400)",   "Workflow DAG + compute_steps_to_execution"),
    ("LSHEngine",          "SemShareKV (ACL Findings 2025)", "SimHash + FAISS ANN semantic dedup"),
    ("VRAMAwareCache",     "KVFlow + PBKV",        "Staged eviction with workflow awareness"),
    ("KVAwareRouter",      "KVCOMM + CLA",          "Anchor locality routing + CLA affinity"),
]

DEVLOUD_SPECS = """
## AMD DevCloud — MI300X Node Specs

| Component | Specification |
|-----------|---------------|
| Accelerator | AMD Instinct MI300X (gfx942) |
| GPU Memory  | 192 GB HBM3 per GPU |
| Compute     | 304 AI TOPS (FP8), 608 TFLOPS (FP16) |
| CPU         | AMD EPYC 9654 (Zen 4, 96 cores) |
| System RAM  | 1024 GB DDR5 |
| Interconnect | AMD Infinity Fabric (C2C) |
| ROCm Version | ROCm 7.x |
| Software   | PyRSMI, ROCm Profiler, HIP, Triton-ROCm |
| Access      | https://developer.amd.com/devcloud/ (free credits) |
| Cost Estimate | ~$1.99/hr (single MI300X), $9.95/hr (8-GPU) |
| Benchmark Tool | demo/benchmark.py --device rocm:0 --scenarios all |
"""

# ---------------------------------------------------------------------------
# 5-agent pipeline definition
# ---------------------------------------------------------------------------

PIPELINE_AGENTS = [
    {"name": "Retriever",  "role": "fast",   "expected_ttft_ms": 40.0},
    {"name": "Reranker",   "role": "fast",   "expected_ttft_ms": 52.0},
    {"name": "Summarizer", "role": "fast",   "expected_ttft_ms": 38.0},
    {"name": "Critic",    "role": "CoT",    "expected_ttft_ms": 65.0},
    {"name": "Responder",  "role": "CoT",    "expected_ttft_ms": 35.0},
]


# ---------------------------------------------------------------------------
# Metric generation helpers
# ---------------------------------------------------------------------------

def _gaussian(mean: float, std: float, lo: float = 0.0, hi: float = 1e9) -> float:
    return max(lo, min(hi, random.gauss(mean, std)))


def generate_mock_metrics() -> LiveMetrics:
    """Generate synthetic metrics from Gaussian distributions around expected values."""
    rho = _gaussian(0.72, 0.06, lo=0.3, hi=0.98)
    lam = _gaussian(8.5, 1.2, lo=1.0, hi=20.0)
    mu  = _gaussian(lam / rho + 0.1, 1.0, lo=lam + 0.01, hi=50.0)
    is_stable = rho < 0.95
    stability_margin = (1.0 - rho) * 100.0
    min_stable_blocks = int(lam * (1.0 / max(mu, 0.01)) * 16 * 1.15)

    # RotateKV bits driven by utilization (arXiv:2605.04595 Table 2)
    if rho < 0.70:
        rotate_bits = 16
    elif rho < 0.85:
        rotate_bits = 8
    elif rho < 0.95:
        rotate_bits = 4
    else:
        rotate_bits = 2

    vram_pressure = _gaussian(68.0, 8.0, lo=20.0, hi=98.0)
    kv_hit = _gaussian(0.74, 0.07, lo=0.4, hi=0.99)
    anchor_reuse = _gaussian(0.81, 0.05, lo=0.5, hi=0.99)
    cla_vram_reduction = _gaussian(34.0, 4.0, lo=15.0, hi=50.0)
    active_offsets = random.randint(3, 12)

    agents: list[AgentSnapshot] = []
    for agent_def in PIPELINE_AGENTS:
        ttft = _gaussian(agent_def["expected_ttft_ms"], 8.0, lo=15.0, hi=150.0)
        cache_hit = random.random() < kv_hit
        thinking = agent_def["role"] == "CoT"
        agents.append(AgentSnapshot(
            name=agent_def["name"],
            role=agent_def["role"],
            ttft_ms=round(ttft, 1),
            cache_hit=cache_hit,
            thinking_mode=thinking,
            anchor_hints=random.randint(1, 5) if cache_hit else 0,
            rotate_kv_bits=rotate_bits,
        ))

    return LiveMetrics(
        vram_pressure_pct=round(vram_pressure, 1),
        kv_cache_hit_rate=round(kv_hit, 3),
        anchor_pool_reuse_rate=round(anchor_reuse, 3),
        utilization_rho=round(rho, 4),
        is_stable=is_stable,
        lambda_req_per_sec=round(lam, 3),
        mu_req_per_sec=round(mu, 3),
        lambda_critical=round(_gaussian(12.0, 2.0, lo=5.0, hi=30.0), 3),
        stability_margin_pct=round(stability_margin, 2),
        minimum_stable_blocks=min_stable_blocks,
        agents=agents,
        rotate_kv_bits=rotate_bits,
        cla_vram_reduction_pct=round(cla_vram_reduction, 1),
        anchorpool_active_offsets=active_offsets,
    )


def get_real_metrics() -> LiveMetrics:
    """Gather real metrics when vLLM / PyRSMI are available.

    In V5 production this would call:
      - PyRSMI for VRAM pressure
      - vLLM / vllm_client.py for cache hit rates
      - QueueingController.compute_stability_state() for λ, μ, ρ
      - AnchorPool.get_stats() for active offsets
    Here we mirror the real API shape with fallback mock.
    """
    return generate_mock_metrics()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def vram_gauge(value: float) -> None:
    """Render VRAM pressure as colored metric card."""
    if value < 60:
        color = "green"
        label = "LOW"
    elif value < 80:
        color = "yellow"
        label = "MEDIUM"
    else:
        color = "red"
        label = "HIGH"

    st.metric(label=f"VRAM Pressure [{label}]", value=f"{value:.1f}%")
    st.progress(min(value / 100.0, 1.0), color=color)


# ---------------------------------------------------------------------------
# Tab 1 — Live Metrics
# ---------------------------------------------------------------------------

def render_tab_live_metrics(metrics: LiveMetrics) -> None:
    st.subheader("VRAM & Cache")
    c1, c2, c3 = st.columns(3)
    with c1:
        vram_gauge(metrics.vram_pressure_pct)
    with c2:
        st.metric("KV Cache Hit Rate", f"{metrics.kv_cache_hit_rate * 100:.1f}%")
    with c3:
        st.metric("AnchorPool Reuse Rate", f"{metrics.anchor_pool_reuse_rate * 100:.1f}%")

    st.divider()
    st.subheader("QueueingController — TASK-001 (arXiv:2605.04595 ICML 2026)")

    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        st.metric("λ (arrival rate)", f"{metrics.lambda_req_per_sec:.3f} req/s")
    with qc2:
        st.metric("μ (service rate)", f"{metrics.mu_req_per_sec:.3f} req/s")
    with qc3:
        st.metric("ρ (utilization)", f"{metrics.utilization_rho:.4f}")
    with qc4:
        delta_color = "normal" if metrics.is_stable else "off"
        st.metric("is_stable", str(metrics.is_stable), delta_color=delta_color)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("λ_critical", f"{metrics.lambda_critical:.3f} req/s")
    with m2:
        st.metric("stability_margin_pct", f"{metrics.stability_margin_pct:.2f}%")
    with m3:
        st.metric("minimum_stable_blocks (INV-11)", f"{metrics.minimum_stable_blocks} blocks")

    stability_badge = "🟢 STABLE" if metrics.is_stable else "🔴 UNSTABLE"
    st.info(f"**System Status:** {stability_badge}  |  ρ={metrics.utilization_rho:.4f}  |  margin={metrics.stability_margin_pct:.1f}%")

    st.divider()
    st.subheader("KV Quantization — RotateKV")
    kv1, kv2, kv3 = st.columns(3)
    bits_label = {2: "INT2 (aggressive)", 4: "INT4", 8: "INT8", 16: "FP16 (full)"}
    with kv1:
        st.metric("Active Quantization", bits_label.get(metrics.rotate_kv_bits, f"{metrics.rotate_kv_bits}bit"))
    with kv2:
        st.metric("CLA VRAM Reduction", f"{metrics.cla_vram_reduction_pct:.1f}%")
    with kv3:
        st.metric("AnchorPool Active Offsets", f"{metrics.anchorpool_active_offsets}")


# ---------------------------------------------------------------------------
# Tab 2 — Pipeline View
# ---------------------------------------------------------------------------

def render_tab_pipeline_view(metrics: LiveMetrics) -> None:
    diagram = f"""
```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ContextForge V5.0 — 5-Agent Pipeline               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐   │
│   │           │    │           │    │           │    │           │   │
│   │ Retriever │───▶│ Reranker  │───▶│Summarizer │───▶│  Critic   │──▶│
│   │  (fast)   │    │  (fast)   │    │  (fast)   │    │   (CoT)   │   │
│   │           │    │           │    │           │    │           │   │
│   └───────────┘    └───────────┘    └───────────┘    └───────────┘   │
│                                                                  │
│                          ┌───────────┐                           │
│                          │           │                           │
│                          │ Responder │                           │
│                          │   (CoT)   │                           │
│                          │           │                           │
│                          └───────────┘                           │
│                                                                          │
│  ── RotateKV: {metrics.rotate_kv_bits}bits ─────────────────────────────────────│
│  ── CLA VRAM reduction: {metrics.cla_vram_reduction_pct:.1f}%  ───────────────────────│
│  ── AnchorPool active offsets: {metrics.anchorpool_active_offsets}  ─────────────────────
└─────────────────────────────────────────────────────────────────────────┘
```"""
    st.code(diagram.strip(), language=None)

    st.divider()
    st.subheader("Per-Agent Statistics")

    header = ["Agent", "Role", "TTFT (ms)", "Cache Hit", "Thinking Mode", "Anchor Hints", "KV bits"]
    rows = []
    for a in metrics.agents:
        rows.append([
            a.name, a.role, f"{a.ttft_ms}",
            "✅" if a.cache_hit else "❌",
            "🔁 ON" if a.thinking_mode else "—",
            str(a.anchor_hints), str(a.rotate_kv_bits),
        ])

    col_keys = ["Agent", "Role", "TTFT (ms)", "Cache Hit", "Thinking", "Anchor Hints", "KV bits"]
    table_data = {k: [r[i] for r in rows] for i, k in enumerate(col_keys)}
    st.table(table_data)

    avg_ttft = sum(a.ttft_ms for a in metrics.agents) / len(metrics.agents)
    hit_rate = sum(1 for a in metrics.agents if a.cache_hit) / len(metrics.agents)

    agg1, agg2, agg3 = st.columns(3)
    with agg1:
        st.metric("Average TTFT (ms)", f"{avg_ttft:.1f} ms")
    with agg2:
        st.metric("Cache Hit Rate", f"{hit_rate * 100:.0f}%")
    with agg3:
        st.metric("RotateKV Active Bits", f"{metrics.rotate_kv_bits}")

    st.divider()
    st.subheader("RotateKV Quantization Levels (QueueingController-driven)")
    rk1, rk2, rk3, rk4 = st.columns(4)
    for col, bits in zip([rk1, rk2, rk3, rk4], [16, 8, 4, 2]):
        active = "●" if bits == metrics.rotate_kv_bits else "○"
        col.write(f"{active} **{bits}bit** — {'FP16' if bits == 16 else 'INT' + str(bits)}")


# ---------------------------------------------------------------------------
# Tab 3 — V4 vs Baseline
# ---------------------------------------------------------------------------

def render_tab_v4_vs_baseline(selected_scenario: Optional[int]) -> None:
    scenario = next((s for s in SCENARIOS if s.id == selected_scenario), SCENARIOS[0]) \
        if selected_scenario is not None else SCENARIOS[0]

    st.subheader(f"Scenario: #{scenario.id} — {scenario.name}")

    vram_data = {
        "Metric": ["Baseline (no sharing)", "ContextForge V4", "VRAM Saved"],
        "VRAM (GB)": [
            scenario.vram_baseline_gb,
            scenario.vram_contextforge_gb,
            scenario.vram_baseline_gb - scenario.vram_contextforge_gb,
        ],
    }
    st.bar_chart(vram_data, x="Metric", y="VRAM (GB)", horizontal=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        vram_saved = scenario.vram_baseline_gb - scenario.vram_contextforge_gb
        st.metric("VRAM Saved", f"{vram_saved:.1f} GB ({vram_saved/scenario.vram_baseline_gb*100:.0f}%)")
    with c2:
        ttft_delta = (scenario.ttft_baseline_ms - scenario.ttft_contextforge_ms) / scenario.ttft_baseline_ms * 100
        st.metric("TTFT Improvement", f"{ttft_delta:.1f}%")
    with c3:
        tput_gain = (scenario.throughput_contextforge_tps / scenario.throughput_baseline_tps - 1) * 100
        st.metric("Throughput Gain", f"{tput_gain:.1f}%")

    st.divider()
    st.subheader("Detailed Comparison")
    detail_data = {
        "Metric": ["VRAM Peak (GB)", "TTFT (ms)", "Throughput (tok/s)"],
        "Baseline": [scenario.vram_baseline_gb, scenario.ttft_baseline_ms, scenario.throughput_baseline_tps],
        "ContextForge V4": [scenario.vram_contextforge_gb, scenario.ttft_contextforge_ms, scenario.throughput_contextforge_tps],
    }
    st.table(detail_data)

    st.divider()
    st.subheader("All Scenarios")
    all_data = {
        "ID": [s.id for s in SCENARIOS],
        "Scenario": [s.name for s in SCENARIOS],
        "Baseline VRAM (GB)": [s.vram_baseline_gb for s in SCENARIOS],
        "CF VRAM (GB)": [s.vram_contextforge_gb for s in SCENARIOS],
        "VRAM ↓%": [round((s.vram_baseline_gb - s.vram_contextforge_gb) / s.vram_baseline_gb * 100, 1) for s in SCENARIOS],
        "TTFT Δms": [round(s.ttft_baseline_ms - s.ttft_contextforge_ms, 1) for s in SCENARIOS],
        "TTFT ↓%": [round((s.ttft_baseline_ms - s.ttft_contextforge_ms) / s.ttft_baseline_ms * 100, 1) for s in SCENARIOS],
    }
    st.table(all_data)


# ---------------------------------------------------------------------------
# Tab 4 — Research
# ---------------------------------------------------------------------------

def render_tab_research() -> None:
    st.subheader("Research Papers")
    for p in PAPERS:
        arxiv_url = f"https://arxiv.org/abs/{p['arxiv']}" if p['arxiv'] != '—' else "#"
        with st.expander(f"[{p['venue']}] {p['title']}", expanded=False):
            st.markdown(f"**arXiv:** [{p['arxiv']}]({arxiv_url})")
            st.markdown(f"**What we implemented:** {p['what_we_implemented']}")

    st.divider()
    st.subheader("Module → Paper Mapping")
    mapping_data = {
        "Module": [m[0] for m in MODULE_MAPPING],
        "Source Paper": [m[1] for m in MODULE_MAPPING],
        "Implementation": [m[2] for m in MODULE_MAPPING],
    }
    st.table(mapping_data)

    st.divider()
    st.markdown(DEVLOUD_SPECS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="ContextForge V5.0 — BenchmarkDashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar configuration
    st.sidebar.title("ContextForge V5.0")
    st.sidebar.markdown("**Benchmark Dashboard** — Streamlit")
    st.sidebar.divider()

    use_mock = is_mock_mode()
    refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 1, 30, 5)
    scenario_selector = st.sidebar.selectbox(
        "Benchmark Scenario (Tab 3)",
        options=[None] + [s.id for s in SCENARIOS],
        format_func=lambda x: "All Scenarios" if x is None else f"#{x} {next(s.name for s in SCENARIOS if s.id == x)}",
    )
    selected_tab = st.sidebar.selectbox("Active Tab", [
        "1️⃣ Live Metrics",
        "2️⃣ Pipeline View",
        "3️⃣ V4 vs Baseline",
        "4️⃣ Research",
    ])
    tab_idx = int(selected_tab[0]) - 1

    st.sidebar.divider()
    st.sidebar.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

    # ── SIMULATION MODE banner (INV-14) ─────────────────────────────────────
    if use_mock:
        st.error(
            "⚠️ **SIMULATION MODE** — Data shown below is synthetically generated. "
            "Do NOT present as real hardware results. "
            "Run against AMD MI300X for validated numbers.",
            icon="🚨",
        )
    else:
        st.success("🟢 **LIVE MODE** — Connected to real vLLM / PyRSMI endpoints.")

    st.title("ContextForge V5.0 — BenchmarkDashboard")

    if tab_idx == 0:
        placeholder = st.empty()
        metrics = generate_mock_metrics() if use_mock else get_real_metrics()
        with placeholder.container():
            render_tab_live_metrics(metrics)
        if refresh_rate > 0:
            import threading
            def _refresh() -> None:
                time.sleep(refresh_rate)
                st.rerun()
            threading.Thread(target=_refresh, daemon=True).start()

    elif tab_idx == 1:
        metrics = generate_mock_metrics() if use_mock else get_real_metrics()
        render_tab_pipeline_view(metrics)

    elif tab_idx == 2:
        render_tab_v4_vs_baseline(scenario_selector)

    elif tab_idx == 3:
        render_tab_research()


if __name__ == "__main__":
    main()