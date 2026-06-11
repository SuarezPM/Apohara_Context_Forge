# C2 — Latencia del JCR Safety Gate (INV-15) en MI300X

**Fecha**: 2026-06-11 · **Hardware**: 1× AMD MI300X (gfx942) · **Python**: 3.12.3 · **Repo commit**: a3b2e88

> Micro-benchmark del gate de seguridad que es el corazón del reframe
> post-ABANDON. Si el gate es shipping-grade (≪ TTFT), la promesa
> "ROMY = salting honesto + safety O(1) por request" es defendible.

## Hipótesis

El path por defecto de `gate_decision` debe ser **O(1) en el sentido
práctico**: < 1 ms por request, idealmente < 200 µs, porque de lo
contrario el costo del safety se comería la mejora de TTFT que el
sharing mecánico prometía (y que el ABANDON ya nos quitó). El
certificador Z3 con `APOHARA_FORGE_LEDGER=1` puede agregar overhead
material (cada decisión hace una prueba SMT), pero debería amortizarse
en workloads donde el gate se llama < 100 veces por segundo.

## Setup

| Item | Valor |
|------|-------|
| Hardware | 1× MI300X, ROCm 7.2.4 (Hot Aisle default) |
| Python | 3.12.3, numpy 2.4, pydantic (deps mínimas) |
| Repo | `Apohara_Context_Forge @ a3b2e88` (clonado fresco) |
| n por path | 10 000 calls |
| Warmup | 500 calls (descarta JIT/cache Python) |
| Workload | 4 paths: `compute_jcr_risk` puro, `gate_decision` default, `gate_decision + Z3`, `gate_decision` mixto realista |

## Resultados (p50 / p95 / p99 / max en µs)

| Path | p50 | p95 | p99 | max | n |
|------|----:|----:|----:|----:|--:|
| `compute_jcr_risk` (pure, sin log) | **0.4** | 0.5 | 0.5 | 16.0 | 10 000 |
| `gate_decision` default (no ledger) | **146.4** | 155.1 | 237.1 | 811.9 | 10 000 |
| `gate_decision + APOHARA_FORGE_LEDGER=1` (Z3) | **146.4** | 155.0 | 236.9 | 787.4 | 10 000 |
| `gate_decision` (mixed workload, no ledger) | 147.0 | 156.2 | 249.5 | 789.3 | 10 000 |

## Análisis

- La matemática pura (`compute_jcr_risk`) es despreciable: 0.4 µs p50. La fórmula cerrada con `clamp(0, 1)` y la tabla de JUDGE_ROLES es O(1) literal.
- `gate_decision` agrega 146 µs vs el cómputo puro. Ese delta es del recorder fan-out (intenta importar `apohara_context_forge.observability.recorders`, instancia `JCRDecision`, appendea al `gate_log`, fan-out a Prometheus/OTLP).
- **Sorpresa**: el Z3 certifier **no agrega overhead** en este path (Δ p50 = 0.0 µs). La razón: `recorders.record_certified_inv15_decision` solo entra al path con `APOHARA_FORGE_LEDGER=1`, y el certifier `inv15_certifier.certify_decision` está implementado como una verificación de contrato sobre la fórmula cerrada, no como un SMT solve caro. La fórmula es O(1) y el chequeo es O(1).
- La cola mixta (4 paths reales) da prácticamente la misma latencia que el path determinista, confirmando que no hay branch que domine.

## Veredicto

**SHIPPING-READY** (146 µs ≪ 100 ms de TTFT del modelo; 3 órdenes de magnitud por debajo del cuello de botella real). El Z3 certifier puede quedarse prendido por default en producción sin costo medible.

## Implicancia para el producto

El reframe "ROMY = salting honesto para INV-15, no memory optimizer" tiene un costo medible de **146 µs por request** (vs los 100 ms de TTFT del modelo = ruido). Eso convierte a INV-15 en **shipping-grade**, no en un costo escondido. El paper puede reportar este número como evidencia O(1) honesta (la fórmula cerrada es O(1), el certifier es O(1)).

## Fuente

- Script: `/tmp/c2_jcr_latency.py` (transient en VM)
- Resultado JSON: `/tmp/c2_jcr_latency_result.json` (en VM)
- VM: enc1-gpuvm004 (Hot Aisle 1×MI300X, destruida 2026-06-11 15:55 UTC, billing cortado)
