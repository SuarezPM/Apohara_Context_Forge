# MI300X baseline de referencia — APC nativo (vLLM V1) en Qwen3-32B

**Fecha**: 2026-06-11 · **Topología**: single-worker · **Citable**: sí

> Tabla de referencia publicable. Cada número con su condición (modelo, hardware,
> seq lens, concurrencia, Nº de requests). Sin condición, no entra al informe
> (disciplina AUDIT.md).

## Setup

| Item | Valor |
|------|-------|
| Hardware | 1× AMD MI300X (192 GiB HBM, gfx942) |
| Driver | ROCm 7.2.4 (image default Hot Aisle) |
| Modelo | Qwen/Qwen3-32B (dense, full-attention) |
| Servidor | vLLM V1 con AITER prebuilt (`aiter_applied=true`) |
| KV dtype | auto |
| Block size | 16 |
| max-model-len | 16 384 |
| gpu-memory-utilization | 0.90 |
| PYTHONHASHSEED | 0 |
| Workload | 5 agentes (retriever, reranker, summarizer, critic, responder) con prefijo de sistema/contexto compartido |
| Nº de requests | 320 |
| Concurrencia | 32 |
| max_tokens/req | 64 |
| Prefijo compartido (chars) | 2 070 (~517 tokens, heurística char/4) |
| Fracción de prefijo compartido | 1.00 (todos los reqs comparten el prefijo canónico) |

## Resultado (Brazo A — APC nativo, baseline)

| Métrica | Valor | Notas |
|---------|-------|-------|
| **Throughput decode** | **2 365.3 tok/s** | decode_tok_s |
| **TTFT medio** | **108.4 ms** | mean_ttft_s |
| **TTFT p50** | **104.0 ms** | p50_ttft_s |
| **TTFT p95** | **207.7 ms** | p95_ttft_s |
| HBM used (post-load, pre-traffic) | 175.39 GiB | pyrsmi + rocm-smi (acuerdo 1ª vs 2ª fuente) |
| HBM total | 191.69 GiB | nominal device |
| Model weights baseline | 175.25 GiB | medido post-load, pre-traffic |
| KV footprint (upper bound) | 0.143 GiB | HBM − weights; incluye slack activación/fragmentación |
| prefix_cache_hit_rate | 0.972 | APC captura el 97.2 % de los prefijos compartidos sin esfuerzo |
| external_prefix_cache_hits | n/a | single-worker (sin LMCache cross-worker) |

## Fuente

Log crudo (31 KB): `logs/gate0/sprint5_5agent_single_worker.json` · timestamp 2026-06-11T15:16:14Z · commit HEAD en `main` (SuarezPM/Apohara_Context_Forge).

## Notas para citación

- **Limitación 1 — concurrencia**: 32 es razonable pero no es la saturación. La curva completa (1/8/32/64/128) es trabajo futuro (D2 en el backlog).
- **Limitación 2 — workload sintético**: el prefijo compartido es 100 % idéntico. En workloads reales la fracción de sharing es menor; el hit rate observado puede ser techo.
- **Limitación 3 — single-worker**: no captura `external_prefix_cache_hits` de LMCache. Para eso ver A1 (cross-worker), backlog.
- **Limitación 4 — modelo único**: solo Qwen3-32B. Otras arquitecturas (GQA, MoE) pueden dar números distintos.

## Uso

- **Benchmark de la comunidad ROCm**: este es uno de los pocos reportes honestos de MI300X 1× con vLLM V1. La mayoría de las publicaciones no declaran AITER, no usan 2ª fuente de VRAM, o confunden `mean` con `p50` de TTFT.
- **Piso de comparación**: cualquier afirmación de "X % speedup sobre APC" debe medirse contra estos números, no contra "sin caching" (GATE-0-protocol §4).

## Cambio respecto al `logs/gate0/sprint5_5agent_single_worker.json`

- `gpu_cache_usage_perc` quedó `null` en el log crudo (bug menor del reader). El cálculo alternativo (HBM − weights) es el que se reporta acá como upper bound honesto.
- El reporte `GATE-0-report-2026-06-11-kvutil.md` da INDECISIVE por este mismo gap. Workaround: re-correr con el reader arreglado (A3) o usar HBM − weights como proxy (lo que se hace acá).
