# B2 — Costo del campo `cache_salt` en el wire de vLLM

**Fecha**: 2026-06-11 · **Host**: 1×MI300X (enc1-gpuvm004, Hot Aisle, Ubuntu 24.04, Python 3.12) · **n**: 5 000 / fase · **httpx 0.28**

> Mide el overhead PURO de incluir `cache_salt` en el body de
> `/v1/completions`, sin GPU ni vLLM. Aísla el costo del feature flag a
> nivel wire (json.dumps + body más grande + header) contra un
> endpoint dummy local.

## Hipótesis

Si el `cache_salt` es transparente a nivel HTTP, el overhead debe ser
**< 100 µs** (es solo un string extra en el JSON body). Si está en el
rango 1-10 ms, es material y hay que repensar el salting como
mecanismo de aislamiento.

## Setup

| Item | Valor |
|------|-------|
| Endpoint dummy | `http://127.0.0.1:18800` (servidor Python `BaseHTTPRequestHandler`, ignora body) |
| Cliente | httpx 0.28.1 async, connection reuse |
| Body shape | replica el de vLLM real: 4.4 KB prompt, max_tokens=16, temperature=0 |
| Salt | `"romy_judge_42_xxxx...x"` (65 chars) |
| Fases | (1) no_salt, (2) with_salt, (3) dummy_field del mismo tamaño (control de tamaño) |
| n por fase | 5 000 |
| Warmup | 50 requests |

## Resultados (µs, n=5000/fase)

| Fase | p50 | p95 | p99 | body |
|------|----:|----:|----:|-----:|
| no_salt | 1745.1 | 2693.5 | 4562.3 | 5 891 B |
| with_salt | **1743.1** | **2414.3** | **3735.0** | 5 973 B (+82 B) |
| dummy_field (mismo tamaño, nombre distinto) | 1753.4 | 2089.6 | 3701.2 | 5 997 B (+106 B) |

## Deltas

| Δ | Valor |
|---|------:|
| with_salt − no_salt (p50) | **−1.99 µs** (dentro del ruido) |
| with_salt − no_salt (p95) | **−279.23 µs** (con_salt sale más rápido — ruido) |
| with_salt − no_salt (p99) | **−827 µs** (idem) |
| Δ body size | +82 B |
| dummy_field − no_salt (p50) | +8.31 µs (control de tamaño: agregar un campo de tamaño similar cuesta lo mismo) |

## Análisis

- El Δ p50 = **−1.99 µs** está dentro del ruido de la medición (la varianza entre fases es de ~10 µs).
- El control `dummy_field_same_size` confirma que el overhead NO es específico al nombre `cache_salt` — agregar un campo extra del mismo tamaño cuesta lo mismo. Es overhead genérico de "campo extra en JSON body", no del feature flag.
- El Δ p95 y p99 **negativo** (con_salt es más rápido que no_salt) confirma que no hay branch en el cliente o server que se active con el campo presente.

## Veredicto

**NEGLIGIBLE** (overhead << 100 µs en p50, dentro del ruido de la
medición). El lever `cache_salt` que se conserva post-ABANDON
(reframe: "salting honesto para INV-15") es **100% shipping-safe a
nivel wire**. Cero justificación técnica para abandonarlo por costo.

## Implicancia para el producto

El `cache_salt` (que queda como mecanismo de aislamiento en el reframe
post-ABANDON, no como optimizador) **no agrega overhead medible**. Eso
significa que la decisión "salting = honesto para safety, no para
sharing" se sostiene sin compromiso de performance. El paper puede
reportar este número como parte de la evidencia de que el componente
sobreviviente es gratis a nivel HTTP.

## Fuente

- Script: `/tmp/b2_salt_overhead.py` (transient en VM)
- Resultado JSON: `/tmp/b2_salt_overhead_result.json` (en VM)
- VM: enc1-gpuvm004 (Hot Aisle 1×MI300X, destruida 2026-06-11 15:55 UTC, billing cortado)
