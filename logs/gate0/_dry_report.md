<!--
GATE-0-report-TEMPLATE.md  —  esqueleto del entregable §9 del GATE #0.

Este archivo es una PLANTILLA. NO contiene números reales: cada hueco está marcado con
{{placeholder}} (lo rellena `analyze.render_report_md` desde el log crudo) o con
[PENDIENTE: ...] (decisión/condición que aún no existe porque el experimento no corrió).

REGLAS DE LLENADO (honestidad = la marca — AUDIT.md):
  - Ningún número se escribe a mano aquí. Todo valor numérico viene del log crudo
    (logs/gate0/<workload>_<topology>.json, esquema en GATE-0-Interface-Contract §9) vía
    `analyze.compute_verdict` + `analyze.render_report_md`.
  - Si una métrica es `null` / no-aislable en A o B, NO se calcula delta: se reporta
    "no aislable" y el veredicto cae a INDECISIVE.
  - El número decisivo es delta = (B − A). NUNCA (B − C). (B − C) es solo la fila de
    sanidad del harness (confirma que medimos *sharing*, no APC sobre el mismo prompt).
  - Toda métrica entra con su condición (bloque §8) o no entra.
  - VRAM solo a través de lecturas honestas; jamás 45.0 / 192.0 ni vram_source en
    {amd_default_192gb, cuda_unavailable, unknown, dry}.

Mapeo plantilla → API de analyze.py (GATE-0-Interface-Contract §10):
  Verdict.{primary_metric,a,b,c,delta_b_minus_a,delta_pct,delta_ci_pct,cut,rationale,quotable}
  ArmMetric.{arm,metric,value,ci,condition,valid}
  CI.{mean,lo,hi,n,method,confidence}
  GateRunResult.{schema_version,timestamp_utc,topology,workload,reuse,conditions,arms,validity,measured}
-->

# GATE #0 — Informe del experimento decisivo (A/B/C)

> **Estado de este informe:** NO citable (dry/validez) `[PENDIENTE: el experimento aún no corrió en MI300X — todo número es placeholder]`
> **Generado por:** `analyze.py --primary kv_used_gb` · **Fecha de generación:** 2026-06-11T14:30:49Z
> **Log(s) crudo(s) fuente:** logs/gate0/_dry_single.json
> **schema_version del log:** 0.1.0 · **measured:** false

---

## 0. La pregunta (una sola, inmutable)

> Dado un workload multi-agente realista con prefijos compartidos, ¿el sharing cross-agent de
> ROMY (`cache_salt` → prefijo byte-idéntico → vLLM APC, + LMCache cross-worker) reduce
> VRAM/KV-footprint o mejora throughput/TTFT **por encima** del prefix-caching nativo de vLLM
> (APC, ya encendido) en MI300X, en un margen que justifique mantenerlo como mecanismo propio?

El número que decide es **`delta = (B − A)`**, con **A = APC ON, sin salt ROMY**. Nunca `(B − C)`,
nunca `(B − sin-cache)`.

## 1. Criterio de corte preregistrado (copiado de GATE-0-protocol.md §2 — NO se modifica tras ver datos)

| Banda de `delta` (B−A, %) | Decisión |
|---|---|
| `delta < 5 %` | **ABANDONAR** el sharing mecánico. ROMY se reduce a "salting honesto INV-15 en el serving" (aislar jueces), no optimizador de memoria. Pivotar 100 % a la capa de safety O(1) + compresión de tokens. |
| `5 % ≤ delta ≤ 15 %` | **ZONA GRIS.** Mantener solo si el costo de mantenimiento es bajo y no compite con upstream que lo cierra (vLLM #26201). Re-evaluar contra el roadmap vLLM/SGLang. |
| `delta > 15 %` sostenido y reproducible | **INVERTIR.** ROMY tiene diferencial; pasar a Fase 2. |

> *Preregistrado el 2026-05-31. Esta tabla es inmutable. Si el resultado incomoda, gana la tabla.*

---

## 2. Condiciones del experimento (bloque §8 — toda métrica de abajo se lee bajo esto)

> Sin condición no hay métrica (disciplina AUDIT.md). Estos valores se copian de
> `GateRunResult.conditions`.

| Campo | Valor |
|---|---|
| `model` | qwen3-32b |
| `hardware_label` | unknown (dry) |
| `vram_source` (primaria) | dry |
| `second_source` (secundaria VRAM) | _pendiente_ |
| `topology` | single_worker |
| `n_agents` | 5 |
| `n_requests` | 320 |
| `concurrency` | 32 |
| `max_tokens` | 64 |
| `approx_prefix_tokens` | 517 |
| `shared_prefix_fraction` | 1.0 |
| `block_size` | 16 |
| `kv_cache_dtype` | auto |
| `max_model_len` | 16384 |
| `gpu_memory_utilization` | 0.9 |
| `aiter_applied` | true |
| `pythonhashseed` | 0 |

**Workload** (`GateRunResult.workload`): `sprint5_5agent` · prefijo canónico hash
`248523bdb03c25ff1d3cbe18ea2c9a82b9c87b5502295beabce1c870579a9e64` · agentes `retriever, reranker, summarizer, critic, responder`.

**Reuse REAL del workload medido** (`GateRunResult.reuse` — condición, nunca asumida):

| `canonical_prefix_chars` | `n_distinct_prefixes` | `shared_prefix_fraction` | `approx_prefix_tokens` | `note` |
|---|---|---|---|---|
| 2070 | 1 | 1.0 | 517 | char/4 heuristic (no tokenizer wired) |

> Para que el gate sea válido, `reuse.n_distinct_prefixes` sobre el prefijo (excluyendo colas)
> debe ser **1**. Si es >1, el workload está roto y `validity.check_shared_prefix_single` lo
> marca → veredicto INDECISIVE. `[PENDIENTE: confirmar con el log real]`

---

## 3. Tabla A/B/C — métrica primaria

> Cada celda: `valor ± CI` (`CI.lo`..`CI.hi`, método `CI.method`, n=`CI.n`, conf=`CI.confidence`).
> `valor` y los extremos del CI provienen EXCLUSIVAMENTE del log crudo. `valid=false` ⇒ la fila
> no es citable y arrastra el veredicto a INDECISIVE.

### 3.1 Primaria 1 — KV-cache / VRAM footprint efectivo (GB) — `metric = kv_used_gb`

| Brazo | `kv_used_gb` ± CI | `method` (aislamiento) | `vram_source` | `valid` |
|---|---|---|---|---|
| **A** — APC ON, sin ROMY | _pendiente_ ± [_pendiente_, _pendiente_] (n=0, none(n<2)) | _pendiente_ | _pendiente_ | false |
| **B** — ROMY (salt compartido + LMCache cross) | _pendiente_ ± [_pendiente_, _pendiente_] (n=0, none(n<2)) | _pendiente_ | _pendiente_ | false |
| **C** — control negativo (salts aislados) | _pendiente_ ± [_pendiente_, _pendiente_] (n=0, none(n<2)) | _pendiente_ | _pendiente_ | false |

> **Caveat de aislamiento (obligatorio):** si `method = hbm_device_wide(NOT_ISOLATED)`, la cifra
> es HBM device-wide y **no aísla el KV**. La corrida previa Qwen3-32B mostró 175.393 GB idénticos
> shared vs isolated: a holgura, el HBM device-wide diluye el delta de KV a ~0. En ese caso
> `kv_used_gb` es `null` y NO se computa delta sobre esta métrica.
> `[PENDIENTE: verificar `method` real en el log; si NOT_ISOLATED, el veredicto KV es "no aislable"]`

### 3.1.b Primaria 1 (robusta a pre-alocación) — Utilización del pool KV — `metric = kv_cache_util` (`gpu_cache_usage_perc`)

> **Métrica KV recomendada en vLLM.** vLLM **pre-aloca** el pool KV al arranque (`gpu_memory_utilization`),
> así que el HBM device-wide (`kv_used_gb`, §3.1) no varía con el KV real y su delta se diluye a ~0. La
> fracción del pool efectivamente en uso (`gpu_cache_usage_perc`, leída de `/metrics`) SÍ baja cuando ROMY
> comparte prefijos al mismo workload — esa es la señal de sharing, visible bajo pre-alocación. Lower is
> better. No depende del VRAM monitor (no aplica la honestidad de `vram_source`).
> `[PENDIENTE: 1ª acción en la VM — confirmar el nombre real del gauge con `curl /metrics | grep -i cache`]`

| Brazo | `gpu_cache_usage_perc` (frac. del pool) | `valid` |
|---|---|---|
| **A** — APC ON, sin ROMY | _pendiente_ | false |
| **B** — ROMY (salt compartido + LMCache cross) | _pendiente_ | false |
| **C** — control negativo (salts aislados) | _pendiente_ | false |

> El veredicto sobre esta primaria se corre con `analyze.py --primary kv_cache_util`. El delta `(B − A)`
> y el corte (§5/§6) se calculan igual; lower-is-better invierte el signo automáticamente.

### 3.2 Primaria 2 — Throughput agregado (tokens/s) a concurrencia fija — `metric = decode_tok_s`

| Brazo | `decode_tok_s` ± CI | `n_requests` | `valid` |
|---|---|---|---|
| **A** | _pendiente_ ± [_pendiente_, _pendiente_] (n=0, none(n<2)) | _pendiente_ | false |
| **B** | _pendiente_ ± [_pendiente_, _pendiente_] (n=0, none(n<2)) | _pendiente_ | false |
| **C** | _pendiente_ ± [_pendiente_, _pendiente_] (n=0, none(n<2)) | _pendiente_ | false |

---

## 4. Tabla A/B/C — métricas secundarias (contexto, NO deciden el corte)

### 4.1 TTFT con prefijo compartido (s) — `mean / p50 / p95`

| Brazo | `mean_ttft_s` ± CI | `p50_ttft_s` | `p95_ttft_s` | `n_requests` |
|---|---|---|---|---|
| **A** | _pendiente_ ± [_pendiente_, _pendiente_] | _pendiente_ | _pendiente_ | _pendiente_ |
| **B** | _pendiente_ ± [_pendiente_, _pendiente_] | _pendiente_ | _pendiente_ | _pendiente_ |
| **C** | _pendiente_ ± [_pendiente_, _pendiente_] | _pendiente_ | _pendiente_ | _pendiente_ |

### 4.2 Prefix-cache hit rate — `metric = hit_rate` (+ externos cross-worker)

| Brazo | `hit_rate` | `queries_delta` | `hits_delta` | `external_hits_delta` | `external_kv_tokens_delta` |
|---|---|---|---|---|---|
| **A** | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ |
| **B** | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ |
| **C** | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ |

> **INV-15 (jueces aislados):** `inv15_fires` por brazo (requests de juez que tomaron el path
> aislado, leído de los salts): A=0 · B=64 ·
> C=0.

---

## 5. Delta ROMY = (B − A) — el número que decide

> Computado por `analyze.compute_verdict` como `delta_pct = (B − A) / A · 100` sobre la métrica
> primaria elegida (`kv_used_gb`), con su CI bootstrap. **Jamás sobre (B − C).**

| Magnitud | Valor |
|---|---|
| Métrica primaria del veredicto | kv_used_gb |
| A (`value` ± CI) | _pendiente_ ± [_pendiente_, _pendiente_] |
| B (`value` ± CI) | _pendiente_ ± [_pendiente_, _pendiente_] |
| **Delta absoluto** (`delta_b_minus_a`) | _no computable_ |
| **Delta porcentual** (`delta_pct`) | _no computable_ % |
| **CI del delta %** (`delta_ci_pct`) | [_pendiente_, _pendiente_] (n=_pendiente_, _pendiente_) |

> Si `verdict.delta_pct` es `null` (métrica primaria no aislable en A o B, p. ej. KV
> device-wide), NO hay delta y el veredicto es INDECISIVE.

### 5.1 Fila de control del harness (sanidad, NO decisión): C

> Esto NO es el delta del gate. Solo confirma que el harness mide *sharing* y no APC sobre el
> mismo prompt. C debe dar hit_rate ≈ 0.

| Sanidad | Valor | Esperado |
|---|---|---|
| `hit_rate` de C | _pendiente_ | ≈ 0 (control negativo válido) |
| `(B − C)` (solo referencia) | _pendiente_ | grande ⇒ el harness *puede* ver sharing |

---

## 6. Veredicto contra el corte preregistrado

> `analyze.compute_verdict` aplica la tabla del §1 sobre `delta_pct` y emite `cut ∈
> {ABANDON, GREY_ZONE, INVEST, INDECISIVE}`. INDECISIVE si la métrica primaria es null/no-aislable
> o si `validity.quotable == false`.

**VEREDICTO:** `INDECISIVE`  `[PENDIENTE: lo fija analyze.py con datos reales]`

**Racional** (`verdict.rationale`):

> INDECISIVE: la corrida no es citable (validity.quotable=False); no se computa delta hasta que pasen los gates requeridos.

**¿Citable?** `quotable = false` (espeja `ValidityReport.quotable`; si es `false`,
el `cut` es INDECISIVE por construcción, sin importar el número).

---

## 7. Decisión y enlace a Fase 1

**Decisión tomada:** INDECISO: no hay número citable para decidir. Resolver la causa de no-citabilidad (gates de validez requeridos, o aislar la métrica primaria) y re-ejecutar antes de cualquier veredicto. El gate permanece abierto; no avanza a Fase 1. `[PENDIENTE: redactar la acción concreta según el veredicto]`

Mapeo veredicto → acción (de PLAN-DEFINITIVO-pivote-2026.md):

- **ABANDON** → ejecutar el **Plan B** del protocolo §8: ROMY pasa a "INV-15 en el serving"
  (salting para aislar jueces, safety O(1)), **no** memory optimizer. Foco 100 % en palancas
  durables (compresión de tokens + safety del reuso INV-15 + hardening ROCm/MI300X). Actualizar
  README/paper: el sharing queda "cubierto por APC/LMCache nativos; ROMY aporta el contrato de
  aislamiento, no el ahorro".
- **GREY_ZONE** → mantener ROMY solo si el costo de mantenimiento es bajo y no compite con
  upstream (vLLM #26201). Re-evaluar contra el roadmap vLLM/SGLang antes de invertir.
- **INVEST** → ROMY tiene diferencial: pasar a **Fase 2** del plan (INV-15 hybrid extension +
  safety del reuso de estado, frente 1).
- **INDECISIVE** → NO decidir. Corregir la causa de no-citabilidad (ver §8) y re-correr; el gate
  permanece abierto.

**Este informe alimenta la Fase 1 del PLAN-DEFINITIVO** (`docs/research/_internal/PLAN-DEFINITIVO-pivote-2026.md`
→ "### FASE 1 — Cosecha + prueba causal del paper"), que asume el lever ya validado/descartado por
este gate.

---

## 8. Validez (lo que arruina el experimento — protocolo §6)

> De `GateRunResult.validity` (`ValidityReport`). Una sola `required` que falle ⇒ el informe NO es
> citable ⇒ veredicto INDECISIVE. Cada fila: `passed`, `required`, `detail` (evidencia), evidencia
> machine-readable.

| Check (`name`) | `passed` | `required` | `detail` (evidencia) |
|---|---|---|---|
| `apc_on` — APC ON en A/B/C (grep `enable_prefix_caching=True` en server log de CADA brazo) | ❌ | sí | no APC server logs supplied: APC-ON could not be verified for any arm. |
| `c_control_zero` — C da hit_rate ≈ 0 (mide sharing, no APC mismo-prompt) | ❌ | sí | arm C prefix metrics not supplied: negative control could not be verified. |
| `aiter_parity` — AITER env idéntico en los 3 brazos | ✅ | sí | AITER env identical across ['A', 'B', 'C'] (aiter_applied=True). |
| `seed_pinned` — `PYTHONHASHSEED=0` en cada env (mandatorio cross-worker) | ✅ | no | PYTHONHASHSEED='0' pinned on every arm ['A', 'B', 'C']. |
| `shared_prefix_single` — `reuse.n_distinct_prefixes == 1` sobre prefijos | ✅ | sí | workload collapses to 1 distinct prefix — shared-prefix assumption holds. |
| `vram_source_honest` — backend real (pyrsmi/drm_sysfs/cuda_nvml/cuda_nvidia_smi), jamás 192GB default | _pendiente_ | _pendiente_ | _pendiente_ |
| `n_requests_sufficient` — N grande para CI estrecho (NO ~28) | ✅ | sí | n_requests=320 >= floor 200 — CI can be tight. |

**Resumen de validez** (`ValidityReport.summary`): NOT quotable — failed required checks: apc_on, c_control_zero

**Sub-bloque de validez exigido por el spec (confirmaciones explícitas):**

- **APC ON confirmado:** no APC server logs supplied: APC-ON could not be verified for any arm. — los 3 brazos arrancan con
  `--enable-prefix-caching`; A es "APC-ON-sin-ROMY", **no** `--no-enable-prefix-caching` (ese
  confound está prohibido por §6). `[PENDIENTE: grep real en los 3 server logs]`
- **C ≈ 0 %:** `hit_rate` de C = _pendiente_ ≤ umbral _pendiente_.
  `[PENDIENTE: confirmar con el log]`
- **VRAM real:** `vram_source = dry` (+ segunda fuente
  `_pendiente_`); ninguna lectura con source en
  {amd_default_192gb, cuda_unavailable, unknown, dry}. `[PENDIENTE: confirmar honestidad de fuente]`
- **N + CI:** `n_requests = 320`; ancho de CI del delta primario =
  [_pendiente_, _pendiente_]. `[PENDIENTE: confirmar N suficiente y CI estrecho]`

---

## 9. Logs crudos y trazabilidad (disciplina AUDIT — ninguna cifra sin fuente)

- **Log(s) crudo(s) commiteados:** logs/gate0/_dry_single.json `[PENDIENTE: rutas reales, p. ej. logs/gate0/sprint5_5agent_single_worker.json]`
- **Veredicto JSON:** logs/gate0/_dry_verdict.json `[PENDIENTE: logs/gate0/verdict.json]`
- **Server logs por brazo** (para grep APC): A=/home/thelinconx/Documentos/Apohara_Context_Forge/logs/gate0/sprint5_5agent_single_worker_A_server.log · B=/home/thelinconx/Documentos/Apohara_Context_Forge/logs/gate0/sprint5_5agent_single_worker_B_server.log · C=/home/thelinconx/Documentos/Apohara_Context_Forge/logs/gate0/sprint5_5agent_single_worker_C_server.log
- **schema_version:** 0.1.0 · **measured:** false (si `measured=false`, este
  informe es de plumbing/dry y **no es citable**).
- **Nota AUDIT:** ninguna cifra de este informe se escribió a mano. Si una celda muestra `null`/`None`,
  la métrica no fue medible y el delta no se computó sobre ella. El número decisivo es siempre
  `(B − A)`; `(B − C)` aparece solo como sanidad del harness.

<!--
CHECKLIST de cierre (para `analyze.render_report_md` / revisión humana — borrar al publicar):
  [ ] §1 corte preregistrado intacto (no editado tras ver datos).
  [ ] §2 todas las condiciones presentes; reuse.n_distinct_prefixes == 1.
  [ ] §3 métrica primaria: si method = NOT_ISOLATED ⇒ kv_used_gb = null ⇒ sin delta KV.
  [ ] §5 delta_pct = (B−A)/A·100 con CI; NUNCA (B−C) como decisión.
  [ ] §6 cut ∈ {ABANDON,GREY_ZONE,INVEST,INDECISIVE}; INDECISIVE si quotable=false.
  [ ] §8 toda required passed; si no, cut = INDECISIVE.
  [ ] §9 measured=true y rutas de log crudo commiteadas.
  [ ] cero números hardcodeados; cero {{placeholder}} sin resolver; [PENDIENTE] solo donde
      la decisión humana aún no existe.
-->
