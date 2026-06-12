# ATOM → ROMY reconciliation

> **Status:** 🟢 Production mapping, source of truth for the Sprint 6
> rename (AUDIT #31a, 2026-06-12).
> **Honest scope:** the rename covers **Python + docs in the in-tree
> `apohara_context_forge/`, `demo/`, `agents/`, `README.md`, and
> `CHANGELOG.md`** surface. The legacy paper source under `paper/`
> (the v3.0-era `inv15_paper.tex` / `references.bib` and the
> `CHANGELOG-paper.md`) is preserved untouched — it is the
> academic-record-of-fact, and any rename there lands in the v5.0
> companion paper (`paper/v5.0/paper.md`) or a v5.1+ revision of the
> Zenodo-bearing artifact, not in a silent commit.

This file is the canonical reference for one question: *"what was the
ATOM called, and what is it called now?"* If a reader sees the bare
`ATOM` identifier anywhere in code or in this repo's tracked docs
outside the allowed zones, it is either (a) a historical audit
citation, (b) a pre-rename artifact, or (c) a bug — open an issue.

## 1. Why the rename happened (one paragraph)

`ATOM` (originally *Anchor-driven Tensor Orchestration for Multi-agent*)
collided with AMD ROCm's [`ROCm/ATOM`](https://github.com/ROCm/ATOM)
engine (*AiTer Optimized Model* — a vLLM acceleration path for
MI300X) in **the same domain** (vLLM-on-MI300X). Two plugins both
named `ATOM` was a recipe for confusion and an implicit (false)
association with AMD's project, so on 2026-05-31 the project was
renamed `ROMY` (*Runtime for Orchestrated Matrix Yields*). The
historical entry is `AUDIT.md` §20. The in-tree rename was completed
in commit `7d…` (pre-Sprint-6, AUDIT #20). The remaining stale
references in this repo's prose and the strict `"ATOM-"` (with
hyphen) brand pattern in code paths are what this sprint cleans up.

## 2. The mapping table

The rename was **identifier-for-identifier and concept-for-concept**.
Where the prior identifier was a Python symbol, the replacement is a
Python symbol in the same module. Where it was a prose brand, the
replacement prose keeps the same role.

| ATOM (was)                              | ROMY (now)                              | Kind         | Where it lives today                                                |
|-----------------------------------------|------------------------------------------|--------------|---------------------------------------------------------------------|
| `ATOM` (plugin brand)                   | `ROMY` (plugin brand)                    | brand name   | `apohara_context_forge/serving/romy_plugin.py`, AUDIT #20           |
| `ATOM` (plugin)                         | `ROMY` (plugin)                          | plugin name  | vLLM entry-point `contextforge_romy`                                |
| `ATOMConfig`                            | `ROMYConfig`                             | Python class | `apohara_context_forge/serving/romy_plugin.py`                      |
| `vLLMAtomPlugin`                        | `vLLMRomyPlugin`                         | Python class | same                                                                |
| `apohara_context_forge/serving/atom_plugin.py` | `apohara_context_forge/serving/romy_plugin.py` | module   | file rename (git history preserves both names)                      |
| `tests/test_atom_plugin.py`             | `tests/test_romy_plugin.py`              | test module  | file rename                                                         |
| `contextforge_atom` (entry-point)       | `contextforge_romy` (entry-point)        | entry-point  | vLLM `vllm.general_plugins` group                                   |
| `atom_plugin_hooks` (dashboard scenario) | `romy_plugin_hooks`                      | scenario id  | `demo/dashboard.py:151`                                             |
| `ATOM Fase 1` (prose)                   | `ROMY Fase 1` (prose)                    | prose        | `agents/pipeline.py:54`                                             |
| `rename ATOM→ROMY` (README prose)       | `ROMY rename completed in code` (prose)  | prose        | `README.md:259`                                                     |
| `vLLM-ATOM` (paper v3.0 prose)          | *(preserved, see §3)*                    | academic ref | `paper/inv15_paper.tex`, `paper/references.bib`                     |
| `AMD ROCm/ATOM` (external product)      | *unchanged* — it's AMD's product, not ours | external    | n/a                                                                 |

> **There is no `ATOM-Cell` / `ATOM-Bus` / `ATOM-MMU` concept in this
> project.** The mechanical-cell vocabulary is from AMD's ROCm/ATOM
> engine (a *different* project, see §1). If a future reader sees those
> terms in this codebase, they are not from us. This row exists to
> forestall false matches.

## 3. What is **not** renamed in this sprint

Per the Sprint 6 brief ("Python/docs only … the .tex/.bib rename is
out of scope"), the following are intentionally **left alone** and
the divergence is documented here so a future reader does not mistake
it for a missed rename:

1. **`paper/inv15_paper.tex`, `paper/references.bib`,
   `paper/README.md`, `paper/zenodo-v3-metadata.json`** — the v3.0
   Zenodo-bearing artifact. The Zenodo DOI for the published v4.2
   paper (`10.5281/zenodo.20412807`) cites the v3.0 source by
   convention; a silent rename of the .tex/.bib would break
   reproducibility without re-depositing. The v5.0 companion
   (`paper/v5.0/paper.md`) writes the rename in prose, but does not
   rewrite the v3.0 source.
2. **`AUDIT.md` historical entries (sections #18, #19, #20, etc.)** —
   the audit ledger is intentionally immutable: an entry describes
   the codebase **as it was on that date**, including its identifier
   choices. Renaming in place would erase the evidence that the
   collision existed and that the rename fixed it. Historical
   citations stay verbatim.
3. **`CHANGELOG-paper.md`** — the paper-changelog mirrors the
   Zenodo record. It is preserved untouched for the same reason
   `paper/` is.
4. **`logs/mi300x_full_pytest_*.json`** — captured benchmark
   outputs from the rename date; they are immutable evidence.

The list above is exhaustive for this sprint. A future `paper/v5.1/`
revision (post-Zenodo re-deposit) is the right home for renaming
items (1) and (3) in a single editorial pass.

## 4. The regression guard

`tests/test_paper_v5_rename.py` is the durable guard. It asserts:

- **Zero** occurrences of the literal string `"ATOM-"` (with the
  hyphen, the brand pattern that read like a "ATOM-Cell"-style
  compound) in the rename target paths:
  `apohara_context_forge/`, `demo/`, `agents/`, `README.md`,
  `CHANGELOG.md`.
- The historical zones are **explicitly allowed**: `paper/` (any
  version) and `AUDIT.md` carry the rename in prose via §1, §2, and
  §3 above, and via the `AUDIT.md` historical entries themselves.
- The `.cocoindex_code/` local index DB is excluded (binary, not
  source).

The test catches the kind of regression the spec was designed to
catch: a future contributor reintroducing the `ATOM-` brand pattern
in a new module under `apohara_context_forge/` will see the test
fail on the next `pytest` run.

## 5. Versioning

- **AUDIT #20** (2026-05-31): the original rename in code, including
  the file rename `atom_plugin.py` → `romy_plugin.py` and the
  entry-point correction. `apohara_context_forge/serving/romy_plugin.py`
  shipped.
- **AUDIT #31a** (2026-06-12, this sprint): the spec target paths
  (`apohara_context_forge/`, `demo/`, `agents/`, `README.md`,
  `CHANGELOG.md`) are now zero-`ATOM-`. The 3 stale prose references
  identified in the spec audit (`README.md:259`,
  `agents/pipeline.py:54`, `demo/dashboard.py:151`) are renamed.
  The reconciliation mapping above is the durable reference.
