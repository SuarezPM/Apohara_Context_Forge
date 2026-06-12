# paper/v5.0/ — Apohara 2.0 companion systems paper

> **Status:** v5.0 source, deposit pending (AUDIT #31b, #31c).
> **Canonical source:** `paper.md` (this directory). The `paper.pdf`
> is a build-time artifact; the markdown source is the source of
> truth.

This is the v5.0 companion systems paper for the Apohara 2.0
release. It is a **short paper (5–8 pages)** that documents the
honest narrative from GATE #0 ABANDON to the new three-layer
compression stack thesis. It is **not** a re-write of the full V6.0
paper; the full paper is the v3.0 Zenodo record
([10.5281/zenodo.20114594](https://doi.org/10.5281/zenodo.20114594))
and the v4.2 re-deposit
([10.5281/zenodo.20412807](https://doi.org/10.5281/zenodo.20412807)).

## Files

| File              | Purpose                                                |
|-------------------|--------------------------------------------------------|
| `paper.md`        | Markdown source (canonical).                           |
| `references.bib`  | Curated 5–10 entry bibliography (subset of v4.2 bib).  |
| `Makefile`        | `pandoc` build wrapper; safe to run without `pandoc`.  |
| `paper.pdf`       | Build artifact (created by `make`, not tracked).       |

## Build

```bash
cd paper/v5.0
make           # produces paper.pdf (skips with a notice if pandoc is missing)
```

### Build-time dependencies (system packages, not pip)

- **`pandoc`** ≥ 2.19 (tested with 3.1.13). Install on Arch: `sudo
  pacman -S pandoc`.
- **`texlive-xetex`** (provides `xelatex`). Install on Arch:
  `sudo pacman -S texlive-xetex texlive-fonts-recommended`.

These are **build-time** dependencies; they are intentionally not
listed in `pyproject.toml`. The `scripts/check_honesty.sh` gate
does not require them; `tests/test_paper_v5_rename.py` does not
require them; the canonical paper source is `paper.md` and the
PDF is a convenience artifact.

## Honest scope

- The `paper.md` sections §2, §3, §4, §5, §6, and §7 are all
  honest about what was **measured** in the development environment
  vs. what requires an H100 / MI300X pivot to measure. The
  "skipped" / "n / a" cells in the headline tables are deliberate
  declarations of *what was not measured*, not TODOs.
- The Zenodo deposit for v5.0 is a one-shot manual step (AUDIT
  #31c) that has not been executed at the time of this commit.
  The `Paper = "https://doi.org/10.5281/zenodo.20412807"` field in
  `pyproject.toml:113` is annotated with a "v5.0 deposit pending"
  comment and **deliberately not updated** until the deposit
  returns its record URL. The `tests/test_paper_v5_rename.py`
  regression guard asserts the v4.2 DOI is still referenced.
- The ATOM → ROMY rename (AUDIT #20, #31a) is documented in
  `docs/research/reconcile/atomy-to-romy.md` (in the repo root).
  The legacy paper source under `paper/` (the v3.0 LaTeX) is
  preserved untouched for the academic record.

## See also

- `../inv15_paper.tex` — v3.0 LaTeX source (preserved, not edited
  by the v5.0 sprint).
- `../references.bib` — v3.0/v4.2 full bibliography (23 entries).
- `../../docs/research/reconcile/atomy-to-romy.md` — the rename
  source of truth.
- `../../docs/research/reconcile/romy-2026-06-11.md` — the GATE
  #0 ABANDON reframe.
- `../../AUDIT.md` entries #20, #27a, #28, #29, #30, #31 — the
  per-decision ledger for the v5.0 release.
