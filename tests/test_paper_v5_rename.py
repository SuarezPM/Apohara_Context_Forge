"""Sprint 6 (AUDIT #31a) — ATOM→ROMY rename regression guard.

The strict rule: the literal string ``"ATOM-"`` (with the hyphen, the
brand pattern that read like "ATOM-Cell"-style compound names) MUST
NOT appear in the spec target paths:

    apohara_context_forge/, demo/, agents/, README.md, CHANGELOG.md

The allowed zones (the rename is intentionally out of scope here, per
the Sprint 6 brief — "Python/docs only … the .tex/.bib rename is out
of scope") are:

    paper/         (the v3.0 LaTeX is preserved for the academic record;
                    the v5.0 companion paper at paper/v5.0/ writes the
                    rename in prose, not as a re-write of the v3.0 source)
    AUDIT.md       (the ledger is intentionally immutable — an entry
                    describes the codebase as it was on that date, and
                    renaming the historical entries would erase the
                    evidence that the collision existed)

This test is a tree-wide grep via :mod:`subprocess` over ``git grep``,
so it is path-aware, gitignore-aware, and the binary ``.cocoindex_code``
index DB is excluded naturally (it is git-ignored). It is the
durable guard that catches a future contributor re-introducing the
``ATOM-`` brand pattern in a new module under
``apohara_context_forge/`` and landing the change without seeing the
regression.

Companion: ``docs/research/reconcile/atomy-to-romy.md`` is the
source-of-truth rename mapping (one row per ATOM concept, including
the "there is no ATOM-Cell / ATOM-Bus concept" negative entry that
forestalls false matches with AMD's ROCm/ATOM engine).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


# Paths where the literal "ATOM-" (with the hyphen) MUST NOT appear.
# Mirrors the Sprint 6 spec — "Python/docs only for this sprint".
FORBIDDEN_TARGETS = (
    "apohara_context_forge/",
    "demo/",
    "agents/",
    "README.md",
    "CHANGELOG.md",
)

# Paths where the literal "ATOM-" is allowed (and documented as such
# in `docs/research/reconcile/atomy-to-romy.md` §3):
#   - `paper/`  → the v3.0 LaTeX source; the v5.0 companion writes the
#                 rename in prose.
#   - `AUDIT.md` → the ledger is intentionally immutable; the rename
#                  is recorded by AUDIT #20 / #31, not by silently
#                  rewriting historical entries.
#   - `tests/`  → this test is allowed to reference the brand pattern
#                 in the FORBIDDEN_TARGETS description and in the
#                 docstring of `_scan()` below.
#   - `docs/`   → the reconciliation doc itself describes the rename
#                 and references historical brand names in prose.
ALLOWED_ZONES = (
    "paper/",
    "AUDIT.md",
    "docs/",
    "tests/test_paper_v5_rename.py",
)


def _git_grep(pattern: str, *paths: str) -> str:
    """Run ``git grep -nE <pattern> -- <paths...>`` from the repo root.

    Returns the combined stdout (empty string if no matches). Stderr
    is suppressed because git grep writes "fatal: bad pattern" to
    stderr on a malformed regex; we want the test to fail with a
    clear pytest assertion, not a noisy stderr dump.
    """
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        "git",
        "grep",
        "-nE",
        pattern,
        "--",
        *paths,
    ]
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    # git grep returns 0 on matches, 1 on no matches, >1 on error.
    # We do not raise — the caller asserts on the output text.
    return result.stdout


def _scan_for_atom_hyphen() -> list[str]:
    """Return the list of repo files (relative paths) that contain
    the literal ``"ATOM-"`` pattern. Used by both positive and
    negative assertions in this module.
    """
    # The literal regex (NOT case-insensitive) — the brand pattern
    # in code is uppercase, and we do not want to over-match prose
    # like "atom-ically" or "atom-by-atom".
    output = _git_grep(r"ATOM-")
    lines = [ln for ln in output.splitlines() if ln.strip()]
    return lines


class TestATOMBrandPatternRemoved:
    """The Sprint 6 spec target paths must be zero-``ATOM-`` (with hyphen)."""

    @pytest.mark.parametrize("target", FORBIDDEN_TARGETS)
    def test_forbidden_target_path_has_no_atom_hyphen(self, target: str) -> None:
        """No ``ATOM-`` (with hyphen) in the rename target paths."""
        output = _git_grep(r"ATOM-", "--", target)
        assert output == "", (
            f"Found forbidden ATOM- (with hyphen) brand pattern in {target!r}.\n"
            f"  Matches:\n{output}\n"
            f"  The rename is a Sprint 6 deliverable (AUDIT #31a).\n"
            f"  See `docs/research/reconcile/atomy-to-romy.md` for the\n"
            f"  full mapping and `tests/test_paper_v5_rename.py` for the\n"
            f"  durable regression guard.\n"
            f"  Allowed zones (where the pattern can stay): {ALLOWED_ZONES}."
        )

    def test_forbidden_target_zero_in_aggregate(self) -> None:
        """Aggregate check — even if a single path is missed by the
        parametrize above, the full-forbidden scan catches it.
        """
        output = _git_grep(r"ATOM-", "--", *FORBIDDEN_TARGETS)
        assert output == "", (
            f"Aggregate ATOM- scan across {FORBIDDEN_TARGETS} returned "
            f"non-empty output:\n{output}"
        )

    def test_legacy_zones_documented_in_reconciliation_doc(self) -> None:
        """The reconciliation doc must exist and reference both the
        rename mapping and the allowed zones, so a future contributor
        can find the source of truth without grep archaeology.
        """
        repo_root = Path(__file__).resolve().parent.parent
        recon = repo_root / "docs" / "research" / "reconcile" / "atomy-to-romy.md"
        assert recon.exists(), (
            f"Reconciliation doc missing: {recon}\n"
            f"The rename source of truth must live in the repo, not\n"
            f"in a comment in this test."
        )
        body = recon.read_text(encoding="utf-8")
        # Both phrases must appear — they are the "allowed zones"
        # entries in the doc.
        assert "AUDIT.md" in body, (
            "atomy-to-romy.md must explicitly call out AUDIT.md as an\n"
            "allowed zone (immutable ledger) so the rationale is on the\n"
            "record, not in this test's docstring alone."
        )
        assert "paper/" in body, (
            "atomy-to-romy.md must explicitly call out `paper/` as an\n"
            "allowed zone (v3.0 LaTeX preserved for the academic record)."
        )

    def test_paper_v5_directory_exists(self) -> None:
        """The v5.0 paper source must exist; this is the file the
        AUDIT #31b entry references. The test pins the location so
        the rename-mapping doc and the new test cannot drift.
        """
        repo_root = Path(__file__).resolve().parent.parent
        v5 = repo_root / "paper" / "v5.0"
        assert v5.is_dir(), (
            f"paper/v5.0/ must exist as the new companion-paper source dir.\n"
            f"  Missing: {v5}"
        )
        assert (v5 / "paper.md").is_file(), (
            f"paper/v5.0/paper.md must exist (canonical source of the v5.0 paper).\n"
            f"  Missing: {v5 / 'paper.md'}"
        )
        assert (v5 / "Makefile").is_file(), (
            f"paper/v5.0/Makefile must exist (build wrapper).\n"
            f"  Missing: {v5 / 'Makefile'}"
        )
        assert (v5 / "references.bib").is_file(), (
            f"paper/v5.0/references.bib must exist (curated bibliography).\n"
            f"  Missing: {v5 / 'references.bib'}"
        )

    def test_pyproject_paper_url_still_v42_doi(self) -> None:
        """The pyproject.toml `Paper = ...` URL MUST still be the
        v4.2 DOI. The deposit-pending comment annotates the field;
        the URL itself is left at v4.2 until the Zenodo deposit
        returns its record URL. A future contributor who updates
        the URL without a confirmed Zenodo record URL will see
        this test fail.
        """
        repo_root = Path(__file__).resolve().parent.parent
        pyproject = repo_root / "pyproject.toml"
        assert pyproject.is_file()
        body = pyproject.read_text(encoding="utf-8")
        assert "10.5281/zenodo.20412807" in body, (
            "pyproject.toml:113 must still reference the v4.2 Zenodo DOI\n"
            "(10.5281/zenodo.20412807) until the v5.0 deposit returns\n"
            "its record URL. AUDIT #31c tracks the deposit as a one-shot\n"
            "manual step."
        )
        # And the deposit-pending comment must be present, so the
        # block is self-documenting for any future reader.
        assert "v5.0 deposit pending" in body, (
            "pyproject.toml must carry the 'v5.0 deposit pending'\n"
            "annotation above the Paper field — the comment is what\n"
            "tells a future contributor the URL is intentionally stale."
        )
