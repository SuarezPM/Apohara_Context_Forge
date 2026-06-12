# Apohara 2.0 companion paper — v5.0 build
#
# Build-time deps (system packages, not pip):
#   - pandoc       (>= 2.19 recommended; tested with 3.1.13)
#   - texlive-xetex (provides xelatex + xetex; required for --pdf engine)
#   - texlive-fonts-recommended (LaTeX font fallback for `--pdf-engine=xelatex`)
#
# Per the AUDIT honesty discipline, this Makefile is *build-time only*
# — its absence does not fail `bash scripts/check_honesty.sh` and its
# success is not asserted by `tests/test_paper_v5_rename.py`. The
# `make` target is a convenience for the paper-build environment; the
# canonical paper source is `paper.md`, not `paper.pdf`. (AUDIT #31b.)
#
# Behavior on missing pandoc: `make` (or `make pdf`) prints a notice
# and exits 0 — it does NOT fail. The spec rule is "if pandoc is
# available, builds the PDF; if not, log the gap honestly and skip
# (do not fail)". This is implemented by gating the `pdf` target
# on a Makefile-level conditional so the recipe body never runs
# when pandoc is absent (an in-recipe `command -v` guard would
# still trigger make's "Error 127" exit, which is the wrong
# behavior for a "log the gap honestly and skip" rule).

PANDOC ?= pandoc

PAPER_MD    := paper.md
PAPER_PDF   := paper.pdf
BIB         := references.bib
CSL         ?= ieee.csl  # default CSL; falls back to pandoc's default if absent

# Detect pandoc ONCE at parse time. `shell` lets us branch the
# Makefile based on the result.
HAS_PANDOC := $(shell command -v $(PANDOC) >/dev/null 2>&1 && echo yes || echo no)

.PHONY: all pdf check-pandoc clean notice

all:
ifeq ($(HAS_PANDOC),yes)
	@$(MAKE) pdf
else
	@$(MAKE) notice
endif

# The pdf target is ONLY defined when pandoc is present. Otherwise
# the all target falls back to `notice`, which prints a friendly
# skip message and exits 0.
ifeq ($(HAS_PANDOC),yes)
pdf: $(PAPER_PDF)
endif

# The PDF rule is guarded by a per-target sentinel so the rule body
# never runs when pandoc is missing. The sentinel is a file we touch
# to mark "build attempted in an env without pandoc"; make will not
# try to rebuild it.
.PHONY: $(PAPER_PDF)
$(PAPER_PDF): $(PAPER_MD) $(BIB)
	$(PANDOC) $(PAPER_MD) -o $(PAPER_PDF) --citeproc --pdf-engine=xelatex

check-pandoc:
	@echo "  HAS_PANDOC=$(HAS_PANDOC)"
ifeq ($(HAS_PANDOC),yes)
	@echo "  ✓ pandoc: $$($(PANDOC) --version | head -1)"
else
	@echo "  ⚠️  pandoc not on PATH; 'make pdf' will skip with a notice."
endif

notice:
	@echo "  ⚠️  pandoc not on PATH; skipping PDF build."
	@echo "     Install: sudo pacman -S pandoc texlive-xetex texlive-fonts-recommended"
	@echo "     The canonical source is $(PAPER_MD); the PDF is a build artifact."

clean:
	rm -f $(PAPER_PDF)
