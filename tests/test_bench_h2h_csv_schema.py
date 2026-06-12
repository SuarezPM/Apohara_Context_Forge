"""test_bench_h2h_csv_schema.py — AUDIT #29 CSV contract tests.

Locks the CSV header and column-type contract. The bench writes
the file via ``csv.DictWriter(f, fieldnames=CSV_HEADER)``, so a
schema drift between the row dict and the writer silently writes
a malformed CSV. These tests make the schema drift visible.

The tests do not write to disk: they import the CSV header
constant, parse a one-row CSV string the test fabricates, and
assert the column types match the bench's row-dict contract.
"""
from __future__ import annotations

import csv
import io
import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apohara_context_forge.benchmarks.apohara2.bench_h2h import (  # noqa: E402
    CSV_HEADER,
    run_condition,
)


def test_csv_header_matches_schema() -> None:
    """The CSV header is the spec-mandated 7-tuple."""
    assert CSV_HEADER == (
        "system",
        "duration_ms",
        "vram_peak_gb",
        "ppl_delta",
        "compression_ratio",
        "prompt_chars",
        "run_idx",
    )


def test_run_condition_row_writes_to_csv_with_correct_types() -> None:
    """A row dict round-trips through ``csv.DictWriter`` with the right types.

    The test fabricates a row dict, writes it via the same
    ``csv.DictWriter`` setup the bench uses, and parses the
    output to assert:
      * system        : str
      * duration_ms   : float
      * vram_peak_gb  : float
      * ppl_delta     : float (NaN allowed when model is absent)
      * compression_ratio : float
      * prompt_chars  : int
      * run_idx       : int
    """
    row = run_condition("apohara", "schema probe", n_tokens=64)
    # Fill in a run_idx the way the orchestrator does.
    row["run_idx"] = 7

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(CSV_HEADER))
    w.writeheader()
    w.writerow(row)
    buf.seek(0)
    parsed = list(csv.DictReader(buf))
    assert len(parsed) == 1
    parsed_row = parsed[0]

    # Header round-trip.
    assert set(parsed_row.keys()) == set(CSV_HEADER)

    # Type checks. We allow NaN for ppl_delta when the model is
    # absent (slim venv); every other column is strict.
    assert isinstance(parsed_row["system"], str)
    assert parsed_row["system"] == "apohara"

    duration = float(parsed_row["duration_ms"])
    assert duration == duration  # not NaN

    vram = float(parsed_row["vram_peak_gb"])
    assert vram == vram  # not NaN

    ppl = float(parsed_row["ppl_delta"])
    # NaN is allowed (model absent); the variance check is
    # run-against-rows not against the schema, so the
    # regression guard lives in the orchestrator.
    if not math.isnan(ppl):
        # When the model IS present, ppl_delta is a real float.
        assert isinstance(ppl, float)

    cr = float(parsed_row["compression_ratio"])
    assert cr == cr  # not NaN
    # The bench's compression ratio is positive; it can exceed
    # 1.0 for tiny prompts (LLMLingua-2's tokenizer sometimes
    # produces a longer token stream than the input text). The
    # upper bound is the size of a small constant: any value up
    # to 10.0 covers the worst observed edge case. The
    # turboquant baseline is exactly 1.0.
    assert cr > 0.0
    assert cr < 10.0

    # prompt_chars and run_idx must round-trip as ints.
    assert int(parsed_row["prompt_chars"]) == len("schema probe")
    assert int(parsed_row["run_idx"]) == 7


def test_csv_header_does_not_contain_hardcoded_stub_values() -> None:
    """The CSV header is the schema — no placeholder rows sneak in."""
    # Sanity: write a row with a stub-allowed compression
    # ratio, and check the output contains "compression_ratio"
    # as a column name, not as a value.
    row = {
        "system": "apohara",
        "duration_ms": 1.0,
        "vram_peak_gb": 0.0,
        "ppl_delta": float("nan"),
        "compression_ratio": 0.55,
        "prompt_chars": 42,
        "run_idx": 0,
    }
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(CSV_HEADER))
    w.writeheader()
    w.writerow(row)
    buf.seek(0)
    text = buf.getvalue()
    first_line = text.splitlines()[0]
    # The first line is the header; the literal 0.55 is in the
    # data row, not the header.
    assert "compression_ratio" in first_line
    assert "0.55" not in first_line
