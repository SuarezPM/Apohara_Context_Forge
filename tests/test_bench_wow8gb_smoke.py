"""test_bench_wow8gb_smoke.py — Sprint 5 (AUDIT #30).

Smoke test for the orchestrator + YAML loader + Markdown emitter.
The bench is invoked in dry-run mode (no model loads), so the
test exercises the real CLI surface and asserts:

* 3 conditions parsed from the YAML.
* Markdown table has 3 rows × 6+ columns.
* No cell equals "N/A" or "TODO" without a paired ``status`` of
  "skipped" (the honesty contract from the spec).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from apohara_context_forge.benchmarks.apohara2 import bench_wow8gb
from apohara_context_forge.benchmarks.apohara2.bench_wow8gb import (
    DEFAULT_PROMPTS,
    BenchRow,
    emit_markdown_table,
    run_condition,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = (
    REPO_ROOT
    / "apohara_context_forge"
    / "benchmarks"
    / "apohara2"
    / "conditions"
    / "wow8gb.yaml"
)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

class TestYamlLoader:
    def test_yaml_file_exists(self) -> None:
        assert YAML_PATH.is_file(), f"missing {YAML_PATH}"

    def test_yaml_loads_three_conditions(self) -> None:
        conds = bench_wow8gb._load_conditions(YAML_PATH)
        assert len(conds) == 3

    def test_conditions_have_non_empty_labels(self) -> None:
        conds = bench_wow8gb._load_conditions(YAML_PATH)
        for c in conds:
            assert c["label"].strip(), f"empty label for {c.get('id')!r}"

    def test_conditions_have_required_keys(self) -> None:
        conds = bench_wow8gb._load_conditions(YAML_PATH)
        required = {"id", "label", "model", "kv_cache_dtype", "compression", "context"}
        for c in conds:
            missing = required - set(c.keys())
            assert not missing, f"condition {c.get('id')!r} missing keys: {missing}"


# ---------------------------------------------------------------------------
# Markdown emitter
# ---------------------------------------------------------------------------

class TestMarkdownEmission:
    def _rows(self) -> list[BenchRow]:
        return [
            BenchRow(
                id="A", label="lbl-A", model="m-A",
                vram_peak_gb=1.5, tokens_per_sec=42.0, ppl_delta=0.02,
                status="ok",
            ),
            BenchRow(
                id="B", label="lbl-B", model="m-B",
                vram_peak_gb=float("nan"), tokens_per_sec=float("nan"),
                ppl_delta=float("nan"),
                status="skipped",
            ),
            BenchRow(
                id="C", label="lbl-C", model="m-C",
                vram_peak_gb=0.0, tokens_per_sec=0.0, ppl_delta=0.0,
                status="ok",
            ),
        ]

    def test_markdown_has_header_and_separator(self) -> None:
        md = emit_markdown_table(self._rows())
        lines = md.strip().splitlines()
        assert lines[0].startswith("| id")
        assert lines[1].startswith("|---")

    def test_markdown_has_three_rows(self) -> None:
        md = emit_markdown_table(self._rows())
        # header + separator + 3 data rows
        assert len(md.strip().splitlines()) == 5

    def test_markdown_has_at_least_six_columns(self) -> None:
        # Count RAW pipe-separated columns (don't filter empties; the
        # spec's "6+ columns" must hold even when numeric cells are
        # blank, because the table layout doesn't collapse).
        md = emit_markdown_table(self._rows())
        for line in md.strip().splitlines():
            cells = [c for c in line.split("|")]
            # `|` at start and end adds 2, so 7 spec columns + 2 = 9.
            assert len(cells) >= 8, f"line {line!r} has only {len(cells)} raw cells"

    def test_no_na_or_todo_in_cells(self) -> None:
        md = emit_markdown_table(self._rows())
        for line in md.strip().splitlines():
            for cell in (c.strip() for c in line.split("|")):
                if cell in {"N/A", "TODO"}:
                    pytest.fail(
                        f"forbidden cell value {cell!r} in row: {line!r}"
                    )

    def test_skipped_rows_have_empty_cells(self) -> None:
        md = emit_markdown_table(self._rows())
        # Row B (skipped): VRAM peak, tokens/s, ΔPPL cells must be empty.
        rows = [r for r in md.strip().splitlines() if r.startswith("| B |")]
        assert rows, "row B missing"
        # Pipe-split: ['', 'B', 'lbl-B', 'm-B', '', '', '', 'skipped', '']
        cells = [c.strip() for c in rows[0].split("|")]
        # cells[4] = VRAM peak, cells[5] = tokens/s, cells[6] = ΔPPL
        for idx in (4, 5, 6):
            assert cells[idx] == "", (
                f"expected empty cell at index {idx}, got {cells[idx]!r}"
            )
        # And the status must be 'skipped'
        assert cells[7] == "skipped"


# ---------------------------------------------------------------------------
# Dry-run end-to-end (subprocess so the CLI is exercised, not the API)
# ---------------------------------------------------------------------------

class TestDryRunSubprocess:
    def test_dry_run_emits_three_conditions(self, tmp_path) -> None:
        out = tmp_path / "wow8gb_test.md"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "apohara_context_forge" / "benchmarks" / "apohara2" / "bench_wow8gb.py"),
            "--conditions", str(YAML_PATH),
            "--output", str(out),
            "--dry-run",
            "--quiet",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert result.returncode == 0, (
            f"non-zero exit {result.returncode}: stderr={result.stderr!r}"
        )
        assert out.is_file(), f"output file {out} not created"
        text = out.read_text(encoding="utf-8")
        # 3 data rows for A, B, C — one per condition.
        for cid in ("A", "B", "C"):
            assert f"| {cid} |" in text, f"row {cid} missing in {text!r}"
        # Header columns present.
        for col in ("VRAM peak", "tokens/s", "ΔPPL vs uncompressed", "status"):
            assert col in text, f"column {col!r} missing in header"
        # Every numeric cell is empty in dry-run mode.
        body_lines = [
            l for l in text.strip().splitlines() if l.startswith("| ") and not l.startswith("| id")
        ]
        for line in body_lines:
            cells = [c.strip() for c in line.split("|")]
            # cells[4:7] = VRAM / tps / ppl — must be empty in dry-run.
            for cell in cells[4:7]:
                assert cell == "", f"expected empty dry-run cell, got {cell!r} in {line!r}"
            # status must be 'dry-run'
            assert cells[7] == "dry-run", f"status cell {cells[7]!r} != 'dry-run'"

    def test_dry_run_writes_json_sidecar(self, tmp_path) -> None:
        out = tmp_path / "wow8gb_test.md"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "apohara_context_forge" / "benchmarks" / "apohara2" / "bench_wow8gb.py"),
            "--conditions", str(YAML_PATH),
            "--output", str(out),
            "--dry-run",
            "--quiet",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert result.returncode == 0, (
            f"non-zero exit {result.returncode}: stderr={result.stderr!r}"
        )
        sidecar = out.with_suffix(".json")
        assert sidecar.is_file(), f"JSON sidecar {sidecar} missing"
        import json as _json
        payload = _json.loads(sidecar.read_text(encoding="utf-8"))
        assert payload["bench"] == "bench_wow8gb"
        assert payload["audit"] == "#30"
        assert payload["dry_run"] is True
        assert len(payload["rows"]) == 3
        for r in payload["rows"]:
            assert r["status"] == "dry-run"


# ---------------------------------------------------------------------------
# run_condition in skip-mode (no model loads)
# ---------------------------------------------------------------------------

class TestRunConditionSkipped:
    def test_dry_run_returns_dry_run_status(self) -> None:
        cond = {
            "id": "X", "label": "test", "model": "fake/model",
            "kv_cache_dtype": "q4_k_m", "compression": "none", "context": 1024,
        }
        row = run_condition(cond, dry_run=True)
        assert row.status == "dry-run"
        # Numeric cells are NaN sentinels.
        import math
        assert math.isnan(row.vram_peak_gb)
        assert math.isnan(row.tokens_per_sec)
        assert math.isnan(row.ppl_delta)

    def test_missing_model_returns_skipped(self) -> None:
        cond = {
            "id": "X", "label": "test", "model": "definitely/not-a-real-model-xyz",
            "kv_cache_dtype": "q4_k_m", "compression": "none", "context": 1024,
        }
        row = run_condition(cond, n_runs=1)
        assert row.status == "skipped"
        import math
        assert math.isnan(row.vram_peak_gb)


def test_default_prompts_non_empty() -> None:
    assert len(DEFAULT_PROMPTS) >= 1
    for p in DEFAULT_PROMPTS:
        assert p.strip()


def test_id_label_pattern() -> None:
    """A / B / C IDs match the spec's three-condition layout."""
    conds = bench_wow8gb._load_conditions(YAML_PATH)
    ids = sorted(c["id"] for c in conds)
    assert ids == ["A", "B", "C"], f"unexpected ids {ids}"
