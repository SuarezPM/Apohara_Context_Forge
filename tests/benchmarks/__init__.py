"""Tests/benchmarks package.

Micro-benchmarks that are runnable from `pytest` directly (no
subprocess needed for the local CPU path). Each bench is a
pytest-discoverable module that emits a JSON contract to stdout
when run with `python -m pytest path/to/bench.py -v -s`.

Cross-references:
- `tests/benchmarks/romy_vs_turboquant_kv.py` (US-007 / Phase 5
  ROMY reconciliation micro-bench).
- `apohara_context_forge/benchmarks/apohara2/bench_*.py` (the
  Apohara 2.0 bench suite — separate location for the CLI-only
  bench scripts that ship with the package).
"""
