"""Apohara 2.0 benchmark suite.

US-002 skeleton. Real benchmarks land in Phases 2-6:

  - `bench_ann.py`     Phase 2: Turbovec recall parity + RAM
  - `bench_compress.py` Phase 3: LLMLingua-2 PPL delta per variant
  - `bench_kv.py`      Phase 4: TurboQuant-KV VRAM + EM
  - `bench_e2e.py`     Phase 6: full bank test (5 tasks × 5 seeds)

Each script ships as a stub with argparse and `--help` working in Phase 0
so the CLI surface is locked down before the real numerics land.
"""
