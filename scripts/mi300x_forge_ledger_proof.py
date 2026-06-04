#!/usr/bin/env python3
"""FORGE-LEDGER hardware proof (S3) — run on the MI300X VM.

Drives the production JCRSafetyGate.gate_decision over the full 1,210-point
Cartesian sweep (5 roles x 11 candidate counts x 11 reuse rates x 2 layouts)
with APOHARA_FORGE_LEDGER=1, so every decision is Z3-certified and appended to
a SHA-256 hash-chained tamper-evident ledger via the real production code path
(gate -> env flag -> record_certified_inv15_decision -> Ledger.append).

Then: verify the chain (ledger_cli, exit 0), aggregate novel metrics
(per-cert Z3 latency p50/p99/mean, certs/sec, % satisfies_inv15, chain-verify
time, ledger bytes), and run a live tamper demo (flip one byte -> exit 2).

Honesty: every number here is measured at runtime on this host. No fabrication.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

FL_DIR = Path("./fl").resolve()
LEDGER = FL_DIR / "inv15_ledger.jsonl"
TAMPERED = FL_DIR / "tampered.jsonl"
OUT = Path("logs/mi300x_p2_forge_ledger.json")

ROLES = ["planner", "researcher", "executor", "critic", "judge"]  # critic+judge are judge-type
CAND = list(range(0, 11))                       # 11
REUSE = [round(0.1 * i, 2) for i in range(0, 11)]  # 11 -> 0.0 .. 1.0
SHUFFLE = [False, True]                          # 2
EXPECTED = len(ROLES) * len(CAND) * len(REUSE) * len(SHUFFLE)  # 1210


def _pct(vals, q):
    """Linear-interpolated percentile (q in [0,100]) over a list of floats."""
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _setup_environment() -> None:
    # Fresh ledger dir; set env BEFORE any recorder singleton is created.
    if FL_DIR.exists():
        shutil.rmtree(FL_DIR)
    FL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["APOHARA_FORGE_LEDGER"] = "1"
    os.environ["APOHARA_OBSERVABILITY_DIR"] = str(FL_DIR)

    from apohara_context_forge.observability import recorders
    recorders._reset_singletons()  # pick up the env we just set


def _drive_sweep(gate) -> tuple[int, float]:
    t0 = time.perf_counter()
    n = 0
    for role in ROLES:
        for c in CAND:
            for r in REUSE:
                for sh in SHUFFLE:
                    gate.gate_decision(role, c, r, sh)
                    n += 1
    wall_s = time.perf_counter() - t0
    return n, wall_s


def _read_ledger_metrics(ledger_path: Path) -> tuple[list[dict], dict]:
    entries = []
    with ledger_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    # Each ledger record wraps the certificate under "payload"; pull it out
    # (fall back to the record itself for resilience to schema changes).
    payloads = [e.get("payload", e) for e in entries]
    elapsed_ms = [p["elapsed_ms"] for p in payloads if "elapsed_ms" in p]
    satisfies = sum(1 for p in payloads if p.get("satisfies_inv15") is True)
    z3_unsat = sum(1 for p in payloads if p.get("z3_status") == "unsat")
    use_dense_true = sum(1 for p in payloads if p.get("observed_use_dense") is True)
    blocks = sum(1 for p in payloads if p.get("gate_action") == "block")
    z3_version = payloads[0].get("z3_version") if payloads else None

    metrics = {
        "satisfies": satisfies,
        "z3_unsat": z3_unsat,
        "use_dense_true": use_dense_true,
        "blocks": blocks,
        "z3_version": z3_version,
        "elapsed_ms": elapsed_ms,
    }
    return entries, metrics


def _verify_chain(ledger_path: Path) -> tuple[int, float, dict]:
    tv0 = time.perf_counter()
    cli = subprocess.run(
        [sys.executable, "-m", "apohara_context_forge.observability.ledger_cli",
         "verify", str(ledger_path)],
        capture_output=True, text=True,
    )
    verify_cli_s = time.perf_counter() - tv0
    verify_json = json.loads(cli.stdout) if cli.stdout.strip() else {}
    return cli.returncode, verify_cli_s, verify_json


def _run_tamper_demo(ledger_path: Path, tampered_path: Path) -> tuple[int, int, dict]:
    shutil.copy(ledger_path, tampered_path)
    raw = tampered_path.read_bytes()
    # Flip a byte ~60% through the file (inside an entry, not the trailing newline).
    pos = int(len(raw) * 0.6)
    flipped = bytearray(raw)
    flipped[pos] = flipped[pos] ^ 0x01
    tampered_path.write_bytes(bytes(flipped))
    tamper = subprocess.run(
        [sys.executable, "-m", "apohara_context_forge.observability.ledger_cli",
         "verify", str(tampered_path)],
        capture_output=True, text=True,
    )
    tamper_json = json.loads(tamper.stdout) if tamper.stdout.strip() else {}
    return pos, tamper.returncode, tamper_json


def _build_report(
    n: int, wall_s: float, entries: list[dict], metrics: dict,
    cli_returncode: int, verify_cli_s: float, verify_json: dict, ledger_bytes: int,
    pos: int, tamper_returncode: int, tamper_json: dict
) -> dict:
    elapsed_ms = metrics["elapsed_ms"]
    return {
        "artifact": "FORGE-LEDGER MI300X hardware proof (S3)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "z3_version": metrics["z3_version"],
        },
        "sweep": {
            "roles": ROLES, "candidate_counts": CAND, "reuse_rates": REUSE,
            "layout_shuffled": SHUFFLE,
            "expected_points": EXPECTED, "decisions_driven": n,
            "ledger_entries": len(entries),
        },
        "integrity": {
            "cli_verify_exit_code": cli_returncode,         # expect 0
            "cli_verify_result": verify_json,                # {valid,entries,broken_at,head}
            "chain_verify_seconds": round(verify_cli_s, 4),
            "ledger_bytes": ledger_bytes,
            "bytes_per_entry": round(ledger_bytes / max(len(entries), 1), 1),
        },
        "tamper_demo": {
            "flipped_byte_offset": pos,
            "cli_verify_exit_code": tamper_returncode,       # expect 2
            "cli_verify_result": tamper_json,                # broken_at set
        },
        "z3_certification": {
            "all_satisfies_inv15": metrics["satisfies"] == len(entries),
            "satisfies_count": metrics["satisfies"],
            "z3_unsat_count": metrics["z3_unsat"],
            "observed_use_dense_count": metrics["use_dense_true"],
            "gate_block_count": metrics["blocks"],
            "latency_ms": {
                "mean": round(statistics.fmean(elapsed_ms), 4) if elapsed_ms else None,
                "p50": round(_pct(elapsed_ms, 50), 4),
                "p99": round(_pct(elapsed_ms, 99), 4),
                "max": round(max(elapsed_ms), 4) if elapsed_ms else None,
                "min": round(min(elapsed_ms), 4) if elapsed_ms else None,
            },
        },
        "throughput": {
            "end_to_end_wall_seconds": round(wall_s, 4),
            "certs_per_second_end_to_end": round(n / wall_s, 1) if wall_s else None,
        },
    }


def main() -> int:
    _setup_environment()

    from apohara_context_forge.safety.jcr_gate import JCRSafetyGate
    gate = JCRSafetyGate()

    # --- Drive the full sweep through the PRODUCTION gate path -------------
    n, wall_s = _drive_sweep(gate)

    # --- Read back the hash-chained ledger --------------------------------
    entries, metrics = _read_ledger_metrics(LEDGER)

    # --- Chain verification via the production CLI (real exit code) -------
    cli_returncode, verify_cli_s, verify_json = _verify_chain(LEDGER)

    # --- Live tamper demo: flip one byte in a middle entry ---------------
    pos, tamper_returncode, tamper_json = _run_tamper_demo(LEDGER, TAMPERED)

    ledger_bytes = LEDGER.stat().st_size

    report = _build_report(
        n, wall_s, entries, metrics,
        cli_returncode, verify_cli_s, verify_json, ledger_bytes,
        pos, tamper_returncode, tamper_json
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    # Human-readable summary to stdout
    print(json.dumps(report, indent=2))
    ok = (
        len(entries) == EXPECTED
        and cli_returncode == 0
        and verify_json.get("valid") is True
        and metrics["satisfies"] == len(entries)
        and tamper_returncode == 2
        and tamper_json.get("valid") is False
    )
    print(f"\nPROOF_OK={ok}  entries={len(entries)}/{EXPECTED}  "
          f"verify_exit={cli_returncode}  tamper_exit={tamper_returncode}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
