#!/usr/bin/env python3
"""Regenerate every table and figure the manuscript includes.

Reads stored data only. No lattice reduction and no point counting are
repeated. Run this and then `git diff --exit-code` to confirm that the
committed output matches what the current code produces.
"""

from __future__ import annotations

import subprocess
import sys

STEPS = [
    ("profile fits and discrepancy tables", ["bkz_experiment.py", "analyze"]),
    ("point-count table", ["node_count_reporting.py"]),
    ("fitting-objective comparison", ["compare_objectives.py"]),
    ("figures", ["generate_revision_figures.py"]),
    ("summation precision audit", ["precision_audit.py"]),
]


def main() -> int:
    for label, argv in STEPS:
        print(f"[regenerate] {label}", flush=True)
        result = subprocess.run([sys.executable, *argv], cwd=str(__import__("pathlib").Path(__file__).parent))
        if result.returncode != 0:
            print(f"[regenerate] FAILED: {label}", file=sys.stderr)
            return result.returncode
    print("[regenerate] done. Now run: git diff --exit-code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
