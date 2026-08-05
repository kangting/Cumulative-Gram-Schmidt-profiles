#!/usr/bin/env python3
"""Rounding sensitivity of the cumulative path and the two proxy readouts.

Recomputes every stored profile three ways: ordinary left-to-right summation,
compensated summation, and exact rational arithmetic on the stored values.
Reports whether the bottleneck level, the dimension for free, or the
admissibility margin change.

This audits the summation and the post-processing only. The stored profile
entries are themselves floating-point Gram-Schmidt output, and their accuracy
is not checked here.
"""

from __future__ import annotations

import json
import math
import pathlib
from fractions import Fraction

import bkz_experiment as bx

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
PROFILE_PATH = DATA_DIR / "profiles.jsonl"
OUTPUT = DATA_DIR / "precision_audit.json"


def cumulative_exact(ell: list[float]) -> list[Fraction]:
    n = len(ell)
    values = [Fraction(x) for x in ell]
    mean = sum(values) / n
    out = [Fraction(0)]
    for value in values:
        out.append(out[-1] + value - mean)
    return out


def cumulative_fsum(ell: list[float]) -> list[float]:
    n = len(ell)
    mean = math.fsum(ell) / n
    out, terms = [0.0], []
    for value in ell:
        terms.append(value - mean)
        out.append(math.fsum(terms))
    return out


def bottleneck_from(cum, n: int) -> int:
    best_k, best = 1, -math.inf
    for k in range(1, n + 1):
        value = bx.ball_constant(n, k) + float(cum[n - k])
        if value > best:
            best_k, best = k, value
    return best_k


def depth_from(cum, n: int) -> int:
    depth = 0
    for d in range(1, n):
        if float(cum[d]) / (n - d) > bx.kappa(n, d):
            break
        depth = d
    return depth


def main() -> None:
    rows = [json.loads(line) for line in PROFILE_PATH.open()]
    rows = [r for r in rows if r.get("ell")]
    max_c_gap = 0.0
    k_changed_fsum = k_changed_exact = 0
    d_changed_fsum = d_changed_exact = 0
    min_margin = math.inf
    for r in rows:
        ell = r["ell"]
        n = len(ell)
        c_ord = bx.cumulative(ell)
        c_fs = cumulative_fsum(ell)
        c_ex = cumulative_exact(ell)
        max_c_gap = max(max_c_gap, max(abs(a - float(b)) for a, b in zip(c_ord, c_ex)))
        k_ord, k_fs, k_ex = (bottleneck_from(c, n) for c in (c_ord, c_fs, c_ex))
        d_ord, d_fs, d_ex = (depth_from(c, n) for c in (c_ord, c_fs, c_ex))
        k_changed_fsum += k_ord != k_fs
        k_changed_exact += k_ord != k_ex
        d_changed_fsum += d_ord != d_fs
        d_changed_exact += d_ord != d_ex
        if d_ex + 1 < n:
            for d in (d_ex, d_ex + 1):
                if d >= 1:
                    margin = abs(float(c_ex[d]) / (n - d) - bx.kappa(n, d))
                    min_margin = min(min_margin, margin)
    result = {
        "profiles": len(rows),
        "max_abs_cumulative_gap_ordinary_vs_exact": max_c_gap,
        "bottleneck_changed_by_compensated_sum": k_changed_fsum,
        "bottleneck_changed_by_exact_sum": k_changed_exact,
        "depth_changed_by_compensated_sum": d_changed_fsum,
        "depth_changed_by_exact_sum": d_changed_exact,
        "min_admissibility_margin_at_crossing": min_margin,
        "scope": "summation and post-processing only, not the stored profile entries",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
