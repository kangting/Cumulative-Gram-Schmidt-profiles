#!/usr/bin/env python3
"""Is the three-zone (Z-shaped) profile the maximizer of C_ell(d)?

Admissible class A (fixed n, Q, L, delta, T):
    ell_1 <= Q,  ell_n >= L,  0 <= ell_j - ell_{j+1} <= delta,  sum_i ell_i = T.

By the decrement identity, with s_j = ell_j - ell_{j+1},
    C_ell(d) = sum_j w_d(j) s_j,   w_d(j) = min(j,d)(n - max(j,d))/n,
and eliminating ell_n through the determinant constraint turns membership in A
into the box 0 <= s_j <= delta plus
    sum_j j s_j       <= T - n L        (this is ell_n >= L)
    sum_j (n - j) s_j <= n Q - T        (this is ell_1 <= Q).

So maximizing C_ell(d) over A is a linear program in s. This script solves it
and asks whether the optimal decrement vector is supported on a single
contiguous run at rate delta, which is exactly a three-zone profile.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog

RATE_TOLERANCE = 1e-7


def weights(n: int, d: int) -> np.ndarray:
    """w_d(j) for j = 1..n-1."""
    j = np.arange(1, n)
    return np.minimum(j, d) * (n - np.maximum(j, d)) / n


def solve_lp(n: int, q_cap: float, floor: float, delta: float,
             total: float, d: int) -> tuple[float, np.ndarray] | None:
    """Maximize C_ell(d) over the admissible class. Returns (value, s)."""
    j = np.arange(1, n, dtype=float)
    result = linprog(
        c=-weights(n, d),
        A_ub=np.vstack([j, n - j]),
        b_ub=np.array([total - n * floor, n * q_cap - total]),
        bounds=[(0.0, delta)] * (n - 1),
        method="highs",
    )
    if not result.success:
        return None
    return -result.fun, result.x


def three_zone_value(n: int, q_cap: float, floor: float, delta: float,
                     total: float, d: int) -> float:
    """Best C_ell(d) over three-zone profiles: plateau, slope delta, plateau.

    A three-zone profile is determined by where the maximal-slope run starts
    and how long it is. Enumerate all such runs and keep the feasible best.
    """
    j = np.arange(1, n, dtype=float)
    w = weights(n, d)
    budget_low = total - n * floor
    budget_high = n * q_cap - total
    best = -np.inf
    for start in range(n - 1):
        for length in range(1, n - start):
            s = np.zeros(n - 1)
            s[start : start + length] = delta
            if s @ j > budget_low + 1e-9:
                continue
            if s @ (n - j) > budget_high + 1e-9:
                continue
            best = max(best, float(w @ s))
    # Allow one partial step at either end of the run, which is what a
    # three-zone profile needs to hit the determinant exactly.
    for start in range(n - 1):
        for length in range(1, n - start):
            for partial_at_end in (True, False):
                for frac in np.linspace(0.0, 1.0, 21):
                    s = np.zeros(n - 1)
                    s[start : start + length] = delta
                    edge = start + length - 1 if partial_at_end else start
                    s[edge] = delta * frac
                    if s @ j > budget_low + 1e-9:
                        continue
                    if s @ (n - j) > budget_high + 1e-9:
                        continue
                    best = max(best, float(w @ s))
    return best


def support_is_one_contiguous_run(s: np.ndarray, delta: float) -> bool:
    """True if the nonzero decrements form one contiguous block at rate delta."""
    active = np.where(s > RATE_TOLERANCE)[0]
    if active.size == 0:
        return True
    if active[-1] - active[0] + 1 != active.size:
        return False
    interior = s[active]
    off_rate = np.abs(interior - delta) > RATE_TOLERANCE
    return int(off_rate.sum()) <= 1


def main() -> None:
    grids = {
        "n": [40, 60],
        "log_q": [4.0, 8.0],
        "delta": [0.15, 0.4],
        "alpha": [0.3, 0.5, 0.7],
        "d_frac": [0.05, 0.15, 0.3, 0.5],
    }
    mismatches = 0
    non_zshape = 0
    trials = 0
    worst_gap = 0.0

    for n, log_q, delta, alpha, d_frac in itertools.product(*grids.values()):
        d = max(1, int(round(d_frac * n)))
        floor = 0.0
        total = alpha * n * log_q
        if not (n * floor <= total <= n * log_q):
            continue
        solved = solve_lp(n, log_q, floor, delta, total, d)
        if solved is None:
            continue
        trials += 1
        lp_value, s = solved
        zone_value = three_zone_value(n, log_q, floor, delta, total, d)
        gap = lp_value - zone_value
        worst_gap = max(worst_gap, gap)
        if gap > 1e-6:
            mismatches += 1
            print(
                f"GAP  n={n} logq={log_q} delta={delta} alpha={alpha} d={d}: "
                f"LP={lp_value:.6f} threezone={zone_value:.6f} gap={gap:.2e}"
            )
        if not support_is_one_contiguous_run(s, delta):
            non_zshape += 1
            active = np.where(s > RATE_TOLERANCE)[0]
            print(
                f"SHAPE n={n} logq={log_q} delta={delta} alpha={alpha} d={d}: "
                f"support {active.min()}..{active.max()} size {active.size}"
            )

    print(f"\ntrials {trials}")
    print(f"three-zone value below LP optimum in {mismatches} cases, "
          f"worst gap {worst_gap:.2e}")
    print(f"LP optimum not a single maximal-slope run in {non_zshape} cases")

    print("\nsimultaneous maximizer check")
    n, log_q, delta, alpha = 40, 8.0, 0.4, 0.5
    total = alpha * n * log_q
    argmax_runs = []
    for d_frac in (0.1, 0.25, 0.5, 0.75):
        d = max(1, int(round(d_frac * n)))
        solved = solve_lp(n, log_q, 0.0, delta, total, d)
        if solved is None:
            continue
        active = np.where(solved[1] > RATE_TOLERANCE)[0]
        argmax_runs.append((d, int(active.min()), int(active.max())))
        print(f"  d={d:3d}  slope run on indices {active.min()}..{active.max()}")
    same = len({(lo, hi) for _, lo, hi in argmax_runs}) == 1
    print(f"  one profile maximizes every d: {same}")


if __name__ == "__main__":
    main()
