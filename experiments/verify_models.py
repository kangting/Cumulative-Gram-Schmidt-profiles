#!/usr/bin/env python3
"""Check the closed-form model formulas of Section 4 against direct computation.

The error-prone part of Section 4 is the exact zone-by-zone expression for the
cumulative profile of a three-zone (ZGSA) basis, equation
eq:zgsa-cumulative-finite, together with the determinant identity
eq:zgsa-exact-volume. Both are checked here by building the model profile from
its definition and comparing against the closed forms.
"""

from __future__ import annotations

import math

TOLERANCE = 1e-9


def zgsa_profile(n: int, n_q: int, n_gsa: int, big_q: float,
                 delta: float) -> list[float]:
    """The profile of eq:zgsa-profile-model."""
    height = big_q - n_gsa * delta
    out = []
    for i in range(1, n + 1):
        if i <= n_q:
            out.append(big_q)
        elif i <= n_q + n_gsa:
            out.append(big_q - (i - n_q) * delta)
        else:
            out.append(height)
    return out


def cumulative(ell: list[float]) -> list[float]:
    n = len(ell)
    mean = sum(ell) / n
    out = [0.0]
    for value in ell:
        out.append(out[-1] + value - mean)
    return out


def closed_form(j: int, n: int, n_q: int, n_gsa: int, n_1: int,
                big_q: float, delta: float, alpha: float) -> float:
    """The three cases of eq:zgsa-cumulative-finite."""
    height = big_q - n_gsa * delta
    if j <= n_q:
        return j * (1.0 - alpha) * big_q
    if j <= n_q + n_gsa:
        u = j - n_q
        return (n_q + u) * (1.0 - alpha) * big_q - 0.5 * delta * u * (u + 1)
    w = j - n_q - n_gsa
    return (n_1 - w) * (alpha * big_q - height)


def check_case(n: int, n_q: int, n_gsa: int, big_q: float,
               delta: float) -> tuple[bool, bool, float, float]:
    n_1 = n - n_q - n_gsa
    ell = zgsa_profile(n, n_q, n_gsa, big_q, delta)
    alpha = sum(ell) / (n * big_q)
    cum = cumulative(ell)

    worst = max(
        abs(cum[j] - closed_form(j, n, n_q, n_gsa, n_1, big_q, delta, alpha))
        for j in range(0, n + 1)
    )

    # eq:zgsa-exact-volume, written as alpha n Q = ... , with H = Q - n_gsa delta
    height = big_q - n_gsa * delta
    volume_lhs = alpha * n * big_q
    volume_rhs = (n_q + (n_gsa - 1) / 2.0) * big_q + (n_1 + (n_gsa + 1) / 2.0) * height
    volume_gap = abs(volume_lhs - volume_rhs)

    return worst < TOLERANCE, volume_gap < TOLERANCE * max(1.0, abs(volume_lhs)), worst, volume_gap


def main() -> None:
    cases = []
    for n in (60, 120, 200):
        for big_q in (math.log(13), math.log(1031), math.log(65537)):
            for n_q_frac in (0.0, 0.1, 0.3):
                for n_gsa_frac in (0.2, 0.5, 0.9):
                    n_q = int(n_q_frac * n)
                    n_gsa = int(n_gsa_frac * n)
                    if n_q + n_gsa > n or n_gsa < 1:
                        continue
                    delta = big_q / n_gsa
                    cases.append((n, n_q, n_gsa, big_q, delta))

    cum_fail = vol_fail = 0
    worst_cum = worst_vol = 0.0
    for n, n_q, n_gsa, big_q, delta in cases:
        cum_ok, vol_ok, worst, gap = check_case(n, n_q, n_gsa, big_q, delta)
        worst_cum = max(worst_cum, worst)
        worst_vol = max(worst_vol, gap)
        if not cum_ok:
            cum_fail += 1
            print(f"CUM FAIL n={n} n_q={n_q} n_gsa={n_gsa} Q={big_q:.3f}: {worst:.3e}")
        if not vol_ok:
            vol_fail += 1
            print(f"VOL FAIL n={n} n_q={n_q} n_gsa={n_gsa} Q={big_q:.3f}: {gap:.3e}")

    print(f"cases {len(cases)}")
    print(f"eq:zgsa-cumulative-finite  failures {cum_fail}, worst {worst_cum:.3e}")
    print(f"eq:zgsa-exact-volume       failures {vol_fail}, worst {worst_vol:.3e}")
    print("PASS" if cum_fail == vol_fail == 0 else "FAIL")


if __name__ == "__main__":
    main()
