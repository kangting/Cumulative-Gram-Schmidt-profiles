#!/usr/bin/env python3
"""Numerical verification of the exact identities of Section 3.

Checks, on random and structured profiles, that
  (i)   log N_k(R) = A_{n,k} + C_B(n-k) - k*omega
  (ii)  Phi_d(B) = C_B(d)/(n-d) and the admissibility rearrangement
  (iii) C is invariant under scaling of the lattice
  (iv)  C_{B_dagger}(j) = C_B(n-j)
  (v)   decrement form  C_B(d) = sum_j w_d(j) s_j
  (vi)  stability identities of Theorem thm:stability

Natural logarithms throughout, matching the paper.
"""

from __future__ import annotations

import math
import random

TOLERANCE = 1e-9
SATURATION_CONSTANT = math.sqrt(4.0 / 3.0)


def log_ball_volume(m: int) -> float:
    """log Vol(unit m-ball)."""
    return 0.5 * m * math.log(math.pi) - math.lgamma(m / 2.0 + 1.0)


def log_gh(m: int) -> float:
    """log of the Gaussian-heuristic constant gh(m) = v_m^{-1/m}."""
    return -log_ball_volume(m) / m


def cumulative_profile(ell: list[float]) -> list[float]:
    """Centered cumulative profile C_B(j) for j = 0..n."""
    n = len(ell)
    mean = sum(ell) / n
    out = [0.0]
    for value in ell:
        out.append(out[-1] + value - mean)
    return out


def log_enum_proxy(ell: list[float], k: int, omega: float) -> float:
    """log N_k(R) computed directly from the definition, R = GH * e^{-omega}."""
    n = len(ell)
    log_det = sum(ell)
    log_radius = log_gh(n) + log_det / n - omega
    return log_ball_volume(k) + k * log_radius - sum(ell[n - k :])


def ball_constant(n: int, k: int) -> float:
    """A_{n,k} = log v_k - (k/n) log v_n."""
    return log_ball_volume(k) - (k / n) * log_ball_volume(n)


def kappa(n: int, d: int) -> float:
    """Projection threshold kappa_{n,d}."""
    return math.log(
        SATURATION_CONSTANT * math.exp(log_gh(n - d))
        / (math.sqrt((n - d) / n) * math.exp(log_gh(n)))
    )


def phi(ell: list[float], d: int) -> float:
    """Projection functional Phi_d(B), from the definition in Section 2."""
    n = len(ell)
    return (sum(ell[:d]) - (d / n) * sum(ell)) / (n - d)


def reversed_dual(ell: list[float]) -> list[float]:
    """Profile of the reversed dual basis."""
    return [-value for value in reversed(ell)]


def decrement_weight(n: int, d: int, j: int) -> float:
    """w_d(j) = min(j,d)(n - max(j,d))/n."""
    return min(j, d) * (n - max(j, d)) / n


def random_profile(n: int, rng: random.Random) -> list[float]:
    """A decreasing profile with random decrements and a random offset."""
    decrements = [rng.uniform(0.0, 0.4) for _ in range(n - 1)]
    ell = [rng.uniform(-3.0, 3.0)]
    for decrement in decrements:
        ell.append(ell[-1] - decrement)
    return ell


def zshape_profile(n: int, log_q: float, head: int, slope_len: int) -> list[float]:
    """Three-zone profile: plateau at log_q, linear descent, bounded tail."""
    delta = log_q / slope_len
    ell = []
    for i in range(1, n + 1):
        if i <= head:
            ell.append(log_q)
        elif i <= head + slope_len:
            ell.append(log_q - (i - head) * delta)
        else:
            ell.append(log_q - slope_len * delta)
    return ell


def check(name: str, worst: float) -> bool:
    ok = worst < TOLERANCE
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<46} max error {worst:.3e}")
    return ok


def run_case(ell: list[float], label: str) -> bool:
    n = len(ell)
    cum = cumulative_profile(ell)
    passed = True
    print(f"{label}  (n = {n})")

    worst = max(abs(cum[0]), abs(cum[n]))
    passed &= check("C(0) = C(n) = 0", worst)

    worst = 0.0
    for omega in (0.0, 0.35, -0.2, 1.7):
        for k in range(1, n + 1):
            lhs = log_enum_proxy(ell, k, omega)
            rhs = ball_constant(n, k) + cum[n - k] - k * omega
            worst = max(worst, abs(lhs - rhs))
    passed &= check("(i) enumeration identity", worst)

    worst = max(abs(phi(ell, d) - cum[d] / (n - d)) for d in range(0, n))
    passed &= check("(ii) projection identity", worst)

    scaled = [value + math.log(7.3) for value in ell]
    worst = max(abs(a - b) for a, b in zip(cum, cumulative_profile(scaled)))
    passed &= check("(iii) scale invariance", worst)

    dual_cum = cumulative_profile(reversed_dual(ell))
    worst = max(abs(dual_cum[j] - cum[n - j]) for j in range(0, n + 1))
    passed &= check("(iv) duality reflection", worst)

    decrements = [ell[j] - ell[j + 1] for j in range(n - 1)]
    worst = 0.0
    for d in range(0, n + 1):
        total = sum(
            decrement_weight(n, d, j + 1) * decrements[j] for j in range(n - 1)
        )
        worst = max(worst, abs(total - cum[d]))
    passed &= check("(v) decrement form", worst)

    return passed


def run_stability(rng: random.Random) -> bool:
    """Theorem thm:stability, on two profiles with the same determinant."""
    n = 120
    ell = random_profile(n, rng)
    model = zshape_profile(n, 6.0, 24, 60)
    shift = (sum(ell) - sum(model)) / n
    model = [value + shift for value in model]

    cum_a = cumulative_profile(ell)
    cum_b = cumulative_profile(model)
    delta_e = max(abs(a - b) for a, b in zip(cum_a, cum_b))
    print(f"stability check  (n = {n}, Delta_E = {delta_e:.4f})")

    passed = True
    worst = 0.0
    gaps = []
    for omega in (0.0, 0.5):
        for k in range(1, n + 1):
            diff = log_enum_proxy(ell, k, omega) - log_enum_proxy(model, k, omega)
            worst = max(worst, abs(diff - (cum_a[n - k] - cum_b[n - k])))
            gaps.append(abs(diff))
    passed &= check("(vi) enumeration error equals cumulative gap", worst)

    passed &= check(
        "(vi) max enumeration error equals Delta_E",
        abs(max(gaps) - delta_e),
    )

    worst = max(
        abs((phi(ell, d) - phi(model, d)) - (cum_a[d] - cum_b[d]) / (n - d))
        for d in range(0, n)
    )
    passed &= check("(vi) projection error identity", worst)

    projection_errors = [
        abs(phi(ell, d) - phi(model, d)) for d in range(n)
    ]
    depth_limit = n // 2
    delta_p = max(projection_errors[: depth_limit + 1])
    cumulative_delta_p = max(
        abs(cum_a[d] - cum_b[d]) / (n - d)
        for d in range(depth_limit + 1)
    )
    passed &= check(
        "(vi) max projection error equals Delta_P(D)",
        abs(delta_p - cumulative_delta_p),
    )
    passed &= check(
        "(vi) Delta_P(D) <= Delta_E/(n-D)",
        max(delta_p - delta_e / (n - depth_limit), 0.0),
    )

    lower = {
        d
        for d in range(n)
        if phi(model, d) <= kappa(n, d) - delta_e / (n - d)
    }
    truth = {d for d in range(n) if phi(ell, d) <= kappa(n, d)}
    upper = {
        d
        for d in range(n)
        if phi(model, d) <= kappa(n, d) + delta_e / (n - d)
    }
    sandwich_ok = lower <= truth <= upper
    print(
        f"  {'PASS' if sandwich_ok else 'FAIL'}  "
        f"{'(vi) certified sandwich on admissible depths':<46} "
        f"|lower|={len(lower)} |true|={len(truth)} |upper|={len(upper)}"
    )
    return passed and sandwich_ok


def main() -> None:
    rng = random.Random(20260731)
    all_passed = True
    for trial in range(3):
        all_passed &= run_case(
            random_profile(80 + 17 * trial, rng), f"random profile {trial + 1}"
        )
    all_passed &= run_case(zshape_profile(150, math.log(3329), 30, 75), "ZGSA profile")
    all_passed &= run_case(
        [(-i) * 0.05 for i in range(100)], "geometric (GSA) profile"
    )
    all_passed &= run_stability(rng)
    print("\nALL CHECKS PASSED" if all_passed else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()
