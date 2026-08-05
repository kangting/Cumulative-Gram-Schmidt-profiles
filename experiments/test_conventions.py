#!/usr/bin/env python3
"""Guard the two conventions that the manuscript states explicitly.

Both were wrong in an earlier revision, and neither is visible in the output
tables, so they are checked directly here.
"""

from __future__ import annotations

import math

import bkz_experiment as bx
import node_counts as nc


def test_tgsa_decrement_matches_the_definition() -> None:
    """The fitted head decrement is eta_beta for every beta, with no guard."""
    assert abs(2.0 * bx.log_gh(2) + math.log(math.pi)) < 1e-14
    ell = [10.0 - 0.1 * i for i in range(60)]
    for beta in (2, 3, 12, 13, 20, 40):
        model = bx.fit_tgsa(ell, beta)
        head = model[: len(ell) - beta]
        eta = 2.0 * bx.log_gh(beta) / (beta - 1)
        assert abs((head[0] - head[1]) - eta) < 1e-12, f"beta={beta}"


def test_tgsa_head_rises_at_block_size_two() -> None:
    """The formal extrapolation to beta = 2 gives an increasing head."""
    ell = [10.0 - 0.1 * i for i in range(30)]
    model = bx.fit_tgsa(ell, 2)
    head = model[: len(ell) - 2]
    assert head[1] > head[0], "beta = 2 must extrapolate to a rising head"


def test_hkz_tail_satisfies_the_shape_heuristic() -> None:
    """Each suffix head equals gh of the suffix rank times the suffix mean."""
    for beta in (2, 3, 5, 12, 13, 20, 40):
        tail = bx.hkz_shape_tail(beta)
        assert abs(sum(tail)) < 1e-12, f"beta={beta} tail is not determinant one"
        for m in range(2, beta + 1):
            suffix = tail[beta - m:]
            expected = bx.log_gh(m) + sum(suffix) / m
            assert abs(tail[beta - m] - expected) < 1e-12, f"beta={beta} m={m}"


def test_tgsa_cumulative_path_matches_the_finite_formula() -> None:
    """The fitted model reproduces the closed form of the manuscript."""
    for n, beta in ((30, 2), (50, 13), (60, 20), (80, 40)):
        ell = [10.0 - 0.1 * i for i in range(n)]
        cumulative = bx.cumulative(bx.fit_tgsa(ell, beta))
        eta = 2.0 * bx.log_gh(beta) / (beta - 1)
        head_len = n - beta
        for j in range(n + 1):
            if j <= head_len:
                expected = eta / 2.0 * j * (n - j)
            elif j < n:
                k = n - j
                expected = k * head_len * eta / 2.0 + k * sum(
                    bx.log_gh(v) / (v - 1) for v in range(k + 1, beta + 1)
                )
            else:
                expected = 0.0
            assert abs(cumulative[j] - expected) < 1e-9, f"n={n} beta={beta} j={j}"


def test_plateau_is_a_leading_run_within_tolerance() -> None:
    q = math.log(97.0)
    tol = bx.PLATEAU_TOLERANCE
    assert bx.plateau_length([q, q, q + 2 * tol, q], q) == 2
    assert bx.plateau_length([q, q - 2 * tol, q], q) == 1
    assert bx.plateau_length([q + 2 * tol, q, q], q) == 0
    assert bx.plateau_length([q, q + tol / 2, q - tol / 2], q) == 3


def test_point_count_plateau_uses_the_same_tolerance() -> None:
    assert nc.PLATEAU_TOLERANCE == bx.PLATEAU_TOLERANCE


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("all conventions hold")
