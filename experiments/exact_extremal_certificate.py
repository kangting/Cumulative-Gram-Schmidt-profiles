#!/usr/bin/env python3
"""Verify one exact counterexample to three-zone extremality.

All arithmetic uses fractions.Fraction. The unrestricted primal and dual
certificates concern the decrement LP in Proposition rem:extremal. The
three-zone checks enumerate every contiguous constant-slope descent. A
floating-point scan.
"""

from __future__ import annotations

from fractions import Fraction

N = 40
DEPTH = 20
CAP = Fraction(4)
FLOOR = Fraction(0)
DECREMENT_CAP = Fraction(2, 5)
TOTAL = Fraction(80)
LOW_BUDGET = TOTAL - N * FLOOR
HIGH_BUDGET = N * CAP - TOTAL


def weight(index: int) -> Fraction:
    """Return the decrement weight at the certified depth."""
    return Fraction(
        min(index, DEPTH) * (N - max(index, DEPTH)),
        N,
    )


def objective(decrements: dict[int, Fraction]) -> Fraction:
    """Return C_ell(DEPTH) from the decrement identity."""
    return sum(
        (weight(index) * value for index, value in decrements.items()),
        Fraction(0),
    )


def budgets(decrements: dict[int, Fraction]) -> tuple[Fraction, Fraction]:
    """Return the floor and cap budget usage."""
    low = sum(
        (index * value for index, value in decrements.items()),
        Fraction(0),
    )
    high = sum(
        ((N - index) * value for index, value in decrements.items()),
        Fraction(0),
    )
    return low, high


def profile_from_decrements(
    decrements: dict[int, Fraction],
) -> list[Fraction]:
    """Recover the determinant-matched profile from its decrements."""
    low_usage, _ = budgets(decrements)
    last_entry = (TOTAL - low_usage) / N
    return [
        last_entry
        + sum(
            (decrements.get(index, Fraction(0)) for index in range(i, N)),
            Fraction(0),
        )
        for i in range(1, N + 1)
    ]


def unrestricted_primal() -> dict[int, Fraction]:
    """Return the explicit non-three-zone optimal decrement vector."""
    return {
        15: Fraction(1, 5),
        **{index: DECREMENT_CAP for index in range(16, 25)},
        25: Fraction(1, 5),
    }


def verify_primal(decrements: dict[int, Fraction]) -> list[Fraction]:
    """Verify every primal constraint and return the resulting profile."""
    profile = profile_from_decrements(decrements)
    low_usage, high_usage = budgets(decrements)
    assert all(
        Fraction(0) <= value <= DECREMENT_CAP
        for value in decrements.values()
    )
    assert low_usage <= LOW_BUDGET
    assert high_usage <= HIGH_BUDGET
    assert profile[0] <= CAP
    assert profile[-1] >= FLOOR
    assert sum(profile, Fraction(0)) == TOTAL
    assert objective(decrements) == Fraction(35)
    return profile


def verify_dual() -> tuple[Fraction, Fraction, dict[int, Fraction]]:
    """Verify a dual solution whose value equals the primal value."""
    low_multiplier = Fraction(3, 16)
    high_multiplier = Fraction(3, 16)
    upper_multipliers = {
        index: max(weight(index) - Fraction(15, 2), Fraction(0))
        for index in range(1, N)
    }
    for index in range(1, N):
        dual_left = (
            index * low_multiplier
            + (N - index) * high_multiplier
            + upper_multipliers[index]
        )
        assert dual_left >= weight(index)
    dual_value = (
        LOW_BUDGET * low_multiplier
        + HIGH_BUDGET * high_multiplier
        + DECREMENT_CAP * sum(upper_multipliers.values(), Fraction(0))
    )
    assert sum(upper_multipliers.values(), Fraction(0)) == Fraction(25, 2)
    assert dual_value == Fraction(35)
    return low_multiplier, high_multiplier, upper_multipliers


def best_constant_slope_three_zone() -> tuple[Fraction, int, int, Fraction]:
    """Enumerate every contiguous descent with one constant slope."""
    best = (Fraction(-1), 0, 0, Fraction(0))
    for start in range(1, N):
        for end in range(start, N):
            indices = range(start, end + 1)
            low_coefficient = sum(indices)
            high_coefficient = sum(N - index for index in range(start, end + 1))
            slope = min(
                DECREMENT_CAP,
                LOW_BUDGET / low_coefficient,
                HIGH_BUDGET / high_coefficient,
            )
            value = slope * sum(weight(index) for index in range(start, end + 1))
            candidate = (value, start, end, slope)
            if candidate[0] > best[0]:
                best = candidate
    assert best == (Fraction(380, 11), 15, 25, Fraction(4, 11))
    return best


def main() -> None:
    """Print the exact primal, dual, and three-zone certificates."""
    primal = unrestricted_primal()
    profile = verify_primal(primal)
    low_multiplier, high_multiplier, upper_multipliers = verify_dual()
    constant_best = best_constant_slope_three_zone()

    print(
        f"parameters n={N} Q={CAP} L={FLOOR} "
        f"delta={DECREMENT_CAP} T={TOTAL} d={DEPTH}"
    )
    print(f"primal decrements={primal}")
    print(f"primal profile={profile}")
    print(f"primal objective={objective(primal)}")
    print(f"dual budget multipliers=({low_multiplier}, {high_multiplier})")
    print(f"dual upper multipliers={upper_multipliers}")
    print("dual objective=35")
    print(f"constant-slope three-zone optimum={constant_best}")


if __name__ == "__main__":
    main()
