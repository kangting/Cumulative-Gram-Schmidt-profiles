#!/usr/bin/env python3
"""Count projected-lattice points at a fixed Gaussian-heuristic radius.

The public ``run`` command generates a scaled version of the manuscript grid.
Every parameter instance runs in an isolated subprocess because FPLLL can
abort at the C level.  The child flushes its reduced profile and one JSON row
after every completed projected level.  A later crash therefore preserves all
earlier measurements, and every terminal condition is recorded.

For projected rank ``level``, fpylll enumeration uses rows ``n - level``
through ``n - 1``.  ``FIRST_N_SOLUTIONS`` keeps the input radius fixed.  The
enumerator returns one member of each nonzero sign pair and omits zero, so the
number of lattice points in the closed ball is twice the solution-list length
plus one.  The ``sanity`` command verifies this convention independently on
integer lattices of ranks two, three, and four.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from typing import Any, Sequence


DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
PROFILE_PATH = DATA_DIR / "fixed_radius_profiles.jsonl"
RAW_PATH = DATA_DIR / "fixed_radius_counts.jsonl"
SANITY_PATH = DATA_DIR / "fixed_radius_sanity.json"

DEFAULT_RANKS = (40,)
REFERENCE_RANK = 120
REFERENCE_MODULI = (13, 31, 127, 1031)
DEFAULT_SEEDS = 3
WORKER_COUNT = 8
NODE_CAP = 100_000_000
SOLUTION_BUDGET = 10_000_000
ENUMERATION_TIME_CAP_SECONDS = 300.0
REDUCTION_TIMEOUT_SECONDS = 900
PROCESS_BUFFER_SECONDS = 20
BKZ_MAX_LOOPS = 8
FLOAT_TYPE = "ld"
PLATEAU_TOLERANCE = 0.05
JSON_PREFIX = "ROW\t"
PROFILE_PREFIX = "PROFILE\t"
STATE_PREFIX = "STATE\t"

JsonRecord = dict[str, Any]
Job = tuple[int, int, int, int, int]


def log_ball_volume(dimension: int) -> float:
    """Return the natural logarithm of the unit-ball volume."""
    return 0.5 * dimension * math.log(math.pi) - math.lgamma(dimension / 2.0 + 1.0)


def log_gh_from_profile(profile: Sequence[float]) -> float:
    """Return log GH(Lambda) from logarithmic Gram--Schmidt norms."""
    dimension = len(profile)
    return sum(profile) / dimension - log_ball_volume(dimension) / dimension


def predicted_log_points(
    profile: Sequence[float], level: int, log_radius: float
) -> float:
    """Return log(v_k R^k / product of the final k projected norms)."""
    return (
        log_ball_volume(level)
        + level * log_radius
        - sum(profile[len(profile) - level :])
    )


def predicted_bottleneck(profile: Sequence[float]) -> tuple[int, float]:
    """Return the level and value maximizing the fixed-radius volume proxy."""
    log_radius = log_gh_from_profile(profile)
    values = [
        predicted_log_points(profile, level, log_radius)
        for level in range(1, len(profile) + 1)
    ]
    best_index = max(range(len(values)), key=values.__getitem__)
    return best_index + 1, values[best_index]


def safe_ratio(log_ratio: float) -> float:
    """Exponentiate a log ratio without producing an exception."""
    if log_ratio > math.log(sys.float_info.max):
        return math.inf
    if log_ratio < math.log(sys.float_info.min):
        return 0.0
    return math.exp(log_ratio)


def beta_values(rank: int) -> tuple[int, ...]:
    """Scale the main-grid block sizes 2, 20, and 40 from rank 120."""
    values = (2, math.ceil(rank / 6), math.ceil(rank / 3))
    return tuple(sorted(set(min(rank, value) for value in values)))


def is_prime(value: int) -> bool:
    """Return whether a small positive integer is prime."""
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def next_prime(value: int) -> int:
    """Return the least prime greater than or equal to value."""
    candidate = max(2, value)
    while not is_prime(candidate):
        candidate += 1
    return candidate


def modulus_values(rank: int) -> tuple[int, ...]:
    """Scale the rank-120 moduli linearly and round each upward to a prime."""
    values = (
        next_prime(max(3, round(modulus * rank / REFERENCE_RANK)))
        for modulus in REFERENCE_MODULI
    )
    return tuple(sorted(set(values)))


def lattice_dimensions(rank: int) -> tuple[int, ...]:
    """Return the two q-ary generator dimensions used by the main grid."""
    return tuple(sorted({rank // 2, (3 * rank) // 4}))


def build_grid(ranks: Sequence[int], seeds: int) -> list[Job]:
    """Return the deterministic scaled parameter grid."""
    return [
        (rank, modulus, lattice_t, beta, seed)
        for rank in ranks
        for modulus in modulus_values(rank)
        for lattice_t in lattice_dimensions(rank)
        for beta in beta_values(rank)
        for seed in range(seeds)
    ]


def job_id(job: Job) -> str:
    """Return a stable textual identifier for one parameter instance."""
    rank, modulus, lattice_t, beta, seed = job
    return f"n={rank}:q={modulus}:t={lattice_t}:beta={beta}:seed={seed}"


def job_fields(job: Job) -> JsonRecord:
    """Return identifying fields shared by every output row."""
    rank, modulus, lattice_t, beta, seed = job
    return {
        "job_id": job_id(job),
        "n": rank,
        "q": modulus,
        "lattice_t": lattice_t,
        "beta": beta,
        "seed": seed,
    }


def emit_state(stage: str, level: int) -> None:
    """Tell the parent which C-level operation is about to run."""
    print(
        STATE_PREFIX + json.dumps({"stage": stage, "level": level}),
        flush=True,
    )


def emit_profile(profile: JsonRecord) -> None:
    """Flush one reduced profile to the parent process."""
    print(PROFILE_PREFIX + json.dumps(profile, sort_keys=True), flush=True)


def emit_row(row: JsonRecord) -> None:
    """Flush one durable measurement row to the parent process."""
    print(JSON_PREFIX + json.dumps(row, sort_keys=True), flush=True)


def alarm_handler(signum: int, frame: Any) -> None:
    """Raise a Python exception when a guarded operation returns to Python."""
    del signum, frame
    raise TimeoutError("operation exceeded its time cap")


def reduce_basis(job: Job) -> tuple[Any, list[float]]:
    """Generate and reduce the q-ary basis for one grid instance."""
    from fpylll import BKZ, FPLLL, GSO, IntegerMatrix, LLL

    rank, modulus, lattice_t, beta, seed = job
    FPLLL.set_random_seed(seed + 1)
    basis = IntegerMatrix.random(
        rank,
        "qary",
        k=lattice_t,
        q=modulus,
    )
    gso = GSO.Mat(basis, float_type=FLOAT_TYPE)
    gso.update_gso()
    lll = LLL.Reduction(gso)
    lll()
    if beta > 2:
        parameters = BKZ.Param(
            block_size=beta,
            max_loops=BKZ_MAX_LOOPS,
            flags=BKZ.MAX_LOOPS | BKZ.GH_BND,
        )
        BKZ.Reduction(gso, lll, parameters)()
    gso.update_gso()
    profile = [
        0.5 * math.log(gso.get_r(index, index))
        for index in range(rank)
    ]
    return basis, profile


def enumerate_job(
    job: Job,
    node_cap: int,
    solution_budget: int,
    enumeration_time_cap: float,
) -> None:
    """Child entry point that reduces one basis and sweeps every level."""
    from fpylll import Enumeration, EnumerationError, EvaluatorStrategy, GSO

    common = job_fields(job)
    rank = int(common["n"])
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(REDUCTION_TIMEOUT_SECONDS)
    reduction_started = time.monotonic()
    emit_state("reduction", 0)
    try:
        basis, profile = reduce_basis(job)
    except Exception as error:
        signal.alarm(0)
        emit_row(
            {
                **common,
                "level": 0,
                "status": "reduction_failed",
                "terminal": True,
                "error": type(error).__name__,
                "detail": str(error)[:240],
                "seconds": round(time.monotonic() - reduction_started, 6),
            }
        )
        return
    finally:
        signal.alarm(0)

    predicted_k, predicted_peak = predicted_bottleneck(profile)
    log_radius = log_gh_from_profile(profile)
    radius_squared = math.exp(2.0 * log_radius)
    plateau_length = sum(
        value > math.log(int(common["q"])) - PLATEAU_TOLERANCE
        for value in profile
    )
    emit_profile(
        {
            **common,
            "ell": profile,
            "log_det": sum(profile),
            "log_radius": log_radius,
            "radius_squared": radius_squared,
            "predicted_bottleneck_k": predicted_k,
            "predicted_bottleneck_log_n": predicted_peak,
            "plateau_length": plateau_length,
            "reduction_seconds": round(
                time.monotonic() - reduction_started, 6
            ),
            "status": "ok",
        }
    )

    gso = GSO.Mat(basis, float_type=FLOAT_TYPE)
    gso.update_gso()
    enumeration_started = time.monotonic()
    for level in range(1, rank + 1):
        elapsed_before = time.monotonic() - enumeration_started
        remaining = enumeration_time_cap - elapsed_before
        if remaining <= 0.0:
            emit_row(
                {
                    **common,
                    "level": level,
                    "status": "time_cap",
                    "terminal": True,
                    "detail": "cap reached before starting this level",
                    "point_count": None,
                    "seconds": 0.0,
                    "enumeration_seconds_total": elapsed_before,
                    "predicted_log_n": predicted_log_points(
                        profile, level, log_radius
                    ),
                    "predicted_bottleneck_k": predicted_k,
                    "predicted_bottleneck_log_n": predicted_peak,
                }
            )
            return

        signal.setitimer(signal.ITIMER_REAL, remaining)
        emit_state("enumeration", level)
        level_started = time.monotonic()
        enumeration = Enumeration(
            gso,
            nr_solutions=solution_budget,
            strategy=EvaluatorStrategy.FIRST_N_SOLUTIONS,
        )
        try:
            solutions = enumeration.enumerate(
                rank - level,
                rank,
                radius_squared,
                0,
            )
        except EnumerationError:
            solutions = []
        except Exception as error:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            status = (
                "time_cap"
                if isinstance(error, TimeoutError)
                else "enumeration_failed"
            )
            emit_row(
                {
                    **common,
                    "level": level,
                    "status": status,
                    "terminal": True,
                    "error": type(error).__name__,
                    "detail": str(error)[:240],
                    "point_count": None,
                    "seconds": round(time.monotonic() - level_started, 6),
                    "enumeration_seconds_total": time.monotonic()
                    - enumeration_started,
                    "predicted_log_n": predicted_log_points(
                        profile, level, log_radius
                    ),
                    "predicted_bottleneck_k": predicted_k,
                    "predicted_bottleneck_log_n": predicted_peak,
                }
            )
            return
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)

        level_seconds = time.monotonic() - level_started
        enumeration_seconds_total = time.monotonic() - enumeration_started
        solution_count = len(solutions)
        point_count = 2 * solution_count + 1
        log_point_count = math.log(point_count)
        proxy_value = predicted_log_points(profile, level, log_radius)
        log_ratio = log_point_count - proxy_value
        nodes = int(enumeration.get_nodes())
        status = "ok"
        terminal = level == rank
        terminal_reason = "rank_complete" if terminal else ""
        if solution_count >= solution_budget:
            status = "solution_cap"
            terminal = True
            terminal_reason = "solution_cap"
        # This is a post-call threshold. It cannot interrupt enumerate().
        elif nodes > node_cap:
            status = "node_cap"
            terminal = True
            terminal_reason = "node_cap"
        elif enumeration_seconds_total >= enumeration_time_cap:
            status = "time_cap"
            terminal = True
            terminal_reason = "time_cap"

        emit_row(
            {
                **common,
                "level": level,
                "status": status,
                "terminal": terminal,
                "terminal_reason": terminal_reason,
                "point_count": point_count,
                "nonzero_sign_orbits": solution_count,
                "log_measured_points": log_point_count,
                "predicted_log_n": proxy_value,
                "ratio": safe_ratio(log_ratio),
                "log_ratio": log_ratio,
                "nodes": nodes,
                "seconds": round(level_seconds, 6),
                "enumeration_seconds_total": round(
                    enumeration_seconds_total, 6
                ),
                "log_radius": log_radius,
                "radius_squared": radius_squared,
                "max_dist_expo": 0,
                "fixed_radius": True,
                "solution_budget": solution_budget,
                "solution_budget_reached": solution_count >= solution_budget,
                "evaluator_strategy": "FIRST_N_SOLUTIONS",
                "predicted_bottleneck_k": predicted_k,
                "predicted_bottleneck_log_n": predicted_peak,
            }
        )
        if terminal:
            return


def parse_child_output(
    output: str,
) -> tuple[JsonRecord | None, list[JsonRecord], JsonRecord]:
    """Parse the reduced profile, rows, and last child state marker."""
    profile: JsonRecord | None = None
    rows: list[JsonRecord] = []
    state: JsonRecord = {"stage": "reduction", "level": 0}
    for line in output.splitlines():
        if line.startswith(PROFILE_PREFIX):
            profile = json.loads(line[len(PROFILE_PREFIX) :])
        elif line.startswith(JSON_PREFIX):
            rows.append(json.loads(line[len(JSON_PREFIX) :]))
        elif line.startswith(STATE_PREFIX):
            state = json.loads(line[len(STATE_PREFIX) :])
    return profile, rows, state


def terminal_row(
    job: Job,
    profile: JsonRecord | None,
    state: JsonRecord,
    status: str,
    detail: str,
) -> JsonRecord:
    """Create a synthetic terminal row after a crash or parent timeout."""
    level = int(state.get("level", 0))
    row: JsonRecord = {
        **job_fields(job),
        "level": level,
        "status": status,
        "terminal": True,
        "detail": detail[-240:],
        "point_count": None,
        "seconds": 0.0,
    }
    if profile is not None:
        row["predicted_bottleneck_k"] = profile["predicted_bottleneck_k"]
        row["predicted_bottleneck_log_n"] = profile[
            "predicted_bottleneck_log_n"
        ]
        if level > 0:
            row["predicted_log_n"] = predicted_log_points(
                profile["ell"], level, float(profile["log_radius"])
            )
    return row


def existing_terminal_jobs(path: pathlib.Path) -> set[str]:
    """Return job identifiers having a durable terminal row."""
    terminal: set[str] = set()
    if not path.exists():
        return terminal
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("terminal"):
                terminal.add(str(row["job_id"]))
    return terminal


def launch_job(
    job: Job,
    node_cap: int,
    solution_budget: int,
    enumeration_time_cap: float,
) -> tuple[JsonRecord | None, list[JsonRecord]]:
    """Run one grid instance in an isolated child process."""
    script = str(pathlib.Path(__file__).resolve())
    command = [
        sys.executable,
        script,
        "job",
        *(str(value) for value in job),
        str(node_cap),
        str(solution_budget),
        repr(enumeration_time_cap),
    ]
    process_timeout = (
        REDUCTION_TIMEOUT_SECONDS
        + enumeration_time_cap
        + PROCESS_BUFFER_SECONDS
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=process_timeout,
            check=False,
        )
        profile, rows, state = parse_child_output(completed.stdout)
        if rows and rows[-1].get("terminal"):
            return profile, rows
        if completed.returncode == 0:
            detail = "child exited without a terminal row"
        else:
            detail = completed.stderr or f"child return code {completed.returncode}"
        stage = str(state["stage"])
        if stage == "enumeration" and "TimeoutError" in detail:
            status = "time_cap"
        else:
            status = (
                "enumeration_crash"
                if stage == "enumeration"
                else "reduction_crash"
            )
        return profile, [
            *rows,
            terminal_row(job, profile, state, status, detail),
        ]
    except subprocess.TimeoutExpired as error:
        standard_output = error.stdout or ""
        if isinstance(standard_output, bytes):
            standard_output = standard_output.decode(errors="replace")
        profile, rows, state = parse_child_output(standard_output)
        stage = str(state["stage"])
        status = (
            "enumeration_timeout" if stage == "enumeration" else "reduction_timeout"
        )
        detail = f"parent timeout after {process_timeout:.1f} seconds"
        return profile, [
            *rows,
            terminal_row(job, profile, state, status, detail),
        ]


def run_experiment(
    ranks: Sequence[int],
    seeds: int,
    raw_path: pathlib.Path,
    profile_path: pathlib.Path,
    workers: int,
    node_cap: int,
    solution_budget: int,
    enumeration_time_cap: float,
) -> None:
    """Run every outstanding scaled-grid instance with thread orchestration."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_grid(ranks, seeds)
    terminal = existing_terminal_jobs(raw_path)
    outstanding = [job for job in jobs if job_id(job) not in terminal]
    outstanding.sort(
        key=lambda item: (-item[3], -item[0], item[1], item[2], item[4])
    )
    print(
        f"{len(outstanding)} outstanding jobs from {len(jobs)} requested "
        f"on {workers} workers",
        flush=True,
    )

    completed_count = 0
    failure_count = 0
    with (
        raw_path.open("a") as raw_handle,
        profile_path.open("a") as profile_handle,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = {
            pool.submit(
                launch_job,
                job,
                node_cap,
                solution_budget,
                enumeration_time_cap,
            ): job
            for job in outstanding
        }
        for future in as_completed(futures):
            profile, rows = future.result()
            if profile is not None:
                profile_handle.write(json.dumps(profile, sort_keys=True) + "\n")
                profile_handle.flush()
            for row in rows:
                raw_handle.write(json.dumps(row, sort_keys=True) + "\n")
            raw_handle.flush()
            completed_count += 1
            status = str(rows[-1]["status"])
            failure_count += status != "ok"
            if completed_count % 8 == 0 or completed_count == len(outstanding):
                print(
                    f"  {completed_count}/{len(outstanding)} done, "
                    f"{failure_count} incomplete",
                    flush=True,
                )


def brute_force_integer_ball(dimension: int, radius_squared: int) -> int:
    """Count integer points in a ball by independent Cartesian search."""
    coordinate_bound = math.isqrt(radius_squared)
    coordinates = range(-coordinate_bound, coordinate_bound + 1)
    return sum(
        sum(value * value for value in vector) <= radius_squared
        for vector in itertools.product(coordinates, repeat=dimension)
    )


def sanity_check(path: pathlib.Path = SANITY_PATH) -> None:
    """Verify fixed-radius complete counts on Z^m for m in 2, 3, and 4."""
    from fpylll import Enumeration, EvaluatorStrategy, GSO, IntegerMatrix

    radius_squared = 5
    rows: list[JsonRecord] = []
    for dimension in (2, 3, 4):
        basis = IntegerMatrix.identity(dimension)
        gso = GSO.Mat(basis, float_type=FLOAT_TYPE)
        gso.update_gso()
        enumeration = Enumeration(
            gso,
            nr_solutions=SOLUTION_BUDGET,
            strategy=EvaluatorStrategy.FIRST_N_SOLUTIONS,
        )
        solutions = enumeration.enumerate(
            0,
            dimension,
            float(radius_squared),
            0,
        )
        enumerated_count = 2 * len(solutions) + 1
        independent_count = brute_force_integer_ball(
            dimension, radius_squared
        )
        if len(solutions) >= SOLUTION_BUDGET:
            raise AssertionError("integer-lattice sanity check hit the solution budget")
        if enumerated_count != independent_count:
            raise AssertionError(
                f"Z^{dimension} mismatch: fpylll={enumerated_count}, "
                f"independent={independent_count}"
            )
        rows.append(
            {
                "dimension": dimension,
                "radius_squared": radius_squared,
                "nonzero_sign_orbits": len(solutions),
                "fpylll_point_count": enumerated_count,
                "independent_point_count": independent_count,
                "match": True,
            }
        )
    oblique_basis = IntegerMatrix(3, 3)
    for row_index, row in enumerate(
        ((1, 1, 0), (1, 0, 0), (0, 0, 2))
    ):
        for column_index, value in enumerate(row):
            oblique_basis[row_index, column_index] = value
    oblique_gso = GSO.Mat(oblique_basis, float_type=FLOAT_TYPE)
    oblique_gso.update_gso()
    oblique_enumeration = Enumeration(
        oblique_gso,
        nr_solutions=SOLUTION_BUDGET,
        strategy=EvaluatorStrategy.FIRST_N_SOLUTIONS,
    )
    oblique_solutions = oblique_enumeration.enumerate(
        1,
        3,
        0.75,
        0,
    )
    oblique_count = 2 * len(oblique_solutions) + 1
    coefficient_range = range(-2, 3)
    exact_projected_count = sum(
        Fraction(first * first, 2) + 4 * second * second
        <= Fraction(3, 4)
        for first, second in itertools.product(
            coefficient_range,
            repeat=2,
        )
    )
    exact_unprojected_count = sum(
        first * first + 4 * second * second <= Fraction(3, 4)
        for first, second in itertools.product(
            coefficient_range,
            repeat=2,
        )
    )
    if oblique_count != exact_projected_count:
        raise AssertionError(
            f"oblique projected mismatch: fpylll={oblique_count}, "
            f"independent={exact_projected_count}"
        )
    if exact_projected_count == exact_unprojected_count:
        raise AssertionError("oblique test does not separate the two lattices")

    result = {
        "evaluator_strategy": "FIRST_N_SOLUTIONS",
        "fixed_radius": True,
        "solution_budget": SOLUTION_BUDGET,
        "sign_convention": "two times nonzero sign orbits plus zero",
        "checks": rows,
        "oblique_projection_check": {
            "basis_rows": [[1, 1, 0], [1, 0, 0], [0, 0, 2]],
            "first": 1,
            "last": 3,
            "radius_squared": 0.75,
            "projected_gso_squared_norms": [0.5, 4.0],
            "fpylll_point_count": oblique_count,
            "independent_projected_point_count": exact_projected_count,
            "independent_unprojected_point_count": exact_unprojected_count,
            "matches_projection_only": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def positive_int(value: str) -> int:
    """Parse a strictly positive integer argument."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    """Parse a strictly positive floating-point argument."""
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def rank_list(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive ranks."""
    ranks = tuple(sorted({positive_int(item) for item in value.split(",")}))
    if any(rank < 4 for rank in ranks):
        raise argparse.ArgumentTypeError("every rank must be at least four")
    return ranks


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch public and internal commands."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "job":
        if len(arguments) != 9:
            raise SystemExit(
                "job requires n, q, t, beta, seed, node threshold, solution budget, "
                "and time cap"
            )
        job = tuple(int(value) for value in arguments[1:6])
        enumerate_job(
            job=job,
            node_cap=int(arguments[6]),
            solution_budget=int(arguments[7]),
            enumeration_time_cap=float(arguments[8]),
        )
        return

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--ranks",
        type=rank_list,
        default=DEFAULT_RANKS,
    )
    run_parser.add_argument("--seeds", type=positive_int, default=DEFAULT_SEEDS)
    run_parser.add_argument("--workers", type=positive_int, default=WORKER_COUNT)
    run_parser.add_argument(
        "--node-cap",
        type=positive_int,
        default=NODE_CAP,
        help="post-call node threshold for stopping a sweep",
    )
    run_parser.add_argument(
        "--solution-budget",
        type=positive_int,
        default=SOLUTION_BUDGET,
    )
    run_parser.add_argument(
        "--time-cap",
        type=positive_float,
        default=ENUMERATION_TIME_CAP_SECONDS,
    )
    run_parser.add_argument("--output", type=pathlib.Path, default=RAW_PATH)
    run_parser.add_argument(
        "--profiles-output",
        type=pathlib.Path,
        default=PROFILE_PATH,
    )
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--input", type=pathlib.Path, default=RAW_PATH)
    analyze_parser.add_argument(
        "--profiles",
        type=pathlib.Path,
        default=PROFILE_PATH,
    )
    sanity_parser = subparsers.add_parser("sanity")
    sanity_parser.add_argument("--output", type=pathlib.Path, default=SANITY_PATH)
    parsed = parser.parse_args(arguments)

    if parsed.command == "run":
        run_experiment(
            ranks=parsed.ranks,
            seeds=parsed.seeds,
            raw_path=parsed.output,
            profile_path=parsed.profiles_output,
            workers=parsed.workers,
            node_cap=parsed.node_cap,
            solution_budget=parsed.solution_budget,
            enumeration_time_cap=parsed.time_cap,
        )
    elif parsed.command == "analyze":
        from node_count_reporting import analyze

        analyze(parsed.input, parsed.profiles)
    else:
        sanity_check(parsed.output)


if __name__ == "__main__":
    main()
