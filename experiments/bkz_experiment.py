#!/usr/bin/env python3
"""Measured Gram-Schmidt profiles from a fixed-tour BKZ procedure.

Two subcommands.

  reduce   run BKZ over a parameter grid and append one JSON record per run
           to data/profiles.jsonl. Resumable, already-present rows are skipped.
  analyze  fit the profile models to each measured profile by minimizing the
           enumeration discrepancy, then emit CSV and LaTeX table bodies.

The fitting objective is the enumeration discrepancy of Theorem thm:stability,
    Delta_E = max_j |C(j) - Chat(j)|.
It is the exact uniform error for the enumeration log-proxy.
Every fitted model is forced to reproduce log det Lambda exactly, otherwise
Chat(n) is nonzero and the comparison is meaningless.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import platform
import sys
import time

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
PROFILE_PATH = DATA_DIR / "profiles.jsonl"
TABLE_DIR = DATA_DIR / "tables"

SATURATION_CONSTANT = math.sqrt(4.0 / 3.0)
BKZ_MAX_LOOPS = 8
PLATEAU_TOLERANCE = 0.05


# ---------------------------------------------------------------- primitives

def log_ball_volume(m: int) -> float:
    return 0.5 * m * math.log(math.pi) - math.lgamma(m / 2.0 + 1.0)


def log_gh(m: int) -> float:
    return -log_ball_volume(m) / m


def cumulative(ell: list[float]) -> list[float]:
    """Centered cumulative profile C(j), j = 0..n."""
    n = len(ell)
    mean = sum(ell) / n
    out = [0.0]
    for value in ell:
        out.append(out[-1] + value - mean)
    return out


def discrepancy(ell: list[float], model: list[float]) -> float:
    """Return Delta_E = max_j |C(j) - Chat(j)|."""
    a, b = cumulative(ell), cumulative(model)
    return max(abs(x - y) for x, y in zip(a, b))


def ball_constant(n: int, k: int) -> float:
    return log_ball_volume(k) - (k / n) * log_ball_volume(n)


def kappa(n: int, d: int) -> float:
    return (
        math.log(SATURATION_CONSTANT)
        + log_gh(n - d)
        - log_gh(n)
        - 0.5 * math.log((n - d) / n)
    )


def bottleneck(ell: list[float], omega: float = 0.0) -> tuple[int, float]:
    """Level k maximizing log N_k, and that maximum."""
    n = len(ell)
    cum = cumulative(ell)
    best_k, best_value = 1, -math.inf
    for k in range(1, n + 1):
        value = ball_constant(n, k) + cum[n - k] - k * omega
        if value > best_value:
            best_k, best_value = k, value
    return best_k, best_value


def admissible_depth(ell: list[float], omega: float = 0.0) -> int:
    """Largest d such that Phi_j <= kappa_{n,j} + omega for every j <= d."""
    n = len(ell)
    cum = cumulative(ell)
    depth = 0
    for d in range(1, n):
        if cum[d] / (n - d) > kappa(n, d) + omega:
            break
        depth = d
    return depth


def match_determinant(model: list[float], log_det: float) -> list[float]:
    """Shift a model profile so that its entries sum to log_det exactly."""
    n = len(model)
    shift = (log_det - sum(model)) / n
    return [value + shift for value in model]


# -------------------------------------------------------------------- models

def fit_gsa(ell: list[float]) -> list[float]:
    """Single slope. One free parameter after the determinant constraint."""
    n = len(ell)
    log_det = sum(ell)
    best, best_delta = None, math.inf
    lo, hi = 0.0, 2.0 * (max(ell) - min(ell)) / max(n - 1, 1) + 1e-6
    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        d1 = discrepancy(ell, match_determinant([-m1 * i for i in range(n)], log_det))
        d2 = discrepancy(ell, match_determinant([-m2 * i for i in range(n)], log_det))
        if d1 < d2:
            hi = m2
        else:
            lo = m1
    slope = 0.5 * (lo + hi)
    best = match_determinant([-slope * i for i in range(n)], log_det)
    best_delta = discrepancy(ell, best)
    return best


def hkz_shape_tail(length: int) -> list[float]:
    """Determinant-one profile satisfying the HKZ shape heuristic.

    Writing g_m = log gh(m) and T_m for the mean of the last m entries, the
    heuristic says L_{length-m+1} = g_m + T_m. Rearranging gives the suffix
    recurrence T_m = T_{m-1} + g_m / (m - 1), and determinant one fixes
    T_length = 0. The entries follow from the two relations.
    """
    if length <= 1:
        return [0.0] * max(length, 0)
    increments = [log_gh(m) / (m - 1) for m in range(2, length + 1)]
    suffix_mean = [0.0] * (length + 1)
    suffix_mean[1] = -sum(increments)
    for m in range(2, length + 1):
        suffix_mean[m] = suffix_mean[m - 1] + increments[m - 2]
    tail = [0.0] * length
    tail[length - 1] = suffix_mean[1]
    for m in range(2, length + 1):
        tail[length - m] = log_gh(m) + suffix_mean[m]
    return tail


def fit_tgsa(ell: list[float], block_size: int) -> list[float]:
    """GSA head of length n - beta joined to an HKZ-shaped tail of length beta."""
    n = len(ell)
    log_det = sum(ell)
    beta = min(max(block_size, 2), n)
    head_len = n - beta
    tail = hkz_shape_tail(beta)
    # Definition of the tail-adapted model. The decrement is nonpositive for
    # beta <= 12, where the modeled head rises. Section 5 reports those cells
    # separately as a formal extrapolation outside the intended regime.
    slope = 2.0 * log_gh(beta) / (beta - 1)
    head = [-slope * i for i in range(head_len)]
    offset = (head[-1] - slope if head else 0.0) - tail[0]
    model = head + [value + offset for value in tail]
    return match_determinant(model, log_det)


def zgsa_profile(n: int, plateau: int, slope_len: int, height: float,
                 step: float) -> list[float]:
    out = []
    for i in range(1, n + 1):
        if i <= plateau:
            out.append(height)
        elif i <= plateau + slope_len:
            out.append(height - (i - plateau) * step)
        else:
            out.append(height - slope_len * step)
    return out


def fit_zgsa(ell: list[float], log_q: float) -> tuple[list[float], int, int]:
    """Plateau at log q, linear descent, bounded tail.

    Searches integer zone lengths. For each pair the descent rate is fixed by
    the determinant constraint, so no continuous search is needed.
    """
    n = len(ell)
    log_det = sum(ell)
    best, best_delta, best_zones = None, math.inf, (0, 0)
    for plateau in range(0, n):
        for slope_len in range(1, n - plateau + 1):
            tail_len = n - plateau - slope_len
            # sum = plateau*Q + sum_{i=1..slope_len}(Q - i*step) + tail*(Q - slope_len*step)
            weight = slope_len * (slope_len + 1) / 2.0 + tail_len * slope_len
            if weight <= 0:
                continue
            step = (n * log_q - log_det) / weight
            # The feasible class of Definition def:qary-zgsa requires a
            # positive descent rate and a tail height in [0, Q). A positive
            # rate with a nonnegative slope length also forces H < Q.
            if step <= 0 or log_q - slope_len * step < 0:
                continue
            model = zgsa_profile(n, plateau, slope_len, log_q, step)
            delta = discrepancy(ell, model)
            if delta < best_delta:
                best, best_delta, best_zones = model, delta, (plateau, slope_len)
    if best is None:
        best, best_zones = fit_gsa(ell), (0, n - 1)
    return best, best_zones[0], best_zones[1]


def rmse(ell: list[float], model: list[float]) -> float:
    """Pointwise root mean square error between two profiles."""
    n = len(ell)
    return (sum((a - b) ** 2 for a, b in zip(ell, model)) / n) ** 0.5


def fit_gsa_by(ell: list[float], objective) -> list[float]:
    """Single slope, fitted by whichever objective is passed in."""
    n = len(ell)
    log_det = sum(ell)
    lo, hi = 0.0, 2.0 * (max(ell) - min(ell)) / max(n - 1, 1) + 1e-6

    def build(slope: float) -> list[float]:
        return match_determinant([-slope * i for i in range(n)], log_det)

    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        if objective(ell, build(m1)) < objective(ell, build(m2)):
            hi = m2
        else:
            lo = m1
    return build(0.5 * (lo + hi))


def fit_zgsa_by(ell: list[float], log_q: float, objective):
    """Three-zone model, fitted by whichever objective is passed in."""
    n = len(ell)
    log_det = sum(ell)
    best, best_score, best_zones = None, math.inf, (0, 0)
    for plateau in range(0, n):
        for slope_len in range(1, n - plateau + 1):
            tail_len = n - plateau - slope_len
            weight = slope_len * (slope_len + 1) / 2.0 + tail_len * slope_len
            if weight <= 0:
                continue
            step = (n * log_q - log_det) / weight
            if step <= 0 or log_q - slope_len * step < 0:
                continue
            model = zgsa_profile(n, plateau, slope_len, log_q, step)
            score = objective(ell, model)
            if score < best_score:
                best, best_score, best_zones = model, score, (plateau, slope_len)
    if best is None:
        best, best_zones = fit_gsa_by(ell, objective), (0, n - 1)
    return best, best_zones[0], best_zones[1]




# ------------------------------------------------------------------- reduce

WORKERS = max(1, min(24, (os.cpu_count() or 4) - 4))


FLOAT_TYPE = "ld"
JOB_TIMEOUT_SECONDS = 900


def reduce_one(job: tuple[int, int, int, int, int]) -> dict:
    """One BKZ reduction. Runs in a worker process.

    FPLLL can raise or hang on nearly flat profiles at larger ranks, so the
    reduction is guarded and any failure is recorded rather than dropped.
    Silently discarding failed cells would bias the reported grid toward the
    parameters that happen to be numerically easy.
    """
    import signal

    from fpylll import BKZ, GSO, IntegerMatrix, LLL, FPLLL

    n, q, t, beta, seed = job
    record = {"n": n, "q": q, "t": t, "beta": beta, "seed": seed}

    def on_alarm(signum, frame):
        raise TimeoutError("reduction exceeded the per-job limit")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(JOB_TIMEOUT_SECONDS)
    started = time.time()
    try:
        FPLLL.set_random_seed(seed + 1)
        basis = IntegerMatrix.random(n, "qary", k=t, q=q)
        gso = GSO.Mat(basis, float_type=FLOAT_TYPE)
        gso.update_gso()
        lll = LLL.Reduction(gso)
        lll()
        if beta > 2:
            BKZ.Reduction(gso, lll, BKZ.Param(
                block_size=beta, max_loops=BKZ_MAX_LOOPS,
                flags=BKZ.MAX_LOOPS | BKZ.GH_BND))()
        gso.update_gso()
        ell = [0.5 * math.log(gso.get_r(i, i)) for i in range(n)]
        record.update({"ell": ell, "log_det": sum(ell), "status": "ok"})
    except Exception as error:
        record.update({"status": "failed", "error": type(error).__name__,
                       "detail": str(error)[:120]})
    finally:
        signal.alarm(0)
    record["seconds"] = round(time.time() - started, 3)
    return record


def run_parallel(todo: list[tuple[int, int, int, int, int]]) -> None:
    """Run the outstanding jobs, one isolated subprocess each.

    FPLLL can abort at the C level on some inputs, which no Python handler can
    catch. Each job therefore runs in its own process, and a crash is recorded
    as a failed row instead of taking the run down with it.
    """
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed

    script = str(pathlib.Path(__file__).resolve())

    def launch(job: tuple[int, int, int, int, int]) -> dict:
        n, q, t, beta, seed = job
        args = [sys.executable, script, "job", str(n), str(q), str(t),
                str(beta), str(seed)]
        try:
            done = subprocess.run(args, capture_output=True, text=True,
                                  timeout=JOB_TIMEOUT_SECONDS)
            for line in reversed(done.stdout.splitlines()):
                if line.startswith("{"):
                    return json.loads(line)
            return {"n": n, "q": q, "t": t, "beta": beta, "seed": seed,
                    "status": "failed", "error": "Crash",
                    "detail": (done.stderr or "no output")[-120:], "seconds": 0.0}
        except subprocess.TimeoutExpired:
            return {"n": n, "q": q, "t": t, "beta": beta, "seed": seed,
                    "status": "failed", "error": "Timeout",
                    "detail": f"exceeded {JOB_TIMEOUT_SECONDS}s",
                    "seconds": float(JOB_TIMEOUT_SECONDS)}

    # Longest jobs first, so the tail of the run is not one straggler.
    todo = sorted(todo, key=lambda j: (-j[3], -j[0]))
    print(f"{len(todo)} runs on {WORKERS} workers", flush=True)
    finished = failures = 0
    with PROFILE_PATH.open("a") as handle, ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(launch, job) for job in todo]
        for future in as_completed(futures):
            row = future.result()
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            finished += 1
            failures += row.get("status") == "failed"
            if finished % 20 == 0 or finished == len(todo):
                print(f"  {finished}/{len(todo)} done, {failures} failed",
                      flush=True)


def run_grid(grid: list[tuple[int, int, int, int]], seeds: int) -> None:
    from fpylll import BKZ, GSO, IntegerMatrix, LLL, FPLLL

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if PROFILE_PATH.exists():
        for line in PROFILE_PATH.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["n"], row["q"], row["t"], row["beta"], row["seed"]))

    todo = [(n, q, t, beta, s) for n, q, t, beta in grid for s in range(seeds)
            if (n, q, t, beta, s) not in done]
    if WORKERS > 1 and todo:
        run_parallel(todo)
        return

    with PROFILE_PATH.open("a") as handle:
        for n, q, t, beta in grid:
            for seed in range(seeds):
                if (n, q, t, beta, seed) in done:
                    continue
                FPLLL.set_random_seed(seed + 1)
                basis = IntegerMatrix.random(n, "qary", k=t, q=q)
                started = time.time()
                LLL.reduction(basis)
                params = BKZ.Param(
                    block_size=beta,
                    max_loops=BKZ_MAX_LOOPS,
                    flags=BKZ.MAX_LOOPS | BKZ.GH_BND,
                )
                BKZ.reduction(basis, params)
                elapsed = time.time() - started
                gso = GSO.Mat(basis, float_type="ld")
                gso.update_gso()
                ell = [0.5 * math.log(gso.get_r(i, i)) for i in range(n)]
                handle.write(json.dumps({
                    "n": n, "q": q, "t": t, "beta": beta, "seed": seed,
                    "ell": ell, "log_det": sum(ell), "seconds": round(elapsed, 3),
                }) + "\n")
                handle.flush()
                print(f"  n={n} q={q} t={t} beta={beta} seed={seed} "
                      f"{elapsed:6.1f}s", flush=True)


def write_tabular(path: pathlib.Path, spec: str, header: str,
                  body: list[str]) -> None:
    """Emit a complete booktabs tabular, ready to \\input inside a table."""
    path.write_text(
        "\\begin{tabular}{" + spec + "}\n"
        "  \\toprule\n"
        "  " + header + " \\\\\n"
        "  \\midrule\n"
        + "\n".join(body) + "\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )


# ------------------------------------------------------------------ analyze

def plateau_length(ell: list[float], log_q: float) -> int:
    """Length of the leading run of entries within PLATEAU_TOLERANCE of log q.

    This is the q-plateau of the three-zone model. Counting every entry near
    log q instead, without requiring a leading run, changes the grid mean by
    under two entries and is available from the stored profiles.
    """
    length = 0
    for value in ell:
        if abs(value - log_q) > PLATEAU_TOLERANCE:
            break
        length += 1
    return length


def write_tgsa_blocksize_table(records: list[dict]) -> None:
    """Report model accuracy separately for beta=2 and beta in {20, 40}."""
    model_names = ("GSA", "TGSA", "ZGSA")
    blocksize_groups = (
        ("beta=2", r"$\beta=2$", (2,)),
        ("beta in {20,40}", r"$\beta\in\{20,40\}$", (20, 40)),
    )

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    summaries: dict[str, dict[str, tuple[float, float, float]]] = {}
    for group_name, _, block_sizes in blocksize_groups:
        summaries[group_name] = {}
        for model_name in model_names:
            subset = [
                record for record in records
                if record["model"] == model_name
                and record["beta"] in block_sizes
            ]
            summaries[group_name][model_name] = (
                mean([record["delta"] for record in subset]),
                mean([
                    abs(record["true_d"] - record["model_d"])
                    for record in subset
                ]),
                mean([
                    abs(record["true_k"] - record["model_k"])
                    for record in subset
                ]),
            )

    body = []
    for model_name in model_names:
        cells = []
        for group_name, _, _ in blocksize_groups:
            cells.extend(summaries[group_name][model_name])
        body.append(
            f"  {model_name} & "
            + " & ".join(f"${value:.2f}$" for value in cells)
            + r" \\"
        )
    group_headers = " & ".join(
        rf"\multicolumn{{3}}{{c}}{{{latex_label}}}"
        for _, latex_label, _ in blocksize_groups
    )
    write_tabular(
        TABLE_DIR / "tgsa_blocksize.tex",
        "@{}lrrrrrr@{}",
        "Model & " + group_headers + r" \\" + "\n"
        r"  \cmidrule(lr){2-4} \cmidrule(l){5-7}" + "\n"
        r"  & mean $\Delta_{\rm E}$ & mean $|\Delta d_{\max}|$ & "
        r"mean $|\Delta k^*|$ & mean $\Delta_{\rm E}$ & "
        r"mean $|\Delta d_{\max}|$ & mean $|\Delta k^*|$",
        body,
    )

    print("\nTGSA block-size split over completed profiles")
    for model_name in model_names:
        for group_name, _, _ in blocksize_groups:
            delta, d_error, k_error = summaries[group_name][model_name]
            print(
                f"  {model_name:<6} {group_name:<16} "
                f"mean Delta_E={delta:.2f}  "
                f"mean |true_d-model_d|={d_error:.2f}  "
                f"mean |true_k-model_k|={k_error:.2f}"
            )

    print("\nTGSA/ZGSA mean Delta_E ratios")
    for group_name, _, _ in blocksize_groups:
        ratio = (
            summaries[group_name]["TGSA"][0]
            / summaries[group_name]["ZGSA"][0]
        )
        print(f"  {group_name}: {ratio:.17g}")

    cell_deltas: dict[tuple[int, int, int, int], dict[str, list[float]]] = {}
    for record in records:
        if record["beta"] not in (20, 40):
            continue
        key = (record["n"], record["q"], record["t"], record["beta"])
        cell_deltas.setdefault(key, {}).setdefault(record["model"], []).append(
            record["delta"]
        )
    cell_ratios = [
        (
            mean(by_model["TGSA"]) / mean(by_model["ZGSA"]),
            key,
        )
        for key, by_model in cell_deltas.items()
    ]
    max_ratio, max_key = max(cell_ratios)
    print(
        "  maximum per-cell TGSA/ZGSA Delta_E ratio for beta in {20,40}: "
        f"{max_ratio:.17g} "
        f"(n={max_key[0]}, q={max_key[1]}, t={max_key[2]}, beta={max_key[3]})"
    )


def analyze() -> None:
    if not PROFILE_PATH.exists():
        sys.exit("no data, run the reduce subcommand first")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = [json.loads(line) for line in PROFILE_PATH.read_text().splitlines()
                if line.strip()]
    rows = [r for r in all_rows if r.get("status", "ok") == "ok"]
    failed = [r for r in all_rows if r.get("status") == "failed"]
    if failed:
        print(f"WARNING {len(failed)} of {len(all_rows)} reductions failed and "
              f"are excluded. Failed cells:")
        seen_fail = {}
        for r in failed:
            seen_fail.setdefault((r["n"], r["q"], r["t"], r["beta"]), []).append(
                r.get("error", "?"))
        for cell, errs in sorted(seen_fail.items()):
            print(f"  n={cell[0]} q={cell[1]} t={cell[2]} beta={cell[3]}: "
                  f"{len(errs)} runs, {sorted(set(errs))}")

    records = []
    for row in rows:
        ell, n = row["ell"], row["n"]
        log_q = math.log(row["q"])
        models = {
            "GSA": fit_gsa(ell),
            "TGSA": fit_tgsa(ell, row["beta"]),
        }
        zgsa, plateau_fit, slope_fit = fit_zgsa(ell, log_q)
        models["ZGSA"] = zgsa

        true_k, true_cost = bottleneck(ell)
        true_d = admissible_depth(ell)
        for name, model in models.items():
            delta = discrepancy(ell, model)
            model_k, model_cost = bottleneck(model)
            model_d = admissible_depth(model)
            records.append({
                "n": n, "q": row["q"], "t": row["t"], "beta": row["beta"],
                "seed": row["seed"], "model": name,
                "alpha": row["log_det"] / (n * log_q),
                "plateau": plateau_length(ell, log_q),
                "fitted_plateau": plateau_fit if name == "ZGSA" else -1,
                "fitted_slope_len": slope_fit if name == "ZGSA" else -1,
                "delta": delta,
                "true_k": true_k, "model_k": model_k,
                "true_cost": true_cost, "model_cost": model_cost,
                "cost_error": abs(true_cost - model_cost),
                "bound_holds": abs(true_cost - model_cost) <= delta + 1e-9,
                "true_d": true_d, "model_d": model_d,
                "seconds": row["seconds"],
            })

    # Referee request: publish every fitted three-zone parameter set.
    zgsa_rows = [r for r in records if r["model"] == "ZGSA"]
    with (DATA_DIR / "zgsa_fits.csv").open("w") as handle:
        handle.write("n,q,t,beta,seed,n_q,n_GSA,n_1,delta_rate,H\n")
        for r in zgsa_rows:
            n = r["n"]
            n_q, n_gsa = r["fitted_plateau"], r["fitted_slope_len"]
            n_1 = n - n_q - n_gsa
            log_q = math.log(r["q"])
            weight = n_gsa * (n_gsa + 1) / 2.0 + n_1 * n_gsa
            log_det = r["alpha"] * n * log_q
            rate = (n * log_q - log_det) / weight if weight else 0.0
            handle.write(f"{n},{r['q']},{r['t']},{r['beta']},{r['seed']},"
                         f"{n_q},{n_gsa},{n_1},{rate:.6f},"
                         f"{log_q - n_gsa * rate:.6f}\n")

    columns = list(records[0].keys())
    csv_path = DATA_DIR / "results.csv"
    with csv_path.open("w") as handle:
        handle.write(",".join(columns) + "\n")
        for record in records:
            handle.write(",".join(str(record[c]) for c in columns) + "\n")

    violations = [r for r in records if not r["bound_holds"]]
    print(f"records {len(records)}, stability-bound violations {len(violations)}")

    print("\nmean Delta_E by model and regime")
    print(f"{'regime':<28}{'GSA':>10}{'TGSA':>10}{'ZGSA':>10}")
    groups: dict[tuple, dict[str, list[float]]] = {}
    for record in records:
        key = (record["n"], record["q"], record["t"], record["beta"])
        groups.setdefault(key, {}).setdefault(record["model"], []).append(record["delta"])
    lines = []
    for key in sorted(groups):
        n, q, t, beta = key
        by_model = groups[key]
        cells = []
        for m in ("GSA", "TGSA", "ZGSA"):
            vals = by_model[m]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
            cells.append((mean, var ** 0.5))
        label = f"n={n} q={q} t={t} beta={beta}"
        print(f"{label:<28}" + "".join(f"{m:>8.2f}+-{s:<5.2f}" for m, s in cells))
        plateaus = [r["plateau"] for r in records if
                    (r["n"], r["q"], r["t"], r["beta"]) == key]
        plateau = sum(plateaus) / len(plateaus)
        lines.append(
            f"  ${n}$ & ${q}$ & ${t}$ & ${beta}$ & ${plateau:.0f}$ & "
            + " & ".join(f"${m:.2f}\\pm{s:.2f}$" for m, s in cells) + r" \\"
        )
    # The Section 5 discrepancy tables are emitted by
    # compact_profile_tables, which owns their row format.

    # Section 5 reports the plateau size of the cells the three-zone model wins
    # against the cells it loses to the geometric model.
    cell_delta: dict[tuple, dict[str, list[float]]] = {}
    cell_plateau: dict[tuple, list[int]] = {}
    for r in records:
        key = (r["n"], r["q"], r["t"], r["beta"])
        cell_delta.setdefault(key, {}).setdefault(r["model"], []).append(r["delta"])
        cell_plateau.setdefault(key, []).append(r["plateau"])
    won, lost = [], []
    for key, by_model in cell_delta.items():
        if not {"GSA", "TGSA", "ZGSA"} <= set(by_model):
            continue
        means = {m: sum(v) / len(v) for m, v in by_model.items()}
        plateau = sum(cell_plateau[key]) / len(cell_plateau[key])
        if means["ZGSA"] == min(means.values()):
            won.append(plateau)
        elif means["GSA"] == min(means.values()):
            lost.append(plateau)
    print(f"\nthree-zone wins {len(won)} of {len(cell_delta)} cells with data")
    if won:
        print(f"  mean plateau in cells it wins  {sum(won) / len(won):6.1f}")
    if lost:
        print(f"  mean plateau in cells it loses {sum(lost) / len(lost):6.1f}")

    print("\nprediction accuracy by model")
    prediction_lines = []
    for name in ("GSA", "TGSA", "ZGSA"):
        subset = [r for r in records if r["model"] == name]
        k_err = sum(abs(r["true_k"] - r["model_k"]) for r in subset) / len(subset)
        d_err = sum(abs(r["true_d"] - r["model_d"]) for r in subset) / len(subset)
        cost_err = sum(r["cost_error"] for r in subset) / len(subset)
        ratio = sum(r["cost_error"] / r["delta"] for r in subset if r["delta"] > 0)
        ratio /= max(sum(1 for r in subset if r["delta"] > 0), 1)
        print(f"  {name:<6} mean |dk*|={k_err:6.2f}  mean |dd_max|={d_err:6.2f}  "
              f"mean cost error={cost_err:8.2f}  mean error/Delta_E={ratio:.3f}")
        prediction_lines.append(
            f"  {name} & ${k_err:.2f}$ & ${d_err:.2f}$ & ${cost_err:.2f}$ & "
            f"${ratio:.3f}$" + r" \\"
        )
    write_tabular(
        TABLE_DIR / "prediction.tex",
        "@{}lrrrr@{}",
        r"Model & mean $|\Delta k^*|$ & mean $|\Delta d_{\max}|$ & "
        r"mean cost error & mean error$/\Delta_{\rm E}$",
        prediction_lines,
    )
    write_tgsa_blocksize_table(records)

    # The Section 6 family table is emitted by compact_profile_tables, which
    # owns the selection rule and the row format.
    from compact_profile_tables import render as render_compact_tables

    render_compact_tables()

    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "bkz_max_loops": BKZ_MAX_LOOPS,
        "saturation_constant": SATURATION_CONSTANT,
        "runs": len(rows),
    }
    try:
        import fpylll
        meta["fpylll"] = fpylll.__version__
    except Exception:
        pass
    (DATA_DIR / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote {csv_path} and the compact manuscript tables")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "job":
        n, q, t, beta, seed = (int(v) for v in sys.argv[2:7])
        print(json.dumps(reduce_one((n, q, t, beta, seed))))
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["reduce", "analyze", "tables"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    if args.command == "analyze":
        analyze()
        return
    if args.command == "tables":
        from compact_profile_tables import render

        render()
        return

    if args.pilot:
        grid = [(60, 521, 30, 20)]
    else:
        # The three regimes of Section 4.3 are reached by moving log q against
        # the spread that the reduction can produce. Small q with weak
        # reduction keeps both plateaus, large q with strong reduction leaves
        # a plain geometric profile, and the middle cells interpolate.
        grid = []
        for n in (120, 160, 200, 256, 320):
            for q in (13, 31, 127, 1031):
                for t in (n // 2, (3 * n) // 4):
                    for beta in (2, 20, 40):
                        grid.append((n, q, t, beta))
    print(f"grid of {len(grid)} settings, {args.seeds} seeds each")
    run_grid(grid, args.seeds)


if __name__ == "__main__":
    main()
