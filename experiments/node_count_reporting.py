"""Reporting helpers for the fixed-radius projected point-count experiment."""

from __future__ import annotations

import csv
import json
import math
import pathlib
import platform
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Sequence

from node_counts import (
    BKZ_MAX_LOOPS,
    DATA_DIR,
    ENUMERATION_TIME_CAP_SECONDS,
    FLOAT_TYPE,
    NODE_CAP,
    PROCESS_BUFFER_SECONDS,
    REDUCTION_TIMEOUT_SECONDS,
    SOLUTION_BUDGET,
    WORKER_COUNT,
    safe_ratio,
)


DETAIL_CSV_PATH = DATA_DIR / "fixed_radius_counts.csv"
INSTANCE_CSV_PATH = DATA_DIR / "fixed_radius_instances.csv"
RANK_CSV_PATH = DATA_DIR / "fixed_radius_ranks.csv"
SUMMARY_PATH = DATA_DIR / "fixed_radius_summary.json"
METADATA_PATH = DATA_DIR / "fixed_radius_metadata.json"
TABLE_DIR = DATA_DIR / "tables"
TABLE_PATH = TABLE_DIR / "node_counts.tex"

JsonRecord = dict[str, Any]


def load_jsonl(path: pathlib.Path) -> list[JsonRecord]:
    """Load nonempty JSON lines from a path."""
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def deduplicate_profiles(path: pathlib.Path) -> dict[str, JsonRecord]:
    """Keep the last reduced profile for each stable job identifier."""
    return {str(row["job_id"]): row for row in load_jsonl(path)}


def normalize_terminal_status(row: JsonRecord) -> JsonRecord:
    """Classify a delayed Python alarm reported through child stderr."""
    if (
        row.get("status") == "enumeration_crash"
        and "TimeoutError" in str(row.get("detail", ""))
    ):
        return {
            **row,
            "status": "time_cap",
            "original_status": "enumeration_crash",
        }
    return row


def deduplicate_rows(path: pathlib.Path) -> list[JsonRecord]:
    """Keep the last record for each job and projected level."""
    rows: dict[tuple[str, int], JsonRecord] = {}
    for raw_row in load_jsonl(path):
        row = normalize_terminal_status(raw_row)
        key = (str(row["job_id"]), int(row["level"]))
        rows[key] = row
    return [rows[key] for key in sorted(rows)]


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a nonempty sequence."""
    return sum(values) / len(values)


def sample_standard_deviation(values: Sequence[float]) -> float:
    """Return the sample standard deviation, or zero for one value."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def quantile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated empirical quantile."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def write_csv(
    path: pathlib.Path,
    rows: Sequence[JsonRecord],
    columns: Sequence[str],
) -> None:
    """Write selected record fields as a CSV file."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def exact_measurements(rows: Sequence[JsonRecord]) -> list[JsonRecord]:
    """Return rows whose point count is complete and below the budget."""
    return [
        row
        for row in rows
        if row.get("point_count") is not None
        and not bool(row.get("solution_budget_reached", False))
    ]


def profile_regime(profile: JsonRecord) -> str:
    """Classify the visible q-plateau as absent, transitional, or long."""
    plateau_length = int(profile.get("plateau_length", 0))
    rank = int(profile["n"])
    if plateau_length == 0:
        return "no_plateau"
    if plateau_length < max(2, rank // 8):
        return "transition"
    return "long_plateau"


def base_instance_fields(
    job_identifier: str,
    profile: JsonRecord | None,
    rows: Sequence[JsonRecord],
) -> JsonRecord:
    """Return identifying fields from the profile or a durable raw row."""
    source = profile if profile is not None else rows[-1]
    return {
        "job_id": job_identifier,
        "n": int(source["n"]),
        "q": int(source["q"]),
        "lattice_t": int(source["lattice_t"]),
        "beta": int(source["beta"]),
        "seed": int(source["seed"]),
    }


def summarize_instances(
    rows: Sequence[JsonRecord],
    profiles: dict[str, JsonRecord],
) -> list[JsonRecord]:
    """Summarize completion, peak agreement, and log-ratio bias per job."""
    grouped: dict[str, list[JsonRecord]] = defaultdict(list)
    for row in rows:
        grouped[str(row["job_id"])].append(row)
    summaries: list[JsonRecord] = []
    for job_identifier in sorted(set(grouped) | set(profiles)):
        job_rows = grouped.get(job_identifier, [])
        profile = profiles.get(job_identifier)
        measured = exact_measurements(job_rows)
        terminals = [row for row in job_rows if row.get("terminal")]
        terminal = terminals[-1] if terminals else {}
        base = base_instance_fields(
            job_identifier,
            profile,
            job_rows,
        )
        rank = int(base["n"])
        terminal_status = str(terminal.get("status", "missing"))
        levels = {int(row["level"]) for row in measured}
        rank_complete = (
            terminal_status == "ok"
            and levels == set(range(1, rank + 1))
        )
        predicted_k = (
            int(profile["predicted_bottleneck_k"])
            if profile is not None
            else None
        )
        log_ratios = [float(row["log_ratio"]) for row in measured]
        if measured:
            measured_peak_count = max(
                int(row["point_count"]) for row in measured
            )
            peak_levels = sorted(
                int(row["level"])
                for row in measured
                if int(row["point_count"]) == measured_peak_count
            )
        else:
            measured_peak_count = None
            peak_levels = []
        if predicted_k is not None and peak_levels:
            nearest_peak = min(
                peak_levels,
                key=lambda level: (abs(level - predicted_k), level),
            )
            peak_offset = nearest_peak - predicted_k
            exact_peak_match = predicted_k in peak_levels
        else:
            nearest_peak = None
            peak_offset = None
            exact_peak_match = False
        max_reached = max(levels, default=0)
        summaries.append(
            {
                **base,
                "regime": profile_regime(profile) if profile else "unknown",
                "plateau_length": (
                    int(profile["plateau_length"]) if profile else None
                ),
                "terminal_status": terminal_status,
                "rank_complete": rank_complete,
                "levels_measured": len(levels),
                "max_reached_k": max_reached,
                "predicted_bottleneck_k": predicted_k,
                "predicted_bottleneck_passed": (
                    max_reached > predicted_k if predicted_k is not None else False
                ),
                "measured_peak_count": measured_peak_count,
                "measured_peak_levels": json.dumps(peak_levels),
                "nearest_measured_peak_k": nearest_peak,
                "peak_offset": peak_offset,
                "absolute_peak_gap": (
                    abs(peak_offset) if peak_offset is not None else None
                ),
                "exact_peak_match": exact_peak_match,
                "mean_log_ratio": mean(log_ratios) if log_ratios else None,
                "sd_log_ratio": (
                    sample_standard_deviation(log_ratios)
                    if log_ratios
                    else None
                ),
                "minimum_log_ratio": min(log_ratios) if log_ratios else None,
                "maximum_log_ratio": max(log_ratios) if log_ratios else None,
                "terminal_detail": terminal.get("detail", ""),
            }
        )
    return summaries


def summarize_rank(
    rank: int,
    rows: Sequence[JsonRecord],
    instances: Sequence[JsonRecord],
) -> JsonRecord:
    """Build one rank-level row from exact completed sweeps."""
    rank_instances = [row for row in instances if int(row["n"]) == rank]
    complete = [row for row in rank_instances if bool(row["rank_complete"])]
    complete_ids = {str(row["job_id"]) for row in complete}
    measured = [
        row
        for row in exact_measurements(rows)
        if str(row["job_id"]) in complete_ids
    ]
    log_ratios = [float(row["log_ratio"]) for row in measured]
    peak_offsets = [int(row["peak_offset"]) for row in complete]
    sweep_means = [float(row["mean_log_ratio"]) for row in complete]
    sweep_mean_sd = (
        sample_standard_deviation(sweep_means) if sweep_means else None
    )
    terminal_statuses = Counter(
        str(row["terminal_status"]) for row in rank_instances
    )
    return {
        "rank": rank,
        "instances_total": len(rank_instances),
        "instances_complete": len(complete),
        "grid_complete": len(complete) == len(rank_instances),
        "measured_pairs": len(measured),
        "predicted_bottleneck_passed": sum(
            bool(row["predicted_bottleneck_passed"]) for row in complete
        ),
        "exact_peak_matches": sum(
            bool(row["exact_peak_match"]) for row in complete
        ),
        "mean_absolute_peak_gap": mean(
            [abs(value) for value in peak_offsets]
        ) if peak_offsets else None,
        "median_absolute_peak_gap": statistics.median(
            [abs(value) for value in peak_offsets]
        ) if peak_offsets else None,
        "mean_peak_offset": mean(peak_offsets) if peak_offsets else None,
        "minimum_peak_offset": min(peak_offsets) if peak_offsets else None,
        "maximum_peak_offset": max(peak_offsets) if peak_offsets else None,
        "mean_log_ratio": mean(log_ratios) if log_ratios else None,
        "sd_log_ratio": (
            sample_standard_deviation(log_ratios) if log_ratios else None
        ),
        "sweep_mean_log_ratio_sd": sweep_mean_sd,
        "sweep_mean_log_ratio_se": (
            sweep_mean_sd / math.sqrt(len(sweep_means))
            if sweep_mean_sd is not None and sweep_means
            else None
        ),
        "q05_log_ratio": quantile(log_ratios, 0.05) if log_ratios else None,
        "median_log_ratio": quantile(log_ratios, 0.5) if log_ratios else None,
        "q95_log_ratio": quantile(log_ratios, 0.95) if log_ratios else None,
        "geometric_mean_ratio": (
            safe_ratio(mean(log_ratios)) if log_ratios else None
        ),
        "terminal_statuses": json.dumps(
            terminal_statuses,
            sort_keys=True,
        ),
    }


def pooled_primary_summary(
    rows: Sequence[JsonRecord],
    instances: Sequence[JsonRecord],
    complete_ranks: Sequence[int],
) -> JsonRecord:
    """Summarize all instances belonging to fully completed rank grids."""
    primary = [
        row
        for row in instances
        if int(row["n"]) in complete_ranks and bool(row["rank_complete"])
    ]
    identifiers = {str(row["job_id"]) for row in primary}
    measured = [
        row
        for row in exact_measurements(rows)
        if str(row["job_id"]) in identifiers
    ]
    log_ratios = [float(row["log_ratio"]) for row in measured]
    peak_offsets = [int(row["peak_offset"]) for row in primary]
    sweep_means = [float(row["mean_log_ratio"]) for row in primary]
    sweep_mean_sd = sample_standard_deviation(sweep_means)
    return {
        "instances": len(primary),
        "measured_pairs": len(measured),
        "predicted_bottleneck_passed": sum(
            bool(row["predicted_bottleneck_passed"]) for row in primary
        ),
        "exact_peak_matches": sum(
            bool(row["exact_peak_match"]) for row in primary
        ),
        "mean_absolute_peak_gap": mean(
            [abs(value) for value in peak_offsets]
        ),
        "median_absolute_peak_gap": statistics.median(
            [abs(value) for value in peak_offsets]
        ),
        "mean_peak_offset": mean(peak_offsets),
        "minimum_peak_offset": min(peak_offsets),
        "maximum_peak_offset": max(peak_offsets),
        "mean_log_ratio": mean(log_ratios),
        "sd_log_ratio": sample_standard_deviation(log_ratios),
        "sweep_mean_log_ratio_sd": sweep_mean_sd,
        "sweep_mean_log_ratio_se": sweep_mean_sd / math.sqrt(len(sweep_means)),
        "q05_log_ratio": quantile(log_ratios, 0.05),
        "median_log_ratio": quantile(log_ratios, 0.5),
        "q95_log_ratio": quantile(log_ratios, 0.95),
        "geometric_mean_ratio": safe_ratio(mean(log_ratios)),
    }


def format_float(value: float | None, digits: int = 2) -> str:
    """Format an optional finite statistic for LaTeX output."""
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_latex_table(rank_rows: Sequence[JsonRecord]) -> None:
    """Emit the rank-stratified table used by the manuscript."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    body: list[str] = []
    for row in rank_rows:
        log_ratio = (
            f"${format_float(row['mean_log_ratio'])}"
            r"\mathbin{\pm}"
            f"{format_float(row['sd_log_ratio'])}$"
        )
        body.append(
            "  "
            + " & ".join(
                [
                    f"${row['rank']}$",
                    f"${row['instances_complete']}/{row['instances_total']}$",
                    f"${row['predicted_bottleneck_passed']}/{row['instances_complete']}$",
                    f"${row['exact_peak_matches']}/{row['instances_complete']}$",
                    f"${format_float(row['mean_absolute_peak_gap'])}$",
                    f"${format_float(row['mean_peak_offset'])}$",
                    log_ratio,
                ]
            )
            + r" \\"
        )
    TABLE_PATH.write_text(
        "\\begin{tabular}{@{}lrrrrrr@{}}\n"
        "  \\toprule\n"
        "  $n$ & complete & $k^*$ passed & peak match & mean abs. gap & "
        "mean offset & mean log ratio " r"\\" "\n"
        "  \\midrule\n"
        + "\n".join(body)
        + "\n  \\bottomrule\n"
        "\\end{tabular}\n"
    )


def analyze(raw_path: pathlib.Path, profile_path: pathlib.Path) -> None:
    """Create detailed reports and headline fixed-radius statistics."""
    if not raw_path.exists() or not profile_path.exists():
        raise SystemExit("fixed-radius data are missing, run the experiment first")
    rows = deduplicate_rows(raw_path)
    profiles = deduplicate_profiles(profile_path)
    instances = summarize_instances(rows, profiles)
    ranks = sorted({int(row["n"]) for row in instances})
    rank_rows = [summarize_rank(rank, rows, instances) for rank in ranks]
    complete_ranks = [
        int(row["rank"])
        for row in rank_rows
        if bool(row["grid_complete"])
    ]
    if not complete_ranks:
        raise SystemExit("no rank has a complete fixed-radius grid")
    primary = pooled_primary_summary(rows, instances, complete_ranks)
    regime_counts = Counter(
        str(row["regime"])
        for row in instances
        if int(row["n"]) in complete_ranks
    )
    all_terminal_statuses = Counter(
        str(row["terminal_status"]) for row in instances
    )
    summary = {
        "attempted_ranks": ranks,
        "complete_ranks": complete_ranks,
        "largest_complete_rank": max(complete_ranks),
        "terminal_statuses": dict(sorted(all_terminal_statuses.items())),
        "regime_counts_on_complete_ranks": dict(sorted(regime_counts.items())),
        "rank_summaries": rank_rows,
        "primary_complete_rank_summary": primary,
    }

    detail_columns = [
        "job_id",
        "n",
        "q",
        "lattice_t",
        "beta",
        "seed",
        "level",
        "status",
        "terminal",
        "terminal_reason",
        "point_count",
        "nonzero_sign_orbits",
        "log_measured_points",
        "predicted_log_n",
        "ratio",
        "log_ratio",
        "nodes",
        "seconds",
        "enumeration_seconds_total",
        "fixed_radius",
        "solution_budget",
        "solution_budget_reached",
        "evaluator_strategy",
        "predicted_bottleneck_k",
        "predicted_bottleneck_log_n",
        "detail",
        "original_status",
    ]
    write_csv(DETAIL_CSV_PATH, rows, detail_columns)
    write_csv(INSTANCE_CSV_PATH, instances, list(instances[0]))
    write_csv(RANK_CSV_PATH, rank_rows, list(rank_rows[0]))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_latex_table(rank_rows)

    metadata: JsonRecord = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "worker_count_default": WORKER_COUNT,
        "node_cap": NODE_CAP,
        "node_cap_semantics": (
            "post-call threshold; does not interrupt active enumeration"
        ),
        "solution_budget": SOLUTION_BUDGET,
        "enumeration_time_cap_seconds": ENUMERATION_TIME_CAP_SECONDS,
        "hard_process_timeout_seconds": (
            REDUCTION_TIMEOUT_SECONDS
            + ENUMERATION_TIME_CAP_SECONDS
            + PROCESS_BUFFER_SECONDS
        ),
        "reduction_timeout_seconds": REDUCTION_TIMEOUT_SECONDS,
        "float_type": FLOAT_TYPE,
        "bkz_max_loops": BKZ_MAX_LOOPS,
        "radius": "GH(Lambda)",
        "max_dist": "GH(Lambda)^2",
        "max_dist_expo": 0,
        "enumeration_strategy": "FIRST_N_SOLUTIONS",
        "radius_update": "disabled",
        "point_count": "2 * len(solutions) + 1",
        "zero_included": True,
        "sign_pair_representatives": "one per nonzero pair",
        "proxy_comparator": "single-level N_k(R)",
    }
    try:
        import fpylll

        metadata["fpylll"] = fpylll.__version__
    except Exception:
        metadata["fpylll"] = "unavailable"
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {DETAIL_CSV_PATH}")
    print(f"wrote {INSTANCE_CSV_PATH}")
    print(f"wrote {RANK_CSV_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {TABLE_PATH}")
