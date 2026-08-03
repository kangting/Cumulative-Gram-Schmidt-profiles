#!/usr/bin/env python3
"""Generate the manuscript figures from stored experiment data."""

from __future__ import annotations

import csv
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any, Sequence

import matplotlib.pyplot as plt

from bkz_experiment import cumulative, discrepancy, fit_gsa, fit_zgsa

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
FIGURE_DIR = DATA_DIR / "figures"
PROFILE_PATH = DATA_DIR / "profiles.jsonl"
RESULTS_PATH = DATA_DIR / "results.csv"
COUNT_PATH = DATA_DIR / "fixed_radius_counts.csv"
INSTANCE_PATH = DATA_DIR / "fixed_radius_instances.csv"
SELECTION_PATH = FIGURE_DIR / "selections.json"

JsonRecord = dict[str, Any]
ProfileKey = tuple[int, int, int, int, int]


def load_csv(path: pathlib.Path) -> list[JsonRecord]:
    """Load a CSV file as dictionaries."""
    with path.open() as handle:
        return list(csv.DictReader(handle))


def profile_key(record: JsonRecord) -> ProfileKey:
    """Return the parameter key for one measured profile."""
    return (
        int(record["n"]),
        int(record["q"]),
        int(record["t"]),
        int(record["beta"]),
        int(record["seed"]),
    )


def load_profiles() -> dict[ProfileKey, JsonRecord]:
    """Load completed profiles and retain the last record for each key."""
    profiles: dict[ProfileKey, JsonRecord] = {}
    for line in PROFILE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status", "ok") != "ok":
            continue
        profiles[profile_key(record)] = record
    return profiles


def model_discrepancies() -> dict[ProfileKey, dict[str, float]]:
    """Return the stored GSA and ZGSA discrepancies for each profile."""
    grouped: dict[ProfileKey, dict[str, float]] = defaultdict(dict)
    for record in load_csv(RESULTS_PATH):
        model = str(record["model"])
        if model not in {"GSA", "ZGSA"}:
            continue
        grouped[profile_key(record)][model] = float(record["delta"])
    return {
        key: values
        for key, values in grouped.items()
        if set(values) == {"GSA", "ZGSA"}
    }


def select_profile_panels() -> list[tuple[str, ProfileKey]]:
    """Select clear GSA, clear ZGSA, and nearest-tie profiles."""
    discrepancies = model_discrepancies()
    gaps = {
        key: values["ZGSA"] - values["GSA"]
        for key, values in discrepancies.items()
    }
    gsa_key = max((key for key, gap in gaps.items() if gap > 0), key=gaps.get)
    zgsa_key = min((key for key, gap in gaps.items() if gap < 0), key=gaps.get)
    excluded = {gsa_key, zgsa_key}
    tie_key = min(
        (key for key in gaps if key not in excluded),
        key=lambda key: abs(gaps[key]),
    )
    return [
        ("geometric model wins", gsa_key),
        ("three-zone model wins", zgsa_key),
        ("geometric model wins narrowly", tie_key),
    ]


def plot_profile_panels(
    selections: Sequence[tuple[str, ProfileKey]],
    profiles: dict[ProfileKey, JsonRecord],
) -> list[JsonRecord]:
    """Plot measured profiles and paths with geometric and three-zone models."""
    discrepancies = model_discrepancies()
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 5.5), sharex="col")
    metadata: list[JsonRecord] = []
    for column, (category, key) in enumerate(selections):
        record = profiles[key]
        ell = [float(value) for value in record["ell"]]
        gsa = fit_gsa(ell)
        zgsa, _, _ = fit_zgsa(ell, math.log(key[1]))
        stored = discrepancies[key]
        assert abs(discrepancy(ell, gsa) - stored["GSA"]) < 1e-6
        assert abs(discrepancy(ell, zgsa) - stored["ZGSA"]) < 1e-6

        profile_axis = axes[0, column]
        path_axis = axes[1, column]
        indices = list(range(1, len(ell) + 1))
        path_indices = list(range(len(ell) + 1))
        profile_axis.plot(indices, ell, color="black", linewidth=1.2, label="measured")
        profile_axis.plot(indices, gsa, color="#2563a7", linestyle="--", label="GSA")
        profile_axis.plot(indices, zgsa, color="#b23a48", linestyle=":", label="ZGSA")
        path_axis.plot(path_indices, cumulative(ell), color="black", linewidth=1.2)
        path_axis.plot(path_indices, cumulative(gsa), color="#2563a7", linestyle="--")
        path_axis.plot(path_indices, cumulative(zgsa), color="#b23a48", linestyle=":")
        rank, modulus, lattice_t, beta, seed = key
        profile_axis.set_title(category, fontsize=8)
        profile_axis.grid(alpha=0.2)
        path_axis.grid(alpha=0.2)
        path_axis.set_xlabel("index")
        if column == 0:
            profile_axis.set_ylabel(r"$\ell_i$")
            path_axis.set_ylabel(r"$C_B(j)$")
        metadata.append(
            {
                "category": category,
                "n": rank,
                "q": modulus,
                "t": lattice_t,
                "beta": beta,
                "seed": seed,
                "gsa_delta_e": stored["GSA"],
                "zgsa_delta_e": stored["ZGSA"],
            }
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(FIGURE_DIR / "profile_fits.pdf", bbox_inches="tight")
    plt.close(figure)
    return metadata


def completed_sweeps() -> dict[int, list[JsonRecord]]:
    """Group completed fixed-radius sweeps by rank."""
    grouped: dict[int, list[JsonRecord]] = defaultdict(list)
    for record in load_csv(INSTANCE_PATH):
        if record["rank_complete"] != "True":
            continue
        grouped[int(record["n"])].append(record)
    return grouped


def select_typical_sweep(rank: int, sweeps: Sequence[JsonRecord]) -> JsonRecord:
    """Select the sweep whose mean log ratio is closest to the rank median."""
    means = [float(record["mean_log_ratio"]) for record in sweeps]
    median = statistics.median(means)
    return min(sweeps, key=lambda record: abs(float(record["mean_log_ratio"]) - median))


def plot_fixed_radius_panels() -> list[JsonRecord]:
    """Plot one typical completed sweep at ranks 40 and 44."""
    sweeps = completed_sweeps()
    selections = [select_typical_sweep(rank, sweeps[rank]) for rank in (40, 44)]
    counts = load_csv(COUNT_PATH)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    metadata: list[JsonRecord] = []
    for axis, selection in zip(axes, selections):
        identifier = str(selection["job_id"])
        rows = [record for record in counts if record["job_id"] == identifier]
        rows.sort(key=lambda record: int(record["level"]))
        levels = [int(record["level"]) for record in rows]
        measured = [float(record["log_measured_points"]) for record in rows]
        predicted = [float(record["predicted_log_n"]) for record in rows]
        predicted_peak = int(selection["predicted_bottleneck_k"])
        measured_peak = int(selection["nearest_measured_peak_k"])
        assert predicted_peak == measured_peak

        axis.plot(
            levels,
            predicted,
            color="#2563a7",
            linewidth=1.4,
            label=r"$\log N_k(R)$",
        )
        axis.plot(
            levels,
            measured,
            color="#b23a48",
            marker="o",
            markersize=2.2,
            linewidth=0.9,
            label="measured log count",
        )
        axis.axvline(predicted_peak, color="0.4", linewidth=0.8, linestyle="--")
        axis.set_title(f"rank {selection['n']}, peak k={predicted_peak}", fontsize=9)
        axis.set_xlabel(r"projected rank $k$")
        axis.grid(alpha=0.2)
        metadata.append(
            {
                "job_id": identifier,
                "rank": int(selection["n"]),
                "predicted_peak": predicted_peak,
                "measured_peak": measured_peak,
                "selection_rule": "mean log ratio nearest the rank median",
            }
        )
    axes[0].set_ylabel("natural logarithm")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(FIGURE_DIR / "fixed_radius_peaks.pdf", bbox_inches="tight")
    plt.close(figure)
    return metadata


def main() -> None:
    """Generate both figures and record the data-derived selections."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles()
    profile_metadata = plot_profile_panels(select_profile_panels(), profiles)
    sweep_metadata = plot_fixed_radius_panels()
    selections = {
        "profile_panels": profile_metadata,
        "fixed_radius_panels": sweep_metadata,
    }
    SELECTION_PATH.write_text(json.dumps(selections, indent=2, sort_keys=True) + "\n")
    print(f"wrote {FIGURE_DIR / 'profile_fits.pdf'}")
    print(f"wrote {FIGURE_DIR / 'fixed_radius_peaks.pdf'}")
    print(f"wrote {SELECTION_PATH}")


if __name__ == "__main__":
    main()
