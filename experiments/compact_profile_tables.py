"""Render page-sized profile tables from the published per-seed CSV."""

from __future__ import annotations

import csv
import math
import pathlib
from collections import defaultdict
from typing import Any, Sequence


DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
RESULTS_PATH = DATA_DIR / "results.csv"
TABLE_DIR = DATA_DIR / "tables"

JsonRecord = dict[str, Any]
CellKey = tuple[int, int, int, int]


def log_ball_volume(dimension: int) -> float:
    """Return the logarithm of the dimension-dimensional unit ball."""
    return (
        0.5 * dimension * math.log(math.pi)
        - math.lgamma(dimension / 2.0 + 1.0)
    )


def log_gh(dimension: int) -> float:
    """Return the logarithmic determinant-one Gaussian heuristic."""
    return -log_ball_volume(dimension) / dimension


def mean(values: Sequence[float]) -> float:
    """Return the mean of a nonempty sequence."""
    return sum(values) / len(values)


def sample_standard_deviation(values: Sequence[float]) -> float:
    """Return the sample standard deviation, or zero for one value."""
    value_mean = mean(values)
    variance = sum((value - value_mean) ** 2 for value in values)
    variance /= max(len(values) - 1, 1)
    return math.sqrt(variance)


def load_records(path: pathlib.Path = RESULTS_PATH) -> list[JsonRecord]:
    """Load numeric fitting records from the published CSV."""
    integer_fields = {
        "n",
        "q",
        "t",
        "beta",
        "seed",
        "plateau",
        "fitted_plateau",
        "true_k",
        "true_d",
    }
    float_fields = {"alpha", "delta"}
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    records: list[JsonRecord] = []
    for row in rows:
        records.append(
            {
                **row,
                **{field: int(float(row[field])) for field in integer_fields},
                **{field: float(row[field]) for field in float_fields},
            }
        )
    return records


def write_tabular(
    path: pathlib.Path,
    specification: str,
    header: str,
    body: Sequence[str],
) -> None:
    """Write a complete booktabs tabular."""
    path.write_text(
        "\\begin{tabular}{" + specification + "}\n"
        "  \\toprule\n"
        "  " + header + " \\\\\n"
        "  \\midrule\n"
        + "\n".join(body)
        + "\n  \\bottomrule\n"
        "\\end{tabular}\n"
    )


def cell_key(record: JsonRecord) -> CellKey:
    """Return the full parameter-cell identifier."""
    return (
        int(record["n"]),
        int(record["q"]),
        int(record["t"]),
        int(record["beta"]),
    )


def render_discrepancy(records: Sequence[JsonRecord]) -> None:
    """Render raw and normalized rank-and-modulus discrepancy rows."""
    body: list[str] = []
    rank_modulus_keys = sorted(
        {(int(record["n"]), int(record["q"])) for record in records}
    )
    for rank, modulus in rank_modulus_keys:
        subset = [
            record
            for record in records
            if (int(record["n"]), int(record["q"])) == (rank, modulus)
        ]
        available_cells = {cell_key(record)[2:] for record in subset}
        plateaus = [
            int(record["plateau"])
            for record in subset
            if record["model"] == "GSA"
        ]
        model_statistics = []
        for model in ("GSA", "TGSA", "ZGSA"):
            values = [
                float(record["delta"])
                for record in subset
                if record["model"] == model
            ]
            scale = math.log(modulus) if model == "ZGSA" else math.log(rank)
            normalized = [value / (rank * scale) for value in values]
            model_statistics.append(
                (
                    mean(values),
                    sample_standard_deviation(values),
                    mean(normalized),
                    sample_standard_deviation(normalized),
                )
            )
        body.append(
            f"  ${rank}$ & ${modulus}$ & ${len(available_cells)}$ & "
            f"${mean(plateaus):.0f}$ & "
            + " & ".join(
                f"$\\substack{{{value_mean:.2f}\\mathbin{{\\pm}}{spread:.2f}"
                f"\\\\{normalized_mean:.4f}\\mathbin{{\\pm}}{normalized_spread:.4f}}}$"
                for value_mean, spread, normalized_mean, normalized_spread
                in model_statistics
            )
            + r" \\"
        )
    write_tabular(
        TABLE_DIR / "discrepancy.tex",
        "@{}rrrrrrr@{}",
        r"$n$ & $q$ & cells & plateau & GSA & TGSA & ZGSA",
        body,
    )


def render_families(records: Sequence[JsonRecord]) -> None:
    """Render clear GSA wins, clear ZGSA wins, and smallest gaps by rank."""
    grouped: dict[CellKey, list[JsonRecord]] = defaultdict(list)
    for record in records:
        grouped[cell_key(record)].append(record)

    def model_mean(key: CellKey, model: str) -> float:
        values = [
            float(record["delta"])
            for record in grouped[key]
            if record["model"] == model
        ]
        return mean(values)

    selected: list[tuple[str, CellKey]] = []
    ranks = sorted({key[0] for key in grouped})
    for rank in ranks:
        rank_keys = [key for key in sorted(grouped) if key[0] == rank]
        gaps = {
            key: model_mean(key, "ZGSA") - model_mean(key, "GSA")
            for key in rank_keys
        }
        gsa_key = max((key for key in rank_keys if gaps[key] > 0), key=gaps.get)
        zgsa_key = min((key for key in rank_keys if gaps[key] < 0), key=gaps.get)
        remaining = [key for key in rank_keys if key not in {gsa_key, zgsa_key}]
        tie_key = min(remaining, key=lambda key: abs(gaps[key]))
        selected.extend(
            [
                ("GSA win", gsa_key),
                ("ZGSA win", zgsa_key),
                ("smallest gap", tie_key),
            ]
        )

    body: list[str] = []
    for selection, key in selected:
        rank, modulus, lattice_t, beta = key
        cell = grouped[key]
        zgsa = [record for record in cell if record["model"] == "ZGSA"]
        alpha = mean([float(record["alpha"]) for record in zgsa])
        gh_over_q = math.exp(
            log_gh(rank) + alpha * math.log(modulus) - math.log(modulus)
        )
        head = mean([int(record["fitted_plateau"]) for record in zgsa])
        true_k = mean([int(record["true_k"]) for record in zgsa])
        true_d = mean([int(record["true_d"]) for record in zgsa])
        gsa_delta = model_mean(key, "GSA")
        zgsa_delta = model_mean(key, "ZGSA")
        body.append(
            f"  ${rank}$ & ${modulus}$ & ${lattice_t}$ & ${beta}$ & "
            f"${gh_over_q:.3f}$ & ${head:.0f}$ & ${true_k:.0f}$ & "
            f"${true_d:.0f}$ & {selection} & "
            f"${gsa_delta:.2f}/{zgsa_delta:.2f}$" + r" \\"
        )
    write_tabular(
        TABLE_DIR / "families.tex",
        "@{}rrrrrrrrlc@{}",
        r"$n$ & $q$ & $t$ & $\beta$ & $\GH/q$ & head & $k^*$ & "
        r"$d_{\max}$ & selection & $\Delta_{\rm E}$ G/Z",
        body,
    )


def render() -> None:
    """Render both compact manuscript tables."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    render_discrepancy(records)
    render_families(records)
    print(f"wrote {TABLE_DIR / 'discrepancy.tex'}")
    print(f"wrote {TABLE_DIR / 'families.tex'}")


if __name__ == "__main__":
    render()
