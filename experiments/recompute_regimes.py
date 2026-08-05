#!/usr/bin/env python3
"""Recompute the point-count regime labels under the leading-run plateau.

The stored labels were produced by an earlier one-sided plateau count. Only
the labels change here. No point count is repeated. Each basis is rebuilt
from its recorded seed, which the generator makes deterministic.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib

import node_counts

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
INSTANCES = DATA_DIR / "fixed_radius_instances.csv"
OUTPUT = DATA_DIR / "fixed_radius_regimes_v2.json"
TOLERANCE = node_counts.PLATEAU_TOLERANCE


def leading_plateau(profile: list[float], log_q: float, tol: float) -> int:
    length = 0
    for value in profile:
        if abs(value - log_q) > tol:
            break
        length += 1
    return length


def regime(plateau: int, rank: int) -> str:
    if plateau == 0:
        return "no_plateau"
    if plateau < max(2, rank // 8):
        return "transition"
    return "long_plateau"


def main() -> None:
    rows = list(csv.DictReader(INSTANCES.open()))
    out, changed = [], 0
    for row in rows:
        job = (
            int(row["n"]), int(row["q"]), int(row["lattice_t"]),
            int(row["beta"]), int(row["seed"]),
        )
        _, profile = node_counts.reduce_basis(job)
        log_q = math.log(job[1])
        plateau = leading_plateau(profile, log_q, TOLERANCE)
        new = regime(plateau, job[0])
        old = row["regime"]
        changed += new != old
        digest = hashlib.sha256(
            ",".join(f"{value:.12g}" for value in profile).encode()
        ).hexdigest()[:16]
        out.append({
            "job_id": row["job_id"],
            "legacy_regime_v1": old,
            "legacy_plateau_length_v1": int(row["plateau_length"]),
            "leading_plateau_length_v2": plateau,
            "regime_v2": new,
            "profile_sha256_16": digest,
        })
        print(f"{row['job_id']:<44} {old:>13} -> {new:<13} "
              f"plateau {row['plateau_length']:>3} -> {plateau}", flush=True)
    OUTPUT.write_text(json.dumps(
        {"plateau_definition_version": 2,
         "tolerance": TOLERANCE,
         "instances": out}, indent=2) + "\n")
    counts: dict[str, int] = {}
    for record in out:
        counts[record["regime_v2"]] = counts.get(record["regime_v2"], 0) + 1
    print(f"\nlabels changed: {changed} of {len(out)}")
    print("regime_v2 counts:", counts)


if __name__ == "__main__":
    main()
