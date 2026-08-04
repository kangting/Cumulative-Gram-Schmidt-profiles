# Cumulative Gram–Schmidt profiles for lattice enumeration and sieving in projected lattices: code and data

Artifact for the paper *Cumulative Gram–Schmidt profiles for lattice
enumeration and sieving in projected lattices*.

Everything reported in the paper is reproducible from the stored data in
this repository. No number in the paper was entered by hand. Every table and
figure is emitted by a script here.

## Requirements

```
python >= 3.12
fpylll == 0.6.4
numpy
scipy
```

`fpylll` is only needed to regenerate lattices and to run reductions or point
counts. The verification scripts and the table generators run on the stored
data with numpy and scipy alone.

## Reproducing the paper

Run in this order. Steps 1 and 3 are the expensive ones.

```bash
# 1. Reduce the parameter grid. 120 cells, 10 seeds, ranks 120 to 320.
#    About 16 CPU-hours, roughly 45 minutes on 24 workers.
python experiments/bkz_experiment.py reduce --seeds 10

# 2. Fit the profile models and emit the discrepancy and family tables.
python experiments/bkz_experiment.py analyze

# 3. Fixed-radius projected point counts at ranks 40 and 44.
python experiments/node_counts.py run

# 4. Emit the point-count table.
python experiments/node_count_reporting.py

# 5. The comparison of fitting objectives.
python experiments/compare_objectives.py

# 6. Figures.
python experiments/generate_revision_figures.py
```

Steps 2 and 4 through 6 read only the stored data, so they reproduce every
table and figure without rerunning any lattice reduction.

## Verification

These check the paper's mathematics against direct computation and should all
pass.

```bash
python experiments/verify_identities.py         # every identity of Section 3
python experiments/verify_models.py             # the ZGSA closed forms
python experiments/exact_extremal_certificate.py # Proposition 3.5, exact rationals
python experiments/extremal_check.py            # the 96-cell linear programme
```

`exact_extremal_certificate.py` is the computer-assisted part of
Proposition 3.5. It enumerates all 780 descent intervals in exact rational
arithmetic and confirms that no common-slope three-zone profile reaches the
unrestricted optimum.

## Data

| File | Contents |
|---|---|
| `profiles.jsonl` | 1200 reduction records. 1162 completed, 38 failed with the failure recorded. |
| `results.csv` | One row per profile, model and fit. |
| `zgsa_fits.csv` | Every fitted three-zone parameter set. |
| `fixed_radius_counts.jsonl` | Levelwise projected point counts at ranks 40 and 44. |
| `fixed_radius_instances.csv` | The instances swept, with their regime. |
| `metadata.json`, `fixed_radius_metadata.json` | Software versions and run parameters. |
| `tables/`, `figures/` | Generated output, included by the manuscript. |

Failed and incomplete runs are kept in the data rather than dropped. Keeping
them makes the completion pattern visible and prevents their silent omission.
The statistics in the paper use completed runs only, and every exclusion is
visible in the tables.

## Notes on reproducibility

Seeds are recorded with every profile, and the generator is deterministic
given a seed, so any stored profile can be rebuilt exactly.

FPLLL can abort at the C level, which no Python handler can catch. Each job
therefore runs in an isolated subprocess with a timeout, and a crash is
recorded as a failed row. This is why 38 of the 1200 reductions carry a
failure status. They fall in seven parameter cells across three rank and
modulus groups. The cause was not identified.
