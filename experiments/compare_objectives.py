"""Does the fitting objective matter? Fit by Delta_E and RMSE, then compare.

This is the experiment that tests the paper's methodological claim directly.
Both fits use the same model classes and the same determinant constraint, so
the only difference is which criterion selects the parameters.
"""
import json
import pathlib, math, sys
sys.path.insert(0, "experiments")
from bkz_experiment import (cumulative, discrepancy, rmse, fit_gsa_by,
                            fit_zgsa_by, bottleneck, admissible_depth)

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
rows = [json.loads(l) for l in (DATA_DIR / "profiles.jsonl").open() if l.strip()]
# Failed reductions carry no profile and are excluded here, as in analyze.
rows = [r for r in rows if r.get("status", "ok") == "ok"]
# One seed per cell keeps the O(n^3) ZGSA search affordable.
seen, sub = set(), []
for r in rows:
    key = (r["n"], r["q"], r["t"], r["beta"])
    if key in seen:
        continue
    seen.add(key)
    sub.append(r)
print(f"cells {len(sub)}")

agg = {}
for r in sub:
    ell, n, lq = r["ell"], r["n"], math.log(r["q"])
    tk, tc = bottleneck(ell)
    td = admissible_depth(ell)
    for name, fit in (("GSA", lambda o: fit_gsa_by(ell, o)),
                      ("ZGSA", lambda o: fit_zgsa_by(ell, lq, o)[0])):
        for obj_name, obj in (("Delta_E", discrepancy), ("RMSE", rmse)):
            m = fit(obj)
            mk, mc = bottleneck(m)
            md = admissible_depth(m)
            a = agg.setdefault((name, obj_name),
                               {"delta": [], "rmse": [], "dk": [], "dd": [], "dc": []})
            a["delta"].append(discrepancy(ell, m))
            a["rmse"].append(rmse(ell, m))
            a["dk"].append(abs(tk - mk))
            a["dd"].append(abs(td - md))
            a["dc"].append(abs(tc - mc))

mean = lambda v: sum(v) / len(v)
print(
    f"\n{'model':<6}{'fitted by':<11}{'Delta_E':>8}{'RMSE':>8}"
    f"{'|dk*|':>8}{'|dd_max|':>10}{'cost err':>10}"
)
lines = []
for (name, obj) in sorted(agg):
    a = agg[(name, obj)]
    print(f"{name:<6}{obj:<11}{mean(a['delta']):>8.2f}{mean(a['rmse']):>8.3f}"
          f"{mean(a['dk']):>8.2f}{mean(a['dd']):>10.2f}{mean(a['dc']):>10.2f}")
    latex_objective = "$\\Delta_{\\rm E}$" if obj == "Delta_E" else obj
    lines.append(
        f"  {name} & {latex_objective} & ${mean(a['delta']):.2f}$ & "
        f"${mean(a['rmse']):.3f}$ & ${mean(a['dk']):.2f}$ & "
        f"${mean(a['dd']):.2f}$ & ${mean(a['dc']):.2f}$" + r" \\"
    )
(DATA_DIR / "tables" / "objective.tex").open("w").write(
    "\\begin{tabular}{@{}llrrrrr@{}}\n  \\toprule\n"
    "  Model & Fitted by & $\\Delta_{\\rm E}$ & RMSE & mean $|\\Delta k^*|$ & "
    "mean $|\\Delta d_{\\max}|$ & mean cost error \\\\\n  \\midrule\n"
    + "\n".join(lines) + "\n  \\bottomrule\n\\end{tabular}\n")
print(f"\nwrote {DATA_DIR / 'tables' / 'objective.tex'}")
