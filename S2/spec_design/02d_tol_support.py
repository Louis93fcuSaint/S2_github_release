# -*- coding: utf-8 -*-
"""Step 02d -- how thick is a point target, and who is the bottleneck.

For every frozen point spec, count the real designs that sit inside a grid of
tolerances, and put that next to the surrogate own median error on the same
material.  The intersection answers the only question that matters for the
protocol: is the tolerance limited by the data or by the surrogate?
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_common as S
import spec_eval as E

GRID = [0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
MIN_SUPPORT = 200      # one design per slot of a submitted shortlist
QUANTILES = [10, 25, 50, 75, 90, 95, 99]


def main():
    started = time.time()
    pool = S.load_pool()
    cs = pool["cs_1_rpm"].to_numpy(dtype=float)
    mat = pool["material"].to_numpy()
    specs = [sp for sp in E.specs(pool) if sp.get("target")]

    cache = os.path.join(E.OUT, "cache_features.npz")
    blob = np.load(cache, allow_pickle=True)
    X, Y = blob["X"], blob["Y"]
    surrogate = S.load_surrogate("mlp_best", threads=8)
    pred = np.asarray(surrogate.predict(X), dtype=np.float64)[:, 0]
    truth1 = Y[:, 0].copy()
    good = np.isfinite(pred) & np.isfinite(truth1) & (truth1 > 0)
    ape = np.full(len(pool), np.nan)
    ape[good] = np.abs(pred[good] - truth1[good]) / truth1[good] * 100.0
    material_ape = {}
    for m in S.MATERIAL_ORDER:
        sel = (mat == m) & np.isfinite(ape)
        material_ape[m] = float(np.median(ape[sel]))
    print("surrogate median APE on cs1 (whole pool): " +
          ", ".join("%s %.2f%%" % (m, material_ape[m]) for m in S.MATERIAL_ORDER))

    report = {"grid": GRID, "material_median_ape_pct": material_ape, "specs": {}}
    print("\n%-16s %-9s %-7s %s" % ("spec", "material", "proxy%",
          "".join("%-9s" % ("+-%.1f%%" % (t * 100)) for t in GRID)))
    for sp in specs:
        target, tol = float(sp["target"]), float(sp["tol"])
        sel = mat == sp["material"]
        rel = np.abs(cs[sel] - target) / target
        counts = {("%.3f" % t): int((rel <= t).sum()) for t in GRID}
        report["specs"][sp["name"]] = {
            "material": sp["material"], "target": target, "tol": tol,
            "support": counts, "proxy_median_ape_pct": material_ape[sp["material"]]}
        print("%-16s %-9s %-7.2f %s"
              % (sp["name"], sp["material"], material_ape[sp["material"]],
                 "".join("%-9d" % counts["%.3f" % t] for t in GRID)))

    # Reachability surface: for an arbitrary target the user may ask for, how
    # many real designs can serve it, and which side is the bottleneck -- the
    # data (too few solutions) or the surrogate (error wider than the band).
    surface = {}
    print("\n%-9s %-6s %-9s %-10s %s"
          % ("material", "quant", "target", "min tol", "support at that tol"))
    for material in S.MATERIAL_ORDER:
        cs_m = cs[mat == material]
        rows = {}
        for q, target in zip(QUANTILES, np.quantile(cs_m, [v / 100.0 for v in QUANTILES])):
            rel = np.abs(cs_m - target) / target
            counts = {("%.4f" % t): int((rel <= t).sum()) for t in GRID}
            smallest = next((t for t in GRID
                             if counts["%.4f" % t] >= MIN_SUPPORT), None)
            rows["p%d" % q] = {
                "target": float(target), "support": counts,
                "smallest_tol_with_min_support": smallest,
                "support_at_smallest": (None if smallest is None
                                        else counts["%.4f" % smallest]),
                "proxy_median_ape_pct": material_ape[material],
            }
            print("%-9s %-6s %-9.0f %-10s %s"
                  % (material, "p%d" % q, target,
                     "-" if smallest is None else "+-%.2f%%" % (smallest * 100),
                     "-" if smallest is None else rows["p%d" % q]["support_at_smallest"]))
        surface[material] = rows
    report["surface"] = {"min_support": MIN_SUPPORT, "quantiles": QUANTILES,
                         "materials": surface}

    S.save_json(os.path.join(E.OUT, "tol_support.json"), report)
    print("\n[out] %s (%.0f s)" % (os.path.join(E.OUT, "tol_support.json"),
                                   time.time() - started))


if __name__ == "__main__":
    main()