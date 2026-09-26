# -*- coding: utf-8 -*-
"""Point-target protocol -- the user names the first critical speed directly.

    input : material, cs1 target (rpm), optional tolerance
    output: a batch of admissible designs whose cs1 sits inside the tolerance

The band version asked for ">= 14400"; this version asks for "18000, give or
take 5 %".  A single-order point target is thick: hundreds of real designs sit
within +-1 % of any target inside the data support, unlike the old protocol's
six-order target, where nothing sat within +-10 %.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_common as S
import spec_eval as E

TOL = 0.05
TARGETS = [("Steel", "mid", 3600.0), ("Steel", "high", 16000.0),
           ("Aluminum", "mid", 5100.0), ("Aluminum", "high", 23000.0),
           ("Titanium", "mid", 4300.0), ("Titanium", "high", 19000.0)]
GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
MIN_SUPPORT = 200


def main():
    pool = S.load_pool()
    cs = pool["cs_1_rpm"].to_numpy(dtype=float)
    mat = pool["material"].to_numpy()
    nd = pool["n_disks"].to_numpy()
    nb = pool["n_bearings"].to_numpy()
    profiles = {"family free": np.ones(len(pool), bool),
                "nd=3, nb=2": (nd == 3) & (nb == 2),
                "nd in 2..4": (nd >= 2) & (nd <= 4)}

    frozen, lines = [], []
    print("point targets, +/- %d%% tolerance, minimum support %d real designs"
          % (TOL * 100, MIN_SUPPORT))
    print("%-16s %-9s %-8s %-9s %-9s %-9s %-9s"
          % ("spec", "material", "target", "family", "nd=3nb=2", "nd2..4", "best tol"))
    for material, tier, target in TARGETS:
        name = "%s_%s" % (material, tier)
        entry = {"name": name, "material": material, "tier": tier,
                 "target": float(target), "tol": TOL,
                 "lower": float(target * (1 - TOL)),
                 "upper": float(target * (1 + TOL))}
        cells = []
        for label, mask in profiles.items():
            sel = mask & (mat == material)
            n = int((np.abs(cs[sel] - target) / target <= TOL).sum())
            entry["support_%s" % label.replace(" ", "_").replace("=", "").replace(",", "")] = n
            cells.append(n)
        best = None
        for tol in GRID:
            if int((np.abs(cs[mat == material] - target) / target <= tol).sum()) >= MIN_SUPPORT:
                best = tol
                break
        entry["achievable_tol"] = best
        frozen.append(entry)
        print("%-16s %-9s %-8.0f %-9d %-9d %-9d %-9s"
              % (name, material, target, cells[0], cells[1], cells[2],
                 ("+-%.1f%%" % (best * 100)) if best is not None else "none"))
        lines.append("| %s | %s | %.0f | +-%.1f%% | %d | %d | %d | %s |"
                     % (name, material, target, TOL * 100, cells[0], cells[1], cells[2],
                        ("+-%.1f%%" % (best * 100)) if best is not None else "none"))

    print("\nthe target has to be reachable: support across the whole range")
    print("%-9s %-9s" % ("material", "target")
          + "".join("%-8s" % ("+-%.1f%%" % (t * 100)) for t in GRID))
    for material in S.MATERIAL_ORDER:
        sel = mat == material
        for target in (2000, 5000, 12000, 20000, 40000, 60000):
            row = "%-9s %-9s" % (material, target)
            for tol in GRID:
                row += "%-8d" % int((np.abs(cs[sel] - target) / target <= tol).sum())
            print(row)
        print()

    S.save_json(E.SPEC_JSON, {"protocol": "point_target", "tol": TOL,
                              "min_support": MIN_SUPPORT,
                              "n_operating_rpm": S.N_OPERATING_RPM,
                              "specs": frozen})
    with open(os.path.join(E.OUT, "spec_table.md"), "w", encoding="utf-8") as fh:
        fh.write("# Frozen point-target specs\n\n")
        fh.write("Target = first critical speed in rpm. Tolerance +-%.0f%%. "
                 "Best tol = the tightest tolerance on the grid that still has "
                 "at least %d real designs behind it.\n\n" % (TOL * 100, MIN_SUPPORT))
        fh.write("| spec | material | target | tol | family free | nd=3 nb=2 | nd 2..4 | best tol |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")
        fh.write("\n".join(lines) + "\n")
    print("[out] %s" % E.SPEC_JSON)


if __name__ == "__main__":
    main()
