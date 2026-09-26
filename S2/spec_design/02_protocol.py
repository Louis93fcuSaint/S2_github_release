# -*- coding: utf-8 -*-
"""Step 02 -- freeze the spec set and report whether each spec is attainable
from *real* designs (that is the retrieval baseline, and the novelty reference).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_common as S
import spec_eval as E


def main():
    ctx = E.pool_ctx()
    pool, truth = ctx["pool"], ctx["Y"][:, 0]
    frozen = E.specs(ctx)

    print("frozen specs: n_operating = %.0f rpm, margin = %.0f%%"
          % (S.N_OPERATING_RPM, S.DEFAULT_MARGIN * 100))
    print("%-14s %-9s %-9s %-9s %-9s %-9s %-9s %-9s"
          % ("spec", "lower", "upper", "real<lb", "in band", "best fam", "n(best)", "attain"))
    lines = ["| spec | lower | upper | real below | in band | best family | n(best) | attainable |",
             "|---|---|---|---|---|---|---|---|"]
    for spec in frozen:
        mask = pool["material"].to_numpy() == spec["material"]
        cs = truth[mask]
        hit = (cs >= spec["lower"]) & (cs <= spec["upper"])
        fams = pool.loc[mask, ["n_disks", "n_bearings"]].to_numpy()
        best, best_n = "-", 0
        for nd in S.ND_CHOICES:
            for nb in S.NB_CHOICES:
                sel = (fams[:, 0] == nd) & (fams[:, 1] == nb)
                if sel.sum() and int(hit[sel].sum()) > best_n:
                    best_n, best = int(hit[sel].sum()), "(%d,%d)" % (nd, nb)
        attain = "yes" if best_n >= 200 else ("rare" if best_n > 0 else "no")
        spec["real_in_band"] = int(hit.sum())
        spec["best_family"] = best
        spec["best_family_n"] = best_n
        spec["attainable"] = attain
        row = (spec["name"], "%.0f" % spec["lower"], "%.0f" % spec["upper"],
               "%d" % int((cs < spec["lower"]).sum()), "%d" % int(hit.sum()),
               best, "%d" % best_n, attain)
        print("%-14s %-9s %-9s %-9s %-9s %-9s %-9s %-9s" % row)
        lines.append("| " + " | ".join(row) + " |")

    print("\nnovelty reference (real designs, median nearest-neighbour distance "
          "in pool-spread units): %.4f" % ctx["novelty_ref"])
    S.save_json(os.path.join(E.OUT, "specs.json"),
                {"n_operating_rpm": S.N_OPERATING_RPM,
                 "margin": S.DEFAULT_MARGIN, "specs": frozen,
                 "novelty_ref_median": ctx["novelty_ref"]})
    with open(os.path.join(E.OUT, "spec_table.md"), "w", encoding="utf-8") as fh:
        fh.write("# Frozen spec set\n\n")
        fh.write("n_operating = %.0f rpm, margin = %.0f%%, lower = %.0f rpm. "
                 "Upper bound is the largest cs1 ever realised for the material.\n\n"
                 % (S.N_OPERATING_RPM, S.DEFAULT_MARGIN * 100, S.DEFAULT_LOWER_RPM))
        fh.write("\n".join(lines) + "\n")
    print("[out] %s" % os.path.join(E.OUT, "specs.json"))


if __name__ == "__main__":
    main()
