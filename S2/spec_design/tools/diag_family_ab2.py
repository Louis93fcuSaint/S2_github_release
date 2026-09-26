# -*- coding: utf-8 -*-
"""Protocol-level table for the family arms: novelty, valid rate, proxy rate."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec_common as S
import spec_eval as E

OUT = os.path.join(E.OUT, "latent_ddpm")
ARMS = [
    ("uniform", "v1"),
    ("learned", "famL_s25"),
    ("pin_d1b4", "pinSH_d1b4"),
    ("pin_d6b2", "pinSH_d6b2"),
    ("pin_d1b4", "pinAH_d1b4"),
    ("pin_d6b2", "pinAH_d6b2"),
]


def main():
    ctx = E.pool_ctx()
    specs = {sp["name"]: sp for sp in E.specs(ctx)}
    surrogate = S.load_surrogate("mlp_best", threads=8)
    rows = {}
    for arm, tag in ARMS:
        for name, spec in specs.items():
            path = os.path.join(OUT, "%s__%s.csv" % (tag, name))
            if not os.path.exists(path):
                continue
            frame = pd.read_csv(path)
            met = E.evaluate_submission(frame, spec, ctx, surrogate)
            rows["%s|%s" % (arm, name)] = {
                "tag": tag, "arm": arm, "spec": name,
                "valid_rate": met["valid_rate"],
                "spec_rate_proxy": met["spec_rate_proxy"],
                "novelty_min_median": met["novelty_min_median"],
                "novelty_min_p10": met["novelty_min_p10"],
                "novelty_ref_median": met["novelty_ref_median"],
                "median_rel_err_proxy": met["median_rel_err_proxy"],
                "n_families": met["n_families"],
                "spec_and_novel_1.0": met["spec_and_novel_1.0"],
                "spec_and_novel_2.0": met["spec_and_novel_2.0"],
            }
    S.save_json(os.path.join(E.OUT, "family_ab_protocol.json"), rows)
    print("%-10s %-14s %7s %8s %9s %9s %8s"
          % ("arm", "spec", "valid", "rate", "nov_med", "nov_p10", "ref"))
    for key in sorted(rows):
        r = rows[key]
        print("%-10s %-14s %7.3f %8.3f %9.3f %9.3f %8.3f"
              % (r["arm"], r["spec"], r["valid_rate"], r["spec_rate_proxy"],
                 r["novelty_min_median"], r["novelty_min_p10"], r["novelty_ref_median"]))
    print("[out] %s" % os.path.join(E.OUT, "family_ab_protocol.json"))


if __name__ == "__main__":
    main()