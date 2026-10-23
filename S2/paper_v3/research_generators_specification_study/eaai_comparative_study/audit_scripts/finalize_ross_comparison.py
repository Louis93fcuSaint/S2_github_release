# -*- coding: utf-8 -*-
"""Finalize the ROSS comparison without re-solving.

Separates (a) numerical equality from (b) same-forward-order association.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
OUT = os.path.join(STUDY, "pilot_runs", "05_ross_version")
COMP = os.path.join(OUT, "ross_version_comparison.csv")
MAN = os.path.join(OUT, "ross_version_comparison_manifest.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    d = pd.read_csv(COMP)
    d["ross161_forward_count_scope"] = np.where(
        d["ross161_source"] == "existing_pilot_1.6.1",
        "stored_first_six_orders_only", "full_worker_output")
    d["mode_order_match_rule"] = "same forward-order index; relative difference < 0.1% (well below adjacent-order separation)"
    for o in (1, 3):
        rel = "rel_diff_161_230_order%d_pct" % o
        match = "forward_order_match_order%d" % o
        if rel in d:
            d[match] = d[rel].notna() & (d[rel] < 0.1)
    d.to_csv(COMP, index=False, encoding="utf-8")
    man = json.load(open(MAN, encoding="utf-8"))
    man["files"]["comparison_sha256"] = sha(COMP)
    man["forward_count_scope_note"] = (
        "For pool samples, the 1.6.1 forward count is the stored first-six prefix, not the full mode count; "
        "do not compare it numerically with the 2.3.0 full count.")
    man["mode_order_match_rule"] = d["mode_order_match_rule"].iloc[0]
    with open(MAN, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, ensure_ascii=False)
    print("[finalize] wrote", COMP)


if __name__ == "__main__":
    main()
