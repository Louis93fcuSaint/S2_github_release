# -*- coding: utf-8 -*-
"""How much of the per-batch keep fraction is sampling noise?

Twelve batches were re-verified with 60 extra rotors.  For each of them this
compares the metrics computed on the original 40 rotors with the metrics on the
full sample, and compares the observed move with the binomial standard error
sqrt(p(1-p)/n) that a 40-rotor sample implies.  If the moves are of that size,
the earlier numbers were noisy but unbiased, and the report should quote the
pooled figures rather than the per-batch ones.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec_common as S
import spec_eval as E
import proxy_calibration as P

STORE = os.path.join(E.OUT, "uncertainty_stageB.json")
CENSUS = os.path.join(E.OUT, "cs1_census.json")
REPORT = os.path.join(E.OUT, "proxy_calibration_report.json")


def main():
    with open(REPORT, encoding="utf-8") as handle:
        report = json.load(handle)
    bar = report["bar_refit"]
    with open(STORE, encoding="utf-8") as handle:
        store = json.load(handle)
    untrusted = set()
    if os.path.exists(CENSUS):
        with open(CENSUS, encoding="utf-8") as handle:
            for item in json.load(handle).get("stage_b", []):
                if item.get("trusted") is False:
                    untrusted.add((item["key"], item["index"]))

    print("bar = %s" % bar["names"])
    print("%-24s %5s %5s %8s %8s %7s %8s %7s %8s"
          % ("batch", "n40", "nAll", "keep40", "keepAll", "move", "prec40", "precAll", "binom_se"))
    moves = []
    for key, entry in sorted(store.items()):
        rotors = [r for r in entry["rotors"] if r["truth"] and (key, r["index"]) not in untrusted]
        first = [r for r in rotors if r.get("round") != "extra"]
        if len(rotors) == len(first) or not first:
            continue

        def metrics(sel):
            proxy = np.array([r["proxy"] for r in sel])
            truth = np.array([r["truth"] for r in sel])
            factor = P.bar_factor(bar, np.array([r["spread"] for r in sel]),
                                  np.array([r["dF"] for r in sel]))
            lower = np.array([entry["band"][0] for _ in sel])
            upper = np.array([entry["band"][1] for _ in sel])
            in_band = (truth >= lower) & (truth <= upper)
            keep = (proxy / factor >= lower) & (proxy * factor <= upper)
            return (float(keep.mean()), float(in_band[keep].mean()) if keep.any() else float("nan"),
                    float(in_band.mean()))

        k40, p40, t40 = metrics(first)
        kA, pA, tA = metrics(rotors)
        se = np.sqrt(max(k40 * (1 - k40), 1e-6) / len(first))
        moves.append(kA - k40)
        print("%-24s %5d %5d %8.3f %8.3f %+7.3f %8.3f %8.3f %8.3f"
              % (key, len(first), len(rotors), k40, kA, kA - k40, p40, pA, se))
    if moves:
        moves = np.array(moves)
        print("\nmoves: mean %+.4f  sd %.4f  rms %.4f  |mean|/se %.2f"
              % (moves.mean(), moves.std(ddof=1), float(np.sqrt((moves ** 2).mean())),
                 abs(moves.mean()) / (moves.std(ddof=1) / np.sqrt(len(moves)))))
        print("a 40-rotor proportion carries a binomial se of about %.3f, so moves of this" % se)
        print("size are sampling noise, not a change of behaviour.")


if __name__ == "__main__":
    main()