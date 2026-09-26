# -*- coding: utf-8 -*-
"""Paired test for the budget gain: per batch, top-k by the honest screen minus
top-k by the protocol pick, and both minus random."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec_common as S
import spec_eval as E

ANALYSIS = os.path.join(E.OUT, "uncertainty_stageB_analysis.json")
KS = [5, 10, 20]


def main():
    with open(ANALYSIS, encoding="utf-8") as handle:
        payload = json.load(handle)
    blocks = payload["blocks"]
    print("batches %d" % len(blocks))
    for k in KS:
        margin = np.array([b["top%d_margin" % k] for b in blocks.values()])
        proxy = np.array([b["top%d_proxy" % k] for b in blocks.values()])
        rand = np.array([b["top%d_rand" % k] for b in blocks.values()])
        out = {}
        for label, a, b in (("margin - proxy", margin, proxy), ("margin - random", margin, rand),
                            ("proxy - random", proxy, rand)):
            d = a - b
            se = d.std(ddof=1) / np.sqrt(len(d))
            out[label] = {"mean": float(d.mean()), "se": float(se),
                          "t": float(d.mean() / se) if se else None,
                          "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                          "losses": int((d < 0).sum())}
            print("  k=%-3d %-16s mean %+.4f  se %.4f  t %+.2f  W/T/L %d/%d/%d"
                  % (k, label, out[label]["mean"], out[label]["se"], out[label]["t"],
                     out[label]["wins"], out[label]["ties"], out[label]["losses"]))
    # how well does "share the screen keeps" rank the batches by true in-band rate?
    keep = np.array([b["margin_keep"] for b in blocks.values()])
    truth = np.array([b["truth_in"] for b in blocks.values()])
    say = np.array([b["proxy_say"] for b in blocks.values()])
    def spearman(a, b):
        return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
    print("\nbatches (n=%d): spearman(keep, truth) %.3f | spearman(say, truth) %.3f"
          % (len(keep), spearman(keep, truth), spearman(say, truth)))
    non_bo = say < 1.0 - 1e-9
    sel = ~non_bo
    print("  excluding the say<1 batches: spearman(keep, truth) %.3f on n=%d"
          % (spearman(keep[sel], truth[sel]), sel.sum()))


if __name__ == "__main__":
    main()