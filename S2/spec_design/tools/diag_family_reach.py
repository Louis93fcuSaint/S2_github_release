# -*- coding: utf-8 -*-
"""Does the choice of family matter for reaching a spec band?

For every (material, family) this asks two separate questions:

  reachability  what cs1 range can the family realise at all?
  support       how many pool designs of that family land inside the band?

If a family cannot reach the band, every sample drawn into it is wasted, which
is what uniform family sampling does today.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec_common as S, spec_eval as E

ctx = E.pool_ctx()
pool = ctx["pool"]
cs1 = ctx["Y"][:, 0]
nd = ctx["nd"]; nb = ctx["nb"]; mat = ctx["material"]
specs = E.specs(ctx)
FAM = [(a, b) for a in S.ND_CHOICES for b in S.NB_CHOICES]
print("families %d, designs %d" % (len(FAM), len(cs1)))

print("\ncs1 range per family (all materials pooled), rpm")
print("  %-8s %8s %8s %8s %8s" % ("family", "n", "p05", "p50", "p95"))
for a, b in FAM:
    sel = (nd == a) & (nb == b)
    v = cs1[sel]
    print("  d%d b%d %8d %8.0f %8.0f %8.0f" % (a, b, len(v), np.percentile(v, 5),
                                               np.median(v), np.percentile(v, 95)))

out = {}
print("\nper spec: which families carry the in-band designs")
print("  %-14s %7s %8s %8s   %s" % ("spec", "support", "families", "eff_unif", "top families (share)"))
for spec in specs:
    lo, hi = spec["lower"], spec["upper"]
    inb = (cs1 >= lo) & (cs1 <= hi) & (mat == spec["material"])
    n = int(inb.sum())
    counts = {}
    for a, b in FAM:
        sel = inb & (nd == a) & (nb == b)
        counts[(a, b)] = int(sel.sum())
    tot = sum(counts.values())
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
    # efficiency of uniform family sampling: the share of the 18 families that
    # holds 90 % of the support, and the expected support per uniform draw
    order = sorted(counts.values(), reverse=True)
    cum = np.cumsum(order) / max(tot, 1)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    # probability that a uniform family draw lands on one that has any support
    p_nonzero = float(np.mean([1.0 if counts[f] else 0.0 for f in FAM]))
    out[spec["name"]] = {"support": n, "counts": {"d%d_b%d" % f: c for f, c in counts.items()},
                         "families_for_90pct": n90, "share_with_support": p_nonzero,
                         "top": [{"family": "d%d_b%d" % f, "n": c, "share": c / max(tot, 1)}
                                 for f, c in top]}
    print("  %-14s %7d %8d %8.3f   %s"
          % (spec["name"], n, n90, p_nonzero,
             ", ".join("d%d_b%d %.2f" % (f[0], f[1], c / max(tot, 1)) for f, c in top[:4])))

print("\np(in band | material, family) x1000, and reachability of the band")
print("  %-14s %s" % ("spec", "  ".join("d%db%d" % f for f in FAM)))
for spec in specs:
    lo, hi = spec["lower"], spec["upper"]
    row = []
    for a, b in FAM:
        sel = (mat == spec["material"]) & (nd == a) & (nb == b)
        if sel.sum() < 30:
            row.append("   --  ")
            continue
        p = float(((cs1 >= lo) & (cs1 <= hi) & sel).sum()) / float(sel.sum())
        row.append("%6.2f" % (p * 1000))
    print("  %-14s %s" % (spec["name"], " ".join(row)))

print("\nreachability: can the family even produce a cs1 inside the band?")
for spec in specs:
    lo, hi = spec["lower"], spec["upper"]
    ok = []
    for a, b in FAM:
        v = cs1[(mat == spec["material"]) & (nd == a) & (nb == b)]
        if len(v) < 30:
            ok.append("--")
            continue
        # the central 99 % of what this family can do
        q1, q9 = np.percentile(v, 0.5), np.percentile(v, 99.5)
        ok.append("Y" if (q1 <= hi and q9 >= lo) else "n")
    print("  %-14s band [%7.0f, %7.0f]  %s"
          % (spec["name"], lo, hi, " ".join("%2s" % o for o in ok)))

S.save_json(os.path.join(E.OUT, "family_reachability.json"), out)
print("\n[out] %s" % os.path.join(E.OUT, "family_reachability.json"))