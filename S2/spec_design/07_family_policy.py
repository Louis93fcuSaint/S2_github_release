# -*- coding: utf-8 -*-
"""Step 07 -- learn the family prior, then check it out of sample.

Two questions, in order:

  1. what does the pool say about which (n_disks, n_bearings) family can serve
     each spec band, and how much of the uniform draw it wastes?
  2. does that survive a five-fold split, where the histogram is fitted on four
     fifths of the pool and scored on the fifth it never saw?

Only if the answer to 2 is yes is it worth wiring the policy into the sampler.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_common as S
import spec_eval as E
import family_policy as F

OUT_JSON = os.path.join(E.OUT, "family_policy_report.json")


def main():
    ctx = E.pool_ctx()
    specs = E.specs(ctx)
    policy = F.fit(ctx)
    F.save(policy)
    print("policy fitted on %d pool designs, %d bins over log cs1"
          % (policy["n_pool"], policy["bins"]), flush=True)

    print("\n1. family weights per spec (uniform would be 1/18 = 0.0556 each)")
    rows = {}
    for spec in specs:
        wrow = F.weights(policy, spec["material"], spec["lower"], spec.get("upper"))
        order = np.argsort(-np.asarray(wrow["w"]))
        rate = F.expected_rate(policy, spec["material"], spec["lower"], spec.get("upper"), wrow)
        # what uniform family sampling achieves on the same pool evidence
        uni = np.mean([F.band_mass(policy, spec["material"], f, spec["lower"],
                                  spec.get("upper")) for f in F.FAMILIES])
        top = ", ".join("%d_%d %.3f" % (wrow["families"][i][0], wrow["families"][i][1],
                                        wrow["w"][i]) for i in order[:6])
        dead = [f for f in F.FAMILIES
                if F.band_mass(policy, spec["material"], f, spec["lower"],
                               spec.get("upper")) <= 0.0]
        print("  %-14s eff_families %4.1f | policy %7.4f vs uniform %7.4f (x%.2f)"
              % (spec["name"], wrow["effective"], rate, uni, rate / max(uni, 1e-9)))
        print("      top: %s" % top)
        print("      cannot reach the band: %s"
              % (", ".join("d%d_b%d" % f for f in dead) if dead else "none"))
        rows[spec["name"]] = {"effective": wrow["effective"], "policy_rate": rate,
                              "uniform_rate": uni, "gain": rate / max(uni, 1e-9),
                              "weights": {"d%d_b%d" % f: v for f, v in
                                          zip(wrow["families"], wrow["w"])},
                              "unreachable": ["d%d_b%d" % f for f in dead]}

    print("\n2. five-fold check, histogram fitted on 4/5 of the pool")
    print("   %-14s %12s %12s %8s" % ("spec", "policy", "uniform", "gain"))
    cv = F.cross_validate(ctx, specs, folds=5)
    for name in sorted(cv):
        print("   %-14s %12.4f %12.4f %8.2f"
              % (name, cv[name]["policy_rate_mean"], cv[name]["uniform_rate_mean"],
                 cv[name]["gain"]))

    print("\n3. the user's three cases on one spec (Steel_high)")
    spec = [sp for sp in specs if sp["name"] == "Steel_high"][0]
    for label, allowed in (("family free", None),
                           ("2..4 disks, 2..4 bearings",
                            [(a, b) for a in (2, 3, 4) for b in (2, 3, 4)]),
                           ("pinned to 3 disks 2 bearings", [(3, 2)])):
        wrow = F.weights(policy, spec["material"], spec["lower"], spec.get("upper"),
                         allowed=allowed)
        print("   %-30s families %2d  eff %4.1f  policy rate %6.4f"
              % (label, len(wrow["families"]), wrow["effective"],
                 F.expected_rate(policy, spec["material"], spec["lower"],
                                 spec.get("upper"), wrow)))

    S.save_json(OUT_JSON, {"pool": policy["n_pool"], "bins": policy["bins"],
                           "per_spec": rows, "cross_validation": cv})
    print("\n[out] %s\n[out] %s" % (F.POLICY_JSON, OUT_JSON))


if __name__ == "__main__":
    main()