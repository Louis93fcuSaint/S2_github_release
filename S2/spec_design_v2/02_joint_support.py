# -*- coding: utf-8 -*-
"""v2 -- is a *joint* interval spec reachable at all, and is it resolvable?

A single-order point target only asks "can this one number happen".  A joint
spec -- "cs1 in [3000,4500] AND cs2 in [6000,9000]", point or interval, any
subset of {cs1, cs2, cs3} -- asks for a cell in 2D/3D, and those cells are much
emptier than the marginals suggest.  Two different questions live here:

  SUPPORT     how many real rotors in the pool actually land in the cell?  If it
              is a handful, the generator is being asked to hit a needle and the
              ROSS gate will reject nearly everything.  This is a data fact, not
              a network problem, and no amount of training fixes it.
  RESOLVABLE  is the band wider than the surrogate's own error on that order?
              The frozen per-order MAPE is 1.28 / 2.19 / 3.22 % (cs1/cs2/cs3).
              A band thinner than that cannot be ranked honestly, so the
              deliverable has to be ROSS-certified.

It also checks the width cap: the denoiser is only asked to fill exp(h_max) ~ 4x,
so a wider band is shrunk towards its centre (see spec_interval.effective).

    python 02_joint_support.py --material Steel --targets "1=[3000,4500],2=[6000,9000]"
    python 02_joint_support.py --material Steel --targets "1=[3000,4500],2=[6000,9000],3=[14000,22000]"
    python 02_joint_support.py --material Steel --targets "1=3680.8,2=[6000,+]"
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(os.path.dirname(HERE), "spec_design")
for path in (V1_DIR, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import spec_common as S          # noqa: E402
import spec_interval as SI       # noqa: E402

OUT = os.path.join(HERE, "outputs")
SURROGATE_MAPE = {1: 1.28, 2: 2.19, 3: 3.22}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True,
                        help='"1=[3000,4500],2=[6000,9000]" or "1=3680.8,2=[6000,+]"')
    parser.add_argument("--material", required=True, choices=list(S.MATERIAL_ORDER))
    parser.add_argument("--disks", default="", help="restrict the search pool, e.g. 3-3")
    parser.add_argument("--bearings", default="", help="e.g. 2-2")
    parser.add_argument("--h-max", type=float, default=0.70)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    specs = SI.parse_specs(args.targets)
    if not specs:
        raise SystemExit("could not parse --targets %r" % args.targets)
    orders = sorted(specs)
    for order in orders:
        if order not in (1, 2, 3):
            raise SystemExit("v2 handles orders 1..3 only (asked for %d)" % order)

    families = None
    if args.disks or args.bearings:
        import importlib.util
        path = os.path.join(V1_DIR, "05_latent_ddpm.py")
        module_spec = importlib.util.spec_from_file_location("v1_latent_ddpm", path)
        v1 = importlib.util.module_from_spec(module_spec)
        sys.modules["v1_latent_ddpm"] = v1
        module_spec.loader.exec_module(v1)
        families = v1.parse_families("%sx%s" % (args.disks or "", args.bearings or ""))

    pool = S.load_pool(materials=[args.material], families=families)
    values = {o: pool["cs_%d_rpm" % o].to_numpy(dtype=np.float64) for o in (1, 2, 3)}
    finite = np.isfinite(values[1]) & np.isfinite(values[2]) & np.isfinite(values[3])
    n_total = int(finite.sum())

    hit = finite.copy()
    rows = []
    fam_text = ("%sx%s" % (args.disks or "", args.bearings or "")) if families else "free"
    print("%s | families %s | pool %d rotors with cs1..cs3 labels"
          % (args.material, fam_text, n_total))
    print("%-6s %-22s %7s %8s %-22s %9s %s"
          % ("order", "band (user)", "log w", "in-band", "band the net fills",
             "half-width", "resolvable"))
    for order in orders:
        lo, hi = specs[order]
        asked_lo, asked_hi = SI.effective(lo, hi, args.h_max)
        c = values[order][finite]
        rate = float(((c >= lo) & (c <= hi)).mean()) if n_total else 0.0
        hit &= (values[order] >= lo) & (values[order] <= hi)
        width = SI.log_half(lo, hi)
        half_pct = 100.0 * (math.exp(width) - 1.0) if width > 0 else 0.0
        mape = SURROGATE_MAPE.get(order)
        if lo == hi:
            verdict = "point spec -> needs ROSS"
        elif half_pct < mape:
            verdict = "half-width %.2f%% < proxy %.2f%% -> ROSS only" % (half_pct, mape)
        else:
            verdict = "half-width %.2f%% >= proxy %.2f%% -> ranking ok" % (half_pct, mape)
        shrunk = " (capped)" if (asked_lo, asked_hi) != (lo, hi) else ""
        logw = float("inf") if hi >= SI.INF_CAP else math.log(hi / lo)
        print("%-6d %-22s %7.3f %8.3f %-22s %8.2f%% %s"
              % (order, SI.band_text(lo, hi), logw, rate,
                 SI.band_text(asked_lo, asked_hi, 0) + shrunk, half_pct, verdict))
        rows.append({"order": order, "band": [lo, min(hi, SI.INF_CAP)],
                     "asked_band": [asked_lo, asked_hi],
                     "log_width": None if hi >= SI.INF_CAP else logw,
                     "in_band_rate": rate, "half_width_pct": half_pct,
                     "proxy_mape_pct": mape})

    joint = int(hit.sum())
    print("")
    print("joint cell %s" % {o: SI.band_text(*specs[o]) for o in orders})
    print("  real rotors inside : %d / %d  (%.4f %%)"
          % (joint, n_total, 100.0 * joint / max(n_total, 1)))
    if joint < 50:
        print("  VERDICT: too thin to be a generation target -- the cell is nearly "
              "empty in the data, the ROSS gate will reject almost everything. "
              "Widen a band or drop an order.")
    elif joint < 500:
        print("  VERDICT: thin but workable -- expect a low delivery count, so bring "
              "--n-generate up.")
    else:
        print("  VERDICT: supported -- the generator has real designs to lean on.")

    per_family = []
    if families is None:
        for nd in sorted(set(int(v) for v in pool["n_disks"])):
            for nb in sorted(set(int(v) for v in pool["n_bearings"])):
                mask = finite & (pool["n_disks"].to_numpy() == nd) & \
                    (pool["n_bearings"].to_numpy() == nb)
                if mask.sum() == 0:
                    continue
                inside = mask.copy()
                for order in orders:
                    lo, hi = specs[order]
                    inside &= (values[order] >= lo) & (values[order] <= hi)
                per_family.append({"disks": nd, "bearings": nb,
                                   "n_pool": int(mask.sum()), "n_inside": int(inside.sum())})
        live = [r for r in per_family if r["n_inside"] > 0]
        print("  families with a real rotor in the cell: %d / %d"
              % (len(live), len(per_family)))
        for r in sorted(per_family, key=lambda r: -r["n_inside"])[:6]:
            print("    %d disks x %d bearings  pool %6d  inside %5d"
                  % (r["disks"], r["bearings"], r["n_pool"], r["n_inside"]))

    tag = args.tag or "joint"
    S.save_json(os.path.join(OUT, "joint_support_%s.json" % tag),
                {"material": args.material, "family": fam_text,
                 "targets": args.targets, "h_max": args.h_max,
                 "n_pool": n_total, "n_joint": joint,
                 "joint_rate": joint / max(n_total, 1), "orders": rows,
                 "per_family": per_family})


if __name__ == "__main__":
    main()