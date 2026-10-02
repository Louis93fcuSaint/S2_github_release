# -*- coding: utf-8 -*-
"""v2 -- what critical speeds can a family (range) actually reach?

The family is an optional condition: the user leaves it free, fixes it (2-2),
or gives a range (1-2 / 2-4).  Fixing it shrinks the design space a lot, and the
speed window an assembly can reach depends strongly on it: for Steel the median
cs1 runs from 2 066 rpm (6 disks, 2 bearings) to 6 050 rpm (1 disk, 4 bearings).

So before asking for a target, ask the data which targets are reachable by
EVERY family in the requested range.  A target outside that common window is not
impossible, it just means part of the batch is wasted on families that cannot
get there, and the delivered count drops.

    python 01_family_window.py --material Steel --disks 1-2 --bearings 2-4
    python 01_family_window.py --material Steel --disks 3-3 --bearings 2-2
    python 01_family_window.py --material Aluminum
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(os.path.dirname(HERE), "spec_design")
if V1_DIR not in sys.path:
    sys.path.insert(0, V1_DIR)

import spec_common as S          # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", required=True, choices=list(S.MATERIAL_ORDER))
    parser.add_argument("--disks", default="", help="1-2 / 2-2 / 3-3 / '' = free")
    parser.add_argument("--bearings", default="", help="2-4 / 2-2 / '' = free")
    parser.add_argument("--orders", default="1,2,3")
    parser.add_argument("--pct", default="25,75", help="percentiles, e.g. 25,75 or 10,90")
    args = parser.parse_args()

    spec = ""
    if args.disks or args.bearings:
        spec = "%sx%s" % (args.disks or "", args.bearings or "")
    import importlib.util
    path = os.path.join(V1_DIR, "05_latent_ddpm.py")
    module_spec = importlib.util.spec_from_file_location("v1_latent_ddpm", path)
    v1 = importlib.util.module_from_spec(module_spec)
    sys.modules["v1_latent_ddpm"] = v1
    module_spec.loader.exec_module(v1)
    families = v1.parse_families(spec) or list(v1.FAMILIES)

    pool = S.load_pool(materials=[args.material], families=families)
    orders = [int(v) for v in str(args.orders).split(",") if v.strip()]
    lo_p, hi_p = [float(v) for v in str(args.pct).split(",")]
    print("%s | families %s (%d)" % (args.material, spec or "free", len(families)))
    print("%-12s %8s %s" % ("family", "n", " ".join(
        "%s p%g-p%g" % ("cs%d" % o, lo_p, hi_p) for o in orders)))

    per_order = {o: [] for o in orders}
    for family in families:
        sub = pool[(pool["n_disks"] == family[0]) & (pool["n_bearings"] == family[1])]
        cells = []
        for o in orders:
            values = sub["cs_%d_rpm" % o].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                cells.append("--")
                continue
            lo, hi = np.percentile(values, [lo_p, hi_p])
            per_order[o].append((lo, hi))
            cells.append("%8.0f-%8.0f" % (lo, hi))
        print("%-12s %8d %s" % ("%d盘%d轴承" % family, len(sub), " ".join(cells)))

    print()
    for o in orders:
        spans = per_order[o]
        if not spans:
            continue
        print("cs%d  common window over every family: %.0f .. %.0f rpm  (width %.2fx)"
              % (o, max(s[0] for s in spans), min(s[1] for s in spans),
                 min(s[1] for s in spans) / max(1.0, max(s[0] for s in spans))))
    print("\nA target outside the common window is still generatable -- the families "
          "that cannot reach it simply contribute nothing, and the delivered count "
          "drops accordingly.")


if __name__ == "__main__":
    main()