# -*- coding: utf-8 -*-
"""Baseline ROSS-truth probe (S2, new approach).

Why this file exists
--------------------
`06_report.py` compares the baselines on PROXY in-band rate, and for the truth
column it only ever used a *random* 20-design control.  The missing row -- and
the one the whole "proxy pre-select + ROSS certify" story rides on -- is:

    what is the ROSS-truth in-band rate of the candidates the baseline ITSELF
    ranks first?

This script answers that with the v2 toolchain (same frozen surrogate, same ROSS
worker, same band arithmetic as `spec_design_v2/v2_ross_verify.py`), for the one
baseline set whose candidate files still exist on disk:

    spec_design/outputs/latent_baselines/<method>__<spec>.csv   (7 x 6, 200 rows)

For every (method, spec) it solves, with ROSS:
  * top-20  -- the 20 the frozen v2 surrogate likes best
  * rand-20 -- unbiased control from the same submission (seed 42)

Output: spec_design_v2/outputs/bl_truth/*.csv and bl_truth_summary.json
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(os.path.dirname(HERE), "spec_design")
for _p in (V1, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spec_common as S        # noqa: E402
import spec_interval as SI     # noqa: E402
import v2_ross_verify as V     # noqa: E402

BL_DIR = os.path.join(V1, "outputs", "latent_baselines")
SPECS = os.path.join(V1, "outputs", "specs.json")
OUT = os.path.join(HERE, "outputs", "bl_truth")
TOLS = [0.01, 0.05]


def load_specs():
    with open(SPECS, encoding="utf-8") as fh:
        return json.load(fh)["specs"]


def proxy_joint(frame, target, surrogate):
    """Joint band deviation of every row, predicted by the frozen v2 surrogate."""
    mat = frame[S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
    full = surrogate.predict(S.canonical_to_model_rows(
        mat,
        [str(m) for m in frame["material"]],
        [int(v) for v in frame["n_disks"]],
        [int(v) for v in frame["n_bearings"]]))
    pred = np.asarray(full, dtype=np.float64)[:, 0]
    lo, hi = SI.parse_specs("1=%g" % target)[1]
    return np.abs(SI.deviation(pred, lo, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--n-base", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default="", help="substring filter, e.g. de__Steel_mid")
    ap.add_argument("--skip-rand", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    V.OUT_DIR = OUT
    vargs = types.SimpleNamespace(workers=args.workers, timeout=args.timeout,
                                  n_base=args.n_base, seed=args.seed)

    print("[bl] loading frozen v2 surrogate ...", flush=True)
    surrogate = S.load_surrogate("mlp_best")
    specs = load_specs()

    rows = []
    methods = sorted(set(os.path.basename(p).split("__")[0]
                         for p in os.listdir(BL_DIR) if p.endswith(".csv")))
    started = time.time()
    for spec in specs:
        for method in methods:
            name = "%s__%s" % (method, spec["name"])
            if args.only and args.only not in name:
                continue
            path = os.path.join(BL_DIR, name + ".csv")
            if not os.path.exists(path):
                print("[bl] missing %s" % path, flush=True)
                continue
            frame = pd.read_csv(path)
            dev = proxy_joint(frame, spec["target"], surrogate)
            ordered = frame.iloc[np.argsort(dev, kind="stable")].reset_index(drop=True)

            entry = {"method": method, "spec": spec["name"],
                     "material": spec["material"], "target": spec["target"],
                     "n_submitted": int(len(frame))}
            top = V.report("bl", ordered.head(20), name + "__top20",
                           {1: tuple(SI.parse_specs("1=%g" % spec["target"])[1])},
                           TOLS, vargs, surrogate)
            entry["top20"] = {"truth_in_band_5": top["truth"]["joint_in_band_tol5"],
                              "truth_in_band_1": top["truth"]["joint_in_band_tol1"],
                              "proxy_in_band_5": top["proxy"]["joint_in_band_tol5"],
                              "truth_median_ape": top["truth"]["order1"]["median_ape_pct"],
                              "rho": top["spearman_order1"]}
            if not args.skip_rand:
                ctrl = frame.sample(n=20, random_state=args.seed).reset_index(drop=True)
                r = V.report("bl", ctrl, name + "__rand20",
                             {1: tuple(SI.parse_specs("1=%g" % spec["target"])[1])},
                             TOLS, vargs, surrogate)
                entry["rand20"] = {"truth_in_band_5": r["truth"]["joint_in_band_tol5"],
                                   "truth_in_band_1": r["truth"]["joint_in_band_tol1"]}
            rows.append(entry)
            print("[bl] %-28s top20 truth+-5%%=%.3f | rand20=%.3f"
                  % (name, entry["top20"]["truth_in_band_5"],
                     entry.get("rand20", {}).get("truth_in_band_5", float("nan"))),
                  flush=True)

    meta = {"elapsed_s": time.time() - started, "workers": args.workers,
            "tols": TOLS, "n_base": args.n_base, "seed": args.seed,
            "baseline_dir": BL_DIR, "rows": rows}
    with open(os.path.join(OUT, "bl_truth_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print("[bl] %d combos in %.0f s -> %s" % (len(rows), meta["elapsed_s"], OUT), flush=True)


if __name__ == "__main__":
    main()