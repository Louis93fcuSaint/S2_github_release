# -*- coding: utf-8 -*-
"""ROSS-verify every 200-design CVAE/DDPM shortlist with the paper's worker."""
from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace

import pandas as pd

import cvae_common as C
import spec_common as S
import spec_interval as SI
import v2_ross_verify as V2R


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="cvae,ddpm")
    parser.add_argument("--seeds", default="42,7,13")
    parser.add_argument("--specs", default="")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--n-base", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    wanted = set(v.strip() for v in args.specs.split(",") if v.strip()) if args.specs else None
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    methods = [v.strip() for v in args.methods.split(",") if v.strip()]
    V2R.OUT_DIR = C.VERIFY_DIR
    surrogate = S.load_surrogate("mlp_best", threads=max(1, args.workers // 2))
    run_args = SimpleNamespace(workers=args.workers, timeout=args.timeout,
                               n_base=args.n_base, seed=42)
    for method in methods:
        for spec in C.SPECS:
            if wanted and spec["name"] not in wanted:
                continue
            frozen = C.frozen_spec(spec)
            specs = {1: (frozen["target"], frozen["target"])}
            for seed in seeds:
                tag = "%s_%s_s%d" % (method, spec["name"], seed)
                csv = os.path.join(C.CAND_DIR, "%s.csv" % tag)
                out_json = os.path.join(C.VERIFY_DIR, "%s.json" % tag)
                if not os.path.exists(csv):
                    print("[missing] %s" % csv, flush=True)
                    continue
                if os.path.exists(out_json) and not args.overwrite:
                    print("[skip] %s" % tag, flush=True)
                    continue
                t0 = time.time()
                frame = pd.read_csv(csv)
                if "rank" in frame:
                    frame = frame.sort_values("rank").reset_index(drop=True)
                entry = V2R.report("v2info", frame.head(C.N_SUBMIT), tag, specs,
                                   [C.TOL], run_args, surrogate, verbose=True)
                payload = {"tag": tag, "method": method, "spec": spec["name"],
                           "material": spec["material"], "target": spec["target"],
                           "seed": seed, "bands": {str(k): list(v) for k, v in specs.items()},
                           "tols": [C.TOL], "n_shortlist": int(len(frame.head(C.N_SUBMIT))),
                           "seconds_wall": float(time.time() - t0), **entry}
                C.save_json(out_json, payload)
                print("[verify] %-28s passed %d/%d" % (tag, entry["deliverable_n"],
                                                       entry["n_shortlist"]), flush=True)
    print("[verify] done", flush=True)


if __name__ == "__main__":
    main()
