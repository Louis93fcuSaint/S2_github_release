# -*- coding: utf-8 -*-
"""More ROSS truth for the batches whose keep fraction rests on only 40 rotors.

The per-batch keep fraction is a proportion estimated from 40 samples, so its
noise is about +/-0.08, which is the same size as some of the effects the report
quotes.  This adds fresh rotors (never verified before) to a chosen list of
batches and appends them to uncertainty_stageB.json, so every downstream number
can be recomputed on the larger sample and the sampling noise quantified.
"""
import argparse
import concurrent.futures as futures
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec_common as S
import spec_eval as E
import uncertainty_signals as U
import _stage_b_uncertainty as SB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="")
    parser.add_argument("--specs", default="")
    parser.add_argument("--extra", type=int, default=60)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--n-base", type=int, default=15)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    ctx = E.pool_ctx()
    surrogate = S.load_surrogate("mlp_best", threads=8)
    bank = U.SignalBank(ctx, surrogate)
    methods = set(args.methods.split(",")) if args.methods else None
    specs = set(args.specs.split(",")) if args.specs else None
    plan = SB.files_for(methods, specs)
    store = SB.load_store()
    print("batches %d | adding up to %d rotors each" % (len(plan), args.extra), flush=True)

    stats = {"timeout": 0, "error": 0}
    lock = threading.Lock()
    run = SB.make_runner(args.timeout, args.n_base, stats, lock)
    started = time.time()

    for index, (method, spec_name, path) in enumerate(plan):
        key = "%s|%s" % (method, spec_name)
        if key not in store:
            print("[skip] %s not in the store" % key, flush=True)
            continue
        frame = pd.read_csv(path).reset_index(drop=True)
        seen = {r["index"] for r in store[key]["rotors"]}
        available = np.array([i for i in range(len(frame)) if i not in seen])
        rng = np.random.default_rng(args.seed + index)
        take = min(args.extra, len(available))
        pick = np.sort(rng.choice(available, take, replace=False))

        x = frame[S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
        rows = S.canonical_to_model_rows(x, frame["material"].tolist(),
                                         frame["n_disks"].tolist(),
                                         frame["n_bearings"].tolist())
        sig = bank.score(rows[pick], x[pick])
        pred = np.exp(sig["log_mean"])
        q = U.margin_q()
        lo = pred / np.exp(q * sig["log_spread"])
        jobs = [(x[i], str(frame["material"].iloc[i]), int(frame["n_disks"].iloc[i]),
                 int(frame["n_bearings"].iloc[i])) for i in pick]
        t0 = time.time()
        with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(run, jobs))
        added = 0
        for j, (i, res) in enumerate(zip(pick, results)):
            truth = res["forward"][0] if res.get("ok") and res.get("forward") else None
            store[key]["rotors"].append({
                "index": int(i), "material": str(frame["material"].iloc[i]),
                "nd": int(frame["n_disks"].iloc[i]), "nb": int(frame["n_bearings"].iloc[i]),
                "proxy": float(pred[j]), "proxy_lo": float(lo[j]),
                "spread": float(sig["log_spread"][j]), "dF": float(sig["dF"][j]),
                "dD": float(sig["dD"][j]), "box": float(sig["box"][j]),
                "truth": None if truth is None else float(truth), "round": "extra",
            })
            added += 1 if truth else 0
        store[key]["n_rotors"] = len(store[key]["rotors"])
        SB.save_store(store)
        print("%-16s %-14s +%d solve, %d with truth | %.0fs"
              % (method, spec_name, len(pick), added, time.time() - t0), flush=True)

    print("[out] %s (%.0f s, timeouts %d errors %d)"
          % (SB.STORE, time.time() - started, stats["timeout"], stats["error"]))


if __name__ == "__main__":
    main()