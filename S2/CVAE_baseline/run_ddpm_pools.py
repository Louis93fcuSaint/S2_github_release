# -*- coding: utf-8 -*-
"""Prepare the DDPM pools under the same common selector as the CVAE."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import cvae_common as C
import spec_common as S
import spec_eval as E
import spec_interval as SI


def run(cmd, log):
    print("[ddpm] " + " ".join(cmd), flush=True)
    with open(log, "w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=C.V2_DIR, stdout=handle,
                              stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        with open(log, encoding="utf-8") as handle:
            raise SystemExit("DDPM sample failed:\n" + handle.read()[-2000:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,7,13")
    parser.add_argument("--specs", default="")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    wanted = set(v.strip() for v in args.specs.split(",") if v.strip()) if args.specs else None
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    ctx = E.pool_ctx()
    manifest_path = os.path.join(C.RESULT_DIR, "ddpm_pool_manifest.json")
    manifest = C.load_json(manifest_path) if os.path.exists(manifest_path) else []
    for spec in C.SPECS:
        if wanted and spec["name"] not in wanted:
            continue
        frozen = C.frozen_spec(spec)
        for seed in seeds:
            tag = "ddpm_%s_s%d" % (spec["name"], seed)
            cand_path = os.path.join(C.CAND_DIR, "%s.csv" % tag)
            pool_path = os.path.join(C.POOL_DIR, "%s_pool.csv" % tag)
            if (not args.overwrite) and os.path.exists(cand_path) and os.path.exists(pool_path):
                print("[skip] %s" % tag, flush=True)
                continue
            t0 = time.time()
            if seed == 42:
                src = os.path.join(C.V2_DIR, "outputs", "nov5k_%s.csv" % spec["name"])
            else:
                src = os.path.join(C.V2_DIR, "outputs", "ddpm_pool_%s_s%d.csv" % (spec["name"], seed))
                if args.overwrite or not os.path.exists(src):
                    out_tag = "ddpm_pool_%s_s%d" % (spec["name"], seed)
                    log = os.path.join(C.LOG_DIR, "ddpm_sample_%s_s%d.log" % (spec["name"], seed))
                    run([sys.executable, os.path.join(C.V2_DIR, "latent_ddpm_v2.py"), "sample",
                         "--tag", "v2i", "--out-tag", out_tag,
                         "--targets", "1=%g" % spec["target"], "--material", spec["material"],
                         "--tol", "0.05", "--use-ema", "1", "--guidance", "1.0",
                         "--guide-lambda", "0.05", "--sample-steps", "25",
                         "--n-generate", str(C.N_GENERATE), "--n-submit", str(C.N_GENERATE),
                         "--threads", str(args.threads), "--seed", str(seed)], log)
            frame = pd.read_csv(src)
            if len(frame) < C.N_GENERATE:
                raise RuntimeError("%s has only %d rows" % (src, len(frame)))
            frame = frame.head(C.N_GENERATE).copy()
            x = frame[S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
            fams = [(int(a), int(b)) for a, b in zip(frame["n_disks"], frame["n_bearings"])]
            if "cs1_pred" in frame:
                pred = frame["cs1_pred"].to_numpy(dtype=np.float64)
            else:
                pred = C.predict_cs1(x, spec["material"], fams, S.load_surrogate("mlp_best"))
            pool = C.make_pool_frame(x, fams, pred, frozen)
            pool.to_csv(pool_path, index=False, encoding="utf-8")
            short, meta = C.select_common(x, fams, pred, frozen, ctx["scales"], C.N_SUBMIT)
            short.to_csv(cand_path, index=False, encoding="utf-8")
            row = {"tag": tag, "method": "ddpm", "spec": spec["name"],
                   "material": spec["material"], "target": spec["target"], "seed": seed,
                   "n_generate": int(len(x)), "n_submit": int(len(short)),
                   "proxy_pool_in_band": float(np.mean(np.abs(
                       SI.deviation(pred, frozen["lower"], frozen["upper"])) <= 1e-12)),
                   "proxy_shortlist_in_band": float(meta["selected_in_band_proxy"] / max(meta["selected"], 1)),
                   "median_dev_pool_pct": float(100.0 * np.median(np.abs(
                       SI.deviation(pred, frozen["lower"], frozen["upper"])))),
                   "seconds": float(time.time() - t0), "source_csv": src,
                   "pool_csv": pool_path, "candidate_csv": cand_path}
            manifest = [m for m in manifest if m.get("tag") != tag] + [row]
            C.save_json(manifest_path, manifest)
            print("[ddpm] %-28s poolIn %.4f shortIn %.3f %.1fs"
                  % (tag, row["proxy_pool_in_band"], row["proxy_shortlist_in_band"], row["seconds"]),
                  flush=True)
    print("[ddpm] manifest %s" % manifest_path, flush=True)


if __name__ == "__main__":
    main()
