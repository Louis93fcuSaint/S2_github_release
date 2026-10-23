# -*- coding: utf-8 -*-
"""Generate CVAE candidate pools and the common 200-design shortlists."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import cvae_common as C
import spec_common as S
import spec_eval as E
import spec_opt as O


def parse_list(text: str):
    return [v.strip() for v in str(text).split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(C.OUT_DIR, "cvae_main", "model_best.pt"))
    parser.add_argument("--seeds", default="42,7,13")
    parser.add_argument("--specs", default="")
    parser.add_argument("--guide-lambda", type=float, default=0.0,
                        help="0 = pure CVAE; >0 adds the diagnostic surrogate guidance")
    parser.add_argument("--guide-steps", type=int, default=C.GUIDE_STEPS)
    parser.add_argument("--prefix", default="cvae")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch
    torch.set_num_threads(int(args.threads))
    model, payload = C.load_checkpoint(args.model)
    scalers = payload["scalers"]
    ctx = E.pool_ctx()
    surrogate = S.load_surrogate("mlp_best", threads=args.threads)
    wanted = set(parse_list(args.specs)) if args.specs else None
    seeds = [int(v) for v in parse_list(args.seeds)]
    manifest = []
    manifest_path = os.path.join(C.RESULT_DIR, "%s_pool_manifest.json" % args.prefix)
    if os.path.exists(manifest_path):
        try:
            manifest = C.load_json(manifest_path)
        except Exception:
            manifest = []
    for spec in C.SPECS:
        if wanted and spec["name"] not in wanted:
            continue
        frozen = C.frozen_spec(spec)
        for seed in seeds:
            tag = "%s_%s_s%d" % (args.prefix, spec["name"], seed)
            pool_path = os.path.join(C.POOL_DIR, "%s_pool.csv" % tag)
            cand_path = os.path.join(C.CAND_DIR, "%s.csv" % tag)
            if (not args.overwrite) and os.path.exists(pool_path) and os.path.exists(cand_path):
                print("[skip] %s" % tag, flush=True)
                continue
            t0 = time.time()
            rng = np.random.default_rng(seed + 100003 * (C.SPECS.index(spec) + 1))
            fams = C.draw_families(C.N_GENERATE, rng)
            cond = C.make_conditions(spec["material"], fams, spec["target"], scalers)
            if args.guide_lambda and args.guide_lambda > 0:
                u, z, diag = C.guided_prior(
                    model, cond, fams, spec["material"], spec["target"], surrogate,
                    strength=args.guide_lambda, steps=args.guide_steps, seed=seed)
            else:
                u, z = C.sample_prior(model, cond, fams, C.N_GENERATE, seed)
                diag = {"guide_fired": 0,
                        "saturated_frac": float(np.mean((u <= 1e-6) | (u >= 1 - 1e-6)))}
            x = C.decode_latents(u, fams)
            valid = O.validity(x, spec["material"], fams)
            pred = C.predict_cs1(x, spec["material"], fams, surrogate)
            pool = C.make_pool_frame(x, fams, pred, frozen)
            pool.to_csv(pool_path, index=False, encoding="utf-8")
            np.save(os.path.join(C.POOL_DIR, "%s_u.npy" % tag), u.astype(np.float32))
            short, meta = C.select_common(x, fams, pred, frozen, ctx["scales"], C.N_SUBMIT)
            short.to_csv(cand_path, index=False, encoding="utf-8")
            row = {
                "tag": tag, "method": args.prefix, "spec": spec["name"],
                "material": spec["material"], "target": spec["target"], "seed": seed,
                "n_generate": int(len(x)), "n_submit": int(len(short)),
                "valid_rate": float(valid.mean()),
                "proxy_pool_in_band": float(np.mean(np.abs(
                    shortcut_dev := __import__("spec_interval").deviation(pred, frozen["lower"], frozen["upper"])) <= 1e-12)),
                "proxy_shortlist_in_band": float(meta["selected_in_band_proxy"] / max(meta["selected"], 1)),
                "median_dev_pool_pct": float(100.0 * np.median(np.abs(
                    __import__("spec_interval").deviation(pred, frozen["lower"], frozen["upper"])))),
                "seconds": float(time.time() - t0),
                "guide_lambda": float(args.guide_lambda),
                "guide_steps": int(args.guide_steps),
                **diag, "pool_csv": pool_path, "candidate_csv": cand_path}
            manifest = [m for m in manifest if m.get("tag") != tag] + [row]
            C.save_json(manifest_path, manifest)
            print("[cvae] %-28s valid %.4f poolIn %.4f shortIn %.3f %.1fs"
                  % (tag, row["valid_rate"], row["proxy_pool_in_band"],
                     row["proxy_shortlist_in_band"], row["seconds"]), flush=True)
    print("[cvae] manifest %s" % manifest_path, flush=True)


if __name__ == "__main__":
    main()


