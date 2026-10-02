# -*- coding: utf-8 -*-
"""v2 -- interval specs: does conditioning on the two edges beat the centre?

A user may now state a target either as a point (`1=3680.8` + a tolerance) or as
a range (`1=[10000,12000]`, `1=[14400,+]` for "at least").  Two checkpoints
answer the same four bands:

  v2m  point layout     condition = one number per order (the band's geometric
                        centre).  This is what a centre-only reading gives.
  v2i  interval layout  condition = both edges per order; trained with random
                        band widths, one-sided bands and exact points mixed in.

The delivered count barely moves.  What moves is the *shape* of the fill: a
centre-conditioned batch piles up at one end of a wide band, an edge-conditioned
one spreads over it.  The quintile shares below are that measurement, computed
in log space inside the band the model can actually fill (see
spec_interval.effective).

Usage:  python run_v2_interval.py [--only narrow,mid] [--n-generate 5000]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
V1_DIR = os.path.join(os.path.dirname(HERE), "spec_design")
for path in (V1_DIR, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import spec_common as S          # noqa: E402
import spec_interval as SI       # noqa: E402

PY = sys.executable
H_MAX = 0.70

# name, targets, relax, note
ARMS = [
    ("narrow", "1=[3500,3870]", "0", "10 % band, narrower than the model's own spread"),
    ("mid", "1=[2600,5200]", "0", "2x band, the ordinary case"),
    ("wide", "1=[1800,7500]", "0", "4.2x band, wider than the model can fill"),
    ("open", "1=[3681,+]", "0", "one-sided: 'at least 3681', the margin-only spec"),
]
MODELS = ["v2m", "v2i"]


def run(cmd, log):
    print("[run] " + " ".join(cmd), flush=True)
    started = time.time()
    with open(log, "w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=HERE)
    print("[run] exit %d in %.0fs -> %s" % (proc.returncode, time.time() - started, log),
          flush=True)
    if proc.returncode != 0:
        with open(log, encoding="utf-8") as handle:
            raise SystemExit("step failed:\n" + handle.read()[-2000:])


def fill_window(lo, hi, h_max=H_MAX):
    """The part of the band the denoiser is actually asked to fill."""
    if hi >= SI.INF_CAP:
        return lo, lo * math.exp(h_max)
    if math.log(hi / lo) > h_max:
        c = SI.centre(lo, hi)
        return c * math.exp(-0.5 * h_max), c * math.exp(0.5 * h_max)
    return lo, hi


def quintiles(values, lo, hi):
    """Share of the in-band designs in each fifth of the fill window (log space)."""
    low, high = fill_window(lo, hi)
    values = np.asarray(values, dtype=float)
    values = values[(values >= lo) & (values <= hi)]
    if len(values) == 0:
        return None, 0
    pos = np.clip((np.log(values) - math.log(low)) / (math.log(high) - math.log(low)),
                  0.0, 0.999999)
    return (np.bincount((pos * 5).astype(int), minlength=5) / len(values)).tolist(), len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--material", default="Steel")
    parser.add_argument("--n-generate", type=int, default=5000)
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument("--guide-lambda", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    wanted = [v for v in args.only.split(",") if v] or None
    logs = os.path.join(OUT_DIR, "logs")
    os.makedirs(logs, exist_ok=True)
    reports = []
    print("%-8s %-18s %-30s %6s %6s" % ("arm", "band", "fill window", "q1", "q5"))
    for name, targets, relax, note in ARMS:
        if wanted and name not in wanted:
            continue
        lo, hi = SI.parse_specs(targets)[1]
        fill = fill_window(lo, hi)
        for model in MODELS:
            label = "int_%s_%s" % (name, model)
            csv = os.path.join(OUT_DIR, "v2_%s.csv" % label)
            if not args.skip_sample:
                run([PY, os.path.join(HERE, "latent_ddpm_v2.py"), "sample",
                     "--tag", model, "--out-tag", "v2_%s" % label,
                     "--targets", targets, "--material", args.material,
                     "--tol", "0", "--use-ema", "1", "--guidance", "1.0",
                     "--guide-lambda", str(args.guide_lambda),
                     "--sample-steps", str(args.sample_steps),
                     "--n-generate", str(args.n_generate),
                     "--n-submit", str(args.n_generate),
                     "--threads", str(args.threads), "--seed", str(args.seed)],
                    os.path.join(logs, "v2_sample_%s.txt" % label))
            if not args.skip_verify:
                run([PY, os.path.join(HERE, "v2_ross_verify.py"),
                     "--csv", csv, "--targets", targets, "--tols", relax,
                     "--top-n", str(args.top_n), "--workers", str(args.workers),
                     "--name", label, "--select", "top", "--seed", str(args.seed)],
                    os.path.join(logs, "v2_verify_%s.txt" % label))

            import pandas as pd
            batch = pd.read_csv(csv)
            solved = pd.read_csv(os.path.join(OUT_DIR, "verify_%s__top%d.csv"
                                              % (label, args.top_n)))
            q_batch, n_batch = quintiles(batch["cs1_pred"].to_numpy(), lo, hi)
            q_true, n_true = quintiles(solved["cs1_ross"].to_numpy(), lo, hi)
            entry = {"arm": name, "model": model, "targets": targets, "note": note,
                     "band": [lo, hi], "fill_window": list(fill),
                     "batch_quintiles": q_batch, "batch_in_band": n_batch,
                     "shortlist_quintiles": q_true, "shortlist_in_band": n_true}
            with open(os.path.join(OUT_DIR, "verify_%s.json" % label), encoding="utf-8") as h:
                report = json.load(h)
            entry["gate"] = report["runs"][0].get("gate_tol0")
            entry["whole_batch_in_band"] = report["proxy_whole_batch"]["joint_in_band_tol0"]
            reports.append(entry)
            print("%-8s %-18s %-30s %6s" % (name, "%g-%g" % (lo, min(hi, 1e9)),
                                            "%g-%g" % fill,
                                            " ".join("%.3f" % v for v in q_batch)),
                  flush=True)

    path = os.path.join(OUT_DIR, "interval_summary.json")
    S.save_json(path, {"h_max": H_MAX, "n_generate": args.n_generate,
                       "models": {"v2m": "point layout (centre only)",
                                  "v2i": "interval layout (both edges)"},
                       "reports": reports})
    print("[interval] wrote %s" % path, flush=True)


if __name__ == "__main__":
    main()