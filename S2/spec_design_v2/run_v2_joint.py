# -*- coding: utf-8 -*-
"""v2 -- interval specs on SEVERAL orders at once, and family before vs after.

Three questions this driver answers, all with ROSS ground truth:

  1. JOINT INTERVAL   can the user pin two or three critical speeds at the same
                      time when each of them is a *range* rather than a number?
                      (`1=[3000,4500],2=[6000,9000]`, + cs3)
  2. MIXED            a point on one order and a one-sided band on another
                      (`1=3680.8,2=[6000,+]`) in the same query.
  3. FAMILY           "disks 3, bearings 2" asked BEFORE sampling (--fam-allowed,
                      conditioned) versus AFTER sampling (--fam-filter, the batch
                      is drawn free and the wrong families are dropped).  Same
                      spec, same shortlist size, so the comparison is fair on
                      delivery -- but not on cost: post-filtering throws away
                      17/18 of the batch, so it is run with 4x the candidates.

Usage:  python run_v2_joint.py [--only cs12,fam] [--n-generate 5000]
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
V1_DIR = os.path.join(os.path.dirname(HERE), "spec_design")
for path in (V1_DIR, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import spec_common as S          # noqa: E402
import spec_interval as SI       # noqa: E402

PY = sys.executable

# name, targets, judge relaxations, n_generate, extra sample flags, note
ARMS = [
    ("cs12", "1=[3000,4500],2=[6000,9000]", "0,0.05", 5000, [],
     "two orders, both given as ranges"),
    ("cs123", "1=[3000,4500],2=[6000,9000],3=[14000,22000]", "0,0.05", 5000, [],
     "three orders, all ranges"),
    ("mix", "1=3680.8,2=[6000,+]", "0.05", 5000, [],
     "one point + one one-sided band in the same query"),
    ("fam_cond", "1=[3000,4500]", "0,0.05", 5000, ["--fam-allowed", "3-3x2-2"],
     "family fixed BEFORE sampling (conditioned)"),
    ("fam_filt", "1=[3000,4500]", "0,0.05", 20000, ["--fam-filter", "3-3x2-2"],
     "family fixed AFTER sampling (free draw, then filtered)"),
]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--material", default="Steel")
    parser.add_argument("--n-generate", type=int, default=0,
                        help="override the per-arm candidate count")
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
    for name, targets, relax, n_gen, extra, note in ARMS:
        if wanted and name not in wanted:
            continue
        n_gen = args.n_generate or n_gen
        label = "joint_%s" % name
        csv = os.path.join(OUT_DIR, "v2_%s.csv" % label)
        print("[arm] %s | %s | judge %s | n=%d" % (name, targets, relax, n_gen), flush=True)
        if not args.skip_sample:
            run([PY, os.path.join(HERE, "latent_ddpm_v2.py"), "sample",
                 "--tag", "v2i", "--out-tag", "v2_%s" % label,
                 "--targets", targets, "--material", args.material,
                 "--tol", "0", "--use-ema", "1", "--guidance", "1.0",
                 "--guide-lambda", str(args.guide_lambda),
                 "--sample-steps", str(args.sample_steps),
                 "--n-generate", str(n_gen), "--n-submit", str(n_gen),
                 "--threads", str(args.threads), "--seed", str(args.seed)] + extra,
                os.path.join(logs, "v2_sample_%s.txt" % label))
        if not args.skip_verify:
            run([PY, os.path.join(HERE, "v2_ross_verify.py"),
                 "--csv", csv, "--targets", targets, "--tols", relax,
                 "--top-n", str(args.top_n), "--workers", str(args.workers),
                 "--name", label, "--select", "top", "--seed", str(args.seed)],
                os.path.join(logs, "v2_verify_%s.txt" % label))
        with open(os.path.join(OUT_DIR, "verify_%s.json" % label), encoding="utf-8") as h:
            verify = json.load(h)
        with open(os.path.join(OUT_DIR, "funnel_v2_%s.json" % label), encoding="utf-8") as h:
            funnel = json.load(h)
        entry = {"arm": name, "targets": targets, "note": note,
                 "bands": {o: SI.band_text(*SI.parse_specs(targets)[int(o)])
                           for o in SI.parse_specs(targets)},
                 "n_generate": n_gen, "sample_flags": extra,
                 "n_kept_after_filter": funnel.get("n_kept_after_filter"),
                 "proxy_whole_batch": verify["proxy_whole_batch"],
                 "runs": []}
        for r in verify["runs"]:
            entry["runs"].append({
                "label": r["label"], "n_shortlist": r["n_shortlist"],
                "n_solved": r["n_solved"], "seconds": r["seconds"],
                "rho": r["spearman_joint_proxy_vs_true"],
                "truth": r["truth"], "proxy": r["proxy"],
                "gate": {k: v for k, v in r.items() if k.startswith("gate_tol")},
                "deliverable_n": r.get("deliverable_n")})
        reports.append(entry)
        top = verify["runs"][0]
        print("[arm] %-9s top%d: solved %d | rho %.3f | truth joint %s | gates %s"
              % (name, args.top_n, top["n_solved"], top["spearman_joint_proxy_vs_true"],
                 {k: round(v, 3) for k, v in top["truth"].items()
                  if k.startswith("joint_in_band")},
                 {k: v["n_passed"] for k, v in top.items() if k.startswith("gate_tol")}),
              flush=True)

    path = os.path.join(OUT_DIR, "joint_summary.json")
    # merge with whatever is already there so a partial rerun (--only fam) does
    # not throw away the arms measured earlier
    seen = set(r["arm"] for r in reports)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            old = json.load(handle)
        reports = [r for r in old.get("reports", []) if r.get("arm") not in seen] + reports
    S.save_json(path, {"model": "v2i", "material": args.material,
                       "arms": [a[0] for a in ARMS], "reports": reports})
    print("[joint] wrote %s" % path, flush=True)


if __name__ == "__main__":
    main()
