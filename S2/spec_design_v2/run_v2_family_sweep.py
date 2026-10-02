# -*- coding: utf-8 -*-
"""v2 -- how many candidates should the family arms be given?

The family (how many disks / bearings) can be fixed in two places:

  CONDITIONED  --fam-allowed 3-3x2-2 : every candidate is drawn inside the family,
                                      so all N of them are usable.
  FILTERED     --fam-filter  3-3x2-2 : the batch is drawn over all 18 families and
                                      the wrong ones are dropped, so only ~1/18 of
                                      N survives -- but the survivors are spread
                                      over the family's real variety instead of
                                      being 200 near-copies of one conditioning
                                      point.

Earlier (README 4.6) the filtered arm was run once at 20 000 and beat the
conditioned one at 5 000: 192/199 against 181/200, at 4x the sampling cost.
That is a single pair.  This sweeps N for both arms with the ROSS budget fixed
at 200 designs per arm, so the table reads as

    given a fixed 200-solve budget, how many designs do I deliver,
    and what does the sampling cost me?

Usage:  python run_v2_family_sweep.py [--only filt_10k] [--threads 10] [--workers 18]
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

PY = sys.executable
TARGETS = "1=[3000,4500]"
RELAX = "0,0.05"

# name, n_generate, extra sample flags, note
ARMS = [
    ("cond_5k", 5000, ["--fam-allowed", "3-3x2-2"],
     "conditioned on the family: every candidate is usable"),
    ("cond_20k", 20000, ["--fam-allowed", "3-3x2-2"],
     "conditioned on the family: every candidate is usable"),
    ("filt_5k", 5000, ["--fam-filter", "3-3x2-2"],
     "free draw, filtered after: ~1/18 of the batch survives"),
    ("filt_10k", 10000, ["--fam-filter", "3-3x2-2"],
     "free draw, filtered after: ~1/18 of the batch survives"),
    ("filt_20k", 20000, ["--fam-filter", "3-3x2-2"],
     "free draw, filtered after: ~1/18 of the batch survives"),
    ("filt_40k", 40000, ["--fam-filter", "3-3x2-2"],
     "free draw, filtered after: ~1/18 of the batch survives"),
]


def run(cmd, log):
    print("[run] " + " ".join(cmd), flush=True)
    started = time.time()
    with open(log, "w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=HERE)
    seconds = time.time() - started
    print("[run] exit %d in %.0fs -> %s" % (proc.returncode, seconds, log), flush=True)
    if proc.returncode != 0:
        with open(log, encoding="utf-8") as handle:
            raise SystemExit("step failed:\n" + handle.read()[-2000:])
    return seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--material", default="Steel")
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

    print("%-9s %8s %7s %6s %6s %8s %7s %9s %s"
          % ("arm", "n_gen", "kept", "solve", "pass0", "pass5", "rho",
             "gen s", "s per delivered"))
    for name, n_gen, extra, note in ARMS:
        if wanted and name not in wanted:
            continue
        label = "sweep_%s" % name
        csv = os.path.join(OUT_DIR, "v2_%s.csv" % label)
        gen_seconds = None
        if not args.skip_sample:
            gen_seconds = run(
                [PY, os.path.join(HERE, "latent_ddpm_v2.py"), "sample",
                 "--tag", "v2i", "--out-tag", "v2_%s" % label,
                 "--targets", TARGETS, "--material", args.material,
                 "--tol", "0", "--use-ema", "1", "--guidance", "1.0",
                 "--guide-lambda", str(args.guide_lambda),
                 "--sample-steps", str(args.sample_steps),
                 "--n-generate", str(n_gen), "--n-submit", str(n_gen),
                 "--threads", str(args.threads), "--seed", str(args.seed)] + extra,
                os.path.join(logs, "v2_sample_%s.txt" % label))
        if not args.skip_verify:
            run([PY, os.path.join(HERE, "v2_ross_verify.py"),
                 "--csv", csv, "--targets", TARGETS, "--tols", RELAX,
                 "--top-n", str(args.top_n), "--workers", str(args.workers),
                 "--name", label, "--select", "top", "--seed", str(args.seed)],
                os.path.join(logs, "v2_verify_%s.txt" % label))

        with open(os.path.join(OUT_DIR, "verify_%s.json" % label), encoding="utf-8") as h:
            verify = json.load(h)
        with open(os.path.join(OUT_DIR, "funnel_v2_%s.json" % label), encoding="utf-8") as h:
            funnel = json.load(h)
        top = verify["runs"][0]
        kept = funnel.get("n_kept_after_filter")
        pass0 = top["gate_tol0"]["n_passed"] if "gate_tol0" in top else None
        pass5 = top["gate_tol5"]["n_passed"] if "gate_tol5" in top else None
        delivered = pass0 if pass0 is not None else pass5
        total = (gen_seconds or 0.0) + top["seconds"]
        entry = {"arm": name, "note": note, "targets": TARGETS,
                 "n_generate": n_gen, "n_kept": kept,
                 "sample_flags": extra,
                 "n_solved": top["n_solved"],
                 "gate_tol0": pass0, "gate_tol5": pass5,
                 "delivered": delivered,
                 "rho": top["spearman_joint_proxy_vs_true"],
                 "truth": top["truth"], "proxy": top["proxy"],
                 "gen_seconds": gen_seconds, "ross_seconds": top["seconds"],
                 "seconds_per_delivered": (total / delivered) if delivered else None}
        reports.append(entry)
        print("%-9s %8d %7s %6d %6s %6d %7.3f %7s %9s"
              % (name, n_gen, kept, top["n_solved"], pass0, pass5,
                 top["spearman_joint_proxy_vs_true"],
                 "%.0f" % gen_seconds if gen_seconds else "n/a",
                 "%.3f" % entry["seconds_per_delivered"]
                 if entry["seconds_per_delivered"] else "n/a"), flush=True)

    path = os.path.join(OUT_DIR, "family_sweep.json")
    seen = set(r["arm"] for r in reports)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            old = json.load(handle)
        reports = [r for r in old.get("reports", []) if r.get("arm") not in seen] + reports
    S.save_json(path, {"model": "v2i", "material": args.material,
                       "targets": TARGETS, "top_n": args.top_n,
                       "arms": [a[0] for a in ARMS], "reports": reports})
    print("[sweep] wrote %s" % path, flush=True)


if __name__ == "__main__":
    main()