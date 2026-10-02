# -*- coding: utf-8 -*-
"""v2 -- one-command evaluation of the multi-order conditioner.

Four conditions, all on Steel, all centred on the same well-supported rotor
neighbourhood (cs1 = 3680.8 rpm is the median of Steel, and 7021 / 17970 are the
in-band medians of cs2 and cs3 once cs1 is pinned at +-1 %, taken from
outputs/feasibility.json):

    cs1         1=3680.8
    cs2                    2=7021
    cs1cs2      1=3680.8   2=7021
    cs123       1=3680.8   2=7021   3=17970.5

Each condition is sampled once with the guidance of v1 (--guide-lambda 0.05) and
the whole 5000-rotor batch is written out, so the same batch answers both
tolerances (+-1 % and +-5 %) and both the shortlist and the unbiased random
control.  Then every batch goes through ROSS.

Usage:  python run_v2_eval.py [--only cs1,cs2] [--n-generate 5000] [--skip-verify]
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
sys.path.insert(0, V1_DIR)
sys.path.insert(0, HERE)

import spec_common as S          # noqa: E402

PY = sys.executable
# label, targets, material, n_generate, family ('' = the disk / bearing counts are free)
CONDITIONS = [
    ("cs1", "1=3680.8", "Steel", 5000, ""),
    ("cs2", "2=7021.0", "Steel", 5000, ""),
    ("cs3", "3=21033.5", "Steel", 5000, ""),
    ("cs1cs3", "1=3680.8,3=18070.4", "Steel", 5000, ""),
    ("cs2cs3", "2=7021.0,3=15857.7", "Steel", 5000, ""),
    ("cs1cs2", "1=3680.8,2=7021.0", "Steel", 5000, ""),
    ("cs123", "1=3680.8,2=7021.0,3=17970.5", "Steel", 5000, ""),
    # the +-5 % operating point: the pair / triple arms are thin in the batch,
    # so they get four times the candidates; al_cs1cs2 is the cross-check on a
    # different material AND a different target point.
    ("cs1cs2_big", "1=3680.8,2=7021.0", "Steel", 20000, ""),
    ("cs123_big", "1=3680.8,2=7021.0,3=17970.5", "Steel", 20000, ""),
    ("al_cs1cs2", "1=3191.1,2=6147.6", "Aluminum", 5000, ""),
    # family as an optional condition: free / fixed (3-3x2-2) / ranged (1-2x2-4).
    # Fixed families are *better* supported than the free pool (narrower
    # manifold); a range has to compromise, because one target has to serve
    # families whose reachable windows only overlap over 3126-6574 rpm.
    ("fam32_cs1", "1=2974", "Steel", 5000, "3-3x2-2"),
    ("fam32_cs12", "1=2974,2=7001", "Steel", 5000, "3-3x2-2"),
    ("fam22_cs2", "2=8180", "Steel", 5000, "2-2x2-2"),
    ("fam12_24_cs12", "1=4642,2=9969", "Steel", 5000, "1-2x2-4"),
    ("fam32_cs123", "1=2974,2=7001,3=20359", "Steel", 20000, "3-3x2-2"),
]


def run(cmd, log):
    print("[run] " + " ".join(cmd), flush=True)
    started = time.time()
    with open(log, "w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT,
                              cwd=HERE)
    print("[run] exit %d in %.0fs -> %s" % (proc.returncode, time.time() - started, log),
          flush=True)
    if proc.returncode != 0:
        with open(log, encoding="utf-8") as handle:
            tail = handle.read()[-2000:]
        raise SystemExit("step failed:\n" + tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v2m")
    parser.add_argument("--only", default="")
    parser.add_argument("--material", default="Steel")
    parser.add_argument("--n-generate", type=int, default=0,
                        help="override every condition's own candidate count")
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument("--guide-lambda", type=float, default=0.05)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=0.05)
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
    summary = []
    for label, targets, material, n_generate, family in CONDITIONS:
        if wanted and label not in wanted:
            continue
        if args.n_generate:
            n_generate = args.n_generate
        csv = os.path.join(OUT_DIR, "v2_%s.csv" % label)
        if not args.skip_sample:
            run([PY, os.path.join(HERE, "latent_ddpm_v2.py"), "sample",
                 "--tag", args.tag, "--out-tag", "v2_%s" % label,
                 "--targets", targets, "--material", material,
                 "--tol", str(args.tol), "--use-ema", "1",
                 "--guidance", str(args.guidance),
                 "--guide-lambda", str(args.guide_lambda),
                 "--sample-steps", str(args.sample_steps),
                 "--n-generate", str(n_generate),
                 "--n-submit", str(n_generate),
                 "--threads", str(args.threads), "--seed", str(args.seed)]
                + (["--fam-allowed", family] if family else []),
                os.path.join(logs, "v2_sample_%s.txt" % label))
        if args.skip_verify:
            continue
        run([PY, os.path.join(HERE, "v2_ross_verify.py"),
             "--csv", csv, "--targets", targets, "--tols", "0.01,0.05",
             "--top-n", str(args.top_n), "--workers", str(args.workers),
             "--name", label, "--select", "both", "--seed", str(args.seed)],
            os.path.join(logs, "v2_verify_%s.txt" % label))
        with open(os.path.join(OUT_DIR, "verify_%s.json" % label), encoding="utf-8") as h:
            summary.append(json.load(h))

    path = os.path.join(OUT_DIR, "v2_summary.json")
    S.save_json(path, {"conditions": [c[0] for c in CONDITIONS if not wanted or c[0] in wanted],
                       "targets": {c[0]: c[1] for c in CONDITIONS},
                       "guide_lambda": args.guide_lambda,
                       "n_generate": args.n_generate, "reports": summary})
    print("[eval] wrote %s" % path, flush=True)


if __name__ == "__main__":
    main()