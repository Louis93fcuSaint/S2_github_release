# -*- coding: utf-8 -*-
"""v2 -- one query, one batch of designs, optionally certified by ROSS.

  user: material + how many disks / bearings (fixed, ranged or free) + a target
        for whichever of {cs1, cs2, cs3} they care about, alone or combined,
        and a tolerance

  us:   condition the v2 latent DDPM on that subset, generate the batch, rank by
        the frozen six-order surrogate, and -- under a tight band -- solve the
        shortlist with ROSS so the delivered list is measured, not claimed.

Examples
--------
python design_by_target_v2.py --targets 1=3680.8 --material Steel\npython design_by_target_v2.py --targets "1=[10000,12000]" --material Steel --verify\npython design_by_target_v2.py --targets "1=[14400,+]" --material Steel --verify
python design_by_target_v2.py --targets 2=7021 --material Steel --tol 0.01 --verify
python design_by_target_v2.py --targets 1=3680.8,2=7021 --material Steel --tol 0.01 --verify
python design_by_target_v2.py --targets 1=3680.8,2=7021,3=17970 --tol 0.05 --verify --disks 3-3 --bearings 2-2
python design_by_target_v2.py --targets "1=[3000,4500],2=[6000,9000]" --material Steel --verify
python design_by_target_v2.py --targets "1=[3000,4500],2=[6000,9000],3=[14000,22000]" --material Steel --verify
python design_by_target_v2.py --targets 1=3680.8,2=[6000,+] --material Steel --verify
python design_by_target_v2.py --targets "1=[3000,4500]" --material Steel --filter-disks 3-3 --filter-bearings 2-2 --verify

The tolerance only changes how the results are *judged and ranked*; the sampled
batch is the same.  Under +-1 % the surrogate cannot tell two shortlisted
candidates apart (its own error is 1.3-3.2 %), so --verify is not optional if
the answer is going to be used: it is what turns a claim into a measurement.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
V1_DIR = os.path.join(os.path.dirname(HERE), "spec_design")
for path in (V1_DIR, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)
import spec_common as S          # noqa: E402

PY = sys.executable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True,
                        help="e.g. 1=3680.8 or 2=7021 or 1=3680.8,2=7021,3=17970")
    parser.add_argument("--material", required=True, choices=list(S.MATERIAL_ORDER))
    parser.add_argument("--tol", type=float, default=None,
                        help="judging relaxation around the band; defaults to 0 for "
                             "an interval target and 0.05 for a point target")
    parser.add_argument("--disks", default="", help="range, e.g. 3-3 or 2-4")
    parser.add_argument("--bearings", default="", help="range, e.g. 2-4")
    parser.add_argument("--filter-disks", default="",
                        help="post-generation filter: generate free, then keep "
                             "only these disk counts, e.g. 3-3 or 2-4")
    parser.add_argument("--filter-bearings", default="",
                        help="post-generation filter on the bearing count")
    parser.add_argument("--tag", default="v2i")
    parser.add_argument("--out-tag", default="")
    parser.add_argument("--n-generate", type=int, default=5000)
    parser.add_argument("--n-submit", type=int, default=200)
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--guide-lambda", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-workers", type=int, default=18)
    parser.add_argument("--verify-timeout", type=float, default=180.0)
    args = parser.parse_args()

    out_tag = args.out_tag or "query"
    if args.tol is None:
        # an interval is the spec itself; a point needs the tolerance to become one
        args.tol = 0.0 if "[" in args.targets else 0.05
    fam = ""
    if args.disks or args.bearings:
        fam = "%sx%s" % (args.disks or "", args.bearings or "")
    fam_filter = ""
    if args.filter_disks or args.filter_bearings:
        fam_filter = "%sx%s" % (args.filter_disks or "", args.filter_bearings or "")
    cmd = [PY, os.path.join(HERE, "latent_ddpm_v2.py"), "sample",
           "--tag", args.tag, "--out-tag", out_tag, "--targets", args.targets,
           "--material", args.material, "--tol", str(args.tol), "--use-ema", "1",
           "--guidance", str(args.guidance), "--guide-lambda", str(args.guide_lambda),
           "--sample-steps", str(args.sample_steps),
           "--n-generate", str(args.n_generate), "--n-submit", str(args.n_submit),
           "--threads", str(args.threads), "--seed", str(args.seed)]
    if fam:
        cmd += ["--fam-allowed", fam]
    if fam_filter:
        cmd += ["--fam-filter", fam_filter]
    print("[v2] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)
    csv = os.path.join(OUT_DIR, "%s.csv" % out_tag)
    if not args.verify:
        print("[v2] shortlist -> %s\n[v2] tight band? rerun with --verify" % csv,
              flush=True)
        return
    subprocess.run([PY, os.path.join(HERE, "v2_ross_verify.py"),
                    "--csv", csv, "--targets", args.targets,
                    "--tols", "%g" % args.tol, "--top-n", str(args.n_submit),
                    "--workers", str(args.verify_workers),
                    "--timeout", str(args.verify_timeout),
                    "--name", out_tag, "--select", "top", "--seed", str(args.seed)],
                   cwd=HERE, check=True)
    print("[v2] deliverable <- outputs/verify_%s.csv (ROSS-certified only)" % out_tag,
          flush=True)


if __name__ == "__main__":
    main()
