# -*- coding: utf-8 -*-
"""Run the point-layout v2m checkpoint on the SAME joint INTERVAL specs used for v2i.

v2m receives the geometric centre of each interval (point layout), which is the only
way this point-conditioned checkpoint can be applied to an interval specification.
Everything else (5000 candidates, top-200 selector, seed 42, guide lambda 0.05,
ROSS 1.6.1) matches the v2i joint runs.
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
LOGS = os.path.join(OUT, "logs")
PY = sys.executable
ARMS = [
    ("joint_cs12_v2mcentre", "1=[3000,4500],2=[6000,9000]", "0,0.05"),
    ("joint_cs123_v2mcentre", "1=[3000,4500],2=[6000,9000],3=[14000,22000]", "0,0.05"),
]

def run(cmd, log):
    print("[run]", " ".join(cmd), flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE).returncode
    print("[run] exit %d in %.0fs" % (rc, time.time() - t0), flush=True)
    if rc != 0:
        with open(log, encoding="utf-8") as fh:
            raise SystemExit(fh.read()[-2000:])

for name, targets, tols in ARMS:
    csv = os.path.join(OUT, "v2_%s.csv" % name)
    run([PY, os.path.join(HERE, "latent_ddpm_v2.py"), "sample",
         "--tag", "v2m", "--out-tag", "v2_%s" % name,
         "--targets", targets, "--material", "Steel", "--tol", "0",
         "--use-ema", "1", "--guidance", "1.0", "--guide-lambda", "0.05",
         "--sample-steps", "25", "--n-generate", "5000", "--n-submit", "5000",
         "--threads", "8", "--seed", "42"],
        os.path.join(LOGS, "v2_sample_%s.txt" % name))
    run([PY, os.path.join(HERE, "v2_ross_verify.py"), "--csv", csv,
         "--targets", targets, "--tols", tols, "--top-n", "200",
         "--workers", "8", "--name", name, "--select", "top", "--seed", "42"],
        os.path.join(LOGS, "v2_verify_%s.txt" % name))
    print("[done]", name, flush=True)
