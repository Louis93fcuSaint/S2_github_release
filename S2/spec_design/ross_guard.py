# -*- coding: utf-8 -*-
"""Is a near-zero first forward mode a real mode, or a solver artefact?

Both the dataset and the generated batches contain rotors whose first forward
critical speed comes back at 0.03, 0.6 or 21 rpm while the rest of the spectrum
is reproduced to five digits.  Solved twice with different modal bases the split
is clean:

    genuine   dataset rows at 64.03, 79.10, 109.87 rpm -> repeat to 0.05 %
    spurious  stage-B rotors at 0.822, 2.534, 21.569   -> 0.822 disappears,
                                                          2.534 becomes 0.405,
                                                          21.569 becomes 0.142

So the test is reproducibility, not a threshold, and it only has to be paid for
on candidates that come back below FLOOR_RPM (3 rotors out of 1426 verified, 172
of 199811 dataset rows).  Everything else is trusted from a single solve.
"""
import concurrent.futures as futures
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "_ross_one_canon.py")

# The pool's 0.1 percentile is 417.9 rpm, so nothing below this is dismissed on
# sight; it is only sent for the second solve.
FLOOR_RPM = 400.0
AGREE = 0.20            # |log(v_a / v_b)| tolerance
N_BASE_A, N_BASE_B = 15, 30
TIMEOUT = 300.0


def _solve(x, material, nd, nb, n_base):
    payload = json.dumps({"x": [float(v) for v in x], "material": material,
                          "nd": int(nd), "nb": int(nb), "n_base": int(n_base)})
    env = dict(os.environ)
    env["ROSS_FAST_RELABEL"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = "1"
    try:
        proc = subprocess.run([sys.executable, WORKER, payload], capture_output=True,
                              timeout=TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout>%gs" % TIMEOUT}
    text = (proc.stdout or b"").decode("utf-8", "replace").strip().splitlines()
    if not text:
        return {"ok": False, "error": (proc.stderr or b"").decode("utf-8", "replace")[-200:]}
    return json.loads(text[-1])


def needs_probe(first_rpm):
    return first_rpm is not None and np.isfinite(first_rpm) and first_rpm < FLOOR_RPM


def probe(x, material, nd, nb, first_rpm=None):
    """Re-solve a suspicious rotor and decide whether its first mode is real.

    Returns truth = the geometric mean of the two solves when they agree, the
    first mode above FLOOR_RPM when they do not, and None when neither solve
    produced anything usable.
    """
    a = _solve(x, material, nd, nb, N_BASE_A)
    b = _solve(x, material, nd, nb, N_BASE_B)
    fa = [float(v) for v in a.get("forward") or []]
    fb = [float(v) for v in b.get("forward") or []]
    va = fa[0] if fa else None
    vb = fb[0] if fb else None
    out = {"first_a": va, "first_b": vb, "first_stored": first_rpm,
           "modes_a": fa[:6], "modes_b": fb[:6], "trusted": None, "truth": None}
    if va is not None and vb is not None and va > 0 and vb > 0:
        agree = abs(np.log(va / vb)) <= np.log(1.0 + AGREE)
        out["trusted"] = bool(agree)
        out["truth"] = float(np.sqrt(va * vb)) if agree else _first_above(fb or fa)
    elif va is not None and vb is None:
        out["trusted"] = False
        out["truth"] = _first_above(fa)
    elif vb is not None and va is None:
        out["trusted"] = False
        out["truth"] = _first_above(fb)
    return out


def _first_above(modes):
    above = [v for v in modes if v is not None and np.isfinite(v) and v >= FLOOR_RPM]
    return float(above[0]) if above else None


def probe_many(jobs, workers=12):
    """jobs: list of (x, material, nd, nb, first_rpm).  Returns the probe dicts."""
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda job: probe(*job), jobs))