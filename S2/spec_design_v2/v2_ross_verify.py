# -*- coding: utf-8 -*-
"""v2 -- ROSS ground truth for the multi-order shortlist.

The v2 conditioner hands back a shortlist whose PROXY error is measured against
a spec the user wrote at +-1 %.  That band is narrower than the surrogate's own
error on every order (1.28 / 2.19 / 3.22 % MAPE), so no amount of ranking turns
the claim into a fact: the only honest deliverable is the subset ROSS itself
certifies.  This file solves the shortlist and, on request, an unbiased random
control drawn from the same generated batch, then reports

  * per-order true APE (median / mean),
  * joint in-band rate at every tolerance asked for,
  * Spearman rho between the proxy ordering and the true ordering,
  * the gate: the rows that really satisfy every requested order, written out.

Everything else (latent space, decoder, surrogate) is the frozen v1 stack.
"""
import argparse
import concurrent.futures as futures
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(os.path.dirname(HERE), "spec_design")
for path in (V1_DIR, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import spec_common as S          # noqa: E402
import spec_opt as O             # noqa: E402
import spec_interval as SI       # noqa: E402

OUT_DIR = os.path.join(HERE, "outputs")
WORKER = os.path.join(V1_DIR, "_ross_one_canon.py")


def parse_targets(text):
    """'1=5000' / '1=[10000,12000]' / '1=[14400,+]' -> {1: (lo, hi)}."""
    return SI.parse_specs(text)


def band_of(specs, order):
    return specs[int(order)]


def make_runner(timeout, n_base, stats, lock):
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["ROSS_FAST_RELABEL"] = "1"

    def run(job):
        x, material, nd, nb = job
        payload = json.dumps({"x": [float(v) for v in x], "material": material,
                              "nd": int(nd), "nb": int(nb), "n_base": n_base})
        try:
            proc = subprocess.run([sys.executable, WORKER, payload],
                                  capture_output=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            with lock:
                stats["timeout"] += 1
            return {"ok": False, "timeout": True, "error": "timeout>%gs" % timeout}
        text = (proc.stdout or b"").decode("utf-8", "replace").strip().splitlines()
        if not text:
            with lock:
                stats["error"] += 1
            return {"ok": False,
                    "error": (proc.stderr or b"").decode("utf-8", "replace")[-300:]}
        try:
            result = json.loads(text[-1])
        except ValueError:
            with lock:
                stats["error"] += 1
            return {"ok": False, "error": text[-1][-300:]}
        if not result.get("ok"):
            with lock:
                stats["error"] += 1
        return result

    return run


def ranks(x):
    order = np.argsort(x, kind="stable")
    out = np.empty(len(x))
    out[order] = np.arange(len(x))
    return out


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def solve(frame, workers, timeout, n_base, seed):
    lock = threading.Lock()
    stats = {"timeout": 0, "error": 0}
    run = make_runner(timeout, n_base, stats, lock)
    matrix = frame[S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
    jobs = [(matrix[i], str(frame["material"].iloc[i]),
             int(frame["n_disks"].iloc[i]), int(frame["n_bearings"].iloc[i]))
            for i in range(len(frame))]
    t0 = time.time()
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, jobs))
    seconds = time.time() - t0
    truth = np.full((len(frame), 6), np.nan)
    for i, res in enumerate(results):
        if res.get("ok") and res.get("forward"):
            forward = list(res["forward"])[:6]
            truth[i, :len(forward)] = forward
    return truth, seconds, stats


def proxy_values(frame, orders, surrogate=None):
    """Per-order proxy values from the CSV when it carries them, else re-predict."""
    pred = np.full((len(frame), 6), np.nan)
    for order in orders:
        column = "cs%d_pred" % order
        if column in frame:
            pred[:, order - 1] = frame[column].to_numpy(dtype=np.float64)
    if np.isnan(pred[:, [o - 1 for o in orders]]).any():
        if surrogate is None:
            surrogate = S.load_surrogate("mlp_best")
        full = surrogate.predict(S.canonical_to_model_rows(
            frame[S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64),
            [str(m) for m in frame["material"]],
            [int(v) for v in frame["n_disks"]],
            [int(v) for v in frame["n_bearings"]]))
        pred = np.asarray(full, dtype=np.float64)[:, :6]
    return pred


def deviations(values, specs, orders):
    """Signed relative distance from each order's band (0 inside)."""
    dev = np.zeros(values.shape, dtype=np.float64)
    for i, order in enumerate(orders):
        lo, hi = specs[int(order)]
        dev[:, i] = SI.deviation(values[:, i], lo, hi)
    return dev


def metrics(values, specs, orders, tols):
    """Band-aware metrics.

    `median_ape_pct` keeps its old name on purpose: for a point band it is the
    absolute percentage error, and for a real band it is the deviation from the
    band -- nothing at all when the design lands inside, which is the number
    that matters once the user has said "anything in this range is fine".
    """
    dev = deviations(values, specs, orders)
    joint = np.sqrt((dev ** 2).sum(axis=1))
    out = {}
    for tol in tols:
        key = "%.0f" % (100.0 * tol)
        hit = np.ones(len(values), dtype=bool)
        per = {}
        for i, order in enumerate(orders):
            lo, hi = specs[int(order)]
            good = SI.inside(values[:, i], lo, hi, tol)
            hit &= good
            per[str(order)] = float(good.mean())
        out["joint_in_band_tol" + key] = float(hit.mean())
        out["per_order_in_band_tol" + key] = per
    for i, order in enumerate(orders):
        out["order%d" % order] = {
            "median_ape_pct": float(100.0 * np.median(np.abs(dev[:, i]))),
            "mean_ape_pct": float(100.0 * np.mean(np.abs(dev[:, i]))),
            "p90_ape_pct": float(100.0 * np.percentile(np.abs(dev[:, i]), 90)),
            "inside_rate": float(np.mean(dev[:, i] == 0.0))}
    out["joint_err"] = {"median_pct": float(100.0 * np.median(joint)),
                        "mean_pct": float(100.0 * np.mean(joint)),
                        "p90_pct": float(100.0 * np.percentile(joint, 90))}
    return out


def report(tag, frame_sel, label, specs, tols, args, surrogate, verbose=True):
    orders = sorted(specs)
    truth, seconds, stats = solve(frame_sel, args.workers, args.timeout,
                                  args.n_base, args.seed)
    keep = np.isfinite(truth[:, [o - 1 for o in orders]]).all(axis=1)
    pred = proxy_values(frame_sel, orders, surrogate)
    proxy_vals = pred[:, [o - 1 for o in orders]]
    truth_vals = truth[:, [o - 1 for o in orders]]
    dev_proxy = deviations(proxy_vals, specs, orders)
    dev_true = deviations(truth_vals, specs, orders)
    joint_proxy = np.sqrt((dev_proxy ** 2).sum(axis=1))
    joint_true = np.sqrt((dev_true ** 2).sum(axis=1))
    entry = {"label": label, "tag": tag, "orders": orders,
             "bands": {str(o): list(specs[o]) for o in orders},
             "tol": list(tols), "n_shortlist": int(len(frame_sel)),
             "n_solved": int(keep.sum()), "n_timeout": int(stats["timeout"]),
             "n_error": int(stats["error"]), "seconds": float(seconds),
             "proxy": metrics(proxy_vals[keep], specs, orders, tols),
             "truth": metrics(truth_vals[keep], specs, orders, tols)}
    rho = spearman(joint_proxy[keep], joint_true[keep])
    entry["spearman_joint_proxy_vs_true"] = rho
    for i, order in enumerate(orders):
        entry["spearman_order%d" % order] = spearman(
            np.abs(dev_proxy[keep, i]), np.abs(dev_true[keep, i]))
    for tol in tols:
        key = "%.0f" % (100.0 * tol)
        hit = np.ones(int(keep.sum()), dtype=bool)
        for i, order in enumerate(orders):
            lo, hi = specs[int(order)]
            hit &= SI.inside(truth_vals[keep, i], lo, hi, tol)
        entry["gate_tol" + key] = {
            "n_passed": int(hit.sum()),
            "rate_of_solved": float(hit.mean()),
            "seconds_per_passed": float(seconds / max(int(hit.sum()), 1))}
    if verbose:
        print("[verify] %-18s n=%d solved=%d (timeout %d, error %d) %.0fs | rho=%.3f"
              % (label, len(frame_sel), int(keep.sum()), stats["timeout"],
                 stats["error"], seconds, rho), flush=True)
        for i, order in enumerate(orders):
            lo, hi = specs[int(order)]
            print("[verify]   cs%d %s  proxy dev %.2f%% | true dev med %.2f%% "
                  "mean %.2f%% p90 %.2f%% | true inside %.3f"
                  % (order, SI.band_text(lo, hi),
                     100.0 * np.median(np.abs(dev_proxy[keep, i])),
                     100.0 * np.median(np.abs(dev_true[keep, i])),
                     100.0 * np.mean(np.abs(dev_true[keep, i])),
                     100.0 * np.percentile(np.abs(dev_true[keep, i]), 90),
                     float(np.mean(dev_true[keep, i] == 0.0))), flush=True)
        for tol in tols:
            key = "%.0f" % (100.0 * tol)
            g = entry["gate_tol" + key]
            print("[verify]   relax +-%-4s joint: proxy %.3f -> true %.3f | "
                  "gate %d/%d passed (%.1f s per delivered design)"
                  % ("%g%%" % (100.0 * tol), entry["proxy"]["joint_in_band_tol" + key],
                     entry["truth"]["joint_in_band_tol" + key], g["n_passed"],
                     int(keep.sum()), g["seconds_per_passed"]), flush=True)
    # deliverable: the rows ROSS certifies, in shortlist order
    out = frame_sel[keep].copy()
    out["joint_err_truth"] = np.round(joint_true[keep], 5)
    for i, order in enumerate(orders):
        out["cs%d_ross" % order] = np.round(truth_vals[keep, i], 2)
        out["cs%d_relerr_truth_pct" % order] = np.round(100.0 * dev_true[keep, i], 3)
    path = os.path.join(OUT_DIR, "verify_%s.csv" % label)
    out.to_csv(path, index=False, encoding="utf-8")
    entry["csv"] = path
    # the deliverable proper: only what passed the tightest requested band.
    tight = min(tols)
    passed = np.ones(int(keep.sum()), dtype=bool)
    for i, order in enumerate(orders):
        lo, hi = specs[int(order)]
        passed &= SI.inside(truth_vals[keep, i], lo, hi, tight)
    entry["deliverable_tol"] = tight
    entry["deliverable_n"] = int(passed.sum())
    if passed.any():
        dpath = os.path.join(OUT_DIR, "verify_%s__passed.csv" % label)
        out[passed].to_csv(dpath, index=False, encoding="utf-8")
        entry["deliverable"] = dpath
    if verbose:
        print("[verify]   deliverable (relax +-%g%%): %d designs -> %s"
              % (100.0 * tight, entry["deliverable_n"],
                 os.path.basename(entry.get("deliverable", "none"))), flush=True)
    return entry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--targets", required=True, help="1=5000 or 1=[10000,12000] or 1=[14400,+]")
    parser.add_argument("--tols", default="0.01,0.05")
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--select", choices=["top", "rand", "both"], default="both")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--n-base", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="")
    parser.add_argument("--skip-rand", action="store_true")
    args = parser.parse_args()

    import pandas as pd
    tols = [float(v) for v in str(args.tols).split(",") if v.strip()]
    specs = parse_targets(args.targets)
    frame = pd.read_csv(args.csv)
    if "rank" in frame:
        frame = frame.sort_values("rank").reset_index(drop=True)
    name = args.name or os.path.splitext(os.path.basename(args.csv))[0]
    surrogate = S.load_surrogate("mlp_best")
    print("[verify] %s | %d rows | bands %s | relax %s | select %s"
          % (name, len(frame),
             {o: SI.band_text(*specs[o]) for o in sorted(specs)}, tols,
             args.select), flush=True)

    reports = []
    top = frame.head(args.top_n)
    reports.append(report("v2info", top, name + "__top%d" % args.top_n, specs, tols,
                          args, surrogate))
    if args.select in ("rand", "both") and not args.skip_rand:
        control = frame.sample(n=min(args.top_n, len(frame)),
                               random_state=args.seed).reset_index(drop=True)
        reports.append(report("v2info", control, name + "__rand%d" % args.top_n,
                              specs, tols, args, surrogate))

    # what the proxy alone claims over the WHOLE generated batch (no ROSS cost)
    orders = sorted(specs)
    pred_all = proxy_values(frame, orders, surrogate)
    whole = metrics(pred_all[:, [o - 1 for o in orders]], specs, orders, tols)
    path = os.path.join(OUT_DIR, "verify_%s.json" % name)
    S.save_json(path, {"name": name, "csv": args.csv,
                       "bands": {str(o): list(specs[o]) for o in orders},
                       "tols": tols, "n_batch": int(len(frame)),
                       "proxy_whole_batch": whole, "runs": reports})
    print("[verify] wrote %s" % path, flush=True)


if __name__ == "__main__":
    main()