# -*- coding: utf-8 -*-
"""Step 05 -- ROSS ground truth for every shortlist, plus a calibration check
that our ROSS invocation still reproduces the stored dataset labels.

Everything upstream of this file is the surrogate talking about itself.  Here
the top of each submission is re-solved by ROSS, one rotor per process, with a
hard timeout, because the search boxes contain designs on which the eigensolver
crawls.  A timeout is recorded, not retried: a design nobody can analyse is not
deployable.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_common as S
import spec_eval as E
import spec_opt as O

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "_ross_one_canon.py")
SRC_DIRS = [os.path.join(E.OUT, "latent_baselines"), os.path.join(E.OUT, "ddpm"),
            os.path.join(E.OUT, "latent_ddpm")]


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


def rank(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r


def calibrate(ctx, n, args, lock):
    """Reproduce stored cs1 labels with the same ROSS call the track uses."""
    rng = np.random.default_rng(args.seed)
    stats = {"timeout": 0, "error": 0}
    run = make_runner(args.timeout, args.n_base, stats, lock)
    picks = []
    for nd in S.ND_CHOICES:
        for nb in S.NB_CHOICES:
            sel = np.where((ctx["nd"] == nd) & (ctx["nb"] == nb))[0]
            if len(sel):
                picks.append(int(rng.choice(sel)))
    picks = picks[:n]
    jobs = [(ctx["designs"][i], str(ctx["material"][i]), int(ctx["nd"][i]),
             int(ctx["nb"][i])) for i in picks]
    print("calibration: %d real rotors through the same ROSS path" % len(jobs), flush=True)
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, jobs))
    devs = []
    for i, res in zip(picks, results):
        if res.get("ok") and res["forward"]:
            truth = ctx["Y"][i, 0]
            devs.append(abs(res["forward"][0] - truth) / truth * 100.0)
    return {"n": len(jobs), "n_ok": len(devs),
            "median_dev_pct": float(np.median(devs)) if devs else None,
            "max_dev_pct": float(np.max(devs)) if devs else None,
            "n_timeout": stats["timeout"], "n_error": stats["error"]}


def write_gate(frame, spec, pred, truth, keep, seconds, method, spec_name,
               verbose=True):
    """Write the verified-and-in-band subset of a solved shortlist.

    A band the surrogate cannot resolve is not deliverable by ranking: the honest
    contract is to solve the shortlist and hand over only what actually passed.
    `keep` marks the rows ROSS could solve at all.  Cost is the price of the
    guarantee, and it is paid per delivered design, not per claimed one, so
    seconds-per-pass is the number that matters when quoting a tight band.
    """
    in_band = keep & (truth >= spec["lower"]) & (truth <= spec["upper"])
    entry = {
        "gate_verified": int(keep.sum()),
        "gate_passed": int(in_band.sum()),
        "gate_rate": float(in_band.sum() / max(int(keep.sum()), 1)),
        "gate_seconds_per_pass": float(seconds / max(int(in_band.sum()), 1)),
    }
    if in_band.any():
        # one proxy column only: cs1_pred is the number the shortlist was ranked
        # on, cs1_ross is the number that decides delivery.
        out = frame[in_band].drop(columns=["_score", "_pred"], errors="ignore").copy()
        out["cs1_pred"] = np.round(pred[in_band], 2)
        aim = spec.get("aim") or spec.get("target")
        if aim:
            out["rel_err_pct"] = np.round(
                100.0 * (pred[in_band] - float(aim)) / float(aim), 3)
        out["cs1_ross"] = truth[in_band]
        gate_path = os.path.join(
            E.OUT, "latent_ddpm", "%s__%s_verified.csv" % (method, spec_name))
        out.to_csv(gate_path, index=False, encoding="utf-8")
        entry["gate_file"] = os.path.basename(gate_path)
        if verbose:
            print("[gate] %-12s %-14s kept %d/%d verified | %d pass | %.2f s per "
                  "passed design -> %s"
                  % (method, spec_name, int(keep.sum()), len(frame), int(in_band.sum()),
                     entry["gate_seconds_per_pass"], entry["gate_file"]), flush=True)
    return entry


def gate_deliverable(frame, spec, workers=18, timeout=180.0, n_base=15,
                     method="design", spec_name="", verbose=True):
    """Solve every row of an ordered shortlist and return the gate bookkeeping.

    This is the one-command path used by `design_by_target.py --verify`: the
    frame is what the sampling stage wrote, which is already in shortlist order,
    so no surrogate reload is needed when the CSV carries cs1_pred.
    """
    lock = threading.Lock()
    stats = {"timeout": 0, "error": 0}
    run = make_runner(timeout, n_base, stats, lock)
    matrix = frame[S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
    jobs = [(matrix[i], str(frame["material"].iloc[i]), int(frame["n_disks"].iloc[i]),
             int(frame["n_bearings"].iloc[i])) for i in range(len(frame))]
    if "cs1_pred" in frame:
        pred = frame["cs1_pred"].to_numpy(dtype=np.float64)
    else:
        pred = E.predict_cs1(frame, S.load_surrogate("mlp_best", threads=8))
    t0 = time.time()
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, jobs))
    truth = np.array([r["forward"][0] if r.get("ok") and r["forward"] else np.nan
                      for r in results])
    keep = np.isfinite(truth)
    seconds = time.time() - t0
    entry = write_gate(frame, spec, pred, truth, keep, seconds, method,
                       spec_name or spec["name"], verbose=verbose)
    entry.update({"n_solved": int(keep.sum()), "n_timeout": int(stats["timeout"]),
                  "n_error": int(stats["error"]), "seconds": float(seconds)})
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--n-base", type=int, default=15)
    parser.add_argument("--methods", default="")
    parser.add_argument("--specs", default="")
    parser.add_argument("--calibrate", type=int, default=12)
    parser.add_argument("--dump-truth", action="store_true",
                        help="also store the per-design proxy and ROSS values, so "
                             "later analysis needs no new solves")
    parser.add_argument("--gate", action="store_true",
                        help="tight bands: write the verified-and-in-band subset as the "
                             "deliverable instead of a shortlist that still needs trust")
    parser.add_argument("--select", choices=["top", "rand"], default="top",
                        help="verify the surrogate-ranked top of each submission, "
                             "or an unbiased random sample of the whole batch")
    parser.add_argument("--deadline-minutes", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ctx = E.pool_ctx()
    surrogate = S.load_surrogate("mlp_best", threads=args.threads)
    lock = threading.Lock()
    started = time.time()
    deadline = started + args.deadline_minutes * 60.0 if args.deadline_minutes else None

    reports = {}
    if args.calibrate:
        reports["_calibration"] = calibrate(ctx, args.calibrate, args, lock)

    files = []
    for folder in SRC_DIRS:
        if os.path.isdir(folder):
            files += [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                      if f.endswith(".csv") and "__" in f]
    methods = set(args.methods.split(",")) if args.methods else None
    specs = set(args.specs.split(",")) if args.specs else None
    plan = []
    for path in files:
        stem = os.path.basename(path)[:-4]
        method, _, spec_name = stem.partition("__")
        if methods and method not in methods:
            continue
        if specs and spec_name not in specs:
            continue
        plan.append((method, spec_name, path))
    print("shortlists to verify: %d (top-%d each)" % (len(plan), args.top_n), flush=True)

    stats = {"timeout": 0, "error": 0}
    run = make_runner(args.timeout, args.n_base, stats, lock)
    spec_by_name = {sp["name"]: sp for sp in E.specs(ctx, with_adhoc=True)}

    for method, spec_name, path in plan:
        import pandas as pd
        if spec_name not in spec_by_name:
            continue
        spec = spec_by_name[spec_name]
        frame = pd.read_csv(path)
        pred = E.predict_cs1(frame, surrogate)
        # the shortlist is "what the proxy would hand to the engineer": under a
        # band spec that is the highest predicted cs1, under a point spec it is
        # the candidate the proxy thinks sits closest to the target.
        if spec.get("target"):
            score = -np.abs(pred - float(spec["target"]))
        else:
            score = pred
        frame = frame.assign(_pred=pred, _score=score).sort_values(
            "_score", ascending=False)
        if args.select == "rand":
            short = frame.sample(n=min(args.top_n, len(frame)),
                                 random_state=args.seed + len(reports))
        else:
            short = frame.head(args.top_n)
        jobs = [(short[S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)[i],
                 str(short["material"].iloc[i]), int(short["n_disks"].iloc[i]),
                 int(short["n_bearings"].iloc[i]))
                for i in range(len(short))]
        if deadline is not None and time.time() > deadline:
            print("[skip] %s / %s -- deadline" % (method, spec_name), flush=True)
            continue
        t0 = time.time()
        with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(run, jobs))
        truth = np.array([r["forward"][0] if r.get("ok") and r["forward"] else np.nan
                          for r in results])
        pred_short = short["_pred"].to_numpy(dtype=np.float64)
        keep = np.isfinite(truth)
        entry = {
            "method": method, "spec": spec_name, "n_shortlist": int(len(short)),
            "n_ok": int(keep.sum()),
            "n_timeout": int(sum(1 for r in results if r.get("timeout"))),
            "n_error": int(sum(1 for r in results
                               if not r.get("ok") and not r.get("timeout"))),
            "band": [spec["lower"], spec["upper"]],
            "mode": spec.get("mode", "band"),
            "family_band": spec.get("family_band"),
            "spec_rate_true": float(np.mean((truth[keep] >= spec["lower"])
                                            & (truth[keep] <= spec["upper"]))) if keep.any() else 0.0,
            "rate_ge_lower_true": float(np.mean(truth[keep] >= spec["lower"])) if keep.any() else 0.0,
            "seconds": time.time() - t0,
        }
        if spec.get("target") and keep.any():
            target = float(spec["target"])
            rel = np.abs(truth[keep] - target) / target
            entry["median_rel_err_true"] = float(np.median(rel))
            entry["mean_rel_err_true"] = float(rel.mean())
            for tol in (0.01, 0.03, 0.05, 0.10):
                entry["hit_true_%d" % int(round(tol * 100))] = float((rel <= tol).mean())
            relp = np.abs(pred_short[keep] - target) / target
            entry["median_rel_err_proxy_short"] = float(np.median(relp))
        if keep.sum() > 2:
            ape = np.abs(pred_short[keep] - truth[keep]) / truth[keep] * 100.0
            entry["MAPE_cs1"] = float(ape.mean())
            entry["median_APE_cs1"] = float(np.median(ape))
            entry["spearman"] = float(np.corrcoef(rank(pred_short[keep]),
                                                  rank(truth[keep]))[0, 1])
            entry["median_true_cs1"] = float(np.median(truth[keep]))
        if args.gate and keep.any():
            if not keep.all():
                print("[gate] %-12s %-14s %d/%d solved; the %d unsolvable designs "
                      "cannot be delivered" % (method, spec_name, int(keep.sum()),
                                               len(short), int((~keep).sum())),
                      flush=True)
            entry.update(write_gate(short, spec, pred_short, truth, keep,
                                    time.time() - t0, method, spec_name))
        if args.dump_truth and keep.any():
            entry["truth_cs1"] = [round(float(v), 4) for v in truth[keep]]
            entry["pred_cs1"] = [round(float(v), 4) for v in pred_short[keep]]
        reports["%s|%s" % (spec_name, method)] = entry
        print("%-12s %-14s ok %2d/%2d | true in-band %.3f | MAPE %5.2f%% | medAPE %5.2f%% | %.0fs"
              % (method, spec_name, entry["n_ok"], len(short), entry["spec_rate_true"],
                 entry.get("MAPE_cs1", float("nan")), entry.get("median_APE_cs1", float("nan")),
                 entry["seconds"]), flush=True)

    name = "ross_verify.json" if args.select == "top" else "ross_verify_rand.json"
    path = os.path.join(E.OUT, name)
    # merge, so a run over a subset of methods does not wipe the earlier ones
    merged = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                merged = json.load(handle)
        except ValueError:
            merged = {}
    merged.update(reports)
    S.save_json(path, merged)
    print("[out] %s  (%.0f s total)" % (path, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
