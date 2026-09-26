# -*- coding: utf-8 -*-
"""Stage B -- does the stage-A error bar transfer to generated submissions?

Stage A calibrated the 5-seed spread and the feature-space novelty against the
held-out truth, where the proxy turns out honest.  The optimism lives on the
submissions: they sit outside the data manifold, so this script re-derives the
signals there and pays for real ROSS truth on a random 40 of each batch.

Every verified rotor is stored with its proxy prediction, seed spread, novelty,
box excursion, the band verdict with and without the uncertainty margin, and
its true cs1.  The screening rule can then be scored on evidence instead of on
the proxy's opinion of itself.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec_common as S
import spec_eval as E
import uncertainty_signals as U

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(HERE, "_ross_one_canon.py")
STORE = os.path.join(E.OUT, "uncertainty_stageB.json")
SRC_DIRS = [os.path.join(E.OUT, "latent_baselines"), os.path.join(E.OUT, "latent_ddpm"),
            os.path.join(E.OUT, "ddpm")]


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
            return {"ok": False, "raw": (proc.stderr or b"").decode("utf-8", "replace")[-200:]}
        try:
            result = json.loads(text[-1])
        except ValueError:
            with lock:
                stats["error"] += 1
            return {"ok": False, "raw": text[-1][-200:]}
        if not result.get("ok"):
            with lock:
                stats["error"] += 1
        return result

    return run


def load_store():
    if os.path.exists(STORE):
        with open(STORE, encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_store(store):
    S.save_json(STORE, store)


def files_for(methods, specs):
    out = []
    for folder in SRC_DIRS:
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not (name.endswith(".csv") and "__" in name):
                continue
            method, _, spec_name = name[:-4].partition("__")
            if methods and method not in methods:
                continue
            if specs and spec_name not in specs:
                continue
            out.append((method, spec_name, os.path.join(folder, name)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="")
    parser.add_argument("--specs", default="")
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--n-base", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry", action="store_true", help="compute signals only, no ROSS")
    args = parser.parse_args()

    import pandas as pd

    ctx = E.pool_ctx()
    surrogate = S.load_surrogate("mlp_best", threads=8)
    bank = U.SignalBank(ctx, surrogate)
    with open(U.CALIB_JSON, encoding="utf-8") as handle:
        calib = json.load(handle)
    spec_by_name = {sp["name"]: sp for sp in E.specs(ctx)}
    methods = set(args.methods.split(",")) if args.methods else None
    specs = set(args.specs.split(",")) if args.specs else None
    plan = files_for(methods, specs)
    print("submissions: %d | q90(|log err|/spread) on the test split = %.2f"
          % (len(plan), calib["q90_err_over_spread"]), flush=True)

    store = load_store()
    stats = {"timeout": 0, "error": 0}
    lock = threading.Lock()
    run = make_runner(args.timeout, args.n_base, stats, lock)
    started = time.time()

    for index, (method, spec_name, path) in enumerate(plan):
        key = "%s|%s" % (method, spec_name)
        if key in store and not args.dry:
            print("[skip] %s (already stored, %d rotors)"
                  % (key, len(store[key]["rotors"])), flush=True)
            continue
        spec = spec_by_name.get(spec_name)
        if spec is None:
            continue
        frame = pd.read_csv(path).reset_index(drop=True)
        x = frame[S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
        rows = S.canonical_to_model_rows(x, frame["material"].tolist(),
                                         frame["n_disks"].tolist(),
                                         frame["n_bearings"].tolist())
        sig = bank.score(rows, x)
        pred = np.exp(sig["log_mean"])
        rng = np.random.default_rng(args.seed + index)
        pick = rng.choice(len(frame), min(args.top_n, len(frame)), replace=False)
        pick = np.sort(pick)
        lower, upper = float(spec["lower"]), float(spec["upper"])
        margin = calib["q90_err_over_spread"]
        lo = np.exp(sig["log_mean"] - margin * sig["log_spread"])
        entry = {
            "method": method, "spec": spec_name, "n_batch": int(len(frame)),
            "band": [lower, upper],
            "proxy_yes_all": float((pred >= lower).mean()),
            "proxy_yes_margin_all": float((lo >= lower).mean()),
            "mean_spread": float(sig["log_spread"].mean()),
            "mean_dF": float(sig["dF"].mean()),
            "mean_dD": float(sig["dD"].mean()),
            "rotors": [],
        }
        if args.dry:
            print("%-16s %-14s batch %4d | proxy in-band all %.3f | margin %.3f | spread %.4f | dF %.2f"
                  % (method, spec_name, len(frame), entry["proxy_yes_all"],
                     entry["proxy_yes_margin_all"], entry["mean_spread"], entry["mean_dF"]),
                  flush=True)
            store[key] = entry
            continue

        jobs = [(x[i], str(frame["material"].iloc[i]), int(frame["n_disks"].iloc[i]),
                 int(frame["n_bearings"].iloc[i])) for i in pick]
        t0 = time.time()
        with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(run, jobs))
        for i, res in zip(pick, results):
            truth = res["forward"][0] if res.get("ok") and res.get("forward") else None
            entry["rotors"].append({
                "index": int(i), "material": str(frame["material"].iloc[i]),
                "nd": int(frame["n_disks"].iloc[i]), "nb": int(frame["n_bearings"].iloc[i]),
                "proxy": float(pred[i]), "proxy_lo": float(lo[i]),
                "spread": float(sig["log_spread"][i]), "dF": float(sig["dF"][i]),
                "dD": float(sig["dD"][i]), "box": float(sig["box"][i]),
                "truth": None if truth is None else float(truth),
            })
        ok = [r for r in entry["rotors"] if r["truth"]]
        if ok:
            ape = np.array([abs(r["proxy"] - r["truth"]) / r["truth"] for r in ok])
            entry["n_ok"] = len(ok)
            entry["proxy_yes_short"] = float(np.mean([r["proxy"] >= lower for r in ok]))
            entry["proxy_yes_margin_short"] = float(np.mean([r["proxy_lo"] >= lower for r in ok]))
            entry["truth_in_band"] = float(np.mean([(r["truth"] >= lower) and (r["truth"] <= upper)
                                                    for r in ok]))
            entry["truth_ge_lower"] = float(np.mean([r["truth"] >= lower for r in ok]))
            entry["MAPE"] = float(ape.mean() * 100)
            entry["median_APE"] = float(np.median(ape) * 100)
            print("%-16s %-14s ok %2d/%2d | proxy %.3f (margin %.3f) | truth %.3f | MAPE %5.2f%% | medAPE %5.2f%% | %.0fs"
                  % (method, spec_name, len(ok), len(pick), entry["proxy_yes_short"],
                     entry["proxy_yes_margin_short"], entry["truth_in_band"],
                     entry["MAPE"], entry["median_APE"], time.time() - t0), flush=True)
        store[key] = entry
        save_store(store)

    print("[out] %s  (%.0f s, ROSS timeouts %d errors %d)"
          % (STORE, time.time() - started, stats["timeout"], stats["error"]))


if __name__ == "__main__":
    main()