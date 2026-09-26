# -*- coding: utf-8 -*-
"""Step 01 -- family-conditional proxy audit for the spec-driven design track.

Answers one question: can the frozen six-order MLP, read out as cs1, *find*
designs whose true first critical speed sits above the spec band, across every
(nd, nb) family and every material?  Everything downstream depends on this.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_common as S
import spec_eval as E

OUT = os.path.join(S.HERE, "outputs")
CACHE = os.path.join(OUT, "cache_features.npz")
KS = [20, 50, 200, 1000]


def features(pool):
    """Engineered (n, 47) matrix and the six-order truth, cached on disk."""
    if os.path.exists(CACHE):
        blob = np.load(CACHE, allow_pickle=True)
        if int(blob["n"]) == len(pool):
            print("features: cache hit (%d rows)" % len(pool))
            return blob["X"], blob["Y"]
    names = list(S.constraint_module().to_features_v4(
        [S.sample_from_canonical(np.zeros(S.DESIGN_DIM), "Steel", 3, 3)])[1])
    started = time.time()
    X, _ = S.mlp_module().build_engineered_features(
        pool[names].to_numpy(dtype=np.float64), names)
    X = np.asarray(X, dtype=np.float64)
    Y = pool[S.ALL_CS].to_numpy(dtype=np.float64)
    np.savez_compressed(CACHE, X=X, Y=Y, n=np.array(len(pool)))
    print("features: built %d x %d in %.1f s"
          % (X.shape[0], X.shape[1], time.time() - started))
    return X, Y


def rank(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    return r


def main():
    started = time.time()
    pool = S.load_pool()
    X, Y = features(pool)
    lower = S.DEFAULT_LOWER_RPM

    surrogate = S.load_surrogate("mlp_best", threads=8)
    print("surrogate: %s\n" % surrogate.source)
    pred = np.asarray(surrogate.predict(X), dtype=np.float64)
    truth = Y[:, 0].copy()
    mat = pool["material"].to_numpy()
    nd_arr = pool["n_disks"].to_numpy()
    nb_arr = pool["n_bearings"].to_numpy()
    specs_list = E.specs(pool)
    point = any(sp.get("target") for sp in specs_list)

    report = {"n_rows": int(len(pool)), "lower_rpm": lower, "families": {},
              "specs": {}, "surrogate": surrogate.source,
              "protocol": "point_target" if point else "band"}

    print("%-8s %-7s %-9s %-9s %-9s %-9s" %
          ("family", "n", "MAPE1%", "medAPE1%", "R2log", "rho"))
    for nd in S.ND_CHOICES:
        for nb in S.NB_CHOICES:
            mask = (pool["n_disks"].to_numpy() == nd) & (pool["n_bearings"].to_numpy() == nb)
            if mask.sum() < 50:
                continue
            p, t = pred[mask, 0], truth[mask]
            good = np.isfinite(p) & np.isfinite(t) & (t > 0)
            p, t = p[good], t[good]
            ape = np.abs(p - t) / t * 100.0
            rho = float(np.corrcoef(rank(p), rank(t))[0, 1])
            log_t = np.log(t)
            r2 = float(1.0 - np.sum((np.log(np.maximum(p, 1e-9)) - log_t) ** 2)
                       / max(np.sum((log_t - log_t.mean()) ** 2), 1e-12))
            if point:
                gidx = np.asarray(mask).nonzero()[0][good]
                hits = 0
                for sp in specs_list:
                    sel = mat[gidx] == sp["material"]
                    hits += int((np.abs(truth[gidx][sel] - sp["target"])
                                 <= sp["tol"] * sp["target"]).sum())
                in_band = hits
            else:
                in_band = int((t >= lower).sum())
            report["families"]["%d_%d" % (nd, nb)] = {
                "n": int(good.sum()), "MAPE1_pct": float(ape.mean()),
                "median_APE1_pct": float(np.median(ape)), "R2_log": r2,
                "spearman": rho, "in_band_true": in_band,
            }
            print("%-8s %-7d %-9.2f %-9.2f %-9.4f %-9.3f"
                  % ("(%d,%d)" % (nd, nb), int(good.sum()), ape.mean(),
                     np.median(ape), r2, rho))

    print("\nranking each slice by proxy distance to the target -- can the proxy"
          " put the true hits on top?")
    print("%-13s %-7s %-8s %-9s %-9s %-9s %-9s" %
          ("spec", "family", "base%", "P@20", "P@50", "P@200", "P@1000"))
    if point:
        cases = [(sp["name"], sp["material"], float(sp["target"]), float(sp["tol"]))
                 for sp in specs_list]
    else:
        cases = [(m, m, None, None) for m in S.MATERIAL_ORDER]
    for case_name, material, target, tol in cases:
        for nd in S.ND_CHOICES:
            for nb in S.NB_CHOICES:
                mask = (mat == material) & (nd_arr == nd) & (nb_arr == nb)
                if mask.sum() < 200:
                    continue
                p, t = pred[mask, 0], truth[mask]
                if target:
                    inside = np.abs(t - target) <= tol * target
                    score = -np.abs(p - target)
                else:
                    inside = t >= lower
                    score = p
                base = float(inside.mean() * 100)
                if base == 0.0:
                    continue
                order = np.argsort(-score)
                entry = {"n": int(mask.sum()), "base_rate_pct": base,
                         "in_band": int(inside.sum())}
                cells = []
                for k in KS:
                    if inside.size < k:
                        entry["P@%d" % k] = float(inside[order].mean() * 100)
                    else:
                        entry["P@%d" % k] = float(inside[order[:k]].mean() * 100)
                    cells.append("%-9.1f" % entry["P@%d" % k])
                report["specs"]["%s_%d_%d" % (case_name, nd, nb)] = entry
                print("%-13s %-7s %-8.2f %s"
                      % (case_name, "(%d,%d)" % (nd, nb), base, " ".join(cells)))

    print("\ntotal %.0f s" % (time.time() - started))
    S.save_json(os.path.join(OUT, "proxy_family_audit.json"), report)
    print("[out] %s" % os.path.join(OUT, "proxy_family_audit.json"))


if __name__ == "__main__":
    main()
