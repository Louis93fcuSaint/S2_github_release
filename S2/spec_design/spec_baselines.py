# -*- coding: utf-8 -*-
"""Step 03 -- baselines for the spec-driven design track.

Each method submits N_SUB admissible candidates for one spec.  The family is
chosen by the method, because in this formulation choosing how many disks and
bearings to use is part of the design act, not part of the request.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_common as S
import spec_eval as E
import spec_opt as O

N_SUB = 200
FAMILIES = [(a, b) for a in S.ND_CHOICES for b in S.NB_CHOICES]
OUT_DIR = os.path.join(E.OUT, "baselines")


def per_family_counts(n_total, keep=None):
    fams = keep or FAMILIES
    base = n_total // len(fams)
    extra = n_total - base * len(fams)
    return {f: base + (1 if i < extra else 0) for i, f in enumerate(fams)}


def predict(x, material, families, sur):
    return sur.predict(S.canonical_to_model_rows(
        x, [material] * len(x), [f[0] for f in families], [f[1] for f in families]))[:, 0]


def score(x, material, families, spec, sur):
    cs1 = predict(x, material, families, sur)
    valid = O.validity(x, material, families)
    return O.objective_from_cs1(cs1, spec, valid), cs1, valid


# --------------------------------------------------------------------------- #
def random_box(ctx, spec, rng):
    box = O.Box(ctx)
    counts = per_family_counts(N_SUB)
    xs, fams = [], []
    for fam, n in counts.items():
        if n == 0:
            continue
        gets = []
        tries = 0
        while sum(len(g) for g in gets) < n and tries < 4:
            cand = box.sample(fam, 400, rng)
            cand = np.array([S.sort_canonical(c, fam[0], fam[1]) for c in cand])
            ok = O.validity(cand, spec["material"], [fam] * len(cand))
            gets.append(cand[ok])
            tries += 1
        pool_cand = np.concatenate(gets) if gets else np.zeros((0, S.DESIGN_DIM))
        if len(pool_cand) < n:
            pool_cand = np.concatenate([pool_cand, box.sample(fam, n - len(pool_cand), rng)])
        xs.append(pool_cand[:n])
        fams += [fam] * n
    x = np.concatenate(xs)
    return x, fams


def retrieval(ctx, spec, rng):
    hit = ((ctx["material"] == spec["material"])
           & (ctx["Y"][:, 0] >= spec["lower"]) & (ctx["Y"][:, 0] <= spec["upper"]))
    idx = np.where(hit)[0]
    take = rng.choice(idx, size=N_SUB, replace=len(idx) < N_SUB)
    return (ctx["designs"][take].copy(),
            [(int(ctx["nd"][i]), int(ctx["nb"][i])) for i in take])


def retrieval_jitter(ctx, spec, rng, sigma=0.05):
    x, fams = retrieval(ctx, spec, rng)
    x = x + rng.normal(0.0, sigma, size=x.shape) * ctx["scales"]
    x = np.array([S.sort_canonical(a, b[0], b[1]) for a, b in zip(x, fams)])
    x, _ = O.repair(x, spec["material"], fams, rng)
    keep = np.where(O.validity(x, spec["material"], fams))[0]
    if len(keep) >= 40:
        cs1 = predict(x, spec["material"], fams, SUR)
        score = O.objective_from_cs1(cs1, spec, None)
        x, fams = x[keep], [fams[i] for i in keep]
        sc = score[keep]
        pick = np.argsort(-sc)[:N_SUB]
        x, fams = x[pick], [fams[i] for i in pick]
    return x, fams


# --------------------------------------------------------------------------- #
def _optimise_adaptive(step_fn, ctx, spec, rng, per_fam, generations):
    box = O.Box(ctx)
    scores, xs, fams = [], [], []
    for fam, n in per_fam.items():
        if n == 0:
            continue
        pop = 64
        init = np.concatenate([box.sample(fam, pop // 2, rng),
                               ctx["designs"][rng.choice(box.index[fam], pop - pop // 2, replace=False)]])
        x = np.array([S.sort_canonical(c, fam[0], fam[1]) for c in init])
        f, cs1, valid = score(x, spec["material"], [fam] * pop, spec, SUR)
        for _ in range(generations):
            trial = step_fn(x, f, box, fam, rng)
            trial = np.array([S.sort_canonical(c, fam[0], fam[1]) for c in trial])
            tf, tcs, tv = score(trial, spec["material"], [fam] * pop, spec, SUR)
            better = tf > f
            x[better], f[better] = trial[better], tf[better]
        order = np.argsort(-f)
        chosen = order[:n]
        scores.append(f[chosen])
        xs.append(x[chosen])
        fams += [fam] * len(chosen)
    return np.concatenate(xs), fams, np.concatenate(scores)


def de_step(x, f, box, fam, rng, F=0.6, CR=0.9):
    n = len(x)
    a = rng.integers(0, n, n)
    b = rng.integers(0, n, n)
    c = rng.integers(0, n, n)
    mutant = x[a] + F * (x[b] - x[c])
    cross = rng.random(x.shape) < CR
    trial = np.where(cross, mutant, x)
    return box.clip(trial, fam)


def ga_step(x, f, box, fam, rng, alpha=0.5, sigma=0.15):
    n = len(x)
    def pick():
        cand = rng.integers(0, n, (n, 3))
        return cand[np.arange(n), np.argmax(f[cand], axis=1)]
    p1, p2 = x[pick()], x[pick()]
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    trial = lo + rng.random(x.shape) * (hi - lo + 1e-12)
    span = box.hi[fam] - box.lo[fam] + 1e-12
    used = box._used(*fam)
    mut = rng.random(x.shape) < 0.15
    trial[:, used] += mut[:, used] * rng.normal(0.0, sigma, (n, len(used))) * span
    return box.clip(trial, fam)


# --------------------------------------------------------------------------- #
def adam_search(ctx, spec, rng, per_fam, steps=100, lr=0.02):
    box = O.Box(ctx)
    smooth = O.SmoothSurrogate(SUR, torch)
    xs, fams, scores = [], [], []
    for fam, n in per_fam.items():
        if n == 0:
            continue
        used = box._used(*fam)
        lo = torch.as_tensor(box.lo[fam], dtype=torch.float64)
        hi = torch.as_tensor(box.hi[fam], dtype=torch.float64)
        span = hi - lo
        pop = 128
        u = torch.rand((pop, len(used)), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([u], lr=lr)
        center = O.log_center(spec)
        for _ in range(steps):
            opt.zero_grad()
            x = torch.zeros((pop, S.DESIGN_DIM), dtype=torch.float64)
            x = x.clone()
            block = lo + u * span
            x[:, used] = block
            cs1 = smooth.cs1(x, spec["material"], fam[0], fam[1])
            band = (torch.log(cs1.clamp(min=1.0)) - center).abs()
            loss = (band + 6.0 * O.soft_constraints(x, spec["material"], fam[0],
                                                    fam[1], torch)).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                u.clamp_(0.0, 1.0)
        with torch.no_grad():
            x = torch.zeros((pop, S.DESIGN_DIM), dtype=torch.float64)
            x[:, used] = lo + u * span
        cand = np.array([S.sort_canonical(c, fam[0], fam[1]) for c in x.numpy()])
        f, cs1, valid = score(cand, spec["material"], [fam] * pop, spec, SUR)
        order = np.argsort(-f)[:n]
        xs.append(cand[order])
        scores.append(f[order])
        fams += [fam] * len(order)
    return np.concatenate(xs), fams, np.concatenate(scores)


# --------------------------------------------------------------------------- #
def bo_search(ctx, spec, rng, per_fam, n_init=30, n_iter=40):
    box = O.Box(ctx)
    xs, fams, scores = [], [], []
    for fam, n in per_fam.items():
        if n == 0:
            continue
        used = box._used(*fam)
        lo, hi = box.lo[fam], box.hi[fam]
        span = hi - lo
        X = rng.random((n_init, len(used)))
        Y = []
        for row in X:
            cand = np.zeros((1, S.DESIGN_DIM))
            cand[0, used] = lo + row * span
            cand[0] = S.sort_canonical(cand[0], fam[0], fam[1])
            f, _, _ = score(cand, spec["material"], [fam], spec, SUR)
            Y.append(f[0])
        Y = np.array(Y)
        for _ in range(n_iter):
            cand_u = rng.random((3000, len(used)))
            pm, pv = _gp_predict(X, Y, cand_u)
            ei = _expected_improvement(pm, np.sqrt(np.maximum(pv, 1e-12)), Y.max())
            pick = int(np.argmax(ei))
            u = cand_u[pick]
            cand = np.zeros((1, S.DESIGN_DIM))
            cand[0, used] = lo + u * span
            cand[0] = S.sort_canonical(cand[0], fam[0], fam[1])
            f, _, _ = score(cand, spec["material"], [fam], spec, SUR)
            X = np.vstack([X, u])
            Y = np.append(Y, f[0])
        order = np.argsort(-Y)[:n]
        out = np.zeros((len(order), S.DESIGN_DIM))
        for i, k in enumerate(order):
            out[i, used] = lo + X[k] * span
            out[i] = S.sort_canonical(out[i], fam[0], fam[1])
        xs.append(out)
        scores.append(Y[order])
        fams += [fam] * len(order)
    return np.concatenate(xs), fams, np.concatenate(scores)


def _kernel(a, b, ls=0.25):
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-0.5 * d2 / ls ** 2)


def _gp(X, Y, noise=1e-3):
    K = _kernel(X, X) + noise * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y - Y.mean()))
    return None, alpha


def _gp_predict(X, Y, Q, noise=1e-3):
    K = _kernel(X, X) + noise * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y - Y.mean()))
    Ks = _kernel(X, Q)
    mu = Ks.T @ alpha + Y.mean()
    v = np.linalg.solve(L, Ks)
    var = 1.0 - (v ** 2).sum(axis=0)
    return mu, np.maximum(var, 1e-9)


def _expected_improvement(mu, sd, best, xi=0.01):
    from math import erf, sqrt
    z = (mu - best - xi) / sd
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))
    return (mu - best - xi) * cdf + sd * phi


# --------------------------------------------------------------------------- #
SUR = None


def _seed_for(name, spec_name, seed):
    import zlib
    return zlib.crc32(("%s|%s|%d" % (name, spec_name, seed)).encode()) % (2 ** 31)


def run_method(name, ctx, spec, rng):
    if name == "random_box":
        return random_box(ctx, spec, rng)
    if name == "retrieval":
        return retrieval(ctx, spec, rng)
    if name == "retrieval_jitter":
        return retrieval_jitter(ctx, spec, rng)
    counts = per_family_counts(N_SUB)
    if name == "de":
        x, fams, _ = _optimise_adaptive(de_step, ctx, spec, rng, counts, 60)
    elif name == "ga":
        x, fams, _ = _optimise_adaptive(ga_step, ctx, spec, rng, counts, 60)
    elif name == "adam":
        x, fams, _ = adam_search(ctx, spec, rng, counts)
    elif name == "bo":
        x, fams, _ = bo_search(ctx, spec, rng, counts)
    else:
        raise ValueError("unknown method: %s" % name)
    return x, fams


def main():
    global SUR
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="random_box,retrieval,retrieval_jitter,de,ga,adam,bo")
    parser.add_argument("--specs", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    ctx = E.pool_ctx()
    SUR = S.load_surrogate("mlp_best", threads=args.threads)
    specs = E.specs(ctx)
    if args.specs:
        wanted = set(args.specs.split(","))
        specs = [sp for sp in specs if sp["name"] in wanted]
    methods = [m for m in args.methods.split(",") if m]
    os.makedirs(OUT_DIR, exist_ok=True)

    print("%-12s %-16s %-8s %-8s %-8s %-8s %-8s %-8s"
          % ("method", "spec", "proxy%", "valid%", "uniq%", "novelty", "fams", "sec"))
    report = {}
    for spec in specs:
        for name in methods:
            rng = np.random.default_rng(_seed_for(name, spec["name"], args.seed))
            started = time.time()
            x, fams = run_method(name, ctx, spec, rng)
            x = np.array([S.sort_canonical(c, f[0], f[1]) for c, f in zip(x, fams)])
            repair_rate = 0.0
            if (~O.validity(x, spec["material"], fams)).any():
                x, repair_rate = O.repair(x, spec["material"], fams, rng)
            frame = O.make_frame(x, spec["material"], fams)
            metrics = E.evaluate_submission(frame, spec, ctx, SUR)
            metrics["seconds"] = time.time() - started
            metrics["repair_rate"] = repair_rate
            report["%s|%s" % (spec["name"], name)] = metrics
            frame.to_csv(os.path.join(OUT_DIR, "%s__%s.csv" % (name, spec["name"])),
                         index=False, encoding="utf-8")
            print("%-12s %-16s %-8.1f %-8.3f %-8.3f %-8.2f %-8d %-8.1f"
                  % (name, spec["name"], metrics["spec_rate_proxy"] * 100,
                     metrics["valid_rate"], metrics["unique_rate"],
                     metrics["novelty_min_median"], metrics["n_families"],
                     metrics["seconds"]))
    path = os.path.join(E.OUT, "baseline_metrics.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                merged = json.load(handle)
        except ValueError:
            merged = {}
        merged.update(report)
        report = merged
    S.save_json(path, report)
    print("[out] %s" % os.path.join(E.OUT, "baseline_metrics.json"))


if __name__ == "__main__":
    main()
