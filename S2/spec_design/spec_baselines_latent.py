# -*- coding: utf-8 -*-
"""Step 03b -- the baseline suite, rewritten inside the latent box.

What changed and why
--------------------
The first version of the baselines (`spec_baselines.py`) searched the canonical
33-vector inside a box fitted to the pool, and whenever a candidate broke one of
H1-H6 it was pushed back by a random walk (`spec_opt.repair`).  The generator has
since moved into the 34 independent latent coordinates of `latent_design.py`,
where the decoder enforces H1-H6 *by construction*.  Keeping the old baselines
would have compared two different feasible sets: the search methods would have
paid a feasibility tax the generator does not pay, and the share of their budget
wasted on infeasible candidates would have been an artefact of the parameterisation
rather than a property of the algorithm.

Here every method submits latent vectors, decoded by the same decoder.  What is
left to compare is the algorithm:

  random_latent     uniform in the latent box (the reference for "no method")
  retrieval         hand back real designs that already sit in the band
  retrieval_jitter  retrieval + local perturbation, then proxy re-ranking
  de                differential evolution (DE/rand/1/bin) on the latent vector
  ga                blend crossover + gaussian mutation, tournament selection
  adam              gradient ascent through the differentiable decoder mirror
  bo                Gaussian-process expected improvement over the latent box

Validity is no longer something a method trades against distance: the decoder
guarantees it, and the script re-checks it (repair_rate should be 0.000; the
structured projection is only a safety net).

Everything here is still scored by the frozen protocol of spec_eval.py, and the
ground truth still comes from `05_ross_verify.py --select rand`.
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
import latent_design as LD
import latent_torch as LT

N_SUB = 200
FAMILIES = [(a, b) for a in S.ND_CHOICES for b in S.NB_CHOICES]
OUT_DIR = os.path.join(E.OUT, "latent_baselines")

# Jitter is applied to the latent vector, whose coordinates are box-normalised
# to [0, 1].  0.015 is the same strength the canonical-space jitter had (5 % of a
# pool standard deviation is roughly 1.5 % of a box span for a box-filling
# coordinate), so "retrieval + a little noise" means the same thing as before.
SIGMA_JITTER = 0.015

SUR = None


def per_family_counts(n_total, keep=None):
    fams = keep or FAMILIES
    base = n_total // len(fams)
    extra = n_total - base * len(fams)
    return {f: base + (1 if i < extra else 0) for i, f in enumerate(fams)}


def random_latent(fam, n, rng):
    u = rng.random((n, LD.LAT_DIM))
    u[:, ~LD.used_mask(fam)] = 0.0
    return u


def decode(u, fams):
    return LT.decode_batch(u, fams)


def proxy_cs1(x, material, fams, sur):
    return sur.predict(S.canonical_to_model_rows(
        x, [material] * len(x), [f[0] for f in fams], [f[1] for f in fams]))[:, 0]


def score(u, fams, material, spec, sur):
    """Proxy objective of a latent batch: the frozen band-centre distance."""
    x = decode(u, fams)
    cs1 = proxy_cs1(x, material, fams, sur)
    return O.objective_from_cs1(cs1, spec, None), cs1, x


def _seed_for(name, spec_name, seed):
    import zlib
    return zlib.crc32(("%s|%s|%d" % (name, spec_name, seed)).encode()) % (2 ** 31)


# --------------------------------------------------------------------------- #
# reference methods
# --------------------------------------------------------------------------- #
def random_latent_method(ctx, spec, rng, args):
    counts = per_family_counts(N_SUB)
    us, fams = [], []
    for fam, n in counts.items():
        if n == 0:
            continue
        us.append(random_latent(fam, n, rng))
        fams += [fam] * n
    return np.concatenate(us), fams


def retrieval(ctx, spec, rng, args):
    hit = ((ctx["material"] == spec["material"])
           & (ctx["Y"][:, 0] >= spec["lower"]) & (ctx["Y"][:, 0] <= spec["upper"]))
    idx = np.where(hit)[0]
    take = rng.choice(idx, size=N_SUB, replace=len(idx) < N_SUB)
    designs = ctx["designs"][take].copy()
    fams = [(int(ctx["nd"][i]), int(ctx["nb"][i])) for i in take]
    u = np.stack([LD.encode(designs[i], fams[i]) for i in range(len(take))])
    return u, fams


def retrieval_jitter(ctx, spec, rng, args):
    u, fams = retrieval(ctx, spec, rng, args)
    u = u + rng.normal(0.0, SIGMA_JITTER, size=u.shape)
    for i, fam in enumerate(fams):
        u[i, ~LD.used_mask(fam)] = 0.0
    u = np.clip(u, 0.0, 1.0)
    # the method's own selection step: keep the candidates its proxy likes best
    f, _, _ = score(u, fams, spec["material"], spec, SUR)
    pick = np.argsort(-f)[:N_SUB]
    return u[pick], [fams[i] for i in pick]


# --------------------------------------------------------------------------- #
# population methods
# --------------------------------------------------------------------------- #
def de_step(u, f, rng, F=0.6, CR=0.9):
    n = len(u)
    a = rng.integers(0, n, n)
    b = rng.integers(0, n, n)
    c = rng.integers(0, n, n)
    mutant = u[a] + F * (u[b] - u[c])
    cross = rng.random(u.shape) < CR
    return np.where(cross, mutant, u)


def ga_step(u, f, rng, alpha=0.5, sigma=0.15):
    n = len(u)

    def pick():
        cand = rng.integers(0, n, (n, 3))
        return cand[np.arange(n), np.argmax(f[cand], axis=1)]

    p1, p2 = u[pick()], u[pick()]
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    trial = lo + rng.random(u.shape) * (hi - lo + 1e-12)
    mut = rng.random(u.shape) < 0.15
    trial = trial + mut * rng.normal(0.0, sigma, u.shape)
    return trial


def _evolve(step_fn, ctx, spec, rng, pop, generations, ctor, init_mix=0.5):
    """One population per family, all of them inside the same latent box."""
    counts = per_family_counts(N_SUB)
    us, fams = [], []
    for fam, n in counts.items():
        if n == 0:
            continue
        u = random_latent(fam, pop, rng)
        if init_mix > 0:
            seeds = ctor(fam, int(pop * init_mix), rng)
            if len(seeds):
                u[int(pop * (1.0 - init_mix)):] = seeds
        f, _, _ = score(u, [fam] * pop, spec["material"], spec, SUR)
        for _ in range(generations):
            trial = step_fn(u, f, rng)
            trial = np.clip(trial, 0.0, 1.0)
            trial[:, ~LD.used_mask(fam)] = 0.0
            tf, _, _ = score(trial, [fam] * pop, spec["material"], spec, SUR)
            better = tf > f
            u[better], f[better] = trial[better], tf[better]
        order = np.argsort(-f)[:n]
        us.append(u[order])
        fams += [fam] * len(order)
    return np.concatenate(us), fams


def _real_designs(ctx, fam, n, rng):
    """A few real designs of this family, encoded -- a warm start for the search."""
    if n <= 0:
        return np.zeros((0, LD.LAT_DIM))
    sel = np.where((ctx["nd"] == fam[0]) & (ctx["nb"] == fam[1]))[0]
    if not len(sel):
        return np.zeros((0, LD.LAT_DIM))
    take = rng.choice(sel, size=min(n, len(sel)), replace=False)
    return np.stack([LD.encode(ctx["designs"][i], fam) for i in take])


def de(ctx, spec, rng, args):
    return _evolve(de_step, ctx, spec, rng, args.pop, args.generations,
                   lambda fam, n, r: _real_designs(ctx, fam, n, r))


def ga(ctx, spec, rng, args):
    return _evolve(ga_step, ctx, spec, rng, args.pop, args.generations,
                   lambda fam, n, r: _real_designs(ctx, fam, n, r))


# --------------------------------------------------------------------------- #
def adam_search(ctx, spec, rng, args):
    """Gradient ascent through the differentiable decoder.

    The old version had to add `soft_constraints` to the loss because Adam has no
    notion of a rejected design.  Inside the latent box that term is gone: the
    decoder cannot leave the admissible set, so the only objective left is the
    distance to the band centre.
    """
    smooth = O.SmoothSurrogate(SUR, torch)
    center = O.log_center(spec)
    counts = per_family_counts(N_SUB)
    us, fams = [], []
    for fam, n in counts.items():
        if n == 0:
            continue
        mask = LD.used_mask(fam)
        start = random_latent(fam, args.pop, rng)
        seeds = _real_designs(ctx, fam, args.pop // 2, rng)
        if len(seeds):
            start[:len(seeds)] = seeds
        u = torch.tensor(start, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([u], lr=0.02)
        for _ in range(args.adam_steps):
            opt.zero_grad()
            x = LT.decode_torch(torch, u, fam)
            cs1 = smooth.cs1(x, spec["material"], fam[0], fam[1])
            band = (torch.log(torch.clamp(cs1, min=1.0)) - center).abs().mean()
            band.backward()
            opt.step()
            with torch.no_grad():
                u.clamp_(0.0, 1.0)
                u[:, ~torch.as_tensor(mask)] = 0.0
        with torch.no_grad():
            cand = u.detach().numpy()
        f, _, _ = score(cand, [fam] * args.pop, spec["material"], spec, SUR)
        order = np.argsort(-f)[:n]
        us.append(cand[order])
        fams += [fam] * len(order)
    return np.concatenate(us), fams


def _kernel(a, b, ls=0.25):
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-0.5 * d2 / ls ** 2)


def _gp_predict(X, Y, Q, noise=1e-3):
    K = _kernel(X, X) + noise * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y - Y.mean()))
    Ks = _kernel(X, Q)
    v = np.linalg.solve(L, Ks)
    var = 1.0 - (v ** 2).sum(axis=0)
    return Ks.T @ alpha + Y.mean(), np.maximum(var, 1e-9)


def _expected_improvement(mu, sd, best, xi=0.01):
    from math import erf, sqrt
    z = (mu - best - xi) / sd
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))
    return (mu - best - xi) * cdf + sd * phi


def bo_search(ctx, spec, rng, args):
    counts = per_family_counts(N_SUB)
    us, fams = [], []
    for fam, n in counts.items():
        if n == 0:
            continue
        mask = LD.used_mask(fam)
        used = np.where(mask)[0]
        n_init, n_iter = args.bo_init, args.bo_iter
        pool_u = rng.random((n_init, LD.LAT_DIM))
        pool_u[:, ~mask] = 0.0
        f, _, _ = score(pool_u, [fam] * n_init, spec["material"], spec, SUR)
        for _ in range(n_iter):
            cand = rng.random((3000, LD.LAT_DIM))
            cand[:, ~mask] = 0.0
            pm, pv = _gp_predict(pool_u[:, used], f, cand[:, used])
            ei = _expected_improvement(pm, np.sqrt(np.maximum(pv, 1e-12)), f.max())
            u_new = cand[int(np.argmax(ei))]
            f_new, _, _ = score(u_new[None, :], [fam], spec["material"], spec, SUR)
            pool_u = np.vstack([pool_u, u_new])
            f = np.append(f, f_new[0])
        order = np.argsort(-f)[:n]
        us.append(pool_u[order])
        fams += [fam] * len(order)
    return np.concatenate(us), fams


METHODS = {"random_latent": random_latent_method, "retrieval": retrieval,
           "retrieval_jitter": retrieval_jitter, "de": de, "ga": ga,
           "adam": adam_search, "bo": bo_search}


# --------------------------------------------------------------------------- #
def main():
    global SUR
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--specs", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--pop", type=int, default=64)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--adam-steps", type=int, default=200)
    parser.add_argument("--bo-init", type=int, default=30)
    parser.add_argument("--bo-iter", type=int, default=40)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    ctx = E.pool_ctx()
    SUR = S.load_surrogate("mlp_best", threads=args.threads)
    specs = E.specs(ctx)
    if args.specs:
        wanted = set(args.specs.split(","))
        specs = [sp for sp in specs if sp["name"] in wanted]
    methods = [m for m in args.methods.split(",") if m]
    os.makedirs(OUT_DIR, exist_ok=True)
    LT.self_test(n=1800, verbose=True)

    print("%-18s %-16s %-8s %-8s %-8s %-8s %-6s %-7s"
          % ("method", "spec", "proxy%", "valid%", "uniq%", "novelty", "fams", "sec"),
          flush=True)
    report = {}
    for spec in specs:
        for name in methods:
            rng = np.random.default_rng(_seed_for(name, spec["name"], args.seed))
            started = time.time()
            u, fams = METHODS[name](ctx, spec, rng, args)
            x = decode(u, fams)
            x = np.array([S.sort_canonical(c, f[0], f[1]) for c, f in zip(x, fams)])
            bad = ~O.validity(x, spec["material"], fams)
            repair_rate = float(bad.mean())
            if bad.any():
                idx = np.where(bad)[0]
                fixed, _ = O.repair_structured(x[idx], spec["material"],
                                               [fams[i] for i in idx], rng=rng)
                x[idx] = fixed
            frame = O.make_frame(x, spec["material"], fams)
            metrics = E.evaluate_submission(frame, spec, ctx, SUR)
            metrics["seconds"] = time.time() - started
            metrics["repair_rate"] = repair_rate
            metrics["n_submitted"] = int(len(frame))
            metrics["space"] = "latent"
            metrics["method"] = name
            report["%s|%s" % (spec["name"], name)] = metrics
            frame.to_csv(os.path.join(OUT_DIR, "%s__%s.csv" % (name, spec["name"])),
                         index=False, encoding="utf-8")
            print("%-18s %-16s %-8.1f %-8.3f %-8.3f %-8.2f %-6d %-7.1f"
                  % (name, spec["name"], metrics["spec_rate_proxy"] * 100,
                     metrics["valid_rate"], metrics["unique_rate"],
                     metrics["novelty_min_median"], metrics["n_families"],
                     metrics["seconds"]), flush=True)
    path = os.path.join(OUT_DIR, "baseline_metrics.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                merged = json.load(handle)
        except ValueError:
            merged = {}
        merged.update(report)
        report = merged
    S.save_json(path, report)
    print("[out] %s" % path, flush=True)


if __name__ == "__main__":
    main()