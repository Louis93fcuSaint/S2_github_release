# -*- coding: utf-8 -*-
"""Differentiable mirror of latent_design.decode, for the baseline rewrite.

Why this module exists
----------------------
The generator no longer emits a design: it emits the 34 independent latent
coordinates of latent_design.py, and the decoder maps them onto the admissible
set by construction.  A baseline that still searched the canonical 33-vector and
then repaired whatever broke H1-H6 was working in a *different* set and paid a
different price for feasibility, so the comparison was not clean.

So every baseline is rewritten to submit latent vectors, decoded by
latent_design.decode.  Differential evolution, the genetic algorithm, BO and the
random baseline only need the decoder as a black box and call the numpy version
directly.  Adam does need a gradient of cs1 with respect to the latent vector,
and the numpy decoder breaks the autograd graph, so this module re-implements the
same arithmetic in torch, batched over the candidates of one family.

Two guards keep the torch copy honest:
  * `self_test()` measures the mismatch against the numpy decoder and the
    validity rate of both on random latent vectors;
  * the submission is *always* rebuilt by the numpy decoder afterwards, so this
    file only has to be accurate enough to point Adam in a useful direction.

`latent_design._snap` is deliberately not reproduced here: the free-interval
construction already keeps every bearing at least 10 mm clear of every disk, and
the guard counter in the round-trip log recorded 0 fires over 207 101 decodes.
"""
import numpy as np

import latent_design as LD
import spec_common as S

H1_FACTOR = LD.H1_FACTOR
H4_SEP = LD.H4_SEP
GAP_EXTRA = LD.GAP_EXTRA
GAP_NUDGE = LD.GAP_NUDGE
W_FLOOR = LD._W_FLOOR

L_LO, L_HI = LD.BOX_L
OD_LO, OD_HI = LD.BOX_OD
LOG_ID_LO, LOG_ID_HI = LD.BOX_LOG_ID
REL_LO, REL_HI = LD.BOX_LOG1P_REL
W_LO, W_HI = LD.BOX_W
LOGK_LO, LOGK_HI = LD.BOX_LOG_K
E_LO, E_HI = LD.BOX_E
LOGC_LO, LOGC_HI = LD.BOX_LOG_C

# The free-length inversion is a fixed point in a piecewise-constant coverage
# function; six sweeps always reach it for a <= 6 disks, and the numpy decoder
# has the last word anyway.
FREE_ITER = 6

CS_LO = 1.0


def _denorm(torch, u, lo, hi):
    return lo + torch.clamp(u, 0.0, 1.0) * (hi - lo)


def decode_torch(torch, u, family):
    """(n, 34) latent -> (n, 33) canonical design, differentiable."""
    a, b = int(family[0]), int(family[1])
    if not torch.is_tensor(u):
        u = torch.as_tensor(np.asarray(u, dtype=np.float64), dtype=torch.float64)
    u = u.to(torch.float64)
    n = int(u.shape[0])
    ones = torch.ones(n, dtype=torch.float64)
    zeros = torch.zeros(n, dtype=torch.float64)

    L = _denorm(torch, u[:, LD.OFF_L], L_LO, L_HI)
    OD = _denorm(torch, u[:, LD.OFF_OD], OD_LO, OD_HI)
    OD = torch.minimum(torch.maximum(OD, torch.maximum(OD_LO * ones, L / 50.0 * (1.0 + 1e-7))),
                       torch.minimum(OD_HI * ones, L / 2.0 * (1.0 - 1e-7)))
    ID = torch.exp(_denorm(torch, u[:, LD.OFF_ID], LOG_ID_LO, LOG_ID_HI))
    ID = torch.minimum(ID, 0.8 * OD)

    if a > 0:
        W = torch.stack([_denorm(torch, u[:, LD.OFF_W + j], W_LO, W_HI)
                         for j in range(a)], dim=1)
        REL = torch.stack([torch.expm1(_denorm(torch, u[:, LD.OFF_REL + j], REL_LO, REL_HI))
                           for j in range(a)], dim=1)
        CUT = torch.stack([torch.clamp(u[:, LD.OFF_CUT + j], 0.0, 1.0)
                           for j in range(a)], dim=1)
    margin = LD.H3_FRAC * L

    if a > 1:
        budget = 0.8 * L - (a + 1) * GAP_NUDGE

        def need_of(block):
            return (0.5 * (block[:, :-1] + block[:, 1:]) + GAP_EXTRA).sum(dim=1)

        need = need_of(W)
        ratio = torch.where(budget > 0.0, budget / torch.clamp(need, min=1e-12), zeros)
        W = torch.where((need > budget)[:, None],
                        torch.clamp(W * ratio[:, None], min=W_FLOOR), W)
        need = need_of(W)
        W = torch.where((need > budget)[:, None], torch.full_like(W, W_FLOOR), W)
        need = need_of(W)
        pairs = 0.5 * (W[:, :-1] + W[:, 1:]) + GAP_EXTRA
    else:
        need = zeros
        pairs = None

    slack = torch.clamp(0.8 * L - need, min=0.0)
    spread = torch.clamp(slack - (a + 1) * GAP_NUDGE, min=0.0)
    CUT, _ = torch.sort(CUT, dim=1)
    edges = torch.cat([torch.zeros(n, 1, dtype=torch.float64), CUT,
                       torch.ones(n, 1, dtype=torch.float64)], dim=1)
    last = torch.maximum(edges[:, a], edges[:, a - 1])
    edges = torch.cat([edges[:, :a], last[:, None], edges[:, a + 1:]], dim=1)
    gaps = torch.clamp(edges[:, 1:] - edges[:, :-1], min=0.0) * spread[:, None] + GAP_NUDGE

    POS = [margin + gaps[:, 0]]
    for j in range(1, a):
        POS.append(POS[-1] + pairs[:, j - 1] + gaps[:, j])
    P = torch.stack([torch.clamp(p, min=margin, max=L - margin) for p in POS], dim=1)

    lo_edge = margin + 1e-9 * L
    hi_edge = L - margin - 1e-9 * L

    # Free space on the shaft = [lo_edge, hi_edge] minus the union of the disk
    # keep-out windows (p_j +- 10 mm).  The union is built from the 2a+2 sorted
    # breakpoints and each elementary segment is kept or dropped by testing its
    # midpoint, so overlapping windows are counted once -- summing per-disk
    # clamps instead would double count them wherever two disks sit within 20 mm.
    breaks = [P - H4_SEP, P + H4_SEP,
              lo_edge[:, None], hi_edge[:, None]]
    bnd, _ = torch.sort(torch.cat(breaks, dim=1), dim=1)
    left = torch.maximum(bnd[:, :-1], lo_edge[:, None])
    right = torch.minimum(bnd[:, 1:], hi_edge[:, None])
    seg = torch.clamp(right - left, min=0.0)
    mid = 0.5 * (left + right)
    blocked = torch.zeros_like(mid)
    for j in range(a):
        blocked = torch.maximum(blocked, (torch.abs(mid - P[:, j][:, None])
                                          < H4_SEP).to(torch.float64))
    free = 1.0 - blocked

    # Cumulative free length at the start / end of every elementary segment.
    # `sel` is the half-open indicator of the segment that carries length y:
    # 1{below <= y} - 1{above <= y}, so exactly one segment is ever selected
    # (ties go to the segment that starts at the boundary), and the inverse is
    # a closed form instead of a slow fixed-point sweep.
    below = torch.cat([torch.zeros(n, 1, dtype=torch.float64),
                       torch.cumsum(seg * free, dim=1)[:, :-1]], dim=1)
    above = below + seg * free
    total = (seg * free).sum(dim=1)

    base = _denorm(torch, u[:, LD.OFF_LOGK], LOGK_LO, LOGK_HI)
    if b > 0:
        E = torch.stack([_denorm(torch, u[:, LD.OFF_E + j], E_LO, E_HI) for j in range(b)], dim=1)
        K = torch.exp(base[:, None] + E)
        CD = torch.exp(torch.stack([_denorm(torch, u[:, LD.OFF_LOGC + j], LOGC_LO, LOGC_HI)
                                    for j in range(b)], dim=1))
    POSB = []
    for j in range(b):
        y = torch.clamp(u[:, LD.OFF_U + j], 0.0, 1.0) * total
        y = torch.minimum(y, total * (1.0 - 1e-15))[:, None]
        sel = (y >= below).to(torch.float64) - (y >= above).to(torch.float64)
        local = torch.minimum(torch.maximum(left + (y - below), left), right)
        POSB.append(torch.clamp((sel * local).sum(dim=1), min=lo_edge, max=hi_edge))

    x = torch.zeros((n, S.DESIGN_DIM), dtype=torch.float64)
    x[:, 0], x[:, 1], x[:, 2] = L, OD, ID
    for j in range(a):
        x[:, 3 + 3 * j] = H1_FACTOR * OD * (1.0 + REL[:, j])
        x[:, 4 + 3 * j] = W[:, j]
        x[:, 5 + 3 * j] = P[:, j]
    for j in range(b):
        x[:, 21 + 3 * j] = K[:, j]
        x[:, 22 + 3 * j] = CD[:, j]
        x[:, 23 + 3 * j] = POSB[j]
    return x


def decode_np(u, family):
    return LD.decode(np.asarray(u, dtype=np.float64), family)


def decode_batch(u, families):
    u = np.asarray(u, dtype=np.float64)
    return np.stack([LD.decode(u[i], families[i]) for i in range(len(u))])


def self_test(n=4000, seed=20260925, verbose=True):
    """Torch vs numpy decode, and the validity of both, on random latents."""
    import torch as _torch

    rng = np.random.default_rng(seed)
    worst, worst_rel, worst_col, hits, n_bad = 0.0, 0.0, 0, 0, 0
    errors, checked = [], 0
    per_family = {}
    for fam in [(a, b) for a in S.ND_CHOICES for b in S.NB_CHOICES]:
        mask = LD.used_mask(fam)
        u = rng.random((max(n // 18, 1), LD.LAT_DIM))
        u[:, ~mask] = 0.0
        x_np = decode_batch(u, [fam] * len(u))
        x_t = decode_torch(_torch, _torch.as_tensor(u), fam).detach().numpy()
        err = np.abs(x_np - x_t)
        rel = err / np.maximum(np.abs(x_np), 1e-12)
        errors.append(float(err.max()))
        # the K / C columns carry values up to 5e8, so agreement has to be
        # judged relative to the column scale, not on an absolute floor.
        bad = rel > 1e-9
        n_bad += int(bad.sum())
        worst_rel = max(worst_rel, float(rel.max()))
        if bad.any():
            cols = np.unique(np.where(bad)[1])
            worst_col = int(cols[np.argmax([rel[:, c].max() for c in cols])])
        valid = [S.is_valid(x, "Steel", fam[0], fam[1])[0] for x in x_np]
        per_family["%d_%d" % fam] = float(np.mean(valid))
        hits += int(np.sum(valid))
        checked += len(u)
        worst = max(worst, float(err.max()))
    out = {"n": checked, "valid_latent": hits / max(checked, 1),
           "max_abs_diff": worst, "max_rel_diff": worst_rel,
           "worst_rel_column": worst_col,
           "worst_rel_column_name": S.CANONICAL_COLUMNS[worst_col],
           "n_rel_gt_1e-9": int(n_bad),
           "per_family_valid": per_family}
    if verbose:
        print("[latent_torch] %d random latents | decoded valid %.4f | "
              "torch-vs-numpy max rel %.3e at %s (cells over 1e-9: %d)"
              % (out["n"], out["valid_latent"], out["max_rel_diff"],
                 out["worst_rel_column_name"], out["n_rel_gt_1e-9"]))
    return out


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(self_test(n=7200), indent=2, ensure_ascii=False))
    sys.exit(0)