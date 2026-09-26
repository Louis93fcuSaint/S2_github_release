# -*- coding: utf-8 -*-
"""Baselines and a differentiable surrogate forward pass for the spec track.

Every optimiser maximises the *same* objective:

    f(x) = -|ln cs1_hat(x) - ln sqrt(lo * hi)|

the log-distance to the geometric centre of the band, minus a large penalty when
the design breaks the hard constraints.  A flat "inside the band" objective has
no gradient and lets an optimiser wander, so the centre is used as the anchor.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_common as S

PENALTY = 10.0
UPSHOT_PENALTY = 20.0   # a 1 % undershoot costs as much as a 20 % overshoot
MAX_DISKS, MAX_BEARINGS = S.MAX_DISKS, S.MAX_BEARINGS


def log_center(spec):
    """Where the ranking aim sits.

    A band spec aims at the geometric centre of its two edges.  A one-sided spec
    ("cs1 at least X") aims at the requirement itself, otherwise the shortlist
    would cluster in the middle of a range the user never asked to reach.
    """
    if spec.get("aim"):
        return np.log(max(float(spec["aim"]), 1.0))
    return 0.5 * (np.log(spec["lower"]) + np.log(spec["upper"]))


def objective_from_cs1(cs1, spec, valid=None):
    """Ranking score.  One-sided specs do not rank symmetrically.

    For "cs1 at least X" an undershoot is a failed design and an overshoot is
    only over-design, so a design that just meets the requirement beats both a
    design just below it and a design far above it.
    """
    log_cs = np.log(np.maximum(cs1, 1.0))
    center = log_center(spec)
    if spec.get("mode") == "lower":
        score = -np.maximum(log_cs - center, 0.0) \
            - UPSHOT_PENALTY * np.maximum(center - log_cs, 0.0)
    else:
        score = -np.abs(log_cs - center)
    if valid is not None:
        score = score - PENALTY * (~np.asarray(valid))
    return score


def objective_from_cs1_torch(cs1, spec, valid_mask, torch):
    score = -(torch.log(torch.clamp(cs1, min=1.0)) - log_center(spec)).abs()
    return score - PENALTY * (~valid_mask).to(cs1.dtype)


def soft_constraints(x, material, nd, nb, torch):
    """Differentiable relaxation of the hard constraints H1-H6.

    Adam has no notion of a rejected design, so the constraints have to enter as
    a penalty; without this term it walks off the admissible region and every
    candidate then fails the S1 filter.
    """
    cmod = S.constraint_module()
    h3 = float(getattr(cmod, "H3_MARGIN_FRAC", 0.10))
    h4 = float(getattr(cmod, "H4_MIN_SEP", 0.01))
    h6 = float(getattr(cmod, "H6_PERT", 0.30))
    L, OD = x[:, 0], x[:, 1]
    disk = x[:, 3:21].reshape(-1, MAX_DISKS, 3)
    brg = x[:, 21:33].reshape(-1, MAX_BEARINGS, 3)
    pos = disk[:, :nd, 2]
    order = torch.argsort(pos, dim=1)
    d_od = torch.gather(disk[:, :nd, 0], 1, order)
    d_w = torch.gather(disk[:, :nd, 1], 1, order)
    d_pos = torch.gather(pos, 1, order)
    b_k = brg[:, :nb, 0]
    b_pos = brg[:, :nb, 2]

    pen = torch.relu(1.05 * OD[:, None] - d_od + 1e-4).sum(dim=1)
    if nd >= 2:
        gap = d_pos[:, 1:] - d_pos[:, :-1]
        need = 0.5 * (d_w[:, 1:] + d_w[:, :-1]) + 0.003
        pen = pen + torch.relu(need - gap).sum(dim=1)
    margin = h3 * L
    pen = pen + torch.relu(margin[:, None] - d_pos).sum(dim=1)
    pen = pen + torch.relu(d_pos - (L - margin)[:, None]).sum(dim=1)
    pen = pen + torch.relu(margin[:, None] - b_pos).sum(dim=1)
    pen = pen + torch.relu(b_pos - (L - margin)[:, None]).sum(dim=1)
    sep = (d_pos[:, :, None] - b_pos[:, None, :]).abs()
    pen = pen + torch.relu(h4 - sep).sum(dim=(1, 2))
    slender = L / torch.clamp(OD, min=1e-9)
    pen = pen + torch.relu(2.0 - slender) + torch.relu(slender - 50.0)
    kmin = b_k.min(dim=1).values
    kmax = b_k.max(dim=1).values
    pen = pen + torch.relu(kmax - ((1.0 + h6) / (1.0 - h6)) * kmin) / torch.clamp(kmin, min=1e3)
    pen = pen + torch.relu(0.5 - d_w).sum(dim=1) + torch.relu(d_w - 5.0).sum(dim=1)
    return pen


# --------------------------------------------------------------------------- #
class Box:
    """Per-family box of the real pool, widened slightly, plus the validity test."""

    def __init__(self, ctx, widen=0.05):
        designs, nd, nb, material = (ctx["designs"], ctx["nd"], ctx["nb"],
                                     ctx["material"])
        self.lo, self.hi, self.designs, self.index = {}, {}, {}, {}
        for a in S.ND_CHOICES:
            for b in S.NB_CHOICES:
                sel = np.where((nd == a) & (nb == b))[0]
                self.index[(a, b)] = sel
                used = self._used(a, b)
                block = designs[sel][:, used]
                span = block.max(axis=0) - block.min(axis=0)
                self.lo[(a, b)] = block.min(axis=0) - widen * span
                self.hi[(a, b)] = block.max(axis=0) + widen * span
                self.used = used
        self.material = material

    @staticmethod
    def _used(a, b):
        used = [0, 1, 2]
        for j in range(a):
            used += [3 + 3 * j, 4 + 3 * j, 5 + 3 * j]
        for j in range(b):
            used += [21 + 3 * j, 22 + 3 * j, 23 + 3 * j]
        return used

    def sample(self, family, n, rng):
        a, b = family
        used = self._used(a, b)
        lo, hi = self.lo[family], self.hi[family]
        block = lo + rng.random((n, len(used))) * (hi - lo)
        x = np.zeros((n, S.DESIGN_DIM))
        x[:, used] = block
        return x

    def clip(self, x, family):
        a, b = family
        used = self._used(a, b)
        out = np.zeros_like(x)
        out[:, used] = np.clip(x[:, used], self.lo[family], self.hi[family])
        return out


# --------------------------------------------------------------------------- #
def make_frame(candidates, material, families):
    """(n, 33) + per-candidate family -> the submission CSV frame."""
    import pandas as pd

    data = {c: candidates[:, i] for i, c in enumerate(S.CANONICAL_COLUMNS)}
    data["material"] = [material] * len(candidates)
    data["n_disks"] = [int(f[0]) for f in families]
    data["n_bearings"] = [int(f[1]) for f in families]
    return pd.DataFrame(data)[E_CSV]


E_CSV = None


def set_columns():
    global E_CSV
    E_CSV = ["material", "n_disks", "n_bearings"] + S.CANONICAL_COLUMNS


set_columns()


def validity(candidates, material, families):
    return np.array([S.is_valid(x, material, int(f[0]), int(f[1]))[0]
                     for x, f in zip(candidates, families)])


def repair(candidates, material, families, rng, rounds=30):
    """Push invalid candidates back inside the hard constraints by shrinking
    perturbations.  Returns the repaired batch and the share that needed it."""
    x = np.array(candidates, dtype=np.float64, copy=True)
    bad = ~validity(x, material, families)
    needed = bad.copy()
    step = 0.10
    scales = np.array([1.0] * S.DESIGN_DIM)
    for _ in range(rounds):
        if not bad.any():
            break
        idx = np.where(bad)[0]
        for i in idx:
            a, b = int(families[i][0]), int(families[i][1])
            used = Box._used(a, b)
            x[i, used] += rng.normal(0.0, step * scales[used])
            x[i] = S.sort_canonical(x[i], a, b)
        bad = ~validity(x, material, families)
        step *= 0.85
    return x, float(needed.mean())


# --------------------------------------------------------------------------- #
# Structured repair: project onto the feasible set instead of random-walking.
# --------------------------------------------------------------------------- #
DISK_SLOT = 3
BRG_SLOT = 3 + 3 * S.MAX_DISKS
H1_FACTOR = 1.05
H3_MARGIN_FRAC = 0.10
H4_MIN_SEP = 0.01
H6_PERT = 0.30
K_MIN, K_MAX = 1.0e6, 5.0e8
W_MIN, W_MAX = 0.5, 5.0
ID_FRAC = 0.8


def _project_sorted(ps, gaps, lo, hi):
    """Closest positions on [lo, hi] with ps[j+1] - ps[j] >= gaps[j].

    Forward then backward interval projection.  Returns None when the gaps do
    not fit at all, which the caller answers with random repair.
    """
    n = len(ps)
    if n == 0:
        return []
    p = list(ps)
    p[0] = max(p[0], lo)
    for j in range(1, n):
        p[j] = max(p[j], p[j - 1] + gaps[j - 1])
    if p[-1] > hi:
        p[-1] = min(p[-1], hi)
        for j in range(n - 2, -1, -1):
            p[j] = min(p[j], p[j + 1] - gaps[j])
    if p[0] < lo:
        p[0] = lo
        for j in range(1, n):
            p[j] = p[j - 1] + gaps[j - 1]
        if p[-1] > hi:
            return None
    return p


def _snap_bearings(ps, disk_ps, lo, hi, sep):
    """Push bearing positions out of the disk exclusion zones, keeping order."""
    out, prev = [], lo
    for p in ps:
        q = max(p, prev)
        for _ in range(4 * len(disk_ps) + 4):
            hit = [d for d in disk_ps if abs(q - d) < sep]
            if not hit:
                break
            q = min(d + sep for d in hit)
        if q > hi:
            return None
        out.append(q)
        prev = q + sep
    return out


def repair_structured(candidates, material, families, box=None, rng=None,
                      rounds=30):
    """Deterministic projection onto H1-H6, then random repair as a fallback.

    Returns (x, stats).  stats counts which rules fired and how many designs
    still needed the old random walk.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.array(candidates, dtype=np.float64, copy=True)
    stats = {"total": 0, "H1": 0, "H2": 0, "H3": 0, "H5": 0, "H6": 0,
             "width": 0, "ID": 0, "fallback": 0}
    stuck = []
    for i in range(len(x)):
        stats["total"] += 1
        a, b = int(families[i][0]), int(families[i][1])
        row = x[i].copy()
        if box is not None:
            # the box is the data manifold: clip first, so that the projections
            # below, which are the last word, still land inside it.
            row = box.clip(row[None, :], (a, b))[0]
        L, OD, ID = float(row[0]), float(row[1]), float(row[2])
        if OD <= 0 or L <= 0:
            stuck.append(i)
            continue

        slender = L / OD
        if slender <= 2.0 or slender >= 50.0:
            L = OD * float(np.clip(slender, 2.0 + 0.01, 50.0 - 0.01))
            stats["H5"] += 1
        if ID > ID_FRAC * OD or ID < 0.0:
            ID = max(0.0, min(ID, ID_FRAC * OD))
            stats["ID"] += 1

        widths = []
        for j in range(a):
            w = float(np.clip(row[DISK_SLOT + 3 * j + 1], W_MIN, W_MAX))
            if w != row[DISK_SLOT + 3 * j + 1]:
                stats["width"] += 1
            widths.append(w)
            row[DISK_SLOT + 3 * j + 1] = w
        for j in range(a):
            od_d = row[DISK_SLOT + 3 * j]
            if od_d <= H1_FACTOR * OD:
                row[DISK_SLOT + 3 * j] = H1_FACTOR * OD * 1.001
                stats["H1"] += 1

        margin = H3_MARGIN_FRAC * L
        old_ps = [float(row[DISK_SLOT + 3 * j + 2]) for j in range(a)]
        gaps = [0.5 * (widths[j] + widths[j + 1]) + 0.0035 for j in range(a - 1)]
        new_ps = _project_sorted(old_ps, gaps, margin, L - margin)
        if new_ps is None:
            stuck.append(i)
            continue
        if any(abs(u - v) > 1e-12 for u, v in zip(old_ps, new_ps)):
            stats["H2"] += int(a > 1)
            stats["H3"] += 1
        for j, q in enumerate(new_ps):
            row[DISK_SLOT + 3 * j + 2] = q

        ks = [float(row[BRG_SLOT + 3 * j]) for j in range(b)]
        pos_b = [float(row[BRG_SLOT + 3 * j + 2]) for j in range(b)]
        finite = [k for k in ks if k > 0]
        if finite:
            centre = float(np.exp(np.mean(np.log(np.maximum(ks, 1.0)))))
            out_k = []
            for k in ks:
                z = np.clip(np.log(max(k, 1.0)) - np.log(centre), -H6_PERT, H6_PERT)
                out_k.append(float(np.clip(np.exp(z) * centre, K_MIN, K_MAX)))
            if any(abs(u - v) > 1e-9 for u, v in zip(ks, out_k)):
                stats["H6"] += 1
            ks = out_k
        snapped = _snap_bearings(pos_b, new_ps, margin, L - margin, H4_MIN_SEP + 1e-3)
        if snapped is None:
            stuck.append(i)
            continue
        if any(abs(u - v) > 1e-12 for u, v in zip(pos_b, snapped)):
            stats["H3"] += 1

        row[0], row[1], row[2] = L, OD, ID
        for j in range(b):
            row[BRG_SLOT + 3 * j] = ks[j]
            row[BRG_SLOT + 3 * j + 2] = snapped[j]
        row = S.sort_canonical(row, a, b)
        x[i] = row

    if stuck:
        stats["fallback"] = len(stuck)
        idx = np.array(stuck, dtype=int)
        fixed, _ = repair(x[idx], material, [families[i] for i in idx], rng,
                          rounds=rounds)
        x[idx] = fixed
    bad = ~validity(x, material, families)
    stats["bad_after_projection"] = int(bad.sum())
    reasons = {}
    for i in np.where(bad)[0][:400]:
        _, why = S.is_valid(x[i], material, int(families[i][0]),
                            int(families[i][1]))
        for w in why:
            key = w.split(":")[0].strip()
            reasons[key] = reasons.get(key, 0) + 1
    stats["reasons"] = reasons
    if bad.any():
        stats["fallback"] += int(bad.sum())
        idx = np.where(bad)[0]
        fixed, _ = repair(x[idx], material, [families[i] for i in idx], rng,
                          rounds=rounds)
        x[idx] = fixed
    return x, stats


# --------------------------------------------------------------------------- #
class SmoothSurrogate:
    """Differentiable cs1(canonical x) for a *fixed* family, mirroring the frozen
    MLP: engineered features -> log on the five log columns -> z-score -> net."""

    def __init__(self, surrogate, torch):
        self.torch = torch
        self.s = surrogate
        self.mats = {m: i for i, m in enumerate(S.MATERIAL_ORDER)}

    def _features(self, x, material, nd, nb):
        torch = self.torch
        L, OD, ID = x[:, 0], x[:, 1], x[:, 2]
        disk = x[:, 3:21].reshape(-1, MAX_DISKS, 3)
        brg = x[:, 21:33].reshape(-1, MAX_BEARINGS, 3)
        pos = disk[:, :nd, 2]
        order = torch.argsort(pos, dim=1)
        d_od = torch.gather(disk[:, :nd, 0], 1, order)
        d_w = torch.gather(disk[:, :nd, 1], 1, order)
        d_pos = torch.gather(pos, 1, order)
        pos_b = brg[:, :nb, 2]
        order_b = torch.argsort(pos_b, dim=1)
        b_k = torch.gather(brg[:, :nb, 0], 1, order_b)
        b_pos = torch.gather(pos_b, 1, order_b)

        n = x.shape[0]
        raw = torch.zeros((n, 39), dtype=x.dtype)
        raw[:, 0], raw[:, 1], raw[:, 2] = L, OD, ID
        raw[:, 3] = float(nd)
        raw[:, 4] = float(nb)
        raw[:, 5 + self.mats[material]] = 1.0
        raw[:, 8:8 + 3 * nd] = disk[:, :nd, :].reshape(n, 3 * nd)
        raw[:, 27:27 + 3 * nb] = brg[:, :nb, :].reshape(n, 3 * nb)

        eps = 1e-6
        slender = L / torch.clamp(OD, min=eps)
        log_L = torch.log(torch.clamp(L, min=eps))
        log_OD = torch.log(torch.clamp(OD, min=eps))
        if nd >= 2:
            d_span = (d_pos.max(dim=1).values - d_pos.min(dim=1).values) / torch.clamp(L, min=eps)
        else:
            d_span = torch.zeros(n, dtype=x.dtype)
        b_span = (b_pos.max(dim=1).values - b_pos.min(dim=1).values) / torch.clamp(L, min=eps)
        log_k_min = torch.log(torch.clamp(b_k.min(dim=1).values, min=1e4))
        log_k_mean = torch.log(torch.clamp(b_k.mean(dim=1), min=1e4))
        shaft_v = np.pi * (OD / 2.0) ** 2 * L
        shaft_v = shaft_v - np.pi * (ID / 2.0) ** 2 * L * (ID > 0).to(x.dtype)
        disk_v = (np.pi * (d_od / 2.0) ** 2 * d_w).sum(dim=1)
        mass_ratio = disk_v / torch.clamp(shaft_v, min=1e-10)
        return torch.cat([raw, torch.stack(
            [slender, log_L, log_OD, d_span, b_span, log_k_min, log_k_mean,
             mass_ratio], dim=1)], dim=1)

    def cs1(self, x, material, nd, nb):
        torch = self.torch
        feats = self._features(x, material, nd, nb)
        logs = []
        for item in self.s.models:
            rows = feats.clone()
            rows[:, self.s.log_columns] = torch.log(torch.clamp(
                rows[:, self.s.log_columns], min=torch.as_tensor(
                    self.s.log_floor, dtype=rows.dtype)))
            scaled = (rows - torch.as_tensor(item["x_mean"], dtype=rows.dtype)) \
                / torch.as_tensor(item["x_scale"], dtype=rows.dtype)
            dtype = next(item["net"].parameters()).dtype
            out = item["net"](scaled.to(dtype))
            logs.append(out[:, 0] * float(item["y_std"][0]) + float(item["y_mean"][0]))
        return torch.exp(torch.stack(logs, dim=0).mean(dim=0))
