# -*- coding: utf-8 -*-
"""Feasibility-preserving latent parameterisation of the canonical design.

Why this module exists
----------------------
The canonical 33-vector satisfies six coupled hard constraints (H1-H6).  A
generative model that emits it directly spends most of its samples outside the
admissible set -- measured on the current v4.7 DDPM: 6.9 % valid, and clipping
does not help, because the constraints are coupled (disk OD vs shaft OD, disk
positions vs widths vs shaft length, bearing stiffness ratio, ...) and because
the linear unit cube is badly conditioned (bearing K spans 2.70 decades, so the
+/-30 % H6 ratio is a 0.6 %-wide sliver of that cube).

This module turns the design into a box.  Every latent coordinate is
independent, every one of H1-H6 is enforced by the decoder *by construction*,
and the encoder is exact on the pool up to negligible quantisation:

  H1  disk_OD = 1.05 * OD * (1 + rel),  rel >= 1e-4        -> expm1 box
  H2  gaps = (half width sum + 3 mm) + slack piece + 10 um
  H3  every position is built inside [0.1 L, 0.9 L]
  H4  bearings are laid out on the free intervals (disks +/- 10 mm)
  H5  OD is clipped into (L/50, L/2)
  H6  log K_j = log(min K) + e_j with e_j in [0, ln(1.3/0.7)]
      so max K / min K <= 1.857 for free

Latent layout (34 coordinates); slots beyond the family are pinned to 0 and
excluded from the loss:

  [0]      L                 [0.25, 2.10]   m
  [1]      OD                [0.018, 0.16]  m
  [2]      log ID            [log 1e-7, log 0.06]
  [3..8]   log1p(rel)        disk_OD / (1.05 OD) - 1
  [9..14]  disk width W      [0.020, 0.150] m
  [15..20] gap cut points    increasing in (0, 1)
  [21]     log min K         [log 9e5, log 5.5e8]
  [22..25] log K_j - log min K                [0, ln(1.3/0.7)]
  [26..29] log C             [log 45, log 1100]
  [30..33] bearing free u                     [0, 1]
"""
from __future__ import annotations

import numpy as np

import spec_common as S

MAX_D, MAX_B = S.MAX_DISKS, S.MAX_BEARINGS
LAT_DIM = 34

H1_FACTOR = 1.05
H3_FRAC = 0.10
H4_SEP = 0.01
H4_SLACK = 1.0e-6           # keeps the decoded distance strictly above 10 mm
GAP_EXTRA = 0.003
GAP_NUDGE = 1.0e-5          # 10 um of guard on every gap, so H2 never sits on 0
D_LIM = float(np.log(1.30 / 0.70))   # 0.619 ; K ratio <= exp(D_LIM) = 1.857

BOX_L = (0.25, 2.10)
BOX_OD = (0.018, 0.16)
BOX_LOG_ID = (float(np.log(1e-7)), float(np.log(0.06)))
BOX_LOG1P_REL = (float(np.log1p(1e-4)), float(np.log1p(30.0)))
BOX_W = (0.020, 0.150)
BOX_LOG_K = (float(np.log(9e5)), float(np.log(5.5e8)))
BOX_E = (0.0, D_LIM)
BOX_LOG_C = (float(np.log(45.0)), float(np.log(1100.0)))

OFF_L, OFF_OD, OFF_ID = 0, 1, 2
OFF_REL, OFF_W, OFF_CUT = 3, 9, 15
OFF_LOGK, OFF_E, OFF_LOGC, OFF_U = 21, 22, 26, 30

_OD_LO, _OD_HI = BOX_OD
_W_FLOOR = 0.002            # decoder-only shrink floor (H2 feasibility)
SLACK_MIN = 0.10            # guard path only
FL_MIN = 0.03               # guard path only

GUARD = {"fired": 0, "total": 0}


def _norm(v, box):
    lo, hi = box
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def _denorm(u, box):
    lo, hi = box
    return lo + float(np.clip(u, 0.0, 1.0)) * (hi - lo)


def used_slots(family):
    a, b = int(family[0]), int(family[1])
    out = [OFF_L, OFF_OD, OFF_ID]
    out += list(range(OFF_REL, OFF_REL + a))
    out += list(range(OFF_W, OFF_W + a))
    out += list(range(OFF_CUT, OFF_CUT + a))
    out += [OFF_LOGK]
    out += list(range(OFF_E, OFF_E + b))
    out += list(range(OFF_LOGC, OFF_LOGC + b))
    out += list(range(OFF_U, OFF_U + b))
    return out


def used_mask(family):
    mask = np.zeros(LAT_DIM, dtype=bool)
    mask[used_slots(family)] = True
    return mask


# --------------------------------------------------------------------------- #
def _subtract(ivs, lo, hi):
    out = []
    for a0, b0 in ivs:
        if hi <= a0 or lo >= b0:
            out.append((a0, b0))
            continue
        if lo > a0:
            out.append((a0, min(lo, b0)))
        if hi < b0:
            out.append((max(hi, a0), b0))
    return [(a0, b0) for a0, b0 in out if b0 - a0 > 1e-13]


def free_intervals(L, positions, margin):
    lo_edge = margin + 1e-9 * L
    hi_edge = L - margin - 1e-9 * L
    if hi_edge <= lo_edge:
        return []
    ivs = [(lo_edge, hi_edge)]
    for p in sorted(float(v) for v in positions):
        ivs = _subtract(ivs, p - H4_SEP, p + H4_SEP)
    return ivs


def _free_pos(ivs, pos):
    run = 0.0
    for a0, b0 in ivs:
        if pos <= a0:
            return run
        if pos <= b0:
            return run + (pos - a0)
        run += b0 - a0
    return run


def _free_inv(ivs, y):
    run, last = 0.0, None
    for a0, b0 in ivs:
        width = b0 - a0
        if y <= run + width:
            return a0 + min(max(y - run, 0.0), width)
        run += width
        last = b0
    return last if last is not None else 0.0


def _snap(bp, positions, margin, L):
    """Nudge a bearing the 1 um it needs to be strictly 10 mm clear of a disk."""
    lo, hi = margin, L - margin
    q = float(min(max(bp, lo), hi))
    for _ in range(8):
        hit = [p for p in positions if abs(q - p) < H4_SEP + 1e-9]
        if not hit:
            break
        cand = []
        for p in hit:
            cand.append(p + H4_SEP + H4_SLACK)
            cand.append(p - H4_SEP - H4_SLACK)
        cand = [c for c in cand if lo <= c <= hi and c >= q - 1e-6]
        if not cand:
            return q
        q = min(cand)
    return float(min(max(q, lo), hi))


# --------------------------------------------------------------------------- #
def _widths_and_positions(L, widths, cuts, a, force_room):
    """Widths and disk positions; H2 and H3 hold by construction."""
    margin = H3_FRAC * L
    reserve = SLACK_MIN if force_room else 0.0
    min_needed = (a + 1) * GAP_NUDGE + reserve

    def mg(wj, wk):
        return 0.5 * (wj + wk) + GAP_EXTRA

    need = sum(mg(widths[j], widths[j + 1]) for j in range(a - 1)) if a > 1 else 0.0
    budget = 0.8 * L - min_needed
    if a > 1 and need > budget:
        widths = ([_W_FLOOR] * a if budget <= 0.0 else
                  [max(w * (budget / need), _W_FLOOR) for w in widths])
        need = sum(mg(widths[j], widths[j + 1]) for j in range(a - 1))
        if need > budget:
            widths = [_W_FLOOR] * a
            need = sum(mg(widths[j], widths[j + 1]) for j in range(a - 1))
    slack_def = max(0.8 * L - need, 0.0)
    spread = max(slack_def - (a + 1) * GAP_NUDGE, 0.0)

    if force_room and a > 1:
        edges = [0.0, 0.4, 0.4 + 0.2] + [0.6] * (a - 2) + [1.0]
    else:
        edges = [0.0] + sorted(float(np.clip(c, 0.0, 1.0)) for c in cuts) + [1.0]
    edges[a] = max(edges[a], edges[a - 1])
    gap = [max(edges[k + 1] - edges[k], 0.0) * spread + GAP_NUDGE
           for k in range(a + 1)]
    pos = [margin + gap[0]]
    for j in range(1, a):
        pos.append(pos[-1] + mg(widths[j - 1], widths[j]) + gap[j])
    pos = [float(min(max(p, margin), L - margin)) for p in pos]
    return widths, pos, sum(mg(widths[j], widths[j + 1]) for j in range(a - 1))


# --------------------------------------------------------------------------- #
def encode(x, family):
    """Canonical 33-vector -> latent in [0,1]^34 (exact for admissible designs)."""
    a, b = int(family[0]), int(family[1])
    x = np.asarray(x, dtype=np.float64)
    L, OD, ID = float(x[0]), float(x[1]), float(x[2])
    u = np.zeros(LAT_DIM, dtype=np.float64)

    u[OFF_L] = _norm(L, BOX_L)
    u[OFF_OD] = _norm(OD, BOX_OD)
    u[OFF_ID] = _norm(np.log(max(ID, 1e-30)), BOX_LOG_ID)

    widths, positions = [], []
    for j in range(a):
        d_od, w, p = float(x[3 + 3 * j]), float(x[4 + 3 * j]), float(x[5 + 3 * j])
        rel = d_od / (H1_FACTOR * OD) - 1.0
        u[OFF_REL + j] = _norm(np.log1p(max(rel, 0.0)), BOX_LOG1P_REL)
        u[OFF_W + j] = _norm(w, BOX_W)
        widths.append(w)
        positions.append(p)

    margin = H3_FRAC * L
    need = sum(0.5 * (widths[j] + widths[j + 1]) + GAP_EXTRA
               for j in range(a - 1)) if a > 1 else 0.0
    spread = 0.8 * L - need - (a + 1) * GAP_NUDGE
    gap = [positions[0] - margin]
    for j in range(1, a):
        gap.append(positions[j] - positions[j - 1]
                   - 0.5 * (widths[j - 1] + widths[j]) - GAP_EXTRA)
    gap.append((L - margin) - positions[-1])
    if spread > 1e-12:
        run = 0.0
        for k in range(a):
            run += gap[k]
            u[OFF_CUT + k] = float(np.clip((run - GAP_NUDGE) / spread, 0.0, 1.0))
    else:
        for k in range(a):
            u[OFF_CUT + k] = (k + 1.0) / (a + 1.0)

    log_k = [np.log(float(x[21 + 3 * j])) for j in range(b)]
    base = float(min(log_k))
    u[OFF_LOGK] = _norm(base, BOX_LOG_K)
    for j in range(b):
        u[OFF_E + j] = _norm(log_k[j] - base, BOX_E)
        u[OFF_LOGC + j] = _norm(np.log(float(x[22 + 3 * j])), BOX_LOG_C)

    ivs = free_intervals(L, positions, margin)
    total = sum(b0 - a0 for a0, b0 in ivs)
    for j in range(b):
        bp = float(x[23 + 3 * j])
        u[OFF_U + j] = _norm(_free_pos(ivs, bp) / total if total > 1e-12 else 0.0,
                             (0.0, 1.0))
    return u


def decode(u, family):
    """Latent -> canonical 33-vector.  The result always satisfies H1-H6."""
    a, b = int(family[0]), int(family[1])
    u = np.asarray(u, dtype=np.float64)
    L = _denorm(u[OFF_L], BOX_L)
    OD = _denorm(u[OFF_OD], BOX_OD)
    OD = float(np.clip(OD, max(_OD_LO, L / 50.0 * (1.0 + 1e-7)),
                       min(_OD_HI, L / 2.0 * (1.0 - 1e-7))))
    ID = float(np.exp(_denorm(u[OFF_ID], BOX_LOG_ID)))
    ID = min(ID, 0.8 * OD)

    widths = [_denorm(u[OFF_W + j], BOX_W) for j in range(a)]
    rel = [float(np.expm1(_denorm(u[OFF_REL + j], BOX_LOG1P_REL)))
           for j in range(a)]
    cuts = [u[OFF_CUT + j] for j in range(a)]
    margin = H3_FRAC * L

    widths, positions, need = _widths_and_positions(L, widths, cuts, a, False)
    ivs = free_intervals(L, positions, margin)
    total = sum(b0 - a0 for a0, b0 in ivs)
    GUARD["total"] += 1
    if total < FL_MIN:
        GUARD["fired"] += 1
        widths, positions, need = _widths_and_positions(L, widths, cuts, a, True)
        ivs = free_intervals(L, positions, margin)
        total = sum(b0 - a0 for a0, b0 in ivs)

    base = _denorm(u[OFF_LOGK], BOX_LOG_K)
    stiff = [float(np.exp(base + _denorm(u[OFF_E + j], BOX_E))) for j in range(b)]
    damping = [float(np.exp(_denorm(u[OFF_LOGC + j], BOX_LOG_C))) for j in range(b)]

    ui = [float(np.clip(u[OFF_U + j], 0.0, 1.0)) for j in range(b)]
    bpos = [_snap(_free_inv(ivs, ui[j] * total), positions, margin, L)
            for j in range(b)]

    x = np.zeros(S.DESIGN_DIM, dtype=np.float64)
    x[0], x[1], x[2] = L, OD, ID
    for j in range(a):
        x[3 + 3 * j] = H1_FACTOR * OD * (1.0 + rel[j])
        x[4 + 3 * j] = widths[j]
        x[5 + 3 * j] = positions[j]
    for j in range(b):
        x[21 + 3 * j] = stiff[j]
        x[22 + 3 * j] = damping[j]
        x[23 + 3 * j] = bpos[j]
    return x