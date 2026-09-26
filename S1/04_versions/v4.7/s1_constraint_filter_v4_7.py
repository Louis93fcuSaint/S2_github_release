"""
s1_constraint_filter_v4_7.py

Sequential conditional LHS: shaft -> disks -> bearings.
1-6 disks, 2-4 bearings, isotropic bearings only.

v4.4 CHANGES (2026-08-22):
  1. Disk OD/width follow the old document global coverage ranges:
     OD [0.10, 0.50] m, width [0.02, 0.15] m.
  2. Bearing type is isotropic only; damping is narrowed to
     [50, 1000] N.s/m (document), stiffness stays at [1e6, 5e8] N/m
     (same as v4.2/v4.3 code).
  3. Disk and bearing positions use sequential conditional LHS.
  4. H6: sample one base log10 stiffness per rotor, then apply a
     +/-30% multiplicative perturbation to each bearing.
  5. Materials aligned with ROSS available_materials.toml (Steel) and
     with the old document material table (Aluminum/Titanium).
  6. H3 end margin is kept at 0.1L (code version, not 0.02L).

v4.5 CHANGES:
  A. Hard/soft separation.  check_hard_v4() decides admissibility (H1-H6);
     check_soft_v4() returns S1-S3 as labels and never rejects.
  B. S1 (disk/shaft volume ratio) and S3 (bearing span) are no longer
     pre-filtered in X_to_samples_v4, and _pick_bearing_endpoints no longer
     pushes the outer bearings apart to force span > 0.4 L.  Bearing
     positions are therefore sampled without bias toward that convention.
  C. S2 uses the single threshold 0.1*OD; the redundant 1 mm test that
     always dominated it has been removed.

v4.6 CHANGES:
  A. Forward-whirl filtering is handled by the ROSS pipeline; this module
     keeps the v4.5 sampling logic and the non-rejecting S1-S3 labels.
  B. H1 uses the raw disk OD values.  The previous max(..., 1.05*OD) clamp
     was dead code because H1 rejects equality with the boundary.
  C. Rejected hard-constraint categories and prefilter reasons are counted
     and returned in sampling statistics for summary.json/histograms.
"""

import numpy as np
from collections import Counter


# Steel matches ROSS available_materials.toml [Materials.Steel].
# Aluminum/Titanium follow the old document material table; the G_s values
# are the commonly used 6061-T6 / Ti-6Al-4V shear moduli.
MATERIALS = {
    "Steel":    {"rho": 7810.0, "E": 2.11e11, "G_s": 8.12e10},
    "Aluminum": {"rho": 2700.0, "E": 7.00e10, "G_s": 2.60e10},
    "Titanium": {"rho": 4430.0, "E": 1.10e11, "G_s": 4.40e10},
}
MAT_LIST = list(MATERIALS.keys())

K_MIN = 1.0e6
K_MAX = 5.0e8
C_MIN = 50.0
C_MAX = 1000.0
DISK_OD_MIN = 0.10
DISK_OD_MAX = 0.50
DISK_W_MIN = 0.02
DISK_W_MAX = 0.15
H3_MARGIN_FRAC = 0.10          # code version: [0.1L, 0.9L]
H4_MIN_SEP = 0.01              # 10 mm disk-bearing separation
H6_PERT = 0.30                 # +/-30% multiplicative stiffness class

# 3 shaft + 6 disk OD + 6 disk W + 1 base log10 K + 4 K pert
# + 4 C + 6 disk pos units + 4 bearing pos units = 34 LHS dimensions.
N_DIMS = 34


def _ranges():
    r = [(0.3, 2.0), (0.02, 0.15), (0.0, 0.05)]
    for _ in range(6):
        r.append((DISK_OD_MIN, DISK_OD_MAX))
    for _ in range(6):
        r.append((DISK_W_MIN, DISK_W_MAX))
    r.append((np.log10(K_MIN), np.log10(K_MAX)))   # base log10 stiffness
    for _ in range(4):
        r.append((-H6_PERT, H6_PERT))              # +/-30% perturbation
    for _ in range(4):
        r.append((C_MIN, C_MAX))
    for _ in range(6):
        r.append((0.0, 1.0))                       # disk position unit
    for _ in range(4):
        r.append((0.0, 1.0))                       # bearing position unit
    return r


def lhs_v4(n, rng=None, seed=42):
    """Stratified LHS over the 34 continuous dimensions."""
    if rng is None:
        rng = np.random.default_rng(seed)
    rr = _ranges()
    X = np.zeros((n, len(rr)))
    for d in range(len(rr)):
        bins = np.linspace(0, 1, n + 1)
        vals = rng.uniform(bins[:-1], bins[1:])
        rng.shuffle(vals)
        lo, hi = rr[d]
        X[:, d] = lo + vals * (hi - lo)
    return X


# ----------------------------------------------------------------------
# Sequential conditional LHS helpers
# ----------------------------------------------------------------------

def _subtract_interval(intervals, lo, hi):
    out = []
    for a, b in intervals:
        if hi <= a or lo >= b:
            out.append((a, b))
        else:
            if lo > a:
                out.append((a, lo))
            if hi < b:
                out.append((hi, b))
    return [(a, b) for a, b in out if b - a > 1e-9]


def _inverse_cdf(u, intervals):
    """Map u in [0,1] to a coordinate inside the union of intervals."""
    total = sum(b - a for a, b in intervals)
    if total <= 0:
        return None
    target = float(u) * total
    for a, b in intervals:
        length = b - a
        if target <= length:
            return a + target
        target -= length
    return intervals[-1][1]


def _lhs_place_disks(nd, L, OD, ods_raw, ws_raw, u_vals):
    """Place disks by sequential LHS with exact H2 gaps.

    Disks are ordered by their LHS unit values.  The k-th sorted disk gets
    x_k = x_min_k + S * v_k, where x_min_k is the minimum position that
    satisfies H2 relative to the previous disks and S is the remaining
    slack.  This keeps the positions stratified while guaranteeing H2/H3.
    """
    margin = H3_MARGIN_FRAC * L
    order = sorted(range(nd), key=lambda i: u_vals[i])
    widths = [ws_raw[i] for i in order]
    v = [u_vals[i] for i in order]

    min_pos = [margin]
    for k in range(1, nd):
        min_gap = (widths[k - 1] + widths[k]) / 2.0 + 0.003
        min_pos.append(min_pos[-1] + min_gap)
    if min_pos[-1] > L - margin:
        return None

    slack = (L - margin) - min_pos[-1]
    positions = [min_pos[k] + slack * v[k] for k in range(nd)]
    return (
        positions,
        [float(ods_raw[i]) for i in order],
        widths,
    )


def _bearing_intervals(L, disk_poses):
    margin = H3_MARGIN_FRAC * L
    intervals = [(margin, L - margin)]
    for dp in disk_poses:
        intervals = _subtract_interval(intervals, dp - H4_MIN_SEP, dp + H4_MIN_SEP)
    return intervals


def _pick_bearing_endpoints(L, intervals, v):
    if not intervals:
        return None, None
    a0, b0 = intervals[0]
    a1, b1 = intervals[-1]
    p0 = a0 + (b0 - a0) * v[0]
    p1 = a1 + (b1 - a1) * v[-1]
    if p0 > p1:
        p0, p1 = p1, p0
    return p0, p1


def _lhs_place_bearings(nb, L, disk_poses, u_vals):
    """Place bearings by sequential conditional LHS.

    The first/last bearings are mapped from the extreme free intervals so
    S3 (span > 0.4L) is satisfied.  Middle bearings are mapped with the
    inverse-CDF over the remaining free intervals after removing 10 mm
    zones around disks and already placed bearings.
    """
    intervals = _bearing_intervals(L, disk_poses)
    if not intervals:
        return None

    order = sorted(range(nb), key=lambda i: u_vals[i])
    v = [u_vals[i] for i in order]
    p0, p1 = _pick_bearing_endpoints(L, intervals, v)
    if p0 is None:
        return None

    positions = [p0, p1]
    if nb > 2:
        for k in range(1, nb - 1):
            free = []
            for a, b in intervals:
                a = max(a, p0 + H4_MIN_SEP)
                b = min(b, p1 - H4_MIN_SEP)
                if b - a > 1e-9:
                    free.append((a, b))
            for bp in positions:
                free = _subtract_interval(free, bp - H4_MIN_SEP, bp + H4_MIN_SEP)
            pos = _inverse_cdf(v[k], free)
            if pos is None:
                return None
            positions.append(pos)

    positions = sorted(positions)
    for dp in disk_poses:
        for bp in positions:
            if abs(dp - bp) < H4_MIN_SEP:
                return None
    return positions


def _bump(counter, key):
    if counter is not None:
        counter[key] += 1


def X_to_samples_v4(X, rng=None, prefilter_counter=None):
    if rng is None:
        rng = np.random.default_rng()
    samples = []

    for row in X:
        L = float(row[0])
        OD = float(row[1])
        ID = min(float(row[2]), OD * 0.8)
        mat = str(rng.choice(MAT_LIST))

        sd = L / max(OD, 1e-6)
        if sd <= 2 or sd >= 50:
            _bump(prefilter_counter, "H5_precheck")
            continue

        nd = int(rng.integers(1, 7))
        ods_raw = [float(row[3 + i]) for i in range(nd)]
        ws_raw = [float(row[9 + i]) for i in range(nd)]
        u_disk = [float(row[24 + i]) for i in range(nd)]

        disk = _lhs_place_disks(nd, L, OD, ods_raw, ws_raw, u_disk)
        if disk is None:
            _bump(prefilter_counter, "disk_placement")
            continue
        dp, d_ods, d_ws = disk

        nb = int(rng.integers(2, 5))
        u_bear = [float(row[30 + i]) for i in range(nb)]
        bp = _lhs_place_bearings(nb, L, dp, u_bear)
        if bp is None:
            _bump(prefilter_counter, "bearing_placement")
            continue

        log_k_base = float(row[15])
        k_vals = []
        for i in range(nb):
            k = (10.0 ** log_k_base) * (1.0 + float(row[16 + i]))
            k_vals.append(float(np.clip(k, K_MIN, K_MAX)))
        c_vals = [float(row[20 + i]) for i in range(nb)]

        shaft_v = np.pi * (OD / 2.0) ** 2 * L
        if ID > 0:
            shaft_v -= np.pi * (ID / 2.0) ** 2 * L
        disk_v = sum(
            np.pi * (d_o / 2.0) ** 2 * d_w
            for d_o, d_w in zip(d_ods, d_ws)
        )
        samples.append({
            "sl": L, "od": OD, "id": ID, "material": mat,
            "bearing_type": "isotropic",
            "nd": nd, "disk_od_list": d_ods,
            "disk_w_list": d_ws, "dp": dp,
            "nb": nb, "bk_list": k_vals,
            "bc_list": c_vals, "bp": bp,
        })

    return samples


def check_hard_v4(s):
    """Hard constraints H1-H6. Returns (ok, reasons). Decides admissibility."""
    reasons = []
    sl, od = s["sl"], s["od"]

    for d_od in s["disk_od_list"]:
        if d_od <= 1.05 * od:
            reasons.append(f"H1: disk_OD {d_od:.4f} <= 1.05*OD")
            break

    for i in range(s["nd"] - 1):
        gap = s["dp"][i + 1] - s["dp"][i]
        min_gap = (s["disk_w_list"][i] + s["disk_w_list"][i + 1]) / 2.0 + 0.003
        if gap < min_gap:
            reasons.append("H2")
            break

    margin = H3_MARGIN_FRAC * sl
    for p in s["dp"] + s["bp"]:
        if p < margin or p > sl - margin:
            reasons.append("H3")
            break

    for dp in s["dp"]:
        for bp in s["bp"]:
            if abs(dp - bp) < H4_MIN_SEP:
                reasons.append("H4")
                break
        if reasons and reasons[-1] == "H4":
            break

    sd = sl / max(od, 1e-6)
    if sd <= 2:
        reasons.append(f"H5: {sd:.1f} <= 2")
    elif sd >= 50:
        reasons.append(f"H5: {sd:.1f} >= 50")

    if s["nb"] >= 2:
        kmin = min(s["bk_list"])
        kmax = max(s["bk_list"])
        ratio_limit = (1.0 + H6_PERT) / (1.0 - H6_PERT) + 1e-9
        if kmin <= 0 or kmax / max(kmin, 1e-12) > ratio_limit:
            reasons.append(f"H6: K ratio {kmax / max(kmin, 1e-12):.3f}")

    return len(reasons) == 0, reasons


def _volumes(s):
    shaft_v = np.pi * (s["od"] / 2.0) ** 2 * s["sl"]
    if s["id"] > 0:
        shaft_v -= np.pi * (s["id"] / 2.0) ** 2 * s["sl"]
    disk_v = sum(
        np.pi * (d_o / 2.0) ** 2 * d_w
        for d_o, d_w in zip(s["disk_od_list"], s["disk_w_list"])
    )
    return shaft_v, disk_v


def check_soft_v4(s):
    """Engineering-preference criteria S1-S3.

    These never reject a sample.  Each returns True when the sample follows
    conventional practice; the values are stored as labels in the dataset.
    """
    shaft_v, disk_v = _volumes(s)

    s1 = (disk_v / max(shaft_v, 1e-10)) < 20.0

    if s["bk_list"] and sum(s["bk_list"]) > 0:
        total_mass = (shaft_v + disk_v) * MATERIALS[s["material"]]["rho"]
        delta = total_mass * 9.81 / sum(s["bk_list"])
        s2 = delta < 0.1 * s["od"]
    else:
        delta = float("nan")
        s2 = False

    span = max(s["bp"]) - min(s["bp"]) if s["nb"] >= 2 else 0.0
    s3 = span > 0.4 * s["sl"]

    return {
        "S1_volume_ratio": bool(s1),
        "S2_support_sag": bool(s2),
        "S3_bearing_span": bool(s3),
        "volume_ratio": float(disk_v / max(shaft_v, 1e-10)),
        "support_sag_m": float(delta),
        "bearing_span_frac": float(span / max(s["sl"], 1e-12)),
    }


def is_valid_v4(s, hard=True, soft=False):
    """Backward-compatible wrapper.  Only hard constraints decide validity."""
    return check_hard_v4(s)


def filter_v4(samples, reject_counter=None):
    """Keep samples passing H1-H6; attach S1-S3 preference labels to them.

    Each rejected sample can violate several hard constraints.  When
    ``reject_counter`` is supplied, every violated H1-H6 category is counted.
    """
    valid, rej = [], 0
    for s in samples:
        ok, reasons = check_hard_v4(s)
        if ok:
            s["soft"] = check_soft_v4(s)
            valid.append(s)
        else:
            rej += 1
            if reject_counter is not None:
                for reason in reasons:
                    reject_counter[reason.split(":", 1)[0].strip()] += 1
    return valid, rej


def generate_filtered_samples(n, seed=42, max_rounds=20, batch_factor=2.0):
    """Iteratively generate and filter LHS candidates until n valid samples.

    This replaces the v4.3 one-shot 2.5x oversample + truncate behavior.
    Each round uses a seed derived from the base seed so the whole run is
    still reproducible.
    """
    valid = []
    reject_counter = Counter()
    prefilter_counter = Counter()
    stats = {
        "n_lhs_rows": 0,
        "n_candidates": 0,
        "n_valid": 0,
        "n_rejected": 0,
        "n_prefiltered_out": 0,
        "n_rounds": 0,
        "rejection_reasons": {},
        "prefilter_reasons": {},
    }
    n_rounds = 0
    while len(valid) < n and n_rounds < max_rounds:
        need = n - len(valid)
        batch = max(200, int(np.ceil(need * batch_factor)))
        round_seed = seed + n_rounds * 7919
        X = lhs_v4(batch, seed=round_seed)
        samples = X_to_samples_v4(
            X,
            rng=np.random.default_rng(round_seed + 1),
            prefilter_counter=prefilter_counter,
        )
        v, rej = filter_v4(samples, reject_counter)
        valid.extend(v)
        stats["n_lhs_rows"] += batch
        stats["n_candidates"] += len(samples)
        stats["n_prefiltered_out"] += batch - len(samples)
        stats["n_rejected"] += rej
        n_rounds += 1

    valid = valid[:n]
    stats["n_valid"] = len(valid)
    stats["n_rounds"] = n_rounds
    stats["rejection_reasons"] = dict(sorted(reject_counter.items()))
    stats["prefilter_reasons"] = dict(sorted(prefilter_counter.items()))
    return valid, stats


def to_features_v4(samples):
    """39 features: 5 shaft + 3 mat + 18 disk + 1 brg + 12 brg."""
    rows = []
    for s in samples:
        row = [s["sl"], s["od"], s["id"], s["nd"], s["nb"]]
        for m in MAT_LIST:
            row.append(1.0 if s["material"] == m else 0.0)
        for j in range(6):
            if j < s["nd"]:
                row += [s["disk_od_list"][j], s["disk_w_list"][j], s["dp"][j]]
            else:
                row += [0.0, 0.0, 0.0]
        row.append(0.0)
        for j in range(4):
            if j < s["nb"]:
                row += [s["bk_list"][j], s["bc_list"][j], s["bp"][j]]
            else:
                row += [0.0, 0.0, 0.0]
        rows.append(row)
    names = (
        ["L", "OD", "ID", "n_disks", "n_bearings"]
        + [f"mat_{m}" for m in MAT_LIST]
        + [f"d{j}_{a}" for j in range(6) for a in ["OD", "W", "pos"]]
        + ["brg_type"]
        + [f"b{j}_{a}" for j in range(4) for a in ["K", "C", "pos"]]
    )
    return np.array(rows), names


def print_sample_v4(s):
    print(f"L={s['sl']:.3f} OD={s['od']:.4f} ID={s['id']:.4f} | {s['material']}")
    print(f"  Disks ({s['nd']}):")
    for i in range(s["nd"]):
        print(
            f"    pos={s['dp'][i]:.3f} OD={s['disk_od_list'][i]:.4f} "
            f"W={s['disk_w_list'][i]:.4f}"
        )
    print(f"  Bearings ({s['nb']}):")
    for i in range(s["nb"]):
        print(
            f"    pos={s['bp'][i]:.3f} K={s['bk_list'][i]:.2e} "
            f"C={s['bc_list'][i]:.1f}"
        )


if __name__ == "__main__":
    valid, stats = generate_filtered_samples(500, seed=42)
    print(
        f"Candidates: {stats['n_candidates']}, Valid: {stats['n_valid']} "
        f"({100 * stats['n_valid'] / max(stats['n_candidates'], 1):.1f}%), "
        f"Rounds: {stats['n_rounds']}"
    )
    if valid:
        print_sample_v4(valid[0])
        Xf, names = to_features_v4(valid)
        print(f"Features: {Xf.shape[0]} x {Xf.shape[1]}")
