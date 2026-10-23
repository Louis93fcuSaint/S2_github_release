# -*- coding: utf-8 -*-
"""Post-generation filtering and sorting for S2 v2.

The generators and the ROSS verifier stay frozen.  This module only decides

  * which candidates are submitted to ROSS, and
  * in what order the delivered shortlist is shown.

Mass uses exactly the same shaft+disk volume formula as S1's
``s1_constraint_filter_v4_7._volumes``: a hollow shaft cylinder plus solid
disk cylinders, multiplied by the material density.
"""
from __future__ import annotations

import numpy as np

# Same densities as S1/04_versions/v4.7/s1_constraint_filter_v4_7.MATERIALS
MATERIAL_RHO = {"Steel": 7810.0, "Aluminum": 2700.0, "Titanium": 4430.0}

MASS_SORTS = ("mass_asc", "mass_desc")
CS_SORTS = tuple("cs%d_err" % i for i in range(1, 4))
SORT_KEYS = ("score",) + MASS_SORTS + CS_SORTS
SORT_HELP = ("score | mass_asc | mass_desc | cs1_err | cs2_err | cs3_err")

DISK_BASE = 3
MAX_DISKS = 6


def total_mass_kg(x, material, nd):
    """Canonical 33-vectors -> total shaft+disk mass in kg.

    ``nd`` is the number of used disk slots per row; unused slots are ignored
    even when the decoder leaves non-zero values there.
    """
    if material not in MATERIAL_RHO:
        raise ValueError("unknown material: %s" % material)
    rho = MATERIAL_RHO[material]
    x = np.atleast_2d(np.asarray(x, dtype=float))
    nd = np.atleast_1d(np.asarray(nd, dtype=int))
    if len(x) != len(nd):
        raise ValueError("x and nd length mismatch: %d vs %d" % (len(x), len(nd)))

    out = np.empty(len(x), dtype=float)
    for row in range(len(x)):
        L = max(float(x[row, 0]), 0.0)
        OD = max(float(x[row, 1]), 0.0)
        ID = max(float(x[row, 2]), 0.0)
        shaft_v = np.pi * (OD / 2.0) ** 2 * L
        if ID > 0:
            shaft_v -= np.pi * (ID / 2.0) ** 2 * L
        disk_v = 0.0
        for i in range(min(int(nd[row]), MAX_DISKS)):
            d_od = max(float(x[row, DISK_BASE + 3 * i]), 0.0)
            d_w = max(float(x[row, DISK_BASE + 3 * i + 1]), 0.0)
            disk_v += np.pi * (d_od / 2.0) ** 2 * d_w
        out[row] = max(shaft_v + disk_v, 0.0) * rho
    return out


def _check_sort(sort_by, specs):
    sort_by = (sort_by or "score").strip()
    if sort_by not in SORT_KEYS:
        raise SystemExit("[sort] 不认识的排序键 %r；可选：%s" % (sort_by, SORT_HELP))
    if sort_by in CS_SORTS:
        order = int(sort_by[2:-4])
        if order not in specs:
            raise SystemExit(
                "[sort] --sort-by %s 需要规格里包含 cs%d（当前规格：%s）"
                % (sort_by, order, ", ".join("cs%d" % o for o in sorted(specs))))
    return sort_by


def validate_sort(sort_by, specs):
    """Fail fast, before spending time on sampling."""
    return _check_sort(sort_by, specs)


def _ensure_mass(frame, material):
    if "mass_kg" in frame.columns:
        return frame["mass_kg"].astype(float)
    cols = ["L", "OD", "ID"]
    for i in range(MAX_DISKS):
        cols += ["d%d_OD" % i, "d%d_W" % i]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise SystemExit("[sort] 这一列缺了，算不出质量：%s" % ", ".join(missing))
    x = frame[["L", "OD", "ID"]
              + [c for i in range(MAX_DISKS)
                 for c in ("d%d_OD" % i, "d%d_W" % i, "d%d_pos" % i)]].to_numpy(float)
    return total_mass_kg(x, material, frame["n_disks"].to_numpy(int))


def sort_frame(frame, sort_by, material=None):
    """Sort an already-built CSV frame (proxy-only or ROSS-verified)."""
    sort_by = (sort_by or "score").strip()
    if sort_by not in SORT_KEYS:
        raise SystemExit("[sort] 不认识的排序键 %r；可选：%s" % (sort_by, SORT_HELP))

    if sort_by == "score":
        column = "joint_err_truth" if "joint_err_truth" in frame.columns else "joint_err"
        if column not in frame.columns:
            column = "joint_err"
        return frame.sort_values(column, kind="stable")

    if sort_by in MASS_SORTS:
        if material is None:
            material = str(frame["material"].iloc[0]) if "material" in frame.columns else None
        mass = _ensure_mass(frame, material)
        out = frame.copy()
        out["mass_kg"] = mass
        return out.sort_values("mass_kg", ascending=(sort_by == "mass_asc"), kind="stable")

    order = int(sort_by[2:-4])
    for column in ("cs%d_relerr_truth_pct" % order, "cs%d_dev_pct" % order):
        if column in frame.columns:
            out = frame.copy()
            out["_abs_err"] = out[column].abs()
            return out.sort_values("_abs_err", kind="stable").drop(columns="_abs_err")
    raise SystemExit("[sort] 这一列缺了，算不出 cs%d 误差" % order)


def describe(sort_by):
    return {
        "score": "代理综合偏差（越小越好）",
        "mass_asc": "总质量从小到大",
        "mass_desc": "总质量从大到小",
    }.get(sort_by, "cs%s 误差从小到大" % sort_by[2:-4])