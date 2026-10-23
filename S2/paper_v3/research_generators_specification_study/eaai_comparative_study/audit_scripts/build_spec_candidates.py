# -*- coding: utf-8 -*-
"""Build calibration / test specification candidates from the frozen v4.7 library.

Read-only with respect to every source asset. The script enumerates a small,
pre-defined grid of single-, two- and three-order specifications, counts their
empirical support in the reference library, and fills pre-declared slots by an
explicit deterministic rule. It never queries a generator or a ROSS run.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "spec_candidates"
DATASET = Path(
    r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划"
    r"\S1\04_versions\v4.7\dataset_v4.7\dataset_v4.7_all.csv"
)

FAMILIES = [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3), (6, 2), (6, 4)]
MATERIALS = ["Steel", "Aluminum", "Titanium"]
ORDERS = [1, 2, 3]
JOINT_SETS = [(1, 2), (1, 3), (2, 3), (1, 2, 3)]
POINT_TOL = 0.05
INTERVAL_HALF_LOG = 0.10
JOINT_INTERVAL_HALF_LOG = 0.08
INF_CAP = 1.0e12

# Pre-declared slots. `target_n` ranks candidates inside a slot; it is not a
# post-hoc model result.
SLOTS = [
    ("calibration", "CAL01", "Steel", "free", (1,), "point", "high", 2000),
    ("calibration", "CAL02", "Aluminum", "free", (1,), "interval", "high", 2000),
    ("calibration", "CAL03", "Titanium", "1d-3b", (1,), "onesided_q95", "medium", 200),
    ("calibration", "CAL04", "Steel", "free", (1, 2), "point", "medium", 200),
    ("calibration", "CAL05", "Aluminum", "6d-2b", (1, 3), "interval", "low", 20),
    ("calibration", "CAL06", "Titanium", "free", (1, 2, 3), "point", "low", 5),
    ("test", "TST01", "Steel", "free", (1,), "onesided_q95", "high", 3000),
    ("test", "TST02", "Aluminum", "3d-3b", (1,), "interval", "high", 2000),
    ("test", "TST03", "Titanium", "1d-3b", (1,), "onesided_q98", "low", 50),
    ("test", "TST04", "Steel", "6d-4b", (1, 2), "point", "low", 20),
    ("test", "TST05", "Aluminum", "6d-2b", (1, 2), "interval", "low", 20),
    ("test", "TST06", "Titanium", "3d-3b", (1, 3), "point", "low", 20),
    ("test", "TST07", "Steel", "free", (1, 2, 3), "point", "low", 5),
    ("test", "TST08", "Aluminum", "2d-3b", (1, 2, 3), "point", "zero", 0),
    ("test", "TST09", "Titanium", "free", (1, 2), "interval", "medium", 500),
]


def load_reference() -> pd.DataFrame:
    cols = ["status", "n_disks", "n_bearings", "mat_Steel", "mat_Aluminum",
            "mat_Titanium", "cs_1_rpm", "cs_2_rpm", "cs_3_rpm"]
    frame = pd.read_csv(DATASET, usecols=cols)
    frame = frame[frame["status"].eq("success")].copy()
    for order in ORDERS:
        frame = frame[np.isfinite(frame["cs_%d_rpm" % order])]
    frame["material"] = np.select(
        [frame["mat_Steel"].eq(1), frame["mat_Aluminum"].eq(1),
         frame["mat_Titanium"].eq(1)],
        MATERIALS, default="UNKNOWN")
    frame = frame[frame["material"].isin(MATERIALS)].reset_index(drop=True)
    frame["family"] = [f"{int(d)}d-{int(b)}b"
                       for d, b in zip(frame["n_disks"], frame["n_bearings"])]
    return frame


def support_mask(frame: pd.DataFrame, material: str, family: str,
                 bands: dict[int, tuple[float, float]]) -> np.ndarray:
    mask = frame["material"].eq(material).to_numpy()
    if family != "free":
        mask &= frame["family"].eq(family).to_numpy()
    for order, (lo, hi) in bands.items():
        values = frame["cs_%d_rpm" % order].to_numpy()
        mask &= (values >= lo) & (values <= min(hi, INF_CAP))
    return mask


def make_bands(frame: pd.DataFrame, material: str, family: str,
               orders: tuple[int, ...], kind: str) -> tuple[dict, dict]:
    base = frame[frame["material"].eq(material)]
    if family != "free":
        base = base[base["family"].eq(family)]
    targets, bands = {}, {}
    for order in orders:
        values = base["cs_%d_rpm" % order].to_numpy()
        q50, q75, q95, q98 = np.percentile(values, [50, 75, 95, 98])
        if kind == "point":
            target = float(q50)
            bands[order] = (target * (1 - POINT_TOL), target * (1 + POINT_TOL))
        elif kind == "interval":
            h = JOINT_INTERVAL_HALF_LOG if len(orders) > 1 else INTERVAL_HALF_LOG
            target = float(q50)
            bands[order] = (target * math.exp(-h), target * math.exp(h))
        elif kind == "onesided_q75":
            target = float(q75)
            bands[order] = (target, INF_CAP)
        elif kind == "onesided_q95":
            target = float(q95)
            bands[order] = (target, INF_CAP)
        elif kind == "onesided_q98":
            target = float(q98)
            bands[order] = (target, INF_CAP)
        else:
            raise ValueError(kind)
        targets[order] = target
    return targets, bands


def enumerate_grid(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for material in MATERIALS:
        n_ref = int(frame["material"].eq(material).sum())
        for family in ["free"] + [f"{d}d-{b}b" for d, b in FAMILIES]:
            fam_n = int((frame["material"].eq(material)
                         & frame["family"].eq(family)).sum()) if family != "free" else n_ref
            if fam_n < 20:
                continue
            for orders in [(o,) for o in ORDERS]:
                for kind in ("point", "interval", "onesided_q75", "onesided_q95",
                             "onesided_q98"):
                    rows.append((material, family, orders, kind, n_ref, fam_n))
            for orders in JOINT_SETS:
                for kind in ("point", "interval"):
                    rows.append((material, family, orders, kind, n_ref, fam_n))
    out = []
    for material, family, orders, kind, n_ref, fam_n in rows:
        try:
            targets, bands = make_bands(frame, material, family, orders, kind)
        except (ValueError, IndexError):
            continue
        mask = support_mask(frame, material, family, bands)
        n_feasible = int(mask.sum())
        out.append({
            "material": material,
            "family": family,
            "family_n": fam_n,
            "orders": "|".join(str(o) for o in orders),
            "kind": kind,
            "targets": json.dumps({str(o): round(targets[o], 6) for o in orders},
                                  ensure_ascii=False),
            "bands": json.dumps({str(o): [round(bands[o][0], 6),
                                           round(min(bands[o][1], INF_CAP), 6)]
                                  for o in orders}, ensure_ascii=False),
            "n_feasible": n_feasible,
            "n_ref": n_ref,
            "p_ref": n_feasible / n_ref if n_ref else float("nan"),
        })
    grid = pd.DataFrame(out).drop_duplicates(
        subset=["material", "family", "orders", "targets", "bands"]).reset_index(drop=True)
    grid["support_class"] = np.select(
        [grid["n_feasible"].eq(0), grid["n_feasible"].lt(100),
         grid["n_feasible"].lt(1000)],
        ["zero", "low", "medium"], default="high")
    grid["spec_key"] = [hashlib.sha1(
        ("|".join([r.material, r.family, r.orders, r.kind, r.targets]))
        .encode("utf-8")).hexdigest()[:10] for r in grid.itertuples()]
    return grid


def select_slots(grid: pd.DataFrame) -> pd.DataFrame:
    used = set()
    chosen = []
    for role, slot_id, material, family, orders, kind, support_class, target_n in SLOTS:
        key_orders = "|".join(str(o) for o in orders)
        cand = grid[
            grid["material"].eq(material)
            & grid["family"].eq(family)
            & grid["orders"].eq(key_orders)
            & grid["kind"].eq(kind)
            & grid["support_class"].eq(support_class)
            & ~grid["spec_key"].isin(used)
        ].copy()
        if cand.empty:
            order = ["high", "medium", "low", "zero"]
            start = order.index(support_class)
            for alt in order[start:] + order[:start]:
                cand = grid[
                    grid["material"].eq(material)
                    & grid["family"].eq(family)
                    & grid["orders"].eq(key_orders)
                    & grid["kind"].eq(kind)
                    & grid["support_class"].eq(alt)
                    & ~grid["spec_key"].isin(used)
                ].copy()
                if not cand.empty:
                    break
        if cand.empty:
            raise RuntimeError("no candidate for slot %s" % slot_id)
        cand["distance"] = (cand["n_feasible"] - target_n).abs()
        cand = cand.sort_values(["distance", "spec_key"]).reset_index(drop=True)
        row = cand.iloc[0].to_dict()
        row.update({"role": role, "slot_id": slot_id,
                    "support_class_target": support_class,
                    "target_n": target_n})
        used.add(row["spec_key"])
        chosen.append(row)
    return pd.DataFrame(chosen)


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_reference()
    grid = enumerate_grid(frame)
    selected = select_slots(grid)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    selected["freeze_timestamp"] = stamp
    selected["freeze_note"] = (
        "support counted on the frozen v4.7 reference library only; "
        "no generator or ROSS run was executed before this freeze"
    )
    selected["spec_id"] = [
        "%s_%s_%s_%s" % (r.slot_id, r.material[:2].upper(), r.family.replace("-", ""),
                         r.orders.replace("|", ""))
        for r in selected.itertuples()
    ]
    cols = ["spec_id", "role", "slot_id", "material", "family", "orders", "kind",
            "targets", "bands", "n_feasible", "n_ref", "p_ref", "support_class",
            "support_class_target", "family_n", "target_n", "spec_key",
            "freeze_timestamp", "freeze_note"]
    selected = selected[cols].sort_values(["role", "slot_id"]).reset_index(drop=True)
    cal = selected[selected["role"].eq("calibration")].reset_index(drop=True)
    test = selected[selected["role"].eq("test")].reset_index(drop=True)
    cal.to_csv(OUT / "calibration_specs.csv", index=False, encoding="utf-8-sig")
    test.to_csv(OUT / "test_specs.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT / "all_candidate_specs.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(OUT / "enumerated_support_grid.csv", index=False, encoding="utf-8-sig")

    hashes = []
    for name in ("calibration_specs.csv", "test_specs.csv", "all_candidate_specs.csv"):
        path = OUT / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.append({"file": name, "bytes": path.stat().st_size, "sha256": digest})
    pd.DataFrame(hashes).to_csv(OUT / "SHA256SUMS.csv", index=False, encoding="utf-8-sig")

    summary = {
        "dataset": str(DATASET),
        "n_reference_rows": int(len(frame)),
        "n_grid": int(len(grid)),
        "n_selected": int(len(selected)),
        "freeze_timestamp": stamp,
        "elapsed_seconds": round(time.time() - started, 2),
        "hashes": hashes,
    }
    (OUT / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(selected[["spec_id", "material", "family", "orders", "kind",
                    "n_feasible", "p_ref", "support_class"]].to_string(index=False))


if __name__ == "__main__":
    main()
