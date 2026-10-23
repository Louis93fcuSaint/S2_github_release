# -*- coding: utf-8 -*-
"""Read-only geometry audit for the Pascal revision.

The audit distinguishes:
  * H4: axial distance between a bearing centre and a disk centre;
  * EDGE10: distance from the bearing centre to the nearest disk-body edge >= 10 mm;
  * NONOVL: bearing centre lies outside the disk-body axial extent (0 mm criterion);
  * DISK_END: disk body stays inside the shaft ends;
  * DISK_GAP: adjacent disk bodies do not overlap.
All lengths in the archived CSVs are metres.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划")
S2 = ROOT / "S2"
OUT = S2 / "paper" / "paper_v3" / "pascal_revision"
DATA = ROOT / "S1" / "04_versions" / "v4.7" / "dataset_v4.7" / "dataset_v4.7_all.csv"
RUNS = S2 / "paper" / "paper_v3" / "research_generators_specification_study" / "03_existing_candidate_analysis" / "four_way_runs_reaggregated.csv"
V2 = S2 / "spec_design_v2" / "outputs"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def geometry_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    n = len(frame)
    h4_min = np.full(n, np.inf)
    edge_min = np.full(n, np.inf)
    disk_end_ok = np.ones(n, dtype=bool)
    disk_gap_ok = np.ones(n, dtype=bool)
    disk_gap_min = np.full(n, np.inf)
    for i, row in enumerate(frame.itertuples(index=False)):
        L = float(row.L)
        a = int(row.n_disks)
        b = int(row.n_bearings)
        dpos = np.array([getattr(row, "d%d_pos" % j) for j in range(a)], dtype=float)
        dw = np.array([getattr(row, "d%d_W" % j) for j in range(a)], dtype=float)
        bpos = np.array([getattr(row, "b%d_pos" % j) for j in range(b)], dtype=float)
        if a and b:
            dist = np.abs(dpos[:, None] - bpos[None, :])
            h4_min[i] = float(dist.min())
            edge_min[i] = float((dist - 0.5 * dw[:, None]).min())
        else:
            h4_min[i] = np.nan
            edge_min[i] = np.nan
        disk_end_ok[i] = bool(np.all(dpos - 0.5 * dw >= -1e-9) and
                              np.all(dpos + 0.5 * dw <= L + 1e-9))
        if a > 1:
            gaps = np.diff(dpos) - 0.5 * (dw[:-1] + dw[1:])
            disk_gap_min[i] = float(gaps.min())
            disk_gap_ok[i] = bool(np.all(gaps >= -1e-9))
        else:
            disk_gap_min[i] = np.nan
            disk_gap_ok[i] = True
    out["h4_min_center_m"] = h4_min
    out["edge_clearance_min_m"] = edge_min
    out["pass_H4_center_10mm"] = h4_min >= 0.01
    out["pass_edge_10mm"] = edge_min >= 0.01
    out["pass_nonoverlap_0mm"] = edge_min >= 0.0
    out["disk_inside_shaft_ends"] = disk_end_ok
    out["adjacent_disk_gap_min_m"] = disk_gap_min
    out["adjacent_disk_nonoverlap"] = disk_gap_ok
    return out


def dataset_audit() -> tuple[pd.DataFrame, dict]:
    cols = ["status", "n_disks", "n_bearings", "L",
            "d0_W", "d0_pos", "d1_W", "d1_pos", "d2_W", "d2_pos",
            "d3_W", "d3_pos", "d4_W", "d4_pos", "d5_W", "d5_pos",
            "b0_pos", "b1_pos", "b2_pos", "b3_pos",
            "cs_1_rpm", "cs_2_rpm", "cs_3_rpm", "cs_4_rpm", "cs_5_rpm", "cs_6_rpm",
            "source", "source_row_id"]
    df = pd.read_csv(DATA, usecols=cols)
    geo = geometry_metrics(df)
    df = pd.concat([df, geo], axis=1)
    success = df["status"].eq("success")
    first3 = np.isfinite(df[["cs_1_rpm", "cs_2_rpm", "cs_3_rpm"]].to_numpy(dtype=float)).all(axis=1)
    six = np.isfinite(df[[f"cs_{k}_rpm" for k in range(1, 7)]].to_numpy(dtype=float)).all(axis=1)
    definitions = {
        "all_rows": int(len(df)),
        "status_success": int(success.sum()),
        "status_rejected_cs1": int(df["status"].eq("rejected_cs1").sum()),
        "success_and_first3_valid": int((success & first3).sum()),
        "all_six_labels_valid": int(six.sum()),
        "success_and_all_six_valid": int((success & six).sum()),
        "rejected_rows": df.loc[df["status"].ne("success"), ["source", "source_row_id", "status", "cs_1_rpm", "cs_2_rpm", "cs_3_rpm"]].to_dict("records"),
    }
    rows = []
    for label, mask in [
        ("all_rows", np.ones(len(df), dtype=bool)),
        ("status_success", success.to_numpy()),
        ("success_and_first3_valid", (success.to_numpy() & first3)),
        ("all_six_labels_valid", six),
    ]:
        sub = df.loc[mask]
        rows.append({
            "pool": label, "n": int(len(sub)),
            "pass_H4_center_10mm_n": int(sub["pass_H4_center_10mm"].sum()),
            "pass_H4_center_10mm_rate": float(sub["pass_H4_center_10mm"].mean()),
            "pass_edge_10mm_n": int(sub["pass_edge_10mm"].sum()),
            "pass_edge_10mm_rate": float(sub["pass_edge_10mm"].mean()),
            "pass_nonoverlap_0mm_n": int(sub["pass_nonoverlap_0mm"].sum()),
            "pass_nonoverlap_0mm_rate": float(sub["pass_nonoverlap_0mm"].mean()),
            "disk_inside_shaft_ends_n": int(sub["disk_inside_shaft_ends"].sum()),
            "disk_inside_shaft_ends_rate": float(sub["disk_inside_shaft_ends"].mean()),
            "adjacent_disk_nonoverlap_n": int(sub["adjacent_disk_nonoverlap"].sum()),
            "adjacent_disk_nonoverlap_rate": float(sub["adjacent_disk_nonoverlap"].mean()),
        })
    return df, {"definitions": definitions, "table": rows}


def case_audit() -> tuple[pd.DataFrame, dict]:
    case_specs = [
        ("Case A", V2 / "v2_cs1.csv", 138),
        ("Case B", V2 / "v2_int_mid_v2i.csv", 0),
        ("Case C", V2 / "v2_joint_cs12.csv", 0),
    ]
    rows = []
    for name, path, rank in case_specs:
        df = pd.read_csv(path)
        row = df.loc[df["rank"].eq(rank)].iloc[[0]].reset_index(drop=True)
        geo = geometry_metrics(row).iloc[0]
        rec = row.iloc[0]
        rows.append({
            "case": name, "source": str(path), "rank": int(rank),
            "material": rec["material"], "n_disks": int(rec["n_disks"]),
            "n_bearings": int(rec["n_bearings"]), "L_m": float(rec["L"]),
            **{k: (bool(geo[k]) if isinstance(geo[k], (bool, np.bool_)) else float(geo[k]))
               for k in geo.index if k not in ("adjacent_disk_nonoverlap",)},
            "adjacent_disk_nonoverlap": bool(geo["adjacent_disk_nonoverlap"]),
        })
    # Exact nearest disk-bearing pair for Case C.
    c = pd.read_csv(V2 / "v2_joint_cs12.csv").query("rank == 0").iloc[0]
    pairs = []
    for j in range(int(c.n_disks)):
        for k in range(int(c.n_bearings)):
            d = float(c[f"d{j}_pos"]); w = float(c[f"d{j}_W"]); b = float(c[f"b{k}_pos"])
            pairs.append((abs(d-b)-0.5*w, j, k, d, w, b, abs(d-b)))
    pairs.sort(key=lambda x: x[0])
    edge, j, k, d, w, b, center = pairs[0]
    details = {
        "case_C_nearest_pair": {
            "disk_index": j, "bearing_index": k,
            "disk_center_m": d, "disk_width_m": w,
            "disk_left_edge_m": d - 0.5*w, "disk_right_edge_m": d + 0.5*w,
            "bearing_center_m": b, "center_distance_m": center,
            "edge_clearance_m": edge,
            "H4_center_rule_margin_m": center - 0.01,
            "EDGE10_margin_m": edge - 0.01,
            "pass_H4": bool(center >= 0.01),
            "pass_EDGE10": bool(edge >= 0.01),
            "pass_NONOVL": bool(edge >= 0.0),
        }
    }
    return pd.DataFrame(rows), details


def historical_geometry_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(RUNS)
    rows = []
    for rec in runs.itertuples(index=False):
        verify = Path(str(rec.verify_csv))
        passed = Path(str(rec.passed_csv))
        if not verify.exists():
            rows.append({"method": rec.method, "spec": rec.spec, "seed": int(rec.seed),
                         "n_shortlist": np.nan, "n_delivered": np.nan,
                         "status": "missing_verify_csv", "verify_csv": str(verify)})
            continue
        df = pd.read_csv(verify)
        geo = geometry_metrics(df)
        delivered_rank = set()
        status = "ok"
        if passed.exists():
            p = pd.read_csv(passed)
            if "rank" in p.columns:
                delivered_rank = set(p["rank"].astype(int).tolist())
            else:
                status = "passed_without_rank"
        else:
            status = "missing_passed_csv"
        delivered = df["rank"].astype(int).isin(delivered_rank).to_numpy()
        rows.append({
            "method": rec.method, "method_label": rec.method_label, "spec": rec.spec,
            "seed": int(rec.seed), "n_records": int(len(df)), "n_shortlist": 200,
            "n_delivered": int(delivered.sum()),
            "n_H4_center_10mm": int(geo["pass_H4_center_10mm"].sum()),
            "n_edge_10mm": int(geo["pass_edge_10mm"].sum()),
            "n_nonoverlap_0mm": int(geo["pass_nonoverlap_0mm"].sum()),
            "n_delivered_and_edge_10mm": int((delivered & geo["pass_edge_10mm"].to_numpy()).sum()),
            "n_delivered_and_nonoverlap_0mm": int((delivered & geo["pass_nonoverlap_0mm"].to_numpy()).sum()),
            "delivery_rate": float(delivered.sum() / 200.0),
            "delivery_rate_after_edge10": float((delivered & geo["pass_edge_10mm"].to_numpy()).sum() / 200.0),
            "delivery_rate_after_nonoverlap": float((delivered & geo["pass_nonoverlap_0mm"].to_numpy()).sum() / 200.0),
            "status": status, "verify_csv": str(verify),
        })
    detail = pd.DataFrame(rows)
    summary = []
    for (method, label), g in detail.groupby(["method", "method_label"], sort=False):
        summary.append({
            "method": method, "method_label": label, "runs": int(len(g)),
            "submitted": int(g["n_shortlist"].sum()),
            "delivered": int(g["n_delivered"].sum()),
            "delivered_and_edge10": int(g["n_delivered_and_edge_10mm"].sum()),
            "delivered_and_nonoverlap": int(g["n_delivered_and_nonoverlap_0mm"].sum()),
            "pooled_delivery_rate": float(g["n_delivered"].sum() / g["n_shortlist"].sum()),
            "pooled_rate_after_edge10": float(g["n_delivered_and_edge_10mm"].sum() / g["n_shortlist"].sum()),
            "pooled_rate_after_nonoverlap": float(g["n_delivered_and_nonoverlap_0mm"].sum() / g["n_shortlist"].sum()),
            "edge10_retention_given_delivered": float(g["n_delivered_and_edge_10mm"].sum() / max(g["n_delivered"].sum(), 1)),
        })
    return detail, pd.DataFrame(summary)


def case_c_alternative() -> pd.DataFrame:
    verify = V2 / "verify_joint_cs12__top200.csv"
    passed = V2 / "verify_joint_cs12__top200__passed.csv"
    df = pd.read_csv(verify)
    geo = geometry_metrics(df)
    p = pd.read_csv(passed)
    delivered = df["rank"].astype(int).isin(set(p["rank"].astype(int)))
    out = df.loc[delivered].copy()
    out["edge_clearance_min_m"] = geo.loc[delivered, "edge_clearance_min_m"].to_numpy()
    out["h4_min_center_m"] = geo.loc[delivered, "h4_min_center_m"].to_numpy()
    out["pass_edge_10mm"] = geo.loc[delivered, "pass_edge_10mm"].to_numpy()
    out = out.sort_values(["pass_edge_10mm", "edge_clearance_min_m"], ascending=[False, False])
    return out


def main():
    out = OUT / "results"
    out.mkdir(parents=True, exist_ok=True)
    df, ds = dataset_audit()
    ds_df = pd.DataFrame(ds["table"])
    ds_df.to_csv(out / "dataset_geometry_rates.csv", index=False, encoding="utf-8")
    (out / "dataset_geometry_meta.json").write_text(
        json.dumps({"source": str(DATA), "sha256": sha256_file(DATA), **ds},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    cases, cdetails = case_audit()
    cases.to_csv(out / "case_ABC_geometry.csv", index=False, encoding="utf-8")

    hist_detail, hist_summary = historical_geometry_audit()
    hist_detail.to_csv(out / "historical_72runs_geometry.csv", index=False, encoding="utf-8")
    hist_summary.to_csv(out / "historical_geometry_summary.csv", index=False, encoding="utf-8")

    alt = case_c_alternative()
    alt.to_csv(out / "case_C_delivered_alternative_candidates.csv", index=False, encoding="utf-8")

    combined = pd.concat([
        ds_df.assign(audit_scope="reference_dataset", group=""),
        cases.assign(audit_scope="representative_case"),
        hist_summary.assign(audit_scope="historical_main_experiment", group=""),
    ], ignore_index=True, sort=False)
    combined.to_csv(OUT / "geometry_audit.csv", index=False, encoding="utf-8")

    cmeta = {
        "source": str(V2 / "v2_joint_cs12.csv"),
        "sha256": sha256_file(V2 / "v2_joint_cs12.csv"),
        "case_C": cdetails,
        "delivered_in_original_200": int(len(alt)),
        "delivered_and_edge10": int(alt["pass_edge_10mm"].sum()),
        "best_edge_candidate": (alt.iloc[0][["rank", "n_disks", "n_bearings", "edge_clearance_min_m"]].to_dict()
                                if len(alt) else None),
    }
    (out / "case_C_geometry_meta.json").write_text(
        json.dumps(cmeta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("dataset rows", len(df), "success", ds["definitions"]["status_success"])
    print(ds_df.to_string(index=False))
    print("\nCases:\n", cases.to_string(index=False))
    print("\nHistorical summary:\n", hist_summary.to_string(index=False))
    print("\nCase C nearest pair:", json.dumps(cdetails, indent=2, ensure_ascii=False))
    print("Case C delivered candidates passing edge10:", int(alt["pass_edge_10mm"].sum()), "/", len(alt))


if __name__ == "__main__":
    main()



