# -*- coding: utf-8 -*-
"""Recompute TST geometry acceptance and accepted-set diversity (read-only).

Inputs are the frozen TST shortlists and ROSS slot records in pascal_revision.
No design is regenerated, no ROSS solve is launched, and no source file is
modified. The two outputs are written to the chapter4_revision folder.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2")
OUT = BASE / "paper" / "paper_v3" / "chapitre4_revision"
PASCAL = BASE / "paper" / "paper_v3" / "pascal_revision"
STUDY = BASE / "paper" / "paper_v3" / "research_generators_specification_study"
sys.path.insert(0, str(STUDY / "10_reproducibility"))
sys.path.insert(0, str(PASCAL / "scripts"))

import research_common as C  # noqa: E402
from geometry_audit import geometry_metrics  # noqa: E402

SPEC_CSV = STUDY / "eaai_comparative_study" / "spec_candidates" / "test_specs.csv"
SHORTLIST_ROOT = PASCAL / "tst_runs" / "02_shortlists"
ROSS_SLOTS = PASCAL / "tst_runs" / "03_ross" / "tst_ross_all_slots.csv"
REF_SCALES = STUDY / "05_valid_diversity" / "reference_scales.npy"
DATA_CSV = C.DATA_CSV

SPECS = ["TST01", "TST02", "TST04", "TST07", "TST09"]
METHODS = ["ddpm", "cvae", "retrieval"]
METHOD_LABEL = {"ddpm": "DDPM (interval-conditioned)",
                "cvae": "CVAE", "retrieval": "Retrieval+jitter"}
SEED = 42
N_SUBMITTED = 200
N_RESAMPLES = 50
MIN_EQUAL_N = 5


def _json(value):
    if isinstance(value, dict):
        return value
    return json.loads(value)


def _band_text(bands, orders):
    pieces = []
    for order in [int(v) for v in str(orders).split("|")]:
        lo, hi = bands[str(order)]
        if hi >= 1e10:
            pieces.append("cs%d >= %.3f" % (order, lo))
        elif abs(hi - lo) < 1e-9:
            pieces.append("cs%d = %.3f" % (order, lo))
        else:
            pieces.append("cs%d in [%.3f, %.3f]" % (order, lo, hi))
    return "; ".join(pieces)


def _condition_type(kind):
    kind = str(kind)
    if kind.startswith("onesided"):
        return "one-sided"
    if kind == "interval":
        return "two-sided interval"
    if kind == "point":
        return "point tolerance (joint)" if "123" in kind else "point tolerance"
    return kind


def _entropy(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[vals > 0]
    if len(vals) == 0:
        return float("nan")
    p = vals / vals.sum()
    return float(-(p * np.log(p)).sum())


def _nearest_reference_for_frame(frame, ref, scales):
    """Nearest same-material, same-family reference distance, active slots only."""
    out = np.full(len(frame), np.nan, dtype=float)
    for (material, nd, nb), qframe in frame.groupby(["material", "n_disks", "n_bearings"], sort=False):
        rsub = ref[(ref["material"] == material)
                   & (ref["n_disks"].astype(int) == int(nd))
                   & (ref["n_bearings"].astype(int) == int(nb))]
        if len(rsub) == 0:
            continue
        idx = np.asarray(C.active_indices(int(nd), int(nb)), dtype=int)
        q = qframe[C.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)[:, idx] / scales[idx]
        r = rsub[C.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)[:, idx] / scales[idx]
        qnorm = (q * q).sum(axis=1)
        rnorm = (r * r).sum(axis=1)
        nearest = np.full(len(q), np.inf)
        for start in range(0, len(r), 20000):
            rr = r[start:start + 20000]
            d2 = qnorm[:, None] + rnorm[None, start:start + 20000] - 2.0 * q @ rr.T
            nearest = np.minimum(nearest, np.sqrt(np.maximum(d2, 0.0)).min(axis=1))
        out[qframe.index.to_numpy()] = nearest
    return out


def main():
    specs = pd.read_csv(SPEC_CSV).set_index("slot_id")
    slots = pd.read_csv(ROSS_SLOTS)
    slots = slots[(slots["seed"].astype(int) == SEED)
                  & (slots["list_type"] == "shortlist")
                  & (slots["spec"].isin(SPECS))].copy()
    slots["ross_pass"] = slots["delivered"].astype(str).str.lower().eq("true")

    ref_cols = ["status", "mat_Steel", "mat_Aluminum", "mat_Titanium",
                "n_disks", "n_bearings"] + C.CANONICAL_COLUMNS
    ref = pd.read_csv(DATA_CSV, usecols=ref_cols)
    ref["material"] = C.material_series(ref)
    ref = ref[ref["status"] == "success"].copy()
    scales = np.load(REF_SCALES)

    geometry_rows = []
    accepted_all = []
    for spec_id in SPECS:
        spec = specs.loc[spec_id]
        bands = _json(spec["bands"])
        for method in METHODS:
            pool_id = "%s_%s_s%d" % (spec_id, method, SEED)
            short = pd.read_csv(SHORTLIST_ROOT / pool_id / "shortlist.csv")
            short.insert(0, "pool_id", pool_id)
            ross = slots[(slots["pool_id"] == pool_id)][
                ["pool_id", "candidate_id", "status", "ross_pass"]]
            merged = short.merge(ross, on=["pool_id", "candidate_id"], how="left", validate="one_to_one")
            if len(merged) != N_SUBMITTED:
                raise RuntimeError("%s has %d shortlist rows, expected %d" %
                                   (pool_id, len(merged), N_SUBMITTED))
            merged["ross_pass"] = merged["ross_pass"].fillna(False).astype(bool)
            missing_ross = merged["status"].isna()
            geo = geometry_metrics(merged)
            merged["geometry_pass"] = geo["pass_edge_10mm"].to_numpy(dtype=bool)
            merged["h4_pass"] = geo["pass_H4_center_10mm"].to_numpy(dtype=bool)
            merged["edge_clearance_m"] = geo["edge_clearance_min_m"].to_numpy(dtype=float)
            merged["ross_and_geometry_pass"] = merged["ross_pass"] & merged["geometry_pass"]
            merged["method"] = method
            merged["spec"] = spec_id
            accepted = merged[merged["ross_and_geometry_pass"]].copy()
            accepted_all.append(accepted)

            ross_n = int(merged["ross_pass"].sum())
            geom_n = int(merged["geometry_pass"].sum())
            both_n = int(merged["ross_and_geometry_pass"].sum())
            geometry_rows.append({
                "row_type": "specification",
                "specification": spec_id,
                "material": spec["material"],
                "family": spec["family"],
                "orders": spec["orders"],
                "condition_type": _condition_type(spec["kind"]),
                "bands": _band_text(bands, spec["orders"]),
                "reference_matches": int(spec["n_feasible"]),
                "method": method,
                "method_label": METHOD_LABEL[method],
                "submitted": N_SUBMITTED,
                "ross_pass": ross_n,
                "geometry_pass": geom_n,
                "ross_and_geometry_pass": both_n,
                "h4_pass": int(merged["h4_pass"].sum()),
                "ross_error_or_timeout": int((~missing_ross & (merged["status"] != "ok")).sum()),
                "missing_ross_record": int(missing_ross.sum()),
                "ross_specification_satisfaction_rate": ross_n / float(N_SUBMITTED),
                "edge_10mm_geometry_rate": geom_n / float(N_SUBMITTED),
                "joint_dynamic_geometric_acceptance_rate": both_n / float(N_SUBMITTED),
                "geometry_retention_given_ross": (both_n / float(ross_n)) if ross_n else float("nan"),
            })

    geo_df = pd.DataFrame(geometry_rows)
    macro_rows = []
    for method in METHODS:
        g = geo_df[(geo_df["row_type"] == "specification") & (geo_df["method"] == method)]
        macro_rows.append({
            "row_type": "macro",
            "specification": "MACRO",
            "material": "", "family": "", "orders": "", "condition_type": "", "bands": "",
            "reference_matches": np.nan,
            "method": method, "method_label": METHOD_LABEL[method],
            "submitted": int(g["submitted"].sum()),
            "ross_pass": int(g["ross_pass"].sum()),
            "geometry_pass": int(g["geometry_pass"].sum()),
            "ross_and_geometry_pass": int(g["ross_and_geometry_pass"].sum()),
            "h4_pass": int(g["h4_pass"].sum()),
            "ross_error_or_timeout": int(g["ross_error_or_timeout"].sum()),
            "missing_ross_record": int(g["missing_ross_record"].sum()),
            "ross_specification_satisfaction_rate": float(g["ross_specification_satisfaction_rate"].mean()),
            "edge_10mm_geometry_rate": float(g["edge_10mm_geometry_rate"].mean()),
            "joint_dynamic_geometric_acceptance_rate": float(g["joint_dynamic_geometric_acceptance_rate"].mean()),
            "geometry_retention_given_ross": (float(g["ross_and_geometry_pass"].sum())
                                              / max(float(g["ross_pass"].sum()), 1.0)),
        })
    geo_out = pd.concat([geo_df, pd.DataFrame(macro_rows)], ignore_index=True)
    geo_out.to_csv(OUT / "tst_geometry_acceptance.csv", index=False, encoding="utf-8-sig")

    accepted = pd.concat(accepted_all, ignore_index=True)
    accepted["reference_nn_active_standardized"] = _nearest_reference_for_frame(accepted, ref, scales)
    accepted_ids = accepted[["spec", "method", "candidate_id", "design_key", "material",
                             "n_disks", "n_bearings", "ross_pass", "geometry_pass",
                             "ross_and_geometry_pass", "edge_clearance_m",
                             "reference_nn_active_standardized"]].copy()
    accepted_ids.to_csv(OUT / "accepted_design_manifest.csv", index=False, encoding="utf-8-sig")

    diversity_rows = []
    for spec_id in SPECS:
        frames = {}
        for method in METHODS:
            frames[method] = accepted[(accepted["spec"] == spec_id)
                                      & (accepted["method"] == method)].copy()
        counts = {m: len(frames[m]) for m in METHODS}
        n_eq = min(counts.values())
        distance_ok = n_eq >= MIN_EQUAL_N
        for method in METHODS:
            f = frames[method]
            fam_counts = f.groupby(["n_disks", "n_bearings"]).size() if len(f) else pd.Series(dtype=int)
            exact_unique = len(set(C.exact_design_keys(f))) if len(f) else 0
            nn_vals, pair_vals, ref_vals, dist_n = [], [], [], []
            if distance_ok:
                rng = np.random.default_rng(20261023 + 100 * SPECS.index(spec_id)
                                            + 10 * METHODS.index(method))
                for rep in range(N_RESAMPLES):
                    idx = np.sort(rng.choice(len(f), size=n_eq, replace=False))
                    sub = f.iloc[idx].reset_index(drop=True)
                    nn, pair, n_dist = C.same_family_nn_and_pair(
                        sub, scales, max_points=n_eq, seed=rep)
                    nn_vals.append(nn)
                    pair_vals.append(pair)
                    dist_n.append(n_dist)
                    ref_vals.append(float(np.median(
                        sub["reference_nn_active_standardized"].to_numpy(dtype=float))))
            diversity_rows.append({
                "row_type": "specification",
                "specification": spec_id,
                "method": method,
                "method_label": METHOD_LABEL[method],
                "n_accepted": int(len(f)),
                "n_exact_unique": int(exact_unique),
                "n_families": int(len(fam_counts)),
                "family_entropy": _entropy(fam_counts.to_numpy(dtype=float)),
                "dominant_family_share": (float(fam_counts.max() / fam_counts.sum())
                                          if len(fam_counts) else float("nan")),
                "common_sample_n": int(n_eq),
                "distance_status": "estimated" if distance_ok else "insufficient_common_sample",
                "median_nn_same_family_active": (float(np.mean(nn_vals)) if nn_vals else float("nan")),
                "median_pair_same_family_active": (float(np.mean(pair_vals)) if pair_vals else float("nan")),
                "median_nn_reference_same_family_active": (float(np.mean(ref_vals)) if ref_vals else float("nan")),
                "distance_resamples": int(len(nn_vals)),
                "distance_sample_n_mean": (float(np.mean(dist_n)) if dist_n else np.nan),
                "notes": ("50 equal-size subsamples; mean of medians" if distance_ok
                          else "common sample size below 5; distances not estimated"),
            })

    div_spec = pd.DataFrame(diversity_rows)
    div_macro = []
    for method in METHODS:
        g = div_spec[div_spec["method"] == method]
        est = g[g["distance_status"] == "estimated"]
        div_macro.append({
            "row_type": "macro",
            "specification": "MACRO",
            "method": method,
            "method_label": METHOD_LABEL[method],
            "n_accepted": int(g["n_accepted"].sum()),
            "n_exact_unique": int(g["n_exact_unique"].sum()),
            "n_families": float(g["n_families"].mean()),
            "family_entropy": float(g["family_entropy"].mean()),
            "dominant_family_share": float(g["dominant_family_share"].mean()),
            "common_sample_n": np.nan,
            "distance_status": "macro_over_%d_specs" % len(est),
            "median_nn_same_family_active": float(est["median_nn_same_family_active"].mean()) if len(est) else np.nan,
            "median_pair_same_family_active": float(est["median_pair_same_family_active"].mean()) if len(est) else np.nan,
            "median_nn_reference_same_family_active": float(est["median_nn_reference_same_family_active"].mean()) if len(est) else np.nan,
            "distance_resamples": int(N_RESAMPLES) if len(est) else 0,
            "distance_sample_n_mean": float(est["distance_sample_n_mean"].mean()) if len(est) else np.nan,
            "notes": "macro mean over specifications with estimable distances; no composite score",
        })
    div_out = pd.concat([div_spec, pd.DataFrame(div_macro)], ignore_index=True)
    div_out.to_csv(OUT / "accepted_set_diversity.csv", index=False, encoding="utf-8-sig")

    print(geo_out.to_string(index=False))
    print("\nDiversity:\n", div_out.to_string(index=False))


if __name__ == "__main__":
    main()
