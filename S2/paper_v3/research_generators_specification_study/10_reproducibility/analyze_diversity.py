# -*- coding: utf-8 -*-
"""Stage 3: raw-pool, shortlist and ROSS-delivered diversity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_common as C

OUT = C.ROOT / "05_valid_diversity"
OUT.mkdir(parents=True, exist_ok=True)

ref_cols = ["status", "mat_Steel", "mat_Aluminum", "mat_Titanium", "n_disks", "n_bearings",
            "cs_1_rpm"] + C.CANONICAL_COLUMNS
ref = pd.read_csv(C.DATA_CSV, usecols=ref_cols)
ref["material"] = C.material_series(ref)
ref = ref[ref.status == "success"].copy()
scales = np.maximum(ref[C.CANONICAL_COLUMNS].to_numpy(float).std(axis=0), 1e-9)
np.save(OUT / "reference_scales.npy", scales)
pd.Series(scales, index=C.CANONICAL_COLUMNS).to_csv(OUT / "reference_scales.csv", header=["scale"])

# A deterministic, stratified-by-material sample of the reference library is not
# used for the distance itself; the full reference is.  This is only for audit.

def layer_metrics(frame, layer, method, spec_name, seed):
    if len(frame) == 0:
        return None
    keys = C.exact_design_keys(frame)
    unique_exact = len(set(keys))
    nn, pair, n_dist = C.same_family_nn_and_pair(frame, scales, max_points=1000)
    nd = C.near_duplicate_fraction(frame, scales, max_points=1000)
    nov = C.nearest_reference_distance(frame, ref, scales, max_queries=100 if layer != "shortlist" else 150)
    fams = frame.groupby(["n_disks", "n_bearings"]).size()
    p = fams.to_numpy(float) / fams.sum()
    return {
        "method": method, "method_label": C.METHOD_LABELS[method], "spec": spec_name,
        "seed": seed, "layer": layer, "n": int(len(frame)), "n_exact_unique": unique_exact,
        "exact_unique_rate": unique_exact / len(frame),
        "family_count": int(len(fams)), "family_entropy": float(-(p * np.log(p)).sum()),
        "family_effective_count": float(np.exp(-(p * np.log(p)).sum())),
        "dominant_family_share": float(p.max()),
        "median_nn_same_family": nn, "median_pair_same_family": pair,
        "distance_sample_n": n_dist,
        "novelty_median_to_reference": float(np.median(nov)),
        "novelty_p10_to_reference": float(np.quantile(nov, 0.10)),
        "novelty_sample_n": int(len(nov)),
        **nd,
    }

rows = []
for method in C.METHODS:
    for spec_name in C.SPEC_NAMES:
        for seed in C.SEEDS:
            pool_p, short_p, ver_p, csv_p, passed_p = C.method_paths(method, spec_name, seed)
            pool = pd.read_csv(pool_p)
            short = pd.read_csv(short_p)
            passed = pd.read_csv(passed_p)
            for frame, layer in ((pool, "raw_pool"), (short, "surrogate_top200"), (passed, "ROSS_delivered")):
                rec = layer_metrics(frame, layer, method, spec_name, seed)
                if rec is not None:
                    rows.append(rec)

metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "diversity_by_run_layer.csv", index=False, encoding="utf-8-sig")

# Summaries by method and layer (descriptive; no composite ranking).
summary = (metrics.groupby(["method", "layer"])
           .agg(runs=("n", "size"), n_mean=("n", "mean"), n_total=("n", "sum"),
                exact_unique_rate=("exact_unique_rate", "mean"),
                family_count=("family_count", "mean"),
                family_entropy=("family_entropy", "mean"),
                dominant_family_share=("dominant_family_share", "mean"),
                median_nn_same_family=("median_nn_same_family", "mean"),
                median_pair_same_family=("median_pair_same_family", "mean"),
                novelty_median=("novelty_median_to_reference", "mean"),
                near_dup_frac_001=("near_dup_frac_le_0.01", "mean"),
                near_dup_frac_005=("near_dup_frac_le_0.05", "mean"))
           .reset_index())
summary.to_csv(OUT / "diversity_summary.csv", index=False, encoding="utf-8-sig")

# Threshold sensitivity: spread across runs and methods.
thr_cols = [c for c in metrics.columns if c.startswith("near_dup_frac_le_")]
thr = metrics.melt(id_vars=["method", "layer", "spec", "seed"], value_vars=thr_cols,
                   var_name="threshold_definition", value_name="fraction")
thr["threshold"] = thr.threshold_definition.str.replace("near_dup_frac_le_", "", regex=False).astype(float)
thr_summary = (thr.groupby(["method", "layer", "threshold"])
               .agg(mean_fraction=("fraction", "mean"), median_fraction=("fraction", "median"),
                    min_fraction=("fraction", "min"), max_fraction=("fraction", "max"),
                    runs=("fraction", "size")).reset_index())
thr_summary.to_csv(OUT / "near_duplicate_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")

# Equal-sample-size comparison across the four methods for each specification.
def combined(method, spec_name, layer):
    frames = []
    for seed in C.SEEDS:
        pool_p, short_p, ver_p, csv_p, passed_p = C.method_paths(method, spec_name, seed)
        p = {"raw_pool": pool_p, "surrogate_top200": short_p, "ROSS_delivered": passed_p}[layer]
        f = pd.read_csv(p)
        f["_seed"] = seed
        frames.append(f)
    return pd.concat(frames, ignore_index=True)

rng = np.random.default_rng(20261021)
eq_rows = []
for layer in ["surrogate_top200", "ROSS_delivered"]:
    for spec_name in C.SPEC_NAMES:
        frames = {m: combined(m, spec_name, layer) for m in C.METHODS}
        n_min = min(len(f) for f in frames.values())
        if n_min < 2:
            continue
        for rep in range(100):
            for method, frame in frames.items():
                idx = rng.choice(len(frame), n_min, replace=False)
                sub = frame.iloc[np.sort(idx)]
                nn, pair, _ = C.same_family_nn_and_pair(sub, scales, max_points=n_min, seed=rep)
                keys = C.exact_design_keys(sub)
                eq_rows.append({"layer": layer, "spec": spec_name, "method": method,
                                "n_equalized": n_min, "replicate": rep,
                                "median_nn_same_family": nn,
                                "median_pair_same_family": pair,
                                "exact_unique_rate": len(set(keys)) / len(sub),
                                "family_entropy": C.family_entropy(sub)})
eq = pd.DataFrame(eq_rows)
eq.to_csv(OUT / "diversity_equalized_replicates.csv", index=False, encoding="utf-8-sig")
eq_summary = (eq.groupby(["layer", "spec", "method", "n_equalized"])
              .agg(nn_mean=("median_nn_same_family", "mean"),
                   nn_low=("median_nn_same_family", lambda x: np.quantile(x, 0.025)),
                   nn_high=("median_nn_same_family", lambda x: np.quantile(x, 0.975)),
                   pair_mean=("median_pair_same_family", "mean"),
                   pair_low=("median_pair_same_family", lambda x: np.quantile(x, 0.025)),
                   pair_high=("median_pair_same_family", lambda x: np.quantile(x, 0.975)),
                   exact_unique_rate=("exact_unique_rate", "mean"),
                   family_entropy=("family_entropy", "mean"))
              .reset_index())
eq_summary.to_csv(OUT / "diversity_equalized_summary.csv", index=False, encoding="utf-8-sig")

# Per-spec method ranking is descriptive only.
wins = (eq_summary[eq_summary.layer == "ROSS_delivered"]
        .sort_values(["spec", "nn_mean"], ascending=[True, False])
        .groupby("spec").head(1)
        .reset_index(drop=True))
wins.to_csv(OUT / "delivered_diversity_best_per_spec_descriptive.csv", index=False, encoding="utf-8-sig")

meta = {
    "distance_definition": "within-family standardized Euclidean distance on actively used slots only",
    "normalization": "per-dimension standard deviation of the full successful reference library",
    "unused_slots": "excluded; zero-filled slots are never counted as design coordinates",
    "cross_family": "not merged into one Euclidean distance; family entropy and coverage are reported separately",
    "exact_duplicate": "same material, same n_disks/n_bearings, used canonical parameters equal to 12 decimals",
    "near_duplicate": "same family and active-slot standardized nearest-neighbour distance <= threshold",
    "novelty": "median nearest exact distance to the full reference library in 33-D standardized space",
    "novelty_subsampling": "100 candidate queries per run (150 for shortlists); full reference library",
    "equalization": "100 repeated samples at the smallest delivered-set size among four methods for each specification",
    "no_composite_score": True,
}
with open(OUT / "diversity_metadata.json", "w", encoding="utf-8") as handle:
    json.dump(meta, handle, ensure_ascii=False, indent=2)
print(summary.to_string(index=False))
print("equalized rows", len(eq_summary))
print(wins.to_string(index=False))
