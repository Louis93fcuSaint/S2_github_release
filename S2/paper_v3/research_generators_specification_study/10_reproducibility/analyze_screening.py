# -*- coding: utf-8 -*-
"""Stage 2b/4b: what surrogate pre-screening changes, with existing controls."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import research_common as C

OUT = C.ROOT / "04_generator_screening_analysis"
OUT.mkdir(parents=True, exist_ok=True)

runs = pd.read_csv(C.ROOT / "03_existing_candidate_analysis" / "four_way_runs_reaggregated.csv")
spec_tab = pd.read_csv(C.ROOT / "02_specification_difficulty" / "p_ref_all_specs.csv")

# 1. Main experiment: raw-pool proxy -> top-200 proxy -> ROSS delivery.
main = runs[["method", "method_label", "spec", "seed", "pool_proxy_in_band",
             "shortlist_proxy_in_band", "delivery_rate", "shortlist_truth_abs_pct_error_mean"]].copy()
main["screening_gain_pp"] = 100.0 * (main.shortlist_proxy_in_band - main.pool_proxy_in_band)
main["delivery_minus_pool_proxy_pp"] = 100.0 * (main.delivery_rate - main.pool_proxy_in_band)
main.to_csv(OUT / "four_way_proxy_funnel_by_run.csv", index=False, encoding="utf-8-sig")

main_summary = (main.groupby(["method", "method_label"])
                .agg(runs=("delivery_rate", "size"),
                     pool_proxy=("pool_proxy_in_band", "mean"),
                     shortlist_proxy=("shortlist_proxy_in_band", "mean"),
                     delivery=("delivery_rate", "mean"),
                     truth_mae=("shortlist_truth_abs_pct_error_mean", "mean"),
                     screening_gain_pp=("screening_gain_pp", "mean"))
                .reset_index())
main_summary.to_csv(OUT / "four_way_proxy_funnel_summary.csv", index=False, encoding="utf-8-sig")

# Rank changes by specification: raw-pool proxy rank vs delivered rank (4 methods).
rank_rows = []
for spec_name in C.SPEC_NAMES:
    sub = main[main.spec == spec_name].groupby("method").agg(
        pool_proxy=("pool_proxy_in_band", "mean"), delivery=("delivery_rate", "mean"))
    sub["raw_rank"] = sub.pool_proxy.rank(ascending=False, method="average")
    sub["delivery_rank"] = sub.delivery.rank(ascending=False, method="average")
    sub["rank_shift"] = sub.delivery_rank - sub.raw_rank
    sub["spec"] = spec_name
    rank_rows.append(sub.reset_index())
rank = pd.concat(rank_rows, ignore_index=True)
rank.to_csv(OUT / "method_rank_shift_after_screening.csv", index=False, encoding="utf-8-sig")
rank_corr = []
for spec_name, sub in rank.groupby("spec"):
    rho, p = stats.spearmanr(sub.raw_rank, sub.delivery_rank)
    rank_corr.append({"spec": spec_name, "spearman_rank": float(rho), "p": float(p),
                      "n_methods": len(sub)})
pd.DataFrame(rank_corr).to_csv(OUT / "method_rank_correlation_by_spec.csv", index=False, encoding="utf-8-sig")

# 2. Direct random-200 vs proxy-top-200 controls from v2 multi-order runs.
v2 = [
    ("cs1", {"1": 3680.8}, "Steel", None, 5000),
    ("cs2", {"2": 7021.0}, "Steel", None, 5000),
    ("cs3", {"3": 21033.5}, "Steel", None, 5000),
    ("cs1cs3", {"1": 3680.8, "3": 18070.4}, "Steel", None, 5000),
    ("cs2cs3", {"2": 7021.0, "3": 15857.7}, "Steel", None, 5000),
    ("cs1cs2", {"1": 3680.8, "2": 7021.0}, "Steel", None, 5000),
    ("cs123", {"1": 3680.8, "2": 7021.0, "3": 17970.5}, "Steel", None, 5000),
    ("cs1cs2_big", {"1": 3680.8, "2": 7021.0}, "Steel", None, 20000),
    ("cs123_big", {"1": 3680.8, "2": 7021.0, "3": 17970.5}, "Steel", None, 20000),
    ("al_cs1cs2", {"1": 3191.1, "2": 6147.6}, "Aluminum", None, 5000),
    ("fam32_cs1", {"1": 2974.0}, "Steel", "3-3x2-2", 5000),
    ("fam32_cs12", {"1": 2974.0, "2": 7001.0}, "Steel", "3-3x2-2", 5000),
    ("fam22_cs2", {"2": 8180.0}, "Steel", "2-2x2-2", 5000),
    ("fam12_24_cs12", {"1": 4642.0, "2": 9969.0}, "Steel", "1-2x2-4", 5000),
    ("fam32_cs123", {"1": 2974.0, "2": 7001.0, "3": 20359.0}, "Steel", "3-3x2-2", 20000),
]
rows = []
for label, targets, material, family, n_batch in v2:
    p = C.V2_OUT / f"verify_{label}.json"
    if not p.exists():
        continue
    obj = C.read_json(p)
    runs_obj = {r["label"].split("__")[-1]: r for r in obj.get("runs", [])}
    top = runs_obj.get("top200")
    rnd = runs_obj.get("rand200")
    if top is None or rnd is None:
        continue
    rows.append({
        "specification_id": "v2_" + label, "material": material,
        "family_condition": family or "free", "n_batch": n_batch,
        "pool_proxy_in_band": obj.get("proxy_whole_batch", {}).get("joint_in_band_tol5", np.nan),
        "top200_proxy_in_band": top.get("proxy", {}).get("joint_in_band_tol5", np.nan),
        "rand200_proxy_in_band": rnd.get("proxy", {}).get("joint_in_band_tol5", np.nan),
        "top200_delivered": top.get("gate_tol5", {}).get("n_passed", -1),
        "top200_submitted": top.get("n_shortlist", 200),
        "rand200_delivered": rnd.get("gate_tol5", {}).get("n_passed", -1),
        "rand200_submitted": rnd.get("n_shortlist", 200),
        "top200_truth_rate": top.get("truth", {}).get("joint_in_band_tol5", np.nan),
        "rand200_truth_rate": rnd.get("truth", {}).get("joint_in_band_tol5", np.nan),
        "top200_seconds": top.get("seconds", np.nan),
        "rand200_seconds": rnd.get("seconds", np.nan),
        "selector_note": "proxy-error rank top-200; not select_common",
        "seed": 42,
    })
ctrl = pd.DataFrame(rows)
ctrl["delivery_lift_pp"] = 100.0 * (ctrl.top200_truth_rate - ctrl.rand200_truth_rate)
ctrl["proxy_lift_pp"] = 100.0 * (ctrl.top200_proxy_in_band - ctrl.rand200_proxy_in_band)
ctrl["top200_precision_vs_random_ratio"] = ctrl.top200_truth_rate / ctrl.rand200_truth_rate.replace(0, np.nan)
ctrl.to_csv(OUT / "random200_vs_proxy_top200_controls.csv", index=False, encoding="utf-8-sig")

desc = {
    "n_conditions": int(len(ctrl)),
    "mean_pool_proxy": float(ctrl.pool_proxy_in_band.mean()),
    "mean_top200_proxy": float(ctrl.top200_proxy_in_band.mean()),
    "mean_random200_proxy": float(ctrl.rand200_proxy_in_band.mean()),
    "mean_top200_delivery": float(ctrl.top200_truth_rate.mean()),
    "mean_random200_delivery": float(ctrl.rand200_truth_rate.mean()),
    "mean_delivery_lift_pp": float(ctrl.delivery_lift_pp.mean()),
    "median_delivery_lift_pp": float(ctrl.delivery_lift_pp.median()),
    "n_positive_delivery_lift": int((ctrl.delivery_lift_pp > 0).sum()),
    "n_negative_delivery_lift": int((ctrl.delivery_lift_pp < 0).sum()),
    "pool_note": "v2 controls are point-conditioned DDPM and one seed; not a CVAE comparison",
    "bootstrap_ci_lift_pp": list(C.bootstrap_mean_ci(ctrl.delivery_lift_pp)),
}
with open(OUT / "screening_metadata.json", "w", encoding="utf-8") as handle:
    json.dump(desc, handle, ensure_ascii=False, indent=2)

# 3. Compare proxy-labelled selection to actual delivery across main runs.
corr_main = stats.spearmanr(main.pool_proxy_in_band, main.delivery_rate)
with open(OUT / "main_proxy_vs_delivery_correlation.txt", "w", encoding="utf-8") as handle:
    handle.write("Spearman rho between raw-pool proxy in-band rate and ROSS delivery across 72 runs: %.6f (p=%.6g)\n"
                 % (corr_main.correlation, corr_main.pvalue))
    handle.write("This is descriptive: runs share checkpoints, specifications and seeds and are not independent.\n")

print(main_summary.to_string(index=False))
print(ctrl[["specification_id", "top200_truth_rate", "rand200_truth_rate", "delivery_lift_pp"]].to_string(index=False))
print(json.dumps(desc, ensure_ascii=False, indent=2))
