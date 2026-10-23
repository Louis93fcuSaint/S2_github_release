# -*- coding: utf-8 -*-
"""Stage 4: specification difficulty vs delivery/diversity and tolerance curves."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import research_common as C

OUT = C.ROOT / "02_specification_difficulty"
OUT.mkdir(parents=True, exist_ok=True)

specs = pd.read_csv(OUT / "frozen_specifications.csv").set_index("specification_id")
runs = pd.read_csv(C.ROOT / "03_existing_candidate_analysis" / "four_way_runs_reaggregated.csv")
div = pd.read_csv(C.ROOT / "05_valid_diversity" / "diversity_by_run_layer.csv")

# A. Main six: p_ref versus delivery and valid-set diversity.
main = (runs.groupby(["method", "spec"])
        .agg(delivery_rate=("delivery_rate", "mean"), delivered=("delivered_n_json", "sum"),
             submitted=("ross_submitted_n", "sum"))
        .reset_index())
main["p_ref"] = main.spec.map(specs.p_ref)
main["n_feasible"] = main.spec.map(specs.n_feasible)
main["support_class"] = main.spec.map(specs.support_class)
main.to_csv(OUT / "main_six_support_vs_delivery.csv", index=False, encoding="utf-8-sig")

corr_rows = []
for method, sub in main.groupby("method"):
    rho, p = stats.spearmanr(sub.p_ref, sub.delivery_rate)
    corr_rows.append({"method": method, "n_specs": len(sub), "spearman_rho": rho, "p": p,
                      "note": "exploratory; six specifications, not a fitted causal model"})
# Pooled 24 cells, but report spec-cluster caveat.
rho, p = stats.spearmanr(main.p_ref, main.delivery_rate)
corr_rows.append({"method": "all_24_method_spec_cells", "n_specs": len(main),
                  "spearman_rho": rho, "p": p,
                  "note": "cells are clustered within six specs; descriptive only"})
pd.DataFrame(corr_rows).to_csv(OUT / "p_ref_vs_delivery_spearman.csv", index=False, encoding="utf-8-sig")

# Diversity of delivered sets vs p_ref, using one row per method/spec (mean over seeds).
dv = (div[(div.layer == "ROSS_delivered")]
      .groupby(["method", "spec"])
      .agg(n_valid=("n", "mean"), nn=("median_nn_same_family", "mean"),
           pair=("median_pair_same_family", "mean"),
           family_entropy=("family_entropy", "mean"),
           novelty=("novelty_median_to_reference", "mean"))
      .reset_index())
dv["p_ref"] = dv.spec.map(specs.p_ref)
dv["n_feasible"] = dv.spec.map(specs.n_feasible)
dv.to_csv(OUT / "p_ref_vs_valid_diversity.csv", index=False, encoding="utf-8-sig")
div_corr = []
for method, sub in dv.groupby("method"):
    rho, p = stats.spearmanr(sub.p_ref, sub.nn)
    div_corr.append({"method": method, "metric": "median_nn_same_family", "rho": rho, "p": p,
                     "n_specs": len(sub)})
pd.DataFrame(div_corr).to_csv(OUT / "p_ref_vs_diversity_spearman.csv", index=False, encoding="utf-8-sig")

# B. Pairwise DDPM-CVAE differences at each specification.
pairs = []
for mode, a, b in [
    ("unguided", "ddpm_unguided", "cvae_pure"),
    ("guided", "ddpm_guided", "cvae_guided"),
    ("DDPM_guided_minus_unguided", "ddpm_guided", "ddpm_unguided"),
    ("CVAE_guided_minus_unguided", "cvae_guided", "cvae_pure"),
]:
    ra = main[main.method == a].set_index("spec").delivery_rate
    rb = main[main.method == b].set_index("spec").delivery_rate
    for spec_name in C.SPEC_NAMES:
        pairs.append({"comparison": mode, "spec": spec_name,
                      "rate_a": float(ra[spec_name]), "rate_b": float(rb[spec_name]),
                      "diff_pp": 100.0 * float(ra[spec_name] - rb[spec_name]),
                      "p_ref": float(specs.loc[spec_name, "p_ref"]),
                      "n_feasible": int(specs.loc[spec_name, "n_feasible"])})
pd.DataFrame(pairs).to_csv(OUT / "spec_pairwise_differences.csv", index=False, encoding="utf-8-sig")

# C. Multi-order point-DDPM experiments (not part of the main interval ranking).
V2_CONDITIONS = [
    ("cs1", 1, None), ("cs2", 1, None), ("cs3", 1, None),
    ("cs1cs3", 2, None), ("cs2cs3", 2, None), ("cs1cs2", 2, None),
    ("cs123", 3, None), ("cs1cs2_big", 2, None), ("cs123_big", 3, None),
    ("al_cs1cs2", 2, None), ("fam32_cs1", 1, "fixed"), ("fam32_cs12", 2, "fixed"),
    ("fam22_cs2", 1, "fixed"), ("fam12_24_cs12", 2, "range"),
    ("fam32_cs123", 3, "fixed"),
]
rows = []
for label, orders, famtype in V2_CONDITIONS:
    p = C.V2_OUT / f"verify_{label}.json"
    if not p.exists():
        continue
    obj = C.read_json(p)
    d = {r["label"].split("__")[-1]: r for r in obj.get("runs", [])}
    top, rnd = d.get("top200"), d.get("rand200")
    sid = "v2_" + label
    rows.append({
        "specification_id": sid, "label": label, "orders": orders,
        "family_type": famtype or ("free" if orders == 1 else "free"),
        "family_condition": specs.loc[sid, "family_condition"] if sid in specs.index else "free",
        "material": specs.loc[sid, "material"] if sid in specs.index else "Steel",
        "p_ref": specs.loc[sid, "p_ref"] if sid in specs.index else np.nan,
        "n_feasible": specs.loc[sid, "n_feasible"] if sid in specs.index else -1,
        "support_class": specs.loc[sid, "support_class"] if sid in specs.index else "",
        "n_batch": obj.get("n_batch", -1),
        "pool_proxy_in_band": obj.get("proxy_whole_batch", {}).get("joint_in_band_tol5", np.nan),
        "top200_delivered": top.get("gate_tol5", {}).get("n_passed", -1) if top else -1,
        "top200_n": top.get("n_shortlist", 200) if top else -1,
        "top200_rate": top.get("truth", {}).get("joint_in_band_tol5", np.nan) if top else np.nan,
        "rand200_delivered": rnd.get("gate_tol5", {}).get("n_passed", -1) if rnd else -1,
        "rand200_n": rnd.get("n_shortlist", 200) if rnd else -1,
        "rand200_rate": rnd.get("truth", {}).get("joint_in_band_tol5", np.nan) if rnd else np.nan,
        "top200_tol1_rate": top.get("truth", {}).get("joint_in_band_tol1", np.nan) if top else np.nan,
        "top200_tol5_rate": top.get("truth", {}).get("joint_in_band_tol5", np.nan) if top else np.nan,
        "selector": "proxy-error top-200",
    })
multi = pd.DataFrame(rows)
multi.to_csv(OUT / "multiorder_point_ddpm.csv", index=False, encoding="utf-8-sig")

order_summary = (multi.groupby("orders")
                 .agg(conditions=("label", "size"),
                      mean_p_ref=("p_ref", "mean"),
                      mean_top200_delivery=("top200_rate", "mean"),
                      mean_rand200_delivery=("rand200_rate", "mean"),
                      mean_pool_proxy=("pool_proxy_in_band", "mean"))
                 .reset_index())
order_summary.to_csv(OUT / "multiorder_by_order_count.csv", index=False, encoding="utf-8-sig")
family_summary = (multi.groupby("family_type")
                  .agg(conditions=("label", "size"), mean_p_ref=("p_ref", "mean"),
                       mean_top200_delivery=("top200_rate", "mean"))
                  .reset_index())
family_summary.to_csv(OUT / "family_condition_summary.csv", index=False, encoding="utf-8-sig")

# D. Tolerance curve for the point-conditioned model (same generated batch scored at 1% and 5%).
tol_curve = multi[["label", "orders", "p_ref", "top200_tol1_rate", "top200_tol5_rate"]].copy()
tol_curve["gain_5_vs_1_pp"] = 100.0 * (tol_curve.top200_tol5_rate - tol_curve.top200_tol1_rate)
tol_curve.to_csv(OUT / "point_tolerance_curve.csv", index=False, encoding="utf-8-sig")

# E. Interval-width diagnostic from the interval checkpoint.
bands = {
    "int_narrow_v2i": (3500.0, 3870.0),
    "int_mid_v2i": (2600.0, 5200.0),
    "int_wide_v2i": (1800.0, 7500.0),
    "int_open_v2i": (3681.0, 1.0e12),
}
wrows = []
for label, (lo, hi) in bands.items():
    sid = "v2_" + label
    ver = C.V2_OUT / f"verify_{label}.json"
    if not ver.exists():
        continue
    obj = C.read_json(ver)
    run0 = (obj.get("runs") or [{}])[0]
    wrows.append({
        "specification_id": sid, "lower": lo, "upper": hi,
        "width_rpm": hi - lo if hi < 1e11 else np.inf,
        "n_feasible": specs.loc[sid, "n_feasible"],
        "p_ref": specs.loc[sid, "p_ref"],
        "pool_proxy_in_band": obj.get("proxy_whole_batch", {}).get("joint_in_band_tol0", np.nan),
        "top200_delivered": run0.get("gate_tol0", {}).get("n_passed", -1),
        "top200_n": run0.get("n_shortlist", 200),
        "top200_rate": run0.get("truth", {}).get("joint_in_band_tol0", np.nan),
        "proxy_top200_rate": run0.get("proxy", {}).get("joint_in_band_tol0", np.nan),
        "ross_seconds": run0.get("seconds", np.nan),
    })
width = pd.DataFrame(wrows)
width.to_csv(OUT / "interval_width_curve.csv", index=False, encoding="utf-8-sig")

meta = {
    "main_six_note": "six specifications are a descriptive set; do not fit or claim a universal law",
    "correlation_rows": corr_rows,
    "multiorder_note": "point-conditioned checkpoint only; not a DDPM-vs-CVAE architecture comparison",
    "selector_note": "v2 top-200 is proxy-error rank, not select_common; cross-experiment comparisons need care",
    "interval_width_note": "interval checkpoint, single seed, top-200 ROSS",
}
with open(OUT / "difficulty_analysis_summary.json", "w", encoding="utf-8") as handle:
    json.dump(meta, handle, ensure_ascii=False, indent=2)

print(main.to_string(index=False))
print(pd.DataFrame(corr_rows).to_string(index=False))
print(order_summary.to_string(index=False))
print(family_summary.to_string(index=False))
print(tol_curve.to_string(index=False))
print(width.to_string(index=False))
