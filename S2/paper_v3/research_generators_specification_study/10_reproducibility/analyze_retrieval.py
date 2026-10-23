# -*- coding: utf-8 -*-
"""Stage 5: retrieval+jitter versus conditional generation under existing protocols."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import research_common as C

OUT = C.ROOT / "06_retrieval_comparison"
OUT.mkdir(parents=True, exist_ok=True)

runs = pd.read_csv(C.ROOT / "03_existing_candidate_analysis" / "four_way_runs_reaggregated.csv")
specs = pd.read_csv(C.ROOT / "02_specification_difficulty" / "p_ref_all_specs.csv").set_index("specification_id")
scales = np.load(C.ROOT / "05_valid_diversity" / "reference_scales.npy")
ref_cols = ["status", "mat_Steel", "mat_Aluminum", "mat_Titanium", "n_disks", "n_bearings", "cs_1_rpm"] + C.CANONICAL_COLUMNS
ref = pd.read_csv(C.DATA_CSV, usecols=ref_cols)
ref["material"] = C.material_series(ref)
ref = ref[ref.status == "success"].copy()

rows = []
for spec_name in C.SPEC_NAMES:
    s = C.spec(spec_name)
    unified_json = C.V2_OUT / f"verify_setobj_jitter__{spec_name}.json"
    center_json = C.V2_OUT / f"verify_jitterphys42__{spec_name}.json"
    u = (C.read_json(unified_json).get("runs") or [{}])[0]
    c = (C.read_json(center_json).get("runs") or [{}])[0]
    base = runs[(runs.spec == spec_name) & (runs.seed == 42)].set_index("method")
    rows.append({
        "spec": spec_name, "material": s["material"], "target": s["target"],
        "p_ref": float(specs.loc[spec_name, "p_ref"]),
        "n_feasible": int(specs.loc[spec_name, "n_feasible"]),
        "support_class": specs.loc[spec_name, "support_class"],
        "ddpm_guided_rate": float(base.loc["ddpm_guided", "delivery_rate"]),
        "ddpm_unguided_rate": float(base.loc["ddpm_unguided", "delivery_rate"]),
        "cvae_pure_rate": float(base.loc["cvae_pure", "delivery_rate"]),
        "cvae_guided_rate": float(base.loc["cvae_guided", "delivery_rate"]),
        "retrieval_jitter_unified_rate": float(u.get("deliverable_n", -1)) / float(u.get("n_shortlist", 200)),
        "retrieval_jitter_unified_delivered": int(u.get("deliverable_n", -1)),
        "retrieval_jitter_unified_n": int(u.get("n_shortlist", 200)),
        "retrieval_jitter_center_rate": float(c.get("deliverable_n", -1)) / float(c.get("n_shortlist", 200)),
        "retrieval_jitter_center_delivered": int(c.get("deliverable_n", -1)),
        "retrieval_jitter_center_n": int(c.get("n_shortlist", 200)),
        "unified_selector": "select_common",
        "center_selector_note": "known in-band references ranked toward band centre; information advantage",
        "seed": 42,
    })
comp = pd.DataFrame(rows)
comp["center_minus_unified_pp"] = 100.0 * (comp.retrieval_jitter_center_rate - comp.retrieval_jitter_unified_rate)
comp["unified_minus_best_generator_pp"] = 100.0 * (
    comp.retrieval_jitter_unified_rate
    - comp[["ddpm_guided_rate", "ddpm_unguided_rate", "cvae_pure_rate", "cvae_guided_rate"]].max(axis=1))
comp["center_minus_best_generator_pp"] = 100.0 * (
    comp.retrieval_jitter_center_rate
    - comp[["ddpm_guided_rate", "ddpm_unguided_rate", "cvae_pure_rate", "cvae_guided_rate"]].max(axis=1))
comp.to_csv(OUT / "retrieval_vs_generators_by_spec.csv", index=False, encoding="utf-8-sig")

summary = {
    "retrieval_jitter_unified_delivered": int(comp.retrieval_jitter_unified_delivered.sum()),
    "retrieval_jitter_unified_n": int(comp.retrieval_jitter_unified_n.sum()),
    "retrieval_jitter_unified_rate": float(comp.retrieval_jitter_unified_delivered.sum() / comp.retrieval_jitter_unified_n.sum()),
    "retrieval_jitter_center_delivered": int(comp.retrieval_jitter_center_delivered.sum()),
    "retrieval_jitter_center_n": int(comp.retrieval_jitter_center_n.sum()),
    "retrieval_jitter_center_rate": float(comp.retrieval_jitter_center_delivered.sum() / comp.retrieval_jitter_center_n.sum()),
    "ddpm_guided_seed42": float(comp.ddpm_guided_rate.mean()),
    "ddpm_unguided_seed42": float(comp.ddpm_unguided_rate.mean()),
    "cvae_pure_seed42": float(comp.cvae_pure_rate.mean()),
    "cvae_guided_seed42": float(comp.cvae_guided_rate.mean()),
    "unified_beats_best_generator_specs": int((comp.unified_minus_best_generator_pp > 0).sum()),
    "center_beats_best_generator_specs": int((comp.center_minus_best_generator_pp > 0).sum()),
}
with open(OUT / "retrieval_comparison_summary.json", "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)

# Support relation (descriptive, six specs).
rho, p = stats.spearmanr(comp.p_ref, comp.retrieval_jitter_unified_rate)
with open(OUT / "retrieval_support_relation.txt", "w", encoding="utf-8") as handle:
    handle.write(f"Spearman p_ref vs unified retrieval+jitter delivery (n=6): rho={rho:.6f}, p={p:.6g}\n")
    handle.write(f"Spearman p_ref vs center-ranking retrieval+jitter delivery (n=6): ")
    rho2, p2 = stats.spearmanr(comp.p_ref, comp.retrieval_jitter_center_rate)
    handle.write(f"rho={rho2:.6f}, p={p2:.6g}\n")
    handle.write("All six specifications have observed support; the study cannot identify a zero-support retrieval regime.\n")

# Diversity of retrieval shortlists and delivered sets, using the same metrics as Stage 3.
def metrics(frame, method, spec_name, layer):
    keys = C.exact_design_keys(frame)
    nn, pair, n_dist = C.same_family_nn_and_pair(frame, scales, max_points=1000)
    nd = C.near_duplicate_fraction(frame, scales, max_points=1000)
    nov = C.nearest_reference_distance(frame, ref, scales, max_queries=150)
    fam = frame.groupby(["n_disks", "n_bearings"]).size().to_numpy(float)
    p = fam / fam.sum()
    return {
        "method": method, "spec": spec_name, "layer": layer, "n": len(frame),
        "exact_unique_rate": len(set(keys)) / len(frame),
        "family_count": len(fam), "family_entropy": float(-(p * np.log(p)).sum()),
        "median_nn_same_family": nn, "median_pair_same_family": pair,
        "novelty_median_to_reference": float(np.median(nov)),
        "near_dup_frac_le_0.01": nd["near_dup_frac_le_0.01"],
        "near_dup_frac_le_0.05": nd["near_dup_frac_le_0.05"],
        "n_distance_sample": n_dist,
    }

div_rows = []
for spec_name in C.SPEC_NAMES:
    u_short = pd.read_csv(C.RETRIEVAL / f"retrieval_jitter__{spec_name}.csv")
    u_pass = pd.read_csv(C.V2_OUT / f"verify_setobj_jitter__{spec_name}__top200__passed.csv")
    c_short = pd.read_csv(C.V2_OUT / f"verify_jitterphys42__{spec_name}__top200.csv")
    c_pass = pd.read_csv(C.V2_OUT / f"verify_jitterphys42__{spec_name}__top200__passed.csv")
    for frame, method, layer in [(u_short, "retrieval_jitter_unified", "shortlist"),
                                 (u_pass, "retrieval_jitter_unified", "delivered"),
                                 (c_short, "retrieval_jitter_center", "shortlist"),
                                 (c_pass, "retrieval_jitter_center", "delivered")]:
        div_rows.append(metrics(frame, method, spec_name, layer))
    # seed-42 generator comparators
    for method in C.METHODS:
        _, short_p, _, _, passed_p = C.method_paths(method, spec_name, 42)
        div_rows.append(metrics(pd.read_csv(short_p), method, spec_name, "shortlist"))
        div_rows.append(metrics(pd.read_csv(passed_p), method, spec_name, "delivered"))
pd.DataFrame(div_rows).to_csv(OUT / "retrieval_diversity_by_spec.csv", index=False, encoding="utf-8-sig")

# Equalized comparison among six methods at seed 42, delivered layer.
methods = ["ddpm_guided", "ddpm_unguided", "cvae_pure", "cvae_guided",
           "retrieval_jitter_unified", "retrieval_jitter_center"]
rng = np.random.default_rng(20261021)
eq_rows = []
for spec_name in C.SPEC_NAMES:
    frames = {}
    for method in methods:
        if method.startswith("retrieval"):
            p = (C.V2_OUT / f"verify_setobj_jitter__{spec_name}__top200__passed.csv"
                 if method.endswith("unified") else C.V2_OUT / f"verify_jitterphys42__{spec_name}__top200__passed.csv")
        else:
            p = C.method_paths(method, spec_name, 42)[4]
        frames[method] = pd.read_csv(p)
    n_min = min(len(v) for v in frames.values())
    for rep in range(100):
        for method, frame in frames.items():
            idx = rng.choice(len(frame), n_min, replace=False)
            sub = frame.iloc[np.sort(idx)]
            nn, pair, _ = C.same_family_nn_and_pair(sub, scales, max_points=n_min, seed=rep)
            eq_rows.append({"spec": spec_name, "method": method, "n_equalized": n_min,
                            "replicate": rep, "nn": nn, "pair": pair,
                            "family_entropy": C.family_entropy(sub)})
eq = pd.DataFrame(eq_rows)
eq.to_csv(OUT / "retrieval_equalized_replicates.csv", index=False, encoding="utf-8-sig")
eq_sum = (eq.groupby(["spec", "method", "n_equalized"])
          .agg(nn_mean=("nn", "mean"), nn_low=("nn", lambda x: np.quantile(x, .025)),
               nn_high=("nn", lambda x: np.quantile(x, .975)),
               pair_mean=("pair", "mean"), pair_low=("pair", lambda x: np.quantile(x, .025)),
               pair_high=("pair", lambda x: np.quantile(x, .975)),
               family_entropy=("family_entropy", "mean"))
          .reset_index())
eq_sum.to_csv(OUT / "retrieval_equalized_summary.csv", index=False, encoding="utf-8-sig")

print(comp.to_string(index=False))
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(eq_sum.to_string(index=False))
