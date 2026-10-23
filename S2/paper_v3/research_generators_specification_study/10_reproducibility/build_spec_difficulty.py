# -*- coding: utf-8 -*-
"""Stage 1: empirical support p_ref, frozen difficulty labels and expansion set."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_common as C

OUT = C.ROOT / "02_specification_difficulty"
OUT.mkdir(parents=True, exist_ok=True)

# Read only the columns needed for support analysis.
cols = ["status", "mat_Steel", "mat_Aluminum", "mat_Titanium", "n_disks", "n_bearings",
        "cs_1_rpm", "cs_2_rpm", "cs_3_rpm"]
frame_all = pd.read_csv(C.DATA_CSV, usecols=cols)
frame_all["material"] = C.material_series(frame_all)
# Quality gate: rejected_cs1 rows are not valid reference designs.
frame = frame_all[frame_all["status"] == "success"].copy()

THRESHOLDS = {
    "frozen_before_model_comparison": True,
    "high_support": "n_feasible >= 1000",
    "medium_support": "100 <= n_feasible < 1000",
    "low_support": "1 <= n_feasible < 100",
    "zero_observed_support": "n_feasible == 0",
    "definition": "counts in the finite v4.7 reference library with finite labels for all constrained orders",
    "note": "zero-observed-support is not physical infeasibility",
}

rows = []

# A. Existing six frozen interval-model specifications.
for s in C.SPECS:
    lo, hi = C.band(s["target"])
    rows.append({
        "specification_id": s["name"], "spec_family": "frozen_main_six",
        "material": s["material"], "constraints": {1: (lo, hi)},
        "family_condition": None, "constraint_orders": 1,
        "source": "fairness_audit_v4", "evaluation_status": "evaluated_interval_DDPM_and_CVAE",
        "model_evidence": "four pipeline x six specs x three sampling seeds",
    })

# B. Existing multi-order / fixed-family point-DDPM experiments.
v2 = [
    ("cs1", {"1": 3680.8}, "Steel", None),
    ("cs2", {"2": 7021.0}, "Steel", None),
    ("cs3", {"3": 21033.5}, "Steel", None),
    ("cs1cs3", {"1": 3680.8, "3": 18070.4}, "Steel", None),
    ("cs2cs3", {"2": 7021.0, "3": 15857.7}, "Steel", None),
    ("cs1cs2", {"1": 3680.8, "2": 7021.0}, "Steel", None),
    ("cs123", {"1": 3680.8, "2": 7021.0, "3": 17970.5}, "Steel", None),
    ("cs1cs2_big", {"1": 3680.8, "2": 7021.0}, "Steel", None),
    ("cs123_big", {"1": 3680.8, "2": 7021.0, "3": 17970.5}, "Steel", None),
    ("al_cs1cs2", {"1": 3191.1, "2": 6147.6}, "Aluminum", None),
    ("fam32_cs1", {"1": 2974.0}, "Steel", "3-3x2-2"),
    ("fam32_cs12", {"1": 2974.0, "2": 7001.0}, "Steel", "3-3x2-2"),
    ("fam22_cs2", {"2": 8180.0}, "Steel", "2-2x2-2"),
    ("fam12_24_cs12", {"1": 4642.0, "2": 9969.0}, "Steel", "1-2x2-4"),
    ("fam32_cs123", {"1": 2974.0, "2": 7001.0, "3": 20359.0}, "Steel", "3-3x2-2"),
]
for label, targets, material, family in v2:
    constraints = {int(k): C.band(v, C.TOL) for k, v in targets.items()}
    rows.append({
        "specification_id": "v2_" + label, "spec_family": "point_DDPM_multitorder",
        "material": material, "constraints": constraints, "family_condition": family,
        "constraint_orders": len(constraints), "source": "v2_multiorder",
        "evaluation_status": "evaluated_point_DDPM_only",
        "model_evidence": "point-conditioned checkpoint; top-200 and random-200 ROSS",
    })

# C. Existing interval-width diagnostics.
intervals = {
    "int_narrow": {1: (3500.0, 3870.0)},
    "int_mid": {1: (2600.0, 5200.0)},
    "int_wide": {1: (1800.0, 7500.0)},
    "int_open": {1: (3681.0, 1.0e12)},
}
for label, constraints in intervals.items():
    rows.append({
        "specification_id": "v2_" + label + "_v2i", "spec_family": "interval_width_diagnostic",
        "material": "Steel", "constraints": constraints, "family_condition": None,
        "constraint_orders": 1, "source": "v2_interval_diagnostics",
        "evaluation_status": "evaluated_interval_DDPM_only",
        "model_evidence": "one sampling seed, top-200 ROSS",
    })

# D. Pre-declared expansion set based only on reference quantiles.
# It is frozen now but deliberately not run in batch 1.
success = frame[frame["status"] == "success"].copy()
for material in ("Steel", "Aluminum", "Titanium"):
    vals = pd.to_numeric(success.loc[success.material == material, "cs_1_rpm"], errors="coerce").dropna()
    for q in (10, 50, 90):
        target = float(np.percentile(vals, q))
        lo, hi = C.band(target, C.TOL)
        rows.append({
            "specification_id": f"expand_{material}_q{q}_cs1", "spec_family": "proposed_expansion",
            "material": material, "constraints": {1: (lo, hi)}, "family_condition": None,
            "constraint_orders": 1, "source": "reference_quantile_predeclared",
            "evaluation_status": "frozen_not_run_batch1",
            "model_evidence": "no model result used for definition",
        })
# One fixed-family expansion anchor, predeclared from the most common family.
counts = success.groupby(["material", "n_disks", "n_bearings"]).size().sort_values(ascending=False)
material, nd, nb = counts.index[0]
vals = pd.to_numeric(success.loc[(success.material == material) & (success.n_disks == nd)
                                & (success.n_bearings == nb), "cs_1_rpm"], errors="coerce").dropna()
target = float(np.median(vals))
rows.append({
    "specification_id": f"expand_{material}_{int(nd)}d{int(nb)}b_cs1_median",
    "spec_family": "proposed_expansion", "material": material,
    "constraints": {1: C.band(target, C.TOL)},
    "family_condition": f"{int(nd)}-{int(nd)}x{int(nb)}-{int(nb)}",
    "constraint_orders": 1, "source": "reference_family_mode_predeclared",
    "evaluation_status": "frozen_not_run_batch1",
    "model_evidence": "no model result used for definition",
})

out = []
for row in rows:
    rec = {k: v for k, v in row.items() if k != "constraints"}
    rec["lower"] = float(min(v[0] for v in row["constraints"].values()))
    rec["upper"] = float(max(v[1] for v in row["constraints"].values()))
    rec.update(C.p_ref(frame, row["material"], row["constraints"], row["family_condition"]))
    rec["support_class"] = C.support_class(rec["n_feasible"])
    rec["support_class_rule_version"] = "counts_v1_2026-10-21"
    out.append(rec)

specs = pd.DataFrame(out)
# Keep a compact JSON encoding of the actual constraints for reproducibility.
specs["constraints_json"] = [
    json.dumps({str(k): [float(a), float(b)] for k, (a, b) in row["constraints"].items()},
               ensure_ascii=False)
    for row in rows
]
front = ["specification_id", "spec_family", "material", "constraint_orders", "family_condition",
         "n_filtered", "n_valid_label", "n_missing_censored", "missing_fraction",
         "n_feasible", "p_ref", "support_class", "support_families", "support_family_counts",
         "evaluation_status", "source", "model_evidence", "constraints_json"]
specs = specs[front + [c for c in specs.columns if c not in front]]
specs.to_csv(OUT / "frozen_specifications.csv", index=False, encoding="utf-8-sig")
specs.to_csv(OUT / "p_ref_all_specs.csv", index=False, encoding="utf-8-sig")

# Six-spec main table, easier to consume in Stage 4.
specs[specs.spec_family == "frozen_main_six"].to_csv(
    OUT / "p_ref_main_six.csv", index=False, encoding="utf-8-sig")

meta = {
    "classification_thresholds": THRESHOLDS,
    "reference_file": str(C.DATA_CSV),
    "raw_rows": int(len(frame_all)),
    "status_counts": {str(k): int(v) for k, v in frame_all.status.value_counts().items()},
    "material_counts": {str(k): int(v) for k, v in frame.material.value_counts().items()},
    "n_specifications": int(len(specs)),
    "support_class_counts": {str(k): int(v) for k, v in specs.support_class.value_counts().items()},
    "important": "p_ref is empirical support in a finite library, not physical feasible volume.",
}
with open(OUT / "specification_protocol.json", "w", encoding="utf-8") as handle:
    json.dump(meta, handle, ensure_ascii=False, indent=2)
print(specs[["specification_id", "material", "n_valid_label", "n_feasible", "p_ref", "support_class"]]
      .to_string(index=False))
print(json.dumps(meta, ensure_ascii=False, indent=2))

