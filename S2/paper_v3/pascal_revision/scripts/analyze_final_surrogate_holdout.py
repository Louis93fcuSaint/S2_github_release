# -*- coding: utf-8 -*-
"""Stratified descriptive metrics for the frozen final surrogate on its fixed test split."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

S2 = Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2")
sys.path.insert(0, str(S2 / "surrogate_v47"))
import common as C

OUT = Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2\paper\paper_v3\pascal_revision\results\surrogate_holdout")
OUT.mkdir(parents=True, exist_ok=True)

z = np.load(S2 / "surrogate_v47" / "outputs" / "mlp_v47_best" / "pred_multi.npz")
idx = z["test_idx"].astype(int)
y_true, y_pred = z["y_test"], z["raw_pred"]
df = pd.read_csv(C.V47_CSV)
meta = df.iloc[idx].reset_index(drop=True)

rows = []
def add(level, group, order, m):
    if m is None or (isinstance(m, dict) and not np.isfinite(m.get("MAPE_pct", np.nan))):
        return
    rows.append({"level": level, "group": group, "order": order, **m})

# overall per order and aggregate
overall = C.evaluate_predictions(y_true, y_pred)
for t, m in overall["per_target"].items():
    add("overall", "all", t, m)
rows.append({"level": "overall", "group": "all_mean", "order": "mean_1_6",
             "n": int(np.sum(np.isfinite(y_true))),
             "MAE_rpm": np.nan,
             "RMSE_rpm": np.nan,
             "MAPE_pct": overall["mean_MAPE_pct"],
             "median_APE_pct": overall["mean_median_APE_pct"],
             "P90_APE_pct": overall["mean_P90_APE_pct"],
             "R2": overall["mean_R2"], "R2_log": overall["mean_R2_log"],
             "RMSLE": overall["mean_RMSLE"]})

# material
mat = np.select([meta.mat_Steel.to_numpy() == 1, meta.mat_Aluminum.to_numpy() == 1,
                 meta.mat_Titanium.to_numpy() == 1], ["Steel", "Aluminum", "Titanium"], default="Other")
for name in ["Steel", "Aluminum", "Titanium"]:
    sel = mat == name
    ev = C.evaluate_predictions(y_true[sel], y_pred[sel])
    for t, m in ev["per_target"].items():
        add("material", name, t, m)
    rows.append({"level": "material", "group": name, "order": "mean_1_6",
                 "n": int(np.sum(np.isfinite(y_true[sel]))), "MAE_rpm": np.nan,
                 "RMSE_rpm": np.nan, "MAPE_pct": ev["mean_MAPE_pct"],
                 "median_APE_pct": ev["mean_median_APE_pct"],
                 "P90_APE_pct": ev["mean_P90_APE_pct"], "R2": ev["mean_R2"],
                 "R2_log": ev["mean_R2_log"], "RMSLE": ev["mean_RMSLE"]})

# family (material | disks | bearings)
fam = np.array(["%s|%dd|%db" % (m, int(d), int(b))
                for m, d, b in zip(mat, meta.n_disks, meta.n_bearings)])
for name in sorted(set(fam)):
    sel = fam == name
    ev = C.evaluate_predictions(y_true[sel], y_pred[sel])
    rows.append({"level": "family", "group": name, "order": "mean_1_6",
                 "n": int(sel.sum()), "MAE_rpm": np.nan, "RMSE_rpm": np.nan,
                 "MAPE_pct": ev["mean_MAPE_pct"], "median_APE_pct": ev["mean_median_APE_pct"],
                 "P90_APE_pct": ev["mean_P90_APE_pct"], "R2": ev["mean_R2"],
                 "R2_log": ev["mean_R2_log"], "RMSLE": ev["mean_RMSLE"]})

# cs1 speed tertiles (all orders within the band defined by true cs1)
valid1 = np.isfinite(y_true[:, 0]) & (y_true[:, 0] > 0)
qs = np.quantile(y_true[valid1, 0], [0, 1/3, 2/3, 1])
labels = ["low", "mid", "high"]
for k, lab in enumerate(labels):
    lo, hi = qs[k], qs[k + 1]
    sel = valid1 & (y_true[:, 0] >= lo) & (y_true[:, 0] <= hi if k == 2 else y_true[:, 0] < hi)
    ev = C.evaluate_predictions(y_true[sel], y_pred[sel])
    for t, m in ev["per_target"].items():
        add("cs1_band", lab, t, m)
    rows.append({"level": "cs1_band", "group": lab, "order": "mean_1_6",
                 "n": int(sel.sum()), "MAE_rpm": np.nan, "RMSE_rpm": np.nan,
                 "MAPE_pct": ev["mean_MAPE_pct"], "median_APE_pct": ev["mean_median_APE_pct"],
                 "P90_APE_pct": ev["mean_P90_APE_pct"], "R2": ev["mean_R2"],
                 "R2_log": ev["mean_R2_log"], "RMSLE": ev["mean_RMSLE"],
                 "band_low_rpm": float(lo), "band_high_rpm": float(hi)})

pd.DataFrame(rows).to_csv(OUT / "surrogate_holdout_metrics_stratified.csv", index=False)

missing = []
for j, t in enumerate(C.TARGETS):
    missing.append({"order": t, "n_total": int(len(y_true)), "n_label_valid": int(np.isfinite(y_true[:, j]).sum()),
                    "n_missing": int((~np.isfinite(y_true[:, j])).sum()),
                    "missing_fraction": float((~np.isfinite(y_true[:, j])).mean())})
pd.DataFrame(missing).to_csv(OUT / "surrogate_holdout_label_coverage.csv", index=False)

meta_out = {
    "source_pred": str(S2 / "surrogate_v47" / "outputs" / "mlp_v47_best" / "pred_multi.npz"),
    "n_test": int(len(y_true)), "n_rows_dataset": int(len(df)),
    "overall_mean_MAPE_pct": overall["mean_MAPE_pct"],
    "overall_mean_median_APE_pct": overall["mean_median_APE_pct"],
    "overall_mean_P90_APE_pct": overall["mean_P90_APE_pct"],
    "overall_mean_R2_log": overall["mean_R2_log"],
    "caveat": "fixed test split used during model selection; descriptive, not an independent blind test",
    "degenerate_cs1_lt_100": int(np.sum(np.isfinite(y_true[:,0]) & (y_true[:,0] < 100))),
    "cs1_band_edges_rpm": qs.tolist(),
}
(OUT / "surrogate_holdout_summary.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
print(json.dumps(meta_out, indent=2))
print(pd.DataFrame(rows).query("level=='material' and order=='mean_1_6'").to_string(index=False))
