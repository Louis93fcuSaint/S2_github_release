# -*- coding: utf-8 -*-
"""Aggregate the final five-fold GroupKFold cross-validation of the deployed surrogate."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "results" / "surrogate_cv_final"
OUT_CSV = CV / "final_surrogate_cv_summary.csv"
OUT_MD = ROOT / "final_surrogate_validation.md"
ORDERS = ["cs_%d_rpm" % i for i in range(1, 7)]

rows = []
for fold in range(5):
    path = CV / ("fold%d" % fold) / "metrics.json"
    if not path.exists():
        continue
    m = json.loads(path.read_text(encoding="utf-8"))
    ens = m["ensemble_metrics"]
    seeds = [s["mean_MAPE_pct"] for s in m["seed_metrics"]]
    row = {
        "fold": fold,
        "n_train": m["train_size"],
        "n_test": m["test_size"],
        "n_test_families": m.get("n_test_families"),
        "ensemble_mean_MAPE_pct": ens["mean_MAPE_pct"],
        "ensemble_mean_median_APE_pct": ens["mean_median_APE_pct"],
        "ensemble_mean_P90_APE_pct": ens["mean_P90_APE_pct"],
        "ensemble_mean_R2": ens["mean_R2"],
        "ensemble_mean_R2_log": ens["mean_R2_log"],
        "ensemble_mean_RMSLE": ens["mean_RMSLE"],
        "seed_MAPE_min": float(np.min(seeds)),
        "seed_MAPE_max": float(np.max(seeds)),
        "seed_MAPE_std": float(np.std(seeds, ddof=1)) if len(seeds) > 1 else np.nan,
        "fit_seconds": m.get("fit_seconds_reported"),
        "wall_seconds": m.get("wall_seconds"),
    }
    for o in ORDERS:
        row["MAPE_" + o] = ens["per_target"][o]["MAPE_pct"]
        row["medianAPE_" + o] = ens["per_target"][o]["median_APE_pct"]
    rows.append(row)

if not rows:
    raise SystemExit("no completed folds")

df = pd.DataFrame(rows).sort_values("fold")
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

complete = len(df) == 5
mean = df.drop(columns=["fold"]).mean(numeric_only=True)
sd = df.drop(columns=["fold"]).std(numeric_only=True, ddof=1)
pooled_mape = float(np.average(df["ensemble_mean_MAPE_pct"], weights=df["n_test"]))

orders = ["MAPE_" + o for o in ORDERS]
lines = []
lines.append("# Final surrogate validation: five-fold GroupKFold CV\n")
lines.append("Status: **%s** (%d/5 folds).\n" % ("complete" if complete else "PARTIAL", len(df)))
lines.append("Protocol: frozen deployed configuration (47 features, 1024-512-256 MLP, L1 loss on "
             "standardised log critical speed, five-member ensemble averaged in log space). "
             "Outer GroupKFold(5) groups on `material|n_disks|n_bearings`; each fold re-selects the "
             "log-input columns, fits the input/target standardisers and the early-stopping split on "
             "the outer training set only. This is a fixed-configuration grouped CV, not a nested CV: "
             "the architecture and training recipe were chosen before this run.\n")
lines.append("## Per-fold results\n")
lines.append("| Fold | Train | Test | Families | Ensemble mean MAPE (%) | Median APE (%) | P90 APE (%) | R2(log) | Seed MAPE range (%) |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for _, r in df.iterrows():
    lines.append("| %d | %d | %d | %s | %.3f | %.3f | %.3f | %.4f | %.2f-%.2f |" % (
        r.fold, r.n_train, r.n_test, r.get("n_test_families"), r.ensemble_mean_MAPE_pct,
        r.ensemble_mean_median_APE_pct, r.ensemble_mean_P90_APE_pct, r.ensemble_mean_R2_log,
        r.seed_MAPE_min, r.seed_MAPE_max))
lines.append("")
lines.append("## Summary\n")
lines.append("- Unweighted fold mean ensemble MAPE: **%.3f%%** (SD %.3f, range %.3f-%.3f)." % (
    mean["ensemble_mean_MAPE_pct"], sd["ensemble_mean_MAPE_pct"],
    df.ensemble_mean_MAPE_pct.min(), df.ensemble_mean_MAPE_pct.max()))
lines.append("- Test-size-weighted pooled MAPE: **%.3f%%**." % pooled_mape)
lines.append("- Median APE: %.3f%%; P90 APE: %.3f%% (fold means)." % (
    mean["ensemble_mean_median_APE_pct"], mean["ensemble_mean_P90_APE_pct"]))
lines.append("- R2 on log targets: %.4f (fold mean)." % mean["ensemble_mean_R2_log"])
lines.append("- Per-order MAPE (fold mean): " + ", ".join(
    "cs%d %.2f%%" % (i + 1, mean["MAPE_" + o]) for i, o in enumerate(ORDERS)) + ".")
lines.append("")
lines.append("## Interpretation limits\n")
lines.append("- The grouped split holds out entire `material|disk|bearing` families, so it measures "
             "transfer to unseen family combinations of materials and topologies present in the data; "
             "it is not a test on physically new rotor concepts.")
lines.append("- The folds are not independent experiments: they reuse the same 199,815-record pool. "
             "The spread across folds describes heterogeneity across held-out families, not sampling "
             "error of an independent campaign.")
lines.append("- This CV does **not** validate the generative model, the selector, or the ROSS delivery "
             "rate. Those are separate evaluations.")
lines.append("- The earlier 11.2%/11.0% values belong to a 256-128 single model on an older data subset "
             "and must not be compared numerically with this table.")
OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", OUT_CSV)
print("wrote", OUT_MD)
print(df[["fold", "ensemble_mean_MAPE_pct", "ensemble_mean_median_APE_pct", "ensemble_mean_P90_APE_pct"]].to_string(index=False))
