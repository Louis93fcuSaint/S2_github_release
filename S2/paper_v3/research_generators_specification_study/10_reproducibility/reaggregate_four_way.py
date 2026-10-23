# -*- coding: utf-8 -*-
"""Stage 2: independent reaggregation of the existing four-pipeline experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import research_common as C

OUT = C.ROOT / "03_existing_candidate_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def in_band(pred, lo, hi):
    pred = np.asarray(pred, dtype=float)
    return (pred >= lo) & (pred <= hi)


def proxy_stats(frame, s):
    lo, hi = C.band(s["target"])
    pred = frame["cs1_pred"].to_numpy(float)
    inside = in_band(pred, lo, hi)
    return {
        "proxy_in_band": float(inside.mean()),
        "proxy_abs_pct_error_target": float(np.mean(np.abs(pred - s["target"]) / s["target"] * 100.0)),
        "pred_min": float(np.min(pred)), "pred_median": float(np.median(pred)),
        "pred_max": float(np.max(pred)),
    }


runs = []
for method in C.METHODS:
    for spec_name in C.SPEC_NAMES:
        s = C.spec(spec_name)
        for seed in C.SEEDS:
            pool_p, short_p, ver_p, csv_p, passed_p = C.method_paths(method, spec_name, seed)
            payload = C.read_json(ver_p)
            short = pd.read_csv(short_p)
            pool = pd.read_csv(pool_p)
            verify = pd.read_csv(csv_p)
            passed_rows = sum(1 for _ in open(passed_p, encoding="utf-8")) - 1 if passed_p.exists() else -1
            deliver_json = int(payload.get("deliverable_n", -1))
            n_submit = int(payload.get("n_shortlist", len(short)))
            rate = deliver_json / n_submit if n_submit else float("nan")
            lo, hi = C.wilson(deliver_json, n_submit)
            ps_pool = proxy_stats(pool, s)
            ps_short = proxy_stats(short, s)
            truth_err = pd.to_numeric(verify.get("cs1_relerr_truth_pct"), errors="coerce")
            runs.append({
                "method": method, "method_label": C.METHOD_LABELS[method],
                "spec": spec_name, "material": s["material"], "target_rpm": s["target"],
                "seed": seed,
                "pool_n": int(len(pool)), "shortlist_n": int(len(short)),
                "ross_submitted_n": n_submit, "ross_solved_n": int(payload.get("n_solved", -1)),
                "ross_timeout_n": int(payload.get("n_timeout", -1)),
                "ross_error_n": int(payload.get("n_error", -1)),
                "delivered_n_json": deliver_json,
                "delivered_n_passed_csv": int(passed_rows),
                "delivered_consistent": bool(passed_rows == deliver_json),
                "delivery_rate": rate,
                "wilson_low": lo, "wilson_high": hi,
                "pool_proxy_in_band": ps_pool["proxy_in_band"],
                "shortlist_proxy_in_band": ps_short["proxy_in_band"],
                "pool_proxy_abs_pct_error_target": ps_pool["proxy_abs_pct_error_target"],
                "shortlist_proxy_abs_pct_error_target": ps_short["proxy_abs_pct_error_target"],
                "shortlist_truth_abs_pct_error_mean": float(truth_err.abs().mean()),
                "shortlist_truth_signed_pct_error_mean": float(truth_err.mean()),
                "pool_path": str(pool_p), "shortlist_path": str(short_p),
                "verify_json": str(ver_p), "verify_csv": str(csv_p), "passed_csv": str(passed_p),
            })

runs_df = pd.DataFrame(runs)
runs_df.to_csv(OUT / "four_way_runs_reaggregated.csv", index=False, encoding="utf-8-sig")

summary = []
for method, sub in runs_df.groupby("method", sort=False):
    k, n = int(sub.delivered_n_json.sum()), int(sub.ross_submitted_n.sum())
    lo, hi = C.wilson(k, n)
    spec_means = sub.groupby("spec")["delivery_rate"].mean().reindex(C.SPEC_NAMES)
    summary.append({
        "method": method, "method_label": C.METHOD_LABELS[method],
        "runs": int(len(sub)), "delivered": k, "submitted": n,
        "pooled_rate": k / n, "wilson_low": lo, "wilson_high": hi,
        "macro_mean_spec_rate": float(spec_means.mean()),
        "spec_rate_sd": float(spec_means.std(ddof=1)),
        "pool_proxy_in_band_mean": float(sub.pool_proxy_in_band.mean()),
        "shortlist_proxy_in_band_mean": float(sub.shortlist_proxy_in_band.mean()),
        "shortlist_truth_mae_pct_mean": float(sub.shortlist_truth_abs_pct_error_mean.mean()),
        "all_passed_rows_consistent": bool(sub.delivered_consistent.all()),
    })
summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT / "four_way_summary_reaggregated.csv", index=False, encoding="utf-8-sig")

# Compare with historical summary file.
hist_path = C.FAIR / "04_four_way_comparison" / "four_way_summary.csv"
hist = pd.read_csv(hist_path)
cons = summary_df.merge(hist[["method", "delivered", "submitted", "pooled_rate"]],
                        on="method", suffixes=("_reaggregated", "_historical"))
cons["delivered_delta"] = cons["delivered_reaggregated"] - cons["delivered_historical"]
cons["rate_delta_pp"] = 100.0 * (cons["pooled_rate_reaggregated"] - cons["pooled_rate_historical"])
cons.to_csv(OUT / "reaggregation_consistency.csv", index=False, encoding="utf-8-sig")

# Spec-level paired comparisons, 6 paired units (not 3,600 independent experiments).
def spec_means(method):
    return runs_df[runs_df.method == method].groupby("spec")["delivery_rate"].mean().reindex(C.SPEC_NAMES).to_numpy(float)


def paired(a, b):
    d = a - b
    if np.allclose(d, 0):
        w, p = 0.0, 1.0
    else:
        w, p = stats.wilcoxon(d, alternative="two-sided")
    rng = np.random.default_rng(42)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(20000)])
    return {
        "n_pairs": len(d), "mean_diff_pp": 100.0 * d.mean(),
        "median_diff_pp": 100.0 * np.median(d),
        "min_diff_pp": 100.0 * d.min(), "max_diff_pp": 100.0 * d.max(),
        "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum()),
        "wilcoxon_w": float(w), "wilcoxon_p": float(p),
        "bootstrap_ci_low_pp": float(100 * np.quantile(boot, 0.025)),
        "bootstrap_ci_high_pp": float(100 * np.quantile(boot, 0.975)),
    }

comparisons = [
    ("primary", "DDPM guidance effect", "ddpm_guided", "ddpm_unguided"),
    ("primary", "CVAE guidance effect", "cvae_guided", "cvae_pure"),
    ("secondary", "Unguided: DDPM vs CVAE", "ddpm_unguided", "cvae_pure"),
    ("secondary", "Guided: DDPM vs CVAE", "ddpm_guided", "cvae_guided"),
]
rows = []
for family, label, a, b in comparisons:
    rec = paired(spec_means(a), spec_means(b))
    rec.update({"family": family, "comparison": label, "method_a": a, "method_b": b,
                "rate_a": float(summary_df.loc[summary_df.method == a, "pooled_rate"].iloc[0]),
                "rate_b": float(summary_df.loc[summary_df.method == b, "pooled_rate"].iloc[0])})
    rows.append(rec)
comp = pd.DataFrame(rows)
for fam in ("primary", "secondary"):
    mask = comp.family == fam
    comp.loc[mask, "holm_p"] = C.holm(comp.loc[mask, "wilcoxon_p"].to_numpy(float))
comp.to_csv(OUT / "four_way_comparisons_reaggregated.csv", index=False, encoding="utf-8-sig")

# 18-pair supplement, explicitly clustered by specification.
run_pairs = []
for _, row in comp.iterrows():
    a = runs_df[runs_df.method == row.method_a].set_index(["spec", "seed"])["delivery_rate"]
    b = runs_df[runs_df.method == row.method_b].set_index(["spec", "seed"])["delivery_rate"]
    d = (a - b).dropna()
    w, p = stats.wilcoxon(d, alternative="two-sided") if not np.allclose(d, 0) else (0.0, 1.0)
    run_pairs.append({"comparison": row.comparison, "n_pairs": len(d),
                      "mean_diff_pp": 100 * d.mean(), "n_positive": int((d > 0).sum()),
                      "n_negative": int((d < 0).sum()), "wilcoxon_w": float(w),
                      "wilcoxon_p": float(p),
                      "unit_warning": "18 spec x seed cells are not independent training replicates"})
pd.DataFrame(run_pairs).to_csv(OUT / "four_way_18pair_supplement.csv", index=False, encoding="utf-8-sig")

# Machine-readable note for downstream stages.
meta = {
    "reaggregated_from": "verify JSON + passed CSV + candidate CSV",
    "runs": int(len(runs_df)), "methods": C.METHODS,
    "statistical_unit": "six paired specifications; three sampling seeds are repeated sampling under one checkpoint",
    "not_independent": "3,600 submitted candidates are not 3,600 independent experiments",
    "all_passed_csv_consistent": bool(runs_df.delivered_consistent.all()),
    "historical_summary": str(hist_path),
}
with open(OUT / "reaggregation_metadata.json", "w", encoding="utf-8") as handle:
    json.dump(meta, handle, ensure_ascii=False, indent=2)

print(summary_df.to_string(index=False))
print(cons.to_string(index=False))
print(comp.to_string(index=False))
