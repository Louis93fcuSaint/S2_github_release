# -*- coding: utf-8 -*-
"""Stage 2b: Wilson intervals and ROSS wall-clock cost per delivery."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

import research_common as C

OUT = C.ROOT / "03_existing_candidate_analysis"
OUT.mkdir(parents=True, exist_ok=True)

runs = pd.read_csv(OUT / "four_way_runs_reaggregated.csv")


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


rows = []
for method in C.METHODS:
    sub = runs[runs.method == method]
    k = int(sub.delivered_n_json.sum())
    n = int(sub.ross_submitted_n.sum())
    low, high = wilson(k, n)
    rows.append({
        "method": method, "method_label": C.METHOD_LABELS[method],
        "runs": len(sub), "delivered": k, "submitted": n,
        "delivery_rate_pct": 100.0 * k / n,
        "wilson_low_pct": 100.0 * low, "wilson_high_pct": 100.0 * high,
        "run_rate_min_pct": 100.0 * sub.delivery_rate.min(),
        "run_rate_max_pct": 100.0 * sub.delivery_rate.max(),
        "macro_mean_run_rate_pct": 100.0 * sub.delivery_rate.mean(),
        "macro_sd_run_rate_pct": 100.0 * sub.delivery_rate.std(ddof=1),
    })
ci = pd.DataFrame(rows)
ci.to_csv(OUT / "four_way_wilson_ci.csv", index=False, encoding="utf-8-sig")

# Per-specification intervals (pooled over three sampling seeds, n = 600).
spec_rows = []
for method in C.METHODS:
    for spec_name in C.SPEC_NAMES:
        sub = runs[(runs.method == method) & (runs.spec == spec_name)]
        k = int(sub.delivered_n_json.sum())
        n = int(sub.ross_submitted_n.sum())
        low, high = wilson(k, n)
        spec_rows.append({"method": method, "method_label": C.METHOD_LABELS[method],
                          "spec": spec_name, "delivered": k, "submitted": n,
                          "delivery_rate_pct": 100.0 * k / n,
                          "wilson_low_pct": 100.0 * low, "wilson_high_pct": 100.0 * high})
pd.DataFrame(spec_rows).to_csv(OUT / "four_way_wilson_ci_by_spec.csv", index=False, encoding="utf-8-sig")

# Wall-clock cost, read from each verify JSON.
cost_rows = []
for _, r in runs.iterrows():
    path = C.Path(r.verify_json)
    seconds = float("nan")
    n_solved = int(r.ross_solved_n)
    n_timeout = int(r.ross_timeout_n)
    n_error = int(r.ross_error_n)
    if path.exists():
        payload = C.read_json(path)
        seconds = float(payload.get("seconds_wall", float("nan")))
    cost_rows.append({
        "method": r.method, "spec": r.spec, "seed": int(r.seed),
        "ross_submitted": int(r.ross_submitted_n), "ross_solved": n_solved,
        "ross_timeout": n_timeout, "ross_error": n_error,
        "delivered": int(r.delivered_n_json),
        "seconds_wall": seconds,
        "seconds_per_submission": seconds / int(r.ross_submitted_n) if seconds == seconds else float("nan"),
        "seconds_per_delivered": seconds / int(r.delivered_n_json) if (seconds == seconds and r.delivered_n_json > 0) else float("nan"),
    })
cost = pd.DataFrame(cost_rows)
cost.to_csv(OUT / "four_way_cost_by_run.csv", index=False, encoding="utf-8-sig")

cost_sum = (cost.groupby("method")
            .agg(runs=("seconds_wall", "size"),
                 seconds_wall_total=("seconds_wall", "sum"),
                 seconds_wall_median=("seconds_wall", "median"),
                 seconds_per_submission_median=("seconds_per_submission", "median"),
                 seconds_per_delivered_median=("seconds_per_delivered", "median"),
                 ross_submitted=("ross_submitted", "sum"),
                 delivered=("delivered", "sum"),
                 ross_timeout=("ross_timeout", "sum"),
                 ross_error=("ross_error", "sum"))
            .reset_index())
cost_sum["seconds_per_delivered_pooled"] = cost_sum.seconds_wall_total / cost_sum.delivered
cost_sum["method_label"] = cost_sum.method.map(C.METHOD_LABELS)
cost_sum = cost_sum[["method", "method_label", "runs", "ross_submitted", "delivered",
                     "ross_timeout", "ross_error", "seconds_wall_total",
                     "seconds_wall_median", "seconds_per_submission_median",
                     "seconds_per_delivered_median", "seconds_per_delivered_pooled"]]
cost_sum.to_csv(OUT / "four_way_cost_summary.csv", index=False, encoding="utf-8-sig")

meta = {
    "wilson_method": "score interval on pooled counts; per-run intervals also in four_way_runs_reaggregated.csv",
    "cost_source": "seconds_wall field of each ROSS verify JSON",
    "cost_caveat": "Batched wall-clock at the recorded thread configuration; not single-solve CPU time.",
    "cost_rows_with_missing_json": int(cost.seconds_wall.isna().sum()),
}
with open(OUT / "cost_ci_metadata.json", "w", encoding="utf-8") as handle:
    json.dump(meta, handle, ensure_ascii=False, indent=2)

print(ci.to_string(index=False))
print(cost_sum.to_string(index=False))
