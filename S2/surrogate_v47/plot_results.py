# -*- coding: utf-8 -*-
"""Figures for the v4.7 six-order surrogate study."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

TARGETS = ["cs_%d_rpm" % i for i in range(1, 7)]
LABELS = ["cs1", "cs2", "cs3", "cs4", "cs5", "cs6"]
RUNS = {
    "MLP multi": os.path.join(OUT, "mlp_v47", "pred_multi.npz"),
    "MLP single": os.path.join(OUT, "mlp_v47", "pred_single.npz"),
    "LightGBM": os.path.join(OUT, "tabular_v47", "pred_lightgbm.npz"),
    "HistGBDT": os.path.join(OUT, "tabular_v47", "pred_hist_gbdt.npz"),
    "ExtraTrees": os.path.join(OUT, "tabular_v47", "pred_extra_trees.npz"),
    "Ridge": os.path.join(OUT, "tabular_v47", "pred_ridge.npz"),
}


def load(name):
    blob = np.load(RUNS[name])
    return blob["y_test"], blob["raw_pred"]


def ape_matrix(y_true, y_pred):
    """(n, 6) absolute percentage error, NaN where unlabelled."""
    out = np.full(y_true.shape, np.nan)
    good = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    out[good] = np.abs(y_true[good] - np.maximum(y_pred[good], 1e-9)) / y_true[good] * 100.0
    return out


# ---------------------------------------------------------------- figure 1
y_true, pred = load("MLP multi")
fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4))
ape = ape_matrix(y_true, pred)
for k, ax in enumerate(axes.ravel()):
    good = np.isfinite(y_true[:, k])
    t, p = y_true[good, k], pred[good, k]
    ax.scatter(t, p, s=1.0, alpha=0.06, color="#1f77b4", rasterized=True)
    lo, hi = max(t.min(), 1.0), t.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    med = np.nanmedian(ape[:, k])
    p90 = np.nanpercentile(ape[:, k], 90)
    ax.set_title("%s   n=%d   median APE %.2f%%   P90 %.1f%%"
                 % (LABELS[k], good.sum(), med, p90), fontsize=10)
    ax.set_xlabel("ROSS truth (rpm)"); ax.set_ylabel("prediction (rpm)")
    ax.grid(alpha=0.25, which="both")
fig.suptitle("MLP multi-output surrogate (v4.7, 3-seed ensemble, 39,963 test rows)",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "01_mlp_multi_parity.png"), dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
width = 0.15
x = np.arange(6)
for i, name in enumerate(RUNS):
    yt, pr = load(name)
    a = ape_matrix(yt, pr)
    med = [np.nanmedian(a[:, k]) for k in range(6)]
    p90 = [np.nanpercentile(a[:, k], 90) for k in range(6)]
    axes[0].bar(x + (i - 2.5) * width, med, width, label=name)
    axes[1].bar(x + (i - 2.5) * width, p90, width, label=name)
for ax, title in zip(axes, ["median APE per order (%)", "P90 APE per order (%)"]):
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_title(title); ax.grid(alpha=0.25, axis="y")
    ax.set_yscale("log")
axes[0].legend(fontsize=8, ncol=2)
fig.suptitle("Error by critical-speed order, all families (log scale)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "02_error_by_order.png"), dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- figure 3
fig, ax = plt.subplots(figsize=(7.6, 4.6))
for name, colour in zip(["MLP multi", "MLP single", "LightGBM", "HistGBDT", "ExtraTrees", "Ridge"],
                        ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd", "#7f7f7f"]):
    yt, pr = load(name)
    a = ape_matrix(yt, pr)
    ax.hist(a.ravel(), bins=np.logspace(-1, 2.2, 80), histtype="step",
            lw=1.6, label=name, color=colour, density=True)
ax.set_xscale("log")
ax.set_xlabel("absolute percentage error (%)")
ax.set_ylabel("density")
ax.set_title("APE distribution over all orders and test rows")
ax.grid(alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "03_ape_distribution.png"), dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- summary text
rows = []
for name in RUNS:
    yt, pr = load(name)
    a = ape_matrix(yt, pr)
    per = []
    for k in range(6):
        good = np.isfinite(yt[:, k])
        t, p = yt[good, k], pr[good, k]
        log_t, log_p = np.log(t), np.log(p)
        ss_res = float(((log_t - log_p) ** 2).sum())
        ss_tot = float(((log_t - log_t.mean()) ** 2).sum())
        per.append({
            "order": LABELS[k],
            "median_APE_pct": float(np.nanmedian(a[:, k])),
            "P90_APE_pct": float(np.nanpercentile(a[:, k], 90)),
            "R2_log": 1.0 - ss_res / ss_tot,
            "n": int(good.sum()),
        })
    rows.append({"family": name, "per_order": per})
with open(os.path.join(FIG, "figure_data.json"), "w", encoding="utf-8") as fh:
    json.dump(rows, fh, indent=2, ensure_ascii=False)

print("figures ->", FIG)
for name in RUNS:
    yt, pr = load(name)
    a = ape_matrix(yt, pr)
    print("%-12s overall median APE %.2f%% | P90 %.2f%% | mean %.2f%%"
          % (name, np.nanmedian(a), np.nanpercentile(a, 90), np.nanmean(a)))