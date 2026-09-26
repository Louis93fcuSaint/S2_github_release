# -*- coding: utf-8 -*-
"""Where does the mean APE come from? Decompose the error tail."""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
LABELS = ["cs%d" % i for i in range(1, 7)]


def load(path):
    blob = np.load(path)
    return blob["y_test"], blob["raw_pred"]


def ape_matrix(y_true, y_pred):
    out = np.full(y_true.shape, np.nan)
    good = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    out[good] = (np.abs(y_true[good] - np.maximum(y_pred[good], 1e-9))
                 / y_true[good] * 100.0)
    return out


y_true, pred = load(os.path.join(OUT, "mlp_v47", "pred_multi.npz"))
ape = ape_matrix(y_true, pred)

print("=== overall ===")
flat = ape.ravel()
flat = flat[np.isfinite(flat)]
print("rows %d | mean %.2f%% | median %.2f%% | P90 %.2f%% | P99 %.2f%% | max %.0f%%"
      % (flat.size, flat.mean(), np.median(flat), np.percentile(flat, 90),
         np.percentile(flat, 99), flat.max()))

print("\n=== how much of the mean comes from the worst rows? ===")
order = np.sort(flat)[::-1]
total = flat.sum()
for frac in (0.001, 0.005, 0.01, 0.05, 0.10, 0.25):
    k = max(1, int(frac * flat.size))
    print("  worst %5.1f%% of rows (n=%6d) contribute %5.1f%% of the mean-APE mass"
          % (frac * 100, k, 100.0 * order[:k].sum() / total))

print("\n=== mean APE if we trim the worst x% per order ===")
print("%-6s %8s %8s %8s %8s %8s" % ("order", "mean", "trim1%", "trim2%", "trim5%", "median"))
for k in range(6):
    col = ape[:, k]
    col = col[np.isfinite(col)]
    s = np.sort(col)[::-1]
    n = col.size
    print("%-6s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%%"
          % (LABELS[k], col.mean(),
             s[int(0.01 * n):].mean(), s[int(0.02 * n):].mean(),
             s[int(0.05 * n):].mean(), np.median(col)))

print("\n=== is the tail concentrated at small true speeds? ===")
rows_t, rows_p = [], []
for k in range(6):
    good = np.isfinite(y_true[:, k])
    rows_t.append(y_true[good, k])
    rows_p.append(ape[good, k])
t = np.concatenate(rows_t)
a = np.concatenate(rows_p)
edges = [0, 1e3, 5e3, 2e4, 1e5, 1e6, 1e9]
print("%-18s %10s %9s %9s %9s" % ("true cs bucket", "n", "mean", "median", "share of mass"))
total = a.sum()
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (t >= lo) & (t < hi)
    if sel.sum() == 0:
        continue
    print("%-18s %10d %8.2f%% %8.2f%% %8.1f%%"
          % ("[%.0f, %.0f)" % (lo, hi), sel.sum(), a[sel].mean(),
             np.median(a[sel]), 100.0 * a[sel].sum() / total))

print("\n=== same bucket table, contribution of the worst 1% inside each bucket ===")
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (t >= lo) & (t < hi)
    if sel.sum() < 100:
        continue
    col = np.sort(a[sel])[::-1]
    k = max(1, int(0.01 * col.size))
    print("  [%.0f, %.0f): mean %.2f%% -> drop worst 1%% -> %.2f%%"
          % (lo, hi, col.mean(), col[k:].mean()))