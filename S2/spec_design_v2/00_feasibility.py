# -*- coding: utf-8 -*-
"""v2 feasibility study -- pure data, no training, no DDPM.

Question: v1 targets only cs1 (+-5 %).  Can v2 let the user target cs2 / cs3 as
well, alone or combined (cs1+cs2, cs1+cs2+cs3), at +-1 %?

Three things have to hold before touching the generator:

  (1) MARGIN   given cs1 is pinned, cs2 must still be free to move, otherwise a
               cs2 target is not an extra handle but a restatement of cs1.
  (2) SUPPORT  the requested joint cell must be reachable from real data,
               otherwise the generator is extrapolating and the ROSS gate will
               reject almost everything.
  (3) PROXY    a +-1 % band has to be resolvable at all, measured against the
               frozen surrogate's own per-order error.

Writes outputs/feasibility.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(os.path.dirname(HERE), "spec_design")
sys.path.insert(0, V1)
import spec_common as S

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
CS_COLS = ["cs_%d_rpm" % i for i in range(1, 7)]
# frozen v4.7 surrogate, DDPM_v47/条件DDPM报告_v4.7.md
SURROGATE_MAPE = {1: 1.28, 2: 2.19, 3: 3.22, 4: 3.97, 5: 3.96, 6: 4.20}


def pool():
    frame = S.load_pool()
    return frame[np.isfinite(frame[CS_COLS]).all(axis=1)].reset_index(drop=True)


def conditional_rows(log_cs, material, targets, tol):
    rows = []
    for target in targets:
        band = np.abs(log_cs[:, 0] - np.log(target)) <= np.log1p(tol)
        n = int(band.sum())
        row = {"material": material, "cs1_target": round(float(target), 1),
               "tol": float(tol), "n_in_cs1_band": n}
        if n >= 30:
            for j in (2, 3):
                ratio = log_cs[band, j - 1] - log_cs[band, 0]
                p = np.percentile(ratio, [5, 25, 50, 75, 95])
                row["cs%d_iqr_pct_of_median" % j] = round(
                    float(100.0 * (np.exp(p[3]) - np.exp(p[1])) / np.exp(p[2])), 1)
                row["cs%d_p5p95_pct_of_median" % j] = round(
                    float(100.0 * (np.exp(p[4]) - np.exp(p[0])) / np.exp(p[2])), 1)
                row["cs%d_ratio_p50" % j] = round(float(np.exp(p[2])), 3)
        rows.append(row)
    return rows


def joint_rows(cs, material, targets, tol, order_list):
    rows = []
    for target in targets:
        keep = np.abs(cs[:, 0] - target) <= tol * target
        row = {"material": material, "combo": "cs1" if not order_list
               else "cs1+" + "+".join("cs%d" % j for j in order_list),
               "cs1_target": round(float(target), 1), "tol": float(tol),
               "n_cs1": int(keep.sum())}
        for j in order_list:
            centre = float(np.median(cs[keep, j - 1])) if keep.sum() >= 20 \
                else float(np.median(cs[:, j - 1]))
            keep = keep & (np.abs(cs[:, j - 1] - centre) <= tol * centre)
            row["centre_cs%d" % j] = round(centre, 1)
            row["n_%s" % row["combo"]] = int(keep.sum())
        rows.append(row)
    return rows


def median_neighbourhood(log_cs, material, targets, tol, order_list):
    """At a +-tol cs1 band, how many designs sit within +-tol of the *median*
    cs2 (and then cs3)?  This is the densest possible cell of that width."""
    rows = []
    for target in targets:
        band = np.abs(log_cs[:, 0] - np.log(target)) <= np.log1p(tol)
        if band.sum() < 30:
            continue
        row = {"material": material, "cs1_target": round(float(target), 1),
               "tol": float(tol), "n_cs1": int(band.sum())}
        keep = band
        label = "cs1"
        for j in order_list:
            med = float(np.median(log_cs[keep, j - 1]))
            keep = keep & (np.abs(log_cs[:, j - 1] - med) <= np.log1p(tol))
            label += "+cs%d" % j
            row["n_%s" % label] = int(keep.sum())
        rows.append(row)
    return rows


def main():
    frame = pool()
    cs = frame[CS_COLS].to_numpy(dtype=float)
    log_cs = np.log(cs)
    print("[feas] pool with six finite orders: %d" % len(frame))
    corr = np.corrcoef(log_cs[:, :3].T)
    print("[feas] corr(log cs1,cs2)=%.3f  corr(cs2,cs3)=%.3f  corr(cs1,cs3)=%.3f"
          % (corr[0, 1], corr[1, 2], corr[0, 2]))

    result = {"pool_rows": int(len(frame)), "log_cs_corr_1_2_3": corr.tolist(),
              "surrogate_mape_pct": SURROGATE_MAPE,
              "surrogate_band_over_mape": {str(k): round(1.0 / v, 2)
                                           for k, v in SURROGATE_MAPE.items()},
              "conditional": [], "joint": [], "median_neighbourhood": []}

    for material in S.MATERIAL_ORDER:
        mask = (frame["material"] == material).to_numpy()
        sub_cs, sub_log = cs[mask], log_cs[mask]
        targets = [float(v) for v in np.percentile(sub_cs[:, 0], [10, 30, 50, 70, 90])]
        print("\n--- %s  (n=%d)  cs1 deciles %s"
              % (material, mask.sum(), ["%.0f" % t for t in targets]))

        for tol in (0.05, 0.01):
            rows = conditional_rows(sub_log, material, targets, tol)
            result["conditional"] += rows
            print("  [cs1 +-%.0f%% band]  cs2/cs1 free? " % (100 * tol))
            for row in rows:
                if row["n_in_cs1_band"] >= 30:
                    print("     cs1=%7.0f  n=%5d | cs2/cs1 IQR %6.1f%% of median "
                          "(p50 ratio %.2f) | cs3/cs1 IQR %6.1f%%"
                          % (row["cs1_target"], row["n_in_cs1_band"],
                             row["cs2_iqr_pct_of_median"], row["cs2_ratio_p50"],
                             row["cs3_iqr_pct_of_median"]))
                else:
                    print("     cs1=%7.0f  n=%5d  (too thin for a statistic)"
                          % (row["cs1_target"], row["n_in_cs1_band"]))

        for tol in (0.01,):
            for order_list in ([], [2], [3], [2, 3]):
                rows = joint_rows(sub_cs, material, targets, tol, order_list)
                result["joint"] += rows
                for row in rows:
                    key = [k for k in row if k.startswith("n_") and k != "n_cs1"]
                    got = row[key[0]] if key else row["n_cs1"]
                    print("  [joint +-1%%] %-14s centre cs1=%7.0f -> n=%d"
                          % (row["combo"], row["cs1_target"], got))

        for tol in (0.01, 0.05):
            rows = median_neighbourhood(sub_log, material, targets, tol, [2, 3])
            result["median_neighbourhood"] += rows
            print("  [densest cell, cs1 +-%.0f%%]" % (100 * tol))
            for row in rows:
                print("     cs1=%7.0f  n(cs1)=%5d -> n(cs1+cs2)=%5d -> n(cs1+cs2+cs3)=%5d"
                      % (row["cs1_target"], row["n_cs1"],
                         row.get("n_cs1+cs2", 0), row.get("n_cs1+cs2+cs3", 0)))

    print("\n--- proxy headroom: 1 %% band vs per-order MAPE ---")
    for k, v in SURROGATE_MAPE.items():
        print("  cs%d : MAPE %5.2f %% -> band/MAPE = %.2f %s"
              % (k, v, 1.0 / v, "(band inside the error bar)" if v > 1.0 else ""))

    path = os.path.join(OUT, "feasibility.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print("\n[feas] wrote %s" % path)


if __name__ == "__main__":
    main()