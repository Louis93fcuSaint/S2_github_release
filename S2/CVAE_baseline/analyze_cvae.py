# -*- coding: utf-8 -*-
"""Summarise CVAE vs DDPM under the common selector and ROSS protocol."""
from __future__ import annotations

import argparse
import math
import os
from itertools import combinations

import numpy as np
import pandas as pd

import cvae_common as C
import spec_eval as E
import spec_common as S
import spec_interval as SI


def wilson(k, n, z=1.959963984540054):
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return centre - half, centre + half


def nn_distance(x, scales):
    if len(x) < 2:
        return float("nan")
    z = np.asarray(x, dtype=np.float64) / np.asarray(scales, dtype=np.float64)
    d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def entropy_from_counts(counts):
    counts = np.asarray([v for v in counts if v > 0], dtype=np.float64)
    if counts.size == 0:
        return float("nan")
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


REF_DDPM = os.path.join(C.HERE, "reference_ddpm")


def resolve_inputs(method, tag):
    """Locate a verification and shortlist pair for a method/seed tag.

    CVAE runs live in the working directories.  DDPM seed-42 results are the
    frozen set-objective reference results, kept verbatim under reference_ddpm
    (the verification JSON is in the original ``runs[0]`` layout).
    """
    verify_candidates = [os.path.join(C.VERIFY_DIR, tag + ".json")]
    cand_candidates = [os.path.join(C.CAND_DIR, tag + ".csv")]
    if method == "ddpm":
        verify_candidates.append(os.path.join(REF_DDPM, "verify_norm", tag + ".json"))
        verify_candidates.append(os.path.join(REF_DDPM, "verify", tag + ".json"))
        cand_candidates.append(os.path.join(REF_DDPM, "candidates", tag + ".csv"))
    jp = next((p for p in verify_candidates if os.path.exists(p)), None)
    cp = next((p for p in cand_candidates if os.path.exists(p)), None)
    return jp, cp


def load_verification(path):
    data = C.load_json(path)
    if isinstance(data, dict) and isinstance(data.get("runs"), list) and data["runs"]:
        merged = dict(data["runs"][0])
        for key, value in data.items():
            merged.setdefault(key, value)
        return merged
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="cvae,ddpm")
    parser.add_argument("--specs", default="")
    parser.add_argument("--seeds", default="42,7,13")
    args = parser.parse_args()
    wanted = set(v.strip() for v in args.specs.split(",") if v.strip()) if args.specs else None
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    methods = [v.strip() for v in args.methods.split(",") if v.strip()]
    specs = [sp for sp in C.SPECS if not wanted or sp["name"] in wanted]
    ctx = E.pool_ctx()
    rows = []
    for method in methods:
        for spec in specs:
            for seed in seeds:
                tag = "%s_%s_s%d" % (method, spec["name"], seed)
                jp, cp = resolve_inputs(method, tag)
                if not jp or not cp:
                    continue
                data = load_verification(jp)
                frame = pd.read_csv(cp)
                x = frame[S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
                fam = [(int(a), int(b)) for a, b in zip(frame["n_disks"], frame["n_bearings"])]
                counts = {}
                for f in fam:
                    counts[f] = counts.get(f, 0) + 1
                passed = int(data.get("deliverable_n", 0))
                n = int(data.get("n_shortlist", len(frame)))
                lo, hi = wilson(passed, n)
                pred = frame["cs1_pred"].to_numpy(dtype=np.float64) if "cs1_pred" in frame else np.full(len(frame), np.nan)
                proxy_hit = int(np.sum(SI.inside(pred, spec["target"] * (1 - C.TOL),
                                                  spec["target"] * (1 + C.TOL), 0.0))) if np.isfinite(pred).all() else 0
                rows.append({
                    "method": method, "spec": spec["name"], "material": spec["material"],
                    "target": spec["target"], "seed": seed, "passed": passed, "n": n,
                    "rate": passed / n if n else float("nan"), "wilson_low": lo,
                    "wilson_high": hi, "proxy_inband_shortlist": proxy_hit / n if n else float("nan"),
                    "proxy_precision": passed / proxy_hit if proxy_hit else float("nan"),
                    "nn_distance_median": nn_distance(x, ctx["scales"]),
                    "family_unique": len(counts), "family_entropy": entropy_from_counts(counts.values()),
                    "ross_s_per_delivered": float(data.get("gate_tol5", {}).get("seconds_per_passed", np.nan)),
                })
    detail = pd.DataFrame(rows)
    detail_path = os.path.join(C.RESULT_DIR, "cvae_ddpm_detail.csv")
    detail.to_csv(detail_path, index=False, encoding="utf-8")

    summary = []
    for method in methods:
        part = detail[detail["method"] == method]
        passed, n = int(part["passed"].sum()), int(part["n"].sum())
        lo, hi = wilson(passed, n)
        summary.append({"method": method, "specs": part["spec"].nunique(),
                        "seeds": part["seed"].nunique(), "passed": passed, "n": n,
                        "pooled_rate": passed / n if n else float("nan"),
                        "wilson_low": lo, "wilson_high": hi,
                        "mean_spec_rate": float(part.groupby("spec")["rate"].mean().mean()),
                        "spec_rate_sd": float(part.groupby("spec")["rate"].mean().std(ddof=1)),
                        "mean_nn_distance": float(part["nn_distance_median"].mean()),
                        "mean_family_unique": float(part["family_unique"].mean()),
                        "mean_proxy_precision": float(part["proxy_precision"].mean())})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(C.RESULT_DIR, "cvae_ddpm_summary.csv"),
                      index=False, encoding="utf-8")

    by_spec = (detail.groupby(["method", "spec"], as_index=False)
               .agg(mean_rate=("rate", "mean"), min_rate=("rate", "min"),
                    max_rate=("rate", "max"), sd_rate=("rate", "std"),
                    passed=("passed", "sum"), n=("n", "sum"),
                    mean_nn_distance=("nn_distance_median", "mean"),
                    mean_family_unique=("family_unique", "mean"),
                    mean_proxy_precision=("proxy_precision", "mean")))
    by_spec.to_csv(os.path.join(C.RESULT_DIR, "cvae_ddpm_by_spec.csv"),
                   index=False, encoding="utf-8")

    paired = []
    if "cvae" in methods and "ddpm" in methods:
        cv = detail[detail["method"] == "cvae"].set_index(["spec", "seed"])
        dd = detail[detail["method"] == "ddpm"].set_index(["spec", "seed"])
        common = sorted(set(cv.index) & set(dd.index))
        diffs = []
        for key in common:
            a = float(cv.loc[key, "rate"])
            b = float(dd.loc[key, "rate"])
            diffs.append(a - b)
            paired.append({"spec": key[0], "seed": key[1], "cvae": a, "ddpm": b,
                           "difference": a - b})
        p = float("nan")
        stat = float("nan")
        try:
            from scipy.stats import wilcoxon
            if diffs and np.any(np.abs(diffs) > 1e-15):
                res = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
                stat, p = float(res.statistic), float(res.pvalue)
        except Exception:
            pos = sum(1 for d in diffs if d > 0)
            neg = sum(1 for d in diffs if d < 0)
            stat, p = float(min(pos, neg)), float(2.0 * min(
                sum(math.comb(len(diffs), k) for k in range(0, min(pos, neg) + 1)) / (2 ** len(diffs)), 1.0)) if diffs else (float("nan"), float("nan"))
        paired_df = pd.DataFrame(paired)
        paired_df.to_csv(os.path.join(C.RESULT_DIR, "cvae_vs_ddpm_paired.csv"),
                         index=False, encoding="utf-8")
        comp = {"n_pairs": len(diffs), "mean_difference": float(np.mean(diffs)) if diffs else None,
                "sd_difference": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
                "wilcoxon_stat": stat, "wilcoxon_p": p,
                "note": "Descriptive paired comparison; no equivalence or superiority claim."}
    else:
        comp = {"note": "One method only; no paired comparison."}
    C.save_json(os.path.join(C.RESULT_DIR, "cvae_vs_ddpm_comparison.json"), comp)
    print(summary_df.to_string(index=False), flush=True)
    print("[analysis] paired: %s" % comp, flush=True)


if __name__ == "__main__":
    main()
