# -*- coding: utf-8 -*-
"""Follow-up diversity analysis using only the frozen pilot ROSS results.

No new ROSS solve is performed.  The distance definition is intentionally the
same as `summarize_eaai_pilot.py`: Euclidean distance over active canonical
coordinates, divided by `ctx["scales"]` (std over the finite v4.7 pool), with
same-family reference sets from the reconstructed training split.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eaai_common as A

THRESHOLDS = [1e-6, 1e-4, 1e-3]
N_BOOT = 200
BOOT_SEED = 20261024
N_EQUAL = 69


def family_label(nd, nb):
    return "%dd-%db" % (int(nd), int(nb))


def active_cols(family):
    a, b = int(family[0]), int(family[1])
    cols = [0, 1, 2]
    cols += [3 + 3 * j + k for j in range(a) for k in (0, 1, 2)]
    cols += [A.S.BRG_BASE + 3 * j + k for j in range(b) for k in (0, 1, 2)]
    return np.asarray(cols, dtype=int)


def normalized_matrix(designs, family, scales):
    cols = active_cols(family)
    z = np.asarray(designs, dtype=np.float64)[:, cols] / np.maximum(scales[cols], 1e-12)
    return np.sqrt(np.maximum(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1), 0.0))


def ref_distance(designs, family, scales, train_designs, train_labels):
    label = family_label(*family)
    ref = train_designs[train_labels == label]
    if len(ref) == 0:
        return np.full(len(designs), np.nan)
    cols = active_cols(family)
    q = np.asarray(designs, dtype=np.float64)[:, cols] / np.maximum(scales[cols], 1e-12)
    r = np.asarray(ref, dtype=np.float64)[:, cols] / np.maximum(scales[cols], 1e-12)
    out = []
    for qi in q:
        out.append(float(np.sqrt(np.maximum(((r - qi) ** 2).sum(1), 0.0)).min()))
    return np.asarray(out)


def _entropy(labels):
    vals, counts = np.unique(labels, return_counts=True)
    if len(vals) <= 1:
        return 0.0
    p = counts.astype(float) / counts.sum()
    return float(-(p * np.log2(p)).sum())


def load_delivered():
    all_slots = pd.read_csv(os.path.join(A.ROSS_DIR, "all_slots.csv"))
    lookup = []
    source_lookup = []
    for pid in sorted(os.listdir(A.SHORT_DIR)):
        p = os.path.join(A.SHORT_DIR, pid, "shortlist.csv")
        if not os.path.exists(p):
            continue
        f = pd.read_csv(p)
        f["pool_id"] = pid
        lookup.append(f[["pool_id", "candidate_id"] + A.S.CANONICAL_COLUMNS])
        pp = os.path.join(A.POOL_DIR, pid, "pool.csv")
        if os.path.exists(pp):
            pf = pd.read_csv(pp, usecols=lambda c: c in ("attempt_id", "source_ref_id"))
            if "source_ref_id" in pf.columns:
                pf["pool_id"] = pid
                source_lookup.append(pf[["pool_id", "attempt_id", "source_ref_id"]])
    lookup = pd.concat(lookup, ignore_index=True)
    d = all_slots[(all_slots["list_type"] == "shortlist") & all_slots["delivered"]].copy()
    d = d.merge(lookup, on=["pool_id", "candidate_id"], how="left")
    if source_lookup:
        sl = pd.concat(source_lookup, ignore_index=True).drop_duplicates(["pool_id", "attempt_id"])
        d = d.merge(sl, on=["pool_id", "attempt_id"], how="left")
    else:
        d["source_ref_id"] = np.nan
    if d[A.S.CANONICAL_COLUMNS].isna().any().any():
        raise RuntimeError("canonical lookup failed")
    return d

def main():
    ctx, train_idx, _, split_meta = A.build_context_split(save=False)
    train_designs = ctx["designs"][train_idx]
    train_labels = np.asarray([family_label(a, b) for a, b in zip(ctx["nd"][train_idx], ctx["nb"][train_idx])])
    scales = np.asarray(ctx["scales"], dtype=np.float64)
    delivered = load_delivered()

    equal_rows = []
    replicate_rows = []
    for method in ("ddpm", "cvae", "retrieval"):
        sub = delivered[(delivered["spec"] == "CAL05") & (delivered["method"] == method)].copy()
        sub = sub.drop_duplicates("design_key").reset_index(drop=True)
        designs = sub[A.S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
        families = [(int(a), int(b)) for a, b in zip(sub["nd"], sub["nb"])]
        if len(designs) < N_EQUAL:
            raise RuntimeError("CAL05 %s has %d delivered designs, fewer than %d" % (method, len(designs), N_EQUAL))
        fam = families[0]
        D = normalized_matrix(designs, fam, scales)
        ref = ref_distance(designs, fam, scales, train_designs, train_labels)
        rng = np.random.default_rng(BOOT_SEED)
        reps = []
        for r in range(N_BOOT):
            idx = np.sort(rng.choice(len(designs), size=N_EQUAL, replace=False))
            subD = D[np.ix_(idx, idx)].copy()
            iu = np.triu_indices(N_EQUAL, k=1)
            np.fill_diagonal(subD, np.inf)
            reps.append({
                "method": method, "replicate": r, "n": N_EQUAL,
                "median_pairwise": float(np.median(subD[iu])),
                "median_nn": float(np.median(subD.min(1))),
                "median_ref": float(np.median(ref[idx])),
                "p10_ref": float(np.percentile(ref[idx], 10)),
            })
        reps = pd.DataFrame(reps)
        replicate_rows.append(reps)
        equal_rows.append({
            "analysis": "CAL05_equalized_69", "spec": "CAL05", "method": method,
            "family": family_label(*fam), "n_available": len(designs), "n_equalized": N_EQUAL,
            "n_replicates": N_BOOT,
            "median_pairwise_base": reps.iloc[0]["median_pairwise"],
            "median_pairwise_median": reps["median_pairwise"].median(),
            "median_pairwise_q05": reps["median_pairwise"].quantile(0.05),
            "median_pairwise_q95": reps["median_pairwise"].quantile(0.95),
            "median_nn_base": reps.iloc[0]["median_nn"],
            "median_nn_median": reps["median_nn"].median(),
            "median_nn_q05": reps["median_nn"].quantile(0.05),
            "median_nn_q95": reps["median_nn"].quantile(0.95),
            "median_ref_base": reps.iloc[0]["median_ref"],
            "median_ref_median": reps["median_ref"].median(),
            "median_ref_q05": reps["median_ref"].quantile(0.05),
            "median_ref_q95": reps["median_ref"].quantile(0.95),
            "family_assignment_source": "external family sequence passed to the generator/spec; not learned by the model",
        })
    pd.concat(replicate_rows, ignore_index=True).to_csv(
        os.path.join(A.STUDY, "pilot_diversity_equalized_replicates.csv"), index=False, encoding="utf-8")

    fam_rows = []
    cal02 = delivered[delivered["spec"] == "CAL02"].copy()
    all_fams = sorted(set(family_label(a, b) for a, b in zip(cal02["nd"], cal02["nb"])))
    for fam_label_ in all_fams:
        for method in ("ddpm", "cvae", "retrieval"):
            sub = cal02[(cal02["method"] == method) &
                        (cal02.apply(lambda r: family_label(r["nd"], r["nb"]) == fam_label_, axis=1))].copy()
            sub = sub.drop_duplicates("design_key")
            n = len(sub)
            row = {"analysis": "CAL02_common_family", "spec": "CAL02", "method": method,
                   "family": fam_label_, "n_available": n, "n_equalized": np.nan,
                   "n_replicates": 0, "family_assignment_source":
                   "external family sequence passed to the generator/spec; not learned by the model"}
            if n >= 2:
                designs = sub[A.S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
                fam = (int(sub["nd"].iloc[0]), int(sub["nb"].iloc[0]))
                d = normalized_matrix(designs, fam, scales)
                iu = np.triu_indices(n, k=1)
                np.fill_diagonal(d, np.inf)
                ref = ref_distance(designs, fam, scales, train_designs, train_labels)
                row.update({
                    "median_pairwise_base": float(np.median(d[iu])),
                    "median_pairwise_median": float(np.median(d[iu])),
                    "median_pairwise_q05": np.nan, "median_pairwise_q95": np.nan,
                    "median_nn_base": float(np.median(d.min(1))),
                    "median_nn_median": float(np.median(d.min(1))),
                    "median_nn_q05": np.nan, "median_nn_q95": np.nan,
                    "median_ref_base": float(np.median(ref)),
                    "median_ref_median": float(np.median(ref)),
                    "median_ref_q05": np.nan, "median_ref_q95": np.nan,
                })
            else:
                for k in ("median_pairwise_base", "median_pairwise_median", "median_pairwise_q05", "median_pairwise_q95",
                          "median_nn_base", "median_nn_median", "median_nn_q05", "median_nn_q95",
                          "median_ref_base", "median_ref_median", "median_ref_q05", "median_ref_q95"):
                    row[k] = np.nan
            fam_rows.append(row)
    equal = pd.concat([pd.DataFrame(equal_rows), pd.DataFrame(fam_rows)], ignore_index=True)
    equal["scale_basis"] = "per-column std over all 199,808 finite v4.7 designs (ctx['scales']), min 1e-9"
    equal["reference_subset"] = "reconstructed training split only, same material/family, active coordinates"
    equal["notes"] = "Bootstrap resampling reflects only sampling variation of the existing delivered sets; not training or cross-spec uncertainty."
    equal.to_csv(os.path.join(A.STUDY, "pilot_diversity_equalized.csv"), index=False, encoding="utf-8")

    src_rows = []
    for spec in ("CAL02", "CAL05"):
        sub = delivered[(delivered["spec"] == spec) & (delivered["method"] == "retrieval")].copy()
        sub = sub.drop_duplicates("design_key").reset_index(drop=True)
        sub = sub[sub["source_ref_id"].notna()].reset_index(drop=True)
        if len(sub) == 0:
            continue
        labels = sub["source_ref_id"].astype(str).to_numpy()
        families = [(int(a), int(b)) for a, b in zip(sub["nd"], sub["nb"])]
        designs = sub[A.S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
        same_src, diff_src, nn_same = [], [], []
        for fam in sorted(set(families)):
            idx = np.where(np.asarray([f == fam for f in families]))[0]
            if len(idx) < 2:
                continue
            d = normalized_matrix(designs[idx], fam, scales)
            np.fill_diagonal(d, np.inf)
            lab = labels[idx]
            for a in range(len(idx)):
                j = int(np.argmin(d[a]))
                nn_same.append(bool(lab[a] == lab[j]))
                for b in range(a + 1, len(idx)):
                    if lab[a] == lab[b]:
                        same_src.append(float(d[a, b]))
                    else:
                        diff_src.append(float(d[a, b]))
        counts = pd.Series(labels).value_counts()
        n = len(sub)
        summary = {
            "level": "summary", "spec": spec, "source_ref_id": "",
            "n_delivered": n, "n_sources": int(len(counts)),
            "max_source_count": int(counts.max()), "max_source_share": float(counts.max() / n),
            "source_entropy": _entropy(labels.tolist()),
            "n_singleton_sources": int((counts == 1).sum()),
            "n_same_source_pairs": len(same_src), "n_diff_source_pairs": len(diff_src),
            "median_same_source_distance": float(np.median(same_src)) if same_src else np.nan,
            "median_diff_source_distance": float(np.median(diff_src)) if diff_src else np.nan,
            "nn_same_source_fraction": float(np.mean(nn_same)) if nn_same else np.nan,
            "notes": "Source labels are retrieval origins, not engineering design clusters; jitter variants from one source are not independent designs.",
        }
        src_rows.append(summary)
        for source, count in counts.items():
            src_rows.append({
                "level": "source", "spec": spec, "source_ref_id": source,
                "n_delivered": n, "n_sources": int(len(counts)),
                "source_count": int(count), "source_share": float(count / n),
                "max_source_count": int(counts.max()), "max_source_share": float(counts.max() / n),
                "source_entropy": summary["source_entropy"],
                "n_singleton_sources": summary["n_singleton_sources"],
                "n_same_source_pairs": summary["n_same_source_pairs"],
                "n_diff_source_pairs": summary["n_diff_source_pairs"],
                "median_same_source_distance": summary["median_same_source_distance"],
                "median_diff_source_distance": summary["median_diff_source_distance"],
                "nn_same_source_fraction": summary["nn_same_source_fraction"],
                "notes": "Per-source contribution to the delivered retrieval shortlist.",
            })
    pd.DataFrame(src_rows).to_csv(os.path.join(A.STUDY, "retrieval_source_cluster_analysis.csv"),
                                  index=False, encoding="utf-8")

    A.save_json(os.path.join(A.METRIC_DIR, "diversity_reference_definition.json"), {
        "distance": "Euclidean on active canonical coordinates, divided by ctx['scales']",
        "scale_source": "np.maximum(designs.std(axis=0), 1e-9) over the finite v4.7 pool",
        "scale_n_designs": int(len(ctx["designs"])),
        "reference_subset": "reconstructed training split, same material and same (n_disks, n_bearings)",
        "train_n": int(len(train_idx)),
        "split_meta": split_meta,
        "zero_padding": "unused disk/bearing slots excluded from active coordinates",
        "interpretation": "Parameter-space novelty, not engineering value or manufacturing feasibility",
    })
    print("[diversity] wrote equalized, source-cluster and reference-definition files", flush=True)


if __name__ == "__main__":
    main()

