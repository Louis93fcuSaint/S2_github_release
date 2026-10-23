# -*- coding: utf-8 -*-
"""Fixed-configuration grouped 5-fold CV of the final 1024-512-256 5-seed surrogate.

The configuration is frozen to the one recorded in
S2/surrogate_v47/outputs/mlp_v47_best/summary_mlp.json.  Outer folds are grouped
by (material, n_disks, n_bearings), so related rotor families never cross the
outer train/test boundary.  Log-input columns, input scaler, target scaler and
early-stopping validation split are fitted/selected inside each outer training
set only.  The outer test fold is never used for early stopping or tuning.

This is a fixed-configuration CV, not a nested CV that repeats model selection.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SURROGATE_DIR = Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2\surrogate_v47")
sys.path.insert(0, str(SURROGATE_DIR))
import common as C  # noqa: E402
import run_mlp as R  # noqa: E402

HIDDEN = (1024, 512, 256)
SEEDS = (42, 43, 44, 45, 46)
N_FOLDS = 5


def sha256_file(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def family_groups(frame: pd.DataFrame) -> np.ndarray:
    mat = np.select(
        [frame["mat_Steel"].to_numpy() == 1,
         frame["mat_Aluminum"].to_numpy() == 1,
         frame["mat_Titanium"].to_numpy() == 1],
        ["Steel", "Aluminum", "Titanium"], default="Other")
    return np.array([
        "%s|d%d|b%d" % (m, int(d), int(b))
        for m, d, b in zip(mat, frame["n_disks"], frame["n_bearings"])
    ])


def make_folds(groups: np.ndarray):
    from sklearn.model_selection import GroupKFold
    folds = []
    for train_idx, test_idx in GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(groups)), groups=groups):
        folds.append((np.sort(train_idx), np.sort(test_idx)))
    return folds


def make_args(fold_id: int, smoke: bool):
    return argparse.Namespace(
        arms=["multi"], seeds=[42] if smoke else list(SEEDS),
        hidden=list(HIDDEN), batch_size=1024,
        max_epochs=2 if smoke else 800, patience=2 if smoke else 80,
        learning_rate=1e-3, weight_decay=1e-5, batchnorm=False, dropout=0.0,
        threads=6, validation_fraction=0.10, split_seed=4200 + fold_id,
        test_size=0.20, out_dir=Path("."), loss="l1", target_mode="log",
        log_inputs=True, huber_beta=0.3, save_models=False, quick=False,
        probe=False, probe_rows=40000, probe_epochs=80,
        probe_test_rows=8000, probe_layouts=[],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    raw_frame = pd.read_csv(C.V47_CSV, usecols=C.RAW_FEATURES)
    groups = family_groups(raw_frame)
    folds = make_folds(groups)
    x, y, meta, feature_names = C.load_dataset(verbose=True)
    log_y = np.full_like(y, np.nan)
    good = np.isfinite(y) & (y > 0)
    log_y[good] = np.log(y[good])

    manifest = {
        "dataset": str(C.V47_CSV),
        "dataset_sha256": sha256_file(C.V47_CSV),
        "n_rows": int(x.shape[0]), "n_features": int(x.shape[1]),
        "feature_names": feature_names,
        "hidden": list(HIDDEN), "seeds": list(SEEDS),
        "loss": "l1", "target": "log(cs_rpm)", "ensemble": "mean in log space",
        "folds": "GroupKFold(5) grouped by material|n_disks|n_bearings",
        "config_source": "S2/surrogate_v47/outputs/mlp_v47_best/summary_mlp.json",
        "note": "fixed-configuration CV; outer test never used for tuning",
        "fold_sizes": [{"fold": i, "train": int(tr.size), "test": int(te.size)}
                       for i, (tr, te) in enumerate(folds)],
    }
    C.save_json(args.out / "cv_manifest.json", manifest)

    for fold_id in args.folds:
        dest = args.out / ("smoke_fold%d" % fold_id if args.smoke else "fold%d" % fold_id)
        if (dest / "metrics.json").exists() and not args.smoke:
            print("[skip] fold %d already complete: %s" % (fold_id, dest), flush=True)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        train_idx, test_idx = folds[fold_id]
        np.savez_compressed(dest / "split.npz", train_idx=train_idx, test_idx=test_idx,
                            groups=groups[test_idx])
        fold_x = np.array(x, copy=True)
        run_args = make_args(fold_id, args.smoke)
        log_cols = R.log_input_columns(fold_x, train_idx)
        if log_cols:
            fold_x = R.apply_log_inputs(fold_x, log_cols)
        R.M.TorchMLP = R.make_mlp_class(False, 0.0)
        R.M.masked_mse = R.make_masked_loss("l1")
        print("\n=== fold %d: train=%d test=%d log_cols=%s ==="
              % (fold_id, train_idx.size, test_idx.size, log_cols), flush=True)
        t0 = time.time()
        raw_pred, sorted_pred, per_seed, seconds = R.run_multi_arm(
            run_args, fold_x, y, log_y, train_idx, test_idx, HIDDEN)
        elapsed = time.time() - t0
        payload = {
            "fold": fold_id, "train_size": int(train_idx.size),
            "test_size": int(test_idx.size), "log_input_columns": [int(i) for i in log_cols],
            "seed_metrics": per_seed,
            "ensemble_metrics": C.evaluate_predictions(y[test_idx], raw_pred),
            "fit_seconds_reported": seconds, "wall_seconds": elapsed,
            "smoke": bool(args.smoke),
            "n_test_families": int(len(set(groups[test_idx]))),
            "test_families": sorted(set(groups[test_idx].tolist())),
        }
        C.save_json(dest / "metrics.json", payload)
        np.savez_compressed(dest / "pred_ensemble.npz",
                            y_true=y[test_idx], y_pred=raw_pred,
                            row_index=test_idx, groups=groups[test_idx])
        pd.DataFrame([{"seed": SEEDS[i] if not args.smoke else 42,
                       "mean_MAPE_pct": m["mean_MAPE_pct"],
                       "mean_median_APE_pct": m["mean_median_APE_pct"],
                       "mean_P90_APE_pct": m["mean_P90_APE_pct"],
                       "mean_R2_log": m["mean_R2_log"]}
                      for i, m in enumerate(per_seed)]).to_csv(
            dest / "seed_metrics.csv", index=False)
        print("[fold %d] done in %.1f s  ensemble MAPE=%.3f%%" %
              (fold_id, elapsed, payload["ensemble_metrics"]["mean_MAPE_pct"]), flush=True)
        del fold_x


if __name__ == "__main__":
    main()
