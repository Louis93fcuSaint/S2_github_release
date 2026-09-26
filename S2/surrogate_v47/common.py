# -*- coding: utf-8 -*-
"""Shared data / feature / metric utilities for the v4.7 six-order surrogate study.

Everything the models see is defined here so that every model family is trained
and scored on exactly the same rows, the same 47 features and the same split.

Feature set (identical to the earlier v4.6 study, so numbers stay comparable):
  * 39 raw geometry columns from the S1 feature table
  * 8 engineered physics aggregates -> 47 total
Targets: cs_1_rpm .. cs_6_rpm (first six positive-damped forward-whirl critical
speeds), predicted in log space, missing orders masked per target.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
S1_DIR = PROJECT_DIR / "S1"
NSE_DIR = PROJECT_DIR / "S2" / "neural_surrogate_experiment"

V47_CSV = S1_DIR / "04_versions" / "v4.7" / "dataset_v4.7" / "dataset_v4.7_all.csv"

TARGETS = ["cs_%d_rpm" % i for i in range(1, 7)]

# canonical order of the 39 raw features (matches features_v4.4_100000.csv)
RAW_FEATURES = [
    "L", "OD", "ID", "n_disks", "n_bearings",
    "mat_Steel", "mat_Aluminum", "mat_Titanium",
    "d0_OD", "d0_W", "d0_pos",
    "d1_OD", "d1_W", "d1_pos",
    "d2_OD", "d2_W", "d2_pos",
    "d3_OD", "d3_W", "d3_pos",
    "d4_OD", "d4_W", "d4_pos",
    "d5_OD", "d5_W", "d5_pos",
    "brg_type",
    "b0_K", "b0_C", "b0_pos",
    "b1_K", "b1_C", "b1_pos",
    "b2_K", "b2_C", "b2_pos",
    "b3_K", "b3_C", "b3_pos",
]
assert len(RAW_FEATURES) == 39

# rows whose cs_1 is below this are treated as degenerate rigid-body modes and
# reported separately (they are kept in training, they are just noisy to score)
DEGENERATE_CS1_RPM = 100.0


def load_mlp_module():
    """Import the earlier six-order MLP module to reuse its feature engineering."""
    path = NSE_DIR / "run_six_order_mlp_relabeled.py"
    spec = importlib.util.spec_from_file_location("six_order_mlp_shared", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MLP = None


def mlp_module():
    global _MLP
    if _MLP is None:
        _MLP = load_mlp_module()
    return _MLP


def load_dataset(path: Path = V47_CSV, verbose: bool = True):
    """Return (X, Y, meta) for the assembled v4.7 table.

    X : (n, 47) float32  -- 39 raw + 8 engineered, same recipe as the v4.6 study
    Y : (n, 6)  float64  -- critical speeds in rpm, NaN where unlabelled
    meta : dict with 'source', 'status', 'n_disks', 'n_bearings', 'row_id'
    """
    import pandas as pd

    frame = pd.read_csv(path)
    missing = [c for c in RAW_FEATURES + TARGETS if c not in frame.columns]
    if missing:
        raise ValueError("v4.7 table is missing columns: %s" % missing)

    raw = np.ascontiguousarray(frame[RAW_FEATURES].to_numpy(dtype=np.float64))
    x, feature_names = mlp_module().build_engineered_features(raw, list(RAW_FEATURES))
    y = frame[TARGETS].to_numpy(dtype=np.float64)

    meta = {
        "source": frame["source"].to_numpy(dtype=object),
        "status": frame["status"].to_numpy(dtype=object),
        "n_disks": frame["n_disks"].to_numpy(dtype=np.float64),
        "n_bearings": frame["n_bearings"].to_numpy(dtype=np.float64),
        "row_id": frame["source_row_id"].to_numpy(dtype=np.float64),
    }
    if verbose:
        labelled = np.isfinite(y)
        print("[data] rows=%d features=%d  per-order labelled=%s"
              % (x.shape[0], x.shape[1], labelled.sum(axis=0).tolist()), flush=True)
        print("[data] features=%s" % feature_names, flush=True)
    return np.ascontiguousarray(x, dtype=np.float64), y, meta, feature_names


def make_split(n: int, meta, seed: int = 42, test_size: float = 0.20):
    """Stratified 80/20 split on (source, n_disks); identical for every model."""
    from sklearn.model_selection import train_test_split

    strat = np.array([
        "%s|d%d|b%d" % (s, int(d), int(b))
        for s, d, b in zip(meta["source"], meta["n_disks"], meta["n_bearings"])
    ])
    index = np.arange(n)
    train_idx, test_idx = train_test_split(
        index, test_size=test_size, random_state=seed, stratify=strat
    )
    return np.sort(train_idx), np.sort(test_idx)


def save_split(path: Path, train_idx, test_idx, seed: int, test_size: float):
    np.savez_compressed(
        str(path), train_idx=train_idx, test_idx=test_idx,
        seed=np.int64(seed), test_size=np.float64(test_size),
    )


def load_split(path: Path):
    blob = np.load(str(path))
    return blob["train_idx"], blob["test_idx"]


METRIC_KEYS = ["n", "MAE_rpm", "RMSE_rpm", "MAPE_pct", "median_APE_pct",
               "P90_APE_pct", "R2", "R2_log", "RMSLE"]


def regression_metrics(y_true, y_pred) -> dict:
    """APEs in %, plus R2 on the log target which is what the models fit."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    good = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    y_true, y_pred = y_true[good], y_pred[good]
    if y_true.size == 0:
        return {k: float("nan") for k in METRIC_KEYS}

    safe_pred = np.maximum(y_pred, 1e-9)
    ape = np.abs(y_true - safe_pred) / y_true * 100.0
    log_true, log_pred = np.log(y_true), np.log(safe_pred)
    ss_res = float(np.sum((log_true - log_pred) ** 2))
    ss_tot = float(np.sum((log_true - log_true.mean()) ** 2))

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "n": int(y_true.size),
        "MAE_rpm": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_rpm": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_pct": float(np.mean(ape)),
        "median_APE_pct": float(np.median(ape)),
        "P90_APE_pct": float(np.quantile(ape, 0.90)),
        "R2": float(r2_score(y_true, y_pred)),
        "R2_log": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "RMSLE": float(np.sqrt(np.mean((log_true - log_pred) ** 2))),
    }


def evaluate_predictions(y_true, y_pred, *, drop_degenerate: bool = False,
                         degenerate_col: int = 0,
                         threshold: float = DEGENERATE_CS1_RPM) -> dict:
    """Per-order metrics + aggregate summary. Missing orders are masked out."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if drop_degenerate:
        keep = ~(np.isfinite(y_true[:, degenerate_col])
                 & (y_true[:, degenerate_col] < threshold))
        y_true, y_pred = y_true[keep], y_pred[keep]

    per_target = {}
    for k, target in enumerate(TARGETS):
        valid = np.isfinite(y_true[:, k])
        if valid.sum() == 0:
            continue
        per_target[target] = regression_metrics(y_true[valid, k], y_pred[valid, k])

    def mean_of(key):
        values = [per_target[t][key] for t in per_target
                  if np.isfinite(per_target[t][key])]
        return float(np.mean(values)) if values else None

    finite_rows = np.isfinite(y_true)
    diff = np.diff(y_pred, axis=1)
    return {
        "per_target": per_target,
        "mean_MAPE_pct": mean_of("MAPE_pct"),
        "mean_median_APE_pct": mean_of("median_APE_pct"),
        "mean_P90_APE_pct": mean_of("P90_APE_pct"),
        "mean_R2": mean_of("R2"),
        "mean_R2_log": mean_of("R2_log"),
        "mean_RMSLE": mean_of("RMSLE"),
        "monotonic_violation_rate": float(np.mean(np.any(diff <= 0, axis=1))),
    }


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(path: Path, payload) -> None:
    os.makedirs(str(Path(path).parent), exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as fh:
        json.dump(jsonable(payload), fh, indent=2, ensure_ascii=False)


def print_metrics_table(name: str, evaluation: dict) -> None:
    print("  %-22s %10s %10s %10s %10s %10s" %
          (name, "R2", "R2_log", "MAPE%", "medAPE%", "P90APE%"), flush=True)
    for target in TARGETS:
        m = evaluation["per_target"].get(target)
        if not m:
            continue
        print("    %-20s %10.5f %10.5f %10.2f %10.2f %10.2f" %
              (target, m["R2"], m["R2_log"], m["MAPE_pct"],
               m["median_APE_pct"], m["P90_APE_pct"]), flush=True)
    print("    %-20s %10.5f %10.5f %10.2f %10.2f %10.2f" %
          ("MEAN", evaluation["mean_R2"], evaluation["mean_R2_log"],
           evaluation["mean_MAPE_pct"], evaluation["mean_median_APE_pct"],
           evaluation["mean_P90_APE_pct"]), flush=True)