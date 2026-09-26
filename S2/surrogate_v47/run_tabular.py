# -*- coding: utf-8 -*-
"""Tabular surrogate models on the v4.7 six-order dataset.

    python run_tabular.py --models lightgbm hist_gbdt extra_trees ridge

Every family is fitted per order on log(cs) and scored on the shared frozen
20% test split. Seeds are ensembled in log space.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

import common as C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["lightgbm", "hist_gbdt", "extra_trees", "ridge"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--out-dir", type=Path, default=C.HERE / "outputs" / "tabular")
    parser.add_argument("--quick", action="store_true",
                        help="tiny smoke run on a subsample")
    return parser.parse_args()


def get_split(args, n, meta):
    path = args.out_dir.parent / "split_v47.npz"
    if path.exists() and not args.quick:
        train_idx, test_idx = C.load_split(path)
        print("[split] loaded %s" % path, flush=True)
    else:
        train_idx, test_idx = C.make_split(n, meta, args.split_seed, args.test_size)
        if not args.quick:
            C.save_split(path, train_idx, test_idx, args.split_seed, args.test_size)
            print("[split] created %s" % path, flush=True)
    return train_idx, test_idx


def fit_predict_lightgbm(x_fit, y_fit, x_test, seed, threads, quick):
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    params = {
        "objective": "regression", "metric": "l2",
        "learning_rate": 0.05 if quick else 0.03,
        "num_leaves": 63, "max_depth": 8, "min_child_samples": 5,
        "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
        "verbosity": -1, "n_jobs": threads, "seed": int(seed),
    }
    idx = np.arange(y_fit.size)
    tr, va = train_test_split(idx, test_size=0.10, random_state=int(seed) + 1009)
    train_set = lgb.Dataset(x_fit[tr], label=y_fit[tr])
    valid_set = lgb.Dataset(x_fit[va], label=y_fit[va], reference=train_set)
    booster = lgb.train(
        params, train_set,
        num_boost_round=300 if quick else 20000,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30 if quick else 200, verbose=False),
                   lgb.log_evaluation(0)],
    )
    best = int(booster.best_iteration or 0)
    return booster.predict(x_test, num_iteration=best), best


def fit_predict_hist_gbdt(x_fit, y_fit, x_test, seed, threads, quick):
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05 if quick else 0.06,
        max_iter=120 if quick else 1200, max_leaf_nodes=63,
        min_samples_leaf=20, l2_regularization=1e-3,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20 if quick else 40, random_state=int(seed),
    )
    model.fit(x_fit, y_fit)
    return model.predict(x_test), int(model.n_iter_)


def fit_predict_extra_trees(x_fit, y_fit, x_test, seed, threads, quick):
    from sklearn.ensemble import ExtraTreesRegressor

    model = ExtraTreesRegressor(
        n_estimators=60 if quick else 300, max_features=0.6,
        min_samples_leaf=2, n_jobs=threads, random_state=int(seed),
    )
    model.fit(x_fit, y_fit)
    return model.predict(x_test), (60 if quick else 300)


def fit_predict_ridge(x_fit, y_fit, x_test, seed, threads, quick):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x_fit, y_fit)
    return model.predict(x_test), 0


FAMILIES = {
    "lightgbm": fit_predict_lightgbm,
    "hist_gbdt": fit_predict_hist_gbdt,
    "extra_trees": fit_predict_extra_trees,
    "ridge": fit_predict_ridge,
}


def main():
    args = parse_args()
    os.makedirs(str(args.out_dir), exist_ok=True)
    np.random.seed(0)

    x, y, meta, feature_names = C.load_dataset()
    train_idx, test_idx = get_split(args, x.shape[0], meta)
    if args.quick:
        train_idx, test_idx = train_idx[:20000], test_idx[:5000]

    log_y = np.full_like(y, np.nan)
    good = np.isfinite(y) & (y > 0)
    log_y[good] = np.log(y[good])
    x_test, y_test = x[test_idx], y[test_idx]

    summary = {"config": dict(vars(args), out_dir=str(args.out_dir)),
               "n_train": int(train_idx.size), "n_test": int(test_idx.size),
               "families": {}}

    for family in args.models:
        if family not in FAMILIES:
            raise SystemExit("unknown family: %s" % family)
        fitter = FAMILIES[family]
        seeds = args.seeds[:1] if args.quick else args.seeds
        print("\n=== %s (seeds=%s) ===" % (family, seeds), flush=True)
        log_pred_by_order, per_order, started = {}, {}, time.time()

        for k, target in enumerate(C.TARGETS):
            mask = np.isfinite(log_y[train_idx, k])
            pool = train_idx[mask]
            x_fit = x[pool]
            y_fit = log_y[pool, k]
            runs = []
            for seed in seeds:
                t0 = time.time()
                pred, extra = fitter(x_fit, y_fit, x_test, seed, args.threads, args.quick)
                runs.append(pred)
                print("  [%s] %s seed=%d n_fit=%d %s %.1fs"
                      % (family, target, seed, pool.size, extra, time.time() - t0),
                      flush=True)
            log_pred_by_order[target] = np.mean(np.stack(runs, axis=0), axis=0)

        for k, target in enumerate(C.TARGETS):
            valid = np.isfinite(y_test[:, k])
            per_order[target] = C.regression_metrics(
                y_test[valid, k], np.exp(log_pred_by_order[target])[valid])

        raw_pred = np.exp(np.stack([log_pred_by_order[t] for t in C.TARGETS], axis=1))
        sorted_pred = np.sort(raw_pred, axis=1)
        result = {
            "family": family, "seeds": seeds,
            "fit_seconds": float(time.time() - started),
            "per_order": per_order,
            "eval_raw": C.evaluate_predictions(y_test, raw_pred),
            "eval_sorted": C.evaluate_predictions(y_test, sorted_pred),
            "eval_raw_drop_degenerate": C.evaluate_predictions(
                y_test, raw_pred, drop_degenerate=True),
        }
        summary["families"][family] = result
        np.savez_compressed(
            str(args.out_dir / ("pred_%s.npz" % family)),
            test_idx=test_idx, raw_pred=raw_pred, sorted_pred=sorted_pred,
            y_test=y_test,
        )
        print("\n-- %s : raw predictions --" % family, flush=True)
        C.print_metrics_table(family, result["eval_raw"])
        C.save_json(args.out_dir / ("metrics_%s.json" % family), result)

    C.save_json(args.out_dir / "summary_tabular.json", summary)
    print("\n[out] %s" % args.out_dir, flush=True)

    print("\n==================== HEADLINE (raw predictions, mean over orders) ====================")
    print("%-14s %9s %9s %9s %9s %9s" % ("family", "R2", "R2_log", "MAPE%", "medAPE%", "P90APE%"))
    for family, result in summary["families"].items():
        e = result["eval_raw"]
        print("%-14s %9.5f %9.5f %9.2f %9.2f %9.2f" %
              (family, e["mean_R2"], e["mean_R2_log"], e["mean_MAPE_pct"],
               e["mean_median_APE_pct"], e["mean_P90_APE_pct"]))


if __name__ == "__main__":
    main()