# -*- coding: utf-8 -*-
"""MLP surrogates on the assembled v4.7 six-order dataset.

Two training strategies, scored on the exact same frozen 80/20 split as
``run_tabular.py`` so numbers are directly comparable with LightGBM:

  multi    one shared network with six masked outputs (per-target balanced
           batches + masked MSE), ensembled over seeds in log space
  single   six independent single-output networks (one per order, trained on
           that order's labelled rows only), ensembled over seeds in log space

    python run_mlp.py --arms multi single --hidden 256 256 128 64

``--probe`` instead runs a hidden-layer sweep on a subsample and exits; that is
how the "narrower per layer, but deeper" layouts requested by our advisor are
chosen before spending hours on the full run.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import common as C

N_ORDERS = len(C.TARGETS)
M = C.mlp_module()

PROBE_LAYOUTS = [
    "256 128",              # v4.6 baseline
    "128 128 128 128",      # narrower, deeper
    "192 192 96 96",
    "256 256 128 64",
    "128 128 128 128 64",
]


def make_mlp_class(batchnorm: bool = False, dropout: float = 0.0):
    activation = nn.ReLU

    class MLP(nn.Module):
        """Plain stack of Linear(+BN/Dropout)+activation, same ctor signature
        as the v4.6 TorchMLP so the reference trainers can use it unchanged."""

        def __init__(self, input_dim: int, hidden, output_dim: int):
            super().__init__()
            layers = []
            previous = input_dim
            for width in hidden:
                layers.append(nn.Linear(previous, width))
                if batchnorm:
                    layers.append(nn.BatchNorm1d(width))
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                previous = width
            layers.append(nn.Linear(previous, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    return MLP


def to_ratio_targets(log_abs):
    """log(cs_k) -> [log(cs1), log(cs2/cs1), ...]; NaN if either neighbour missing."""
    out = np.full_like(log_abs, np.nan)
    out[:, 0] = log_abs[:, 0]
    out[:, 1:] = log_abs[:, 1:] - log_abs[:, :-1]
    return out


def ratio_to_abs(ratio_log):
    """Inverse of to_ratio_targets: cumsum gives log(cs_k) again."""
    return np.cumsum(ratio_log, axis=1)


def log_input_columns(x, fit_idx, ratio_threshold=100.0):
    """Pick columns that are strictly positive but span >100x (skewed for an MLP)."""
    block = x[fit_idx]
    lo, hi = block.min(axis=0), block.max(axis=0)
    return [i for i in range(x.shape[1])
            if lo[i] > 1e-12 and hi[i] / lo[i] > ratio_threshold]


def apply_log_inputs(x, columns):
    if not columns:
        return x
    out = np.array(x, dtype=np.float64, copy=True)
    out[:, columns] = np.log(out[:, columns])
    return out


def make_masked_loss(kind="mse", beta=1.0):
    """Masked per-target training loss.

    The metric we care about is APE, i.e. a *relative* error.  In log space the
    residual is log(pred/true), so L1 there is the natural surrogate for APE,
    whereas MSE (the default) over-weights the few largest log errors.
    """
    import torch as _torch
    import torch.nn.functional as _F

    def masked(prediction, target, mask):
        diff = prediction - target
        if kind == "l1":
            element = diff.abs()
        elif kind == "huber":
            element = _F.smooth_l1_loss(diff, _torch.zeros_like(diff),
                                        reduction="none", beta=beta)
        else:
            element = diff ** 2
        element = element * mask
        counts = mask.sum(dim=0)
        present = counts > 0
        if not bool(present.any()):
            return prediction.sum() * 0.0
        return (element.sum(dim=0)[present] / counts[present]).mean()

    return masked


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["multi", "single"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 128, 64])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--out-dir", type=Path, default=C.HERE / "outputs" / "mlp_v47")
    parser.add_argument("--loss", default="mse", choices=["mse", "l1", "huber"])
    parser.add_argument("--target-mode", default="log", choices=["log", "ratio"],
                        help="ratio: predict log(cs1) then log(cs_k/cs_{k-1})")
    parser.add_argument("--log-inputs", action="store_true",
                        help="log-transform strictly positive, wide-range inputs")
    parser.add_argument("--huber-beta", type=float, default=0.3)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--probe", action="store_true",
                        help="run the hidden-layer sweep on a subsample and exit")
    parser.add_argument("--probe-rows", type=int, default=40000)
    parser.add_argument("--probe-epochs", type=int, default=80)
    parser.add_argument("--probe-test-rows", type=int, default=8000)
    parser.add_argument("--probe-layouts", nargs="+", default=PROBE_LAYOUTS)
    return parser.parse_args()


def get_split(args, n, meta):
    path = C.HERE / "outputs" / "split_v47.npz"   # canonical, shared by every run
    if path.exists() and not args.quick:
        train_idx, test_idx = C.load_split(path)
        print("[split] loaded %s" % path, flush=True)
    else:
        train_idx, test_idx = C.make_split(n, meta, args.split_seed, args.test_size)
        if not args.quick and not args.probe:
            C.save_split(path, train_idx, test_idx, args.split_seed, args.test_size)
            print("[split] created %s" % path, flush=True)
    return train_idx, test_idx


def parse_layout(text):
    return tuple(int(v) for v in text.split())


def fit_val_split(train_idx, validation_fraction, split_seed):
    from sklearn.model_selection import train_test_split
    return train_test_split(train_idx, test_size=validation_fraction,
                            random_state=split_seed + 1009)


def run_multi_arm(args, x, y, log_y, train_idx, test_idx, hidden):
    ratio_mode = args.target_mode == "ratio"
    target_log = to_ratio_targets(log_y) if ratio_mode else log_y
    target_mask = np.isfinite(target_log)
    fit_idx, val_idx = fit_val_split(train_idx, args.validation_fraction, args.split_seed)
    seeds = args.seeds[:1] if args.quick else args.seeds
    max_epochs = 20 if args.quick else args.max_epochs
    patience = 6 if args.quick else args.patience

    log_runs, per_seed_metrics, started = [], [], time.time()
    for seed in seeds:
        t0 = time.time()
        run = M.train_multi_model(
            x, target_log, target_mask, fit_idx, val_idx, test_idx,
            seed=seed, hidden=tuple(hidden), batch_size=args.batch_size,
            max_epochs=max_epochs, patience=patience,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        )
        run_log = ratio_to_abs(run["test_log_pred"]) if ratio_mode else run["test_log_pred"]
        log_runs.append(run_log)
        per_seed_metrics.append(C.evaluate_predictions(y[test_idx], np.exp(run_log)))
        if args.save_models:
            _save_state(args, "multi_seed%d" % seed, run["model"], x.shape[1],
                        tuple(hidden), N_ORDERS,
                        {"x_mean": run["x_scaler"].mean_,
                         "x_scale": run["x_scaler"].scale_,
                         "y_mean": run["target_means_log"],
                         "y_std": run["target_stds_log"]})
        del run["model"]
        print("  [multi] seed=%d best_epoch=%d val_loss=%.5f %.1fs"
              % (seed, run["best_epoch"], run["best_val_loss"], time.time() - t0),
              flush=True)

    log_pred = np.mean(np.stack(log_runs, axis=0), axis=0)
    raw_pred = np.exp(log_pred)
    return raw_pred, np.sort(raw_pred, axis=1), per_seed_metrics, float(time.time() - started)


def run_single_arm(args, x, y, log_y, train_idx, test_idx, hidden):
    seeds = args.seeds[:1] if args.quick else args.seeds
    max_epochs = 20 if args.quick else args.max_epochs
    patience = 6 if args.quick else args.patience

    log_by_order, per_seed_metrics, started = {}, [], time.time()
    for order_index, target in enumerate(C.TARGETS):
        runs = []
        for seed in seeds:
            t0 = time.time()
            result = M.train_independent_model(
                x, log_y, order_index, train_idx, test_idx,
                seed=seed, hidden=tuple(hidden), batch_size=args.batch_size,
                max_epochs=max_epochs, patience=patience,
                learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                validation_fraction=args.validation_fraction,
            )
            runs.append(result["all_test_log"])
            if args.save_models:
                _save_state(args, "%s_seed%d" % (target, seed), result["model"],
                            x.shape[1], tuple(hidden), 1,
                            {"x_mean": result["x_scaler"].mean_,
                             "x_scale": result["x_scaler"].scale_,
                             "y_mean": result["y_scaler"].mean_,
                             "y_std": result["y_scaler"].scale_})
            del result["model"]
            print("  [single] %-9s seed=%d best_epoch=%d val_loss=%.5f %.1fs"
                  % (target, seed, result["best_epoch"], result["best_val_loss"],
                     time.time() - t0), flush=True)
        log_by_order[target] = np.mean(np.stack(runs, axis=0), axis=0)

    log_pred = np.stack([log_by_order[t] for t in C.TARGETS], axis=1)
    raw_pred = np.exp(log_pred)
    per_seed_metrics.append(C.evaluate_predictions(y[test_idx], raw_pred))
    return raw_pred, np.sort(raw_pred, axis=1), per_seed_metrics, float(time.time() - started)


def _save_state(args, name, model, input_dim, hidden, output_dim, extras=None):
    """Persist weights plus the input/target scalers so the model can be
    reloaded for inference (e.g. inside the DDPM loop)."""
    directory = Path(args.out_dir) / "models"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(),
               "config": {"input_dim": int(input_dim),
                          "hidden": [int(w) for w in hidden],
                          "output_dim": int(output_dim)}}
    if extras:
        payload.update(extras)
    torch.save(payload, str(directory / ("%s.pt" % name)))


ARMS = {"multi": run_multi_arm, "single": run_single_arm}


def run_probe(args, x, y, log_y, train_idx, test_idx):
    """Hidden-layer sweep with the single-output arm, one seed, on a subsample."""
    rng = np.random.default_rng(args.split_seed)
    fit_pool = rng.choice(train_idx, size=min(args.probe_rows, train_idx.size),
                          replace=False)
    probe_test = rng.choice(test_idx, size=min(args.probe_test_rows, test_idx.size),
                            replace=False)
    probe_test = np.sort(probe_test)

    print("\n===== hidden-layer probe (single-output arm, 1 seed, "
          "train=%d test=%d) =====" % (fit_pool.size, probe_test.size), flush=True)
    print("%-22s %10s %10s %10s %9s" % ("hidden", "R2_log", "medAPE%", "P90APE%", "sec"))
    rows, best = [], None
    for text in args.probe_layouts:
        hidden = parse_layout(text)
        t0 = time.time()
        log_pred = np.full((probe_test.size, N_ORDERS), np.nan)
        for order_index in range(N_ORDERS):
            result = M.train_independent_model(
                x, log_y, order_index, fit_pool, probe_test,
                seed=args.seeds[0], hidden=hidden, batch_size=args.batch_size,
                max_epochs=args.probe_epochs, patience=args.patience,
                learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                validation_fraction=args.validation_fraction,
            )
            log_pred[:, order_index] = result["all_test_log"]
            del result["model"]
        seconds = time.time() - t0
        evaluation = C.evaluate_predictions(y[probe_test], np.exp(log_pred))
        print("%-22s %10.5f %10.2f %10.2f %9.1f"
              % (text, evaluation["mean_R2_log"], evaluation["mean_median_APE_pct"],
                 evaluation["mean_P90_APE_pct"], seconds), flush=True)
        rows.append({"hidden": text, "seconds": seconds,
                     "mean_R2_log": evaluation["mean_R2_log"],
                     "mean_median_APE_pct": evaluation["mean_median_APE_pct"],
                     "mean_P90_APE_pct": evaluation["mean_P90_APE_pct"],
                     "per_target": evaluation["per_target"]})
        if best is None or evaluation["mean_R2_log"] > best["mean_R2_log"]:
            best = rows[-1]
    print("\n[probe] best layout by R2_log: %s (%.5f)"
          % (best["hidden"], best["mean_R2_log"]), flush=True)
    C.save_json(Path(args.out_dir) / "probe_hidden.json",
                {"probe_rows": int(fit_pool.size), "probe_epochs": int(args.probe_epochs),
                 "results": rows, "best_hidden": best["hidden"]})
    return best


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    x, y, meta, feature_names = C.load_dataset()
    train_idx, test_idx = get_split(args, x.shape[0], meta)
    if args.quick:
        train_idx, test_idx = train_idx[:20000], test_idx[:5000]

    log_y = np.full_like(y, np.nan)
    good = np.isfinite(y) & (y > 0)
    log_y[good] = np.log(y[good])
    y_test = y[test_idx]

    M.TorchMLP = make_mlp_class(args.batchnorm, args.dropout)
    M.masked_mse = make_masked_loss(args.loss, args.huber_beta)

    log_columns = log_input_columns(x, train_idx) if args.log_inputs else []
    if log_columns:
        x = apply_log_inputs(x, log_columns)
    if args.target_mode == "ratio" and "single" in args.arms:
        print("[warn] ratio targets need all six orders at once; dropping the single arm",
              flush=True)
        args.arms = [a for a in args.arms if a != "single"]

    if args.probe:
        run_probe(args, x, y, log_y, train_idx, test_idx)
        return

    summary = {
        "config": dict(vars(args), out_dir=str(args.out_dir), hidden=list(args.hidden),
                         log_input_columns=[int(i) for i in log_columns]),
        "n_train": int(train_idx.size), "n_test": int(test_idx.size),
        "feature_names": feature_names, "arms": {},
    }

    for arm in args.arms:
        if arm not in ARMS:
            raise SystemExit("unknown arm: %s" % arm)
        print("\n=== arm %s (hidden=%s) ===" % (arm, list(args.hidden)), flush=True)
        raw_pred, sorted_pred, per_seed, seconds = ARMS[arm](
            args, x, y, log_y, train_idx, test_idx, args.hidden)
        result = {
            "arm": arm, "seeds": args.seeds[:1] if args.quick else args.seeds,
            "hidden": list(args.hidden), "batchnorm": bool(args.batchnorm),
            "dropout": float(args.dropout), "fit_seconds": seconds,
            "per_seed_metrics": per_seed,
            "eval_raw": C.evaluate_predictions(y_test, raw_pred),
            "eval_sorted": C.evaluate_predictions(y_test, sorted_pred),
            "eval_raw_drop_degenerate": C.evaluate_predictions(
                y_test, raw_pred, drop_degenerate=True),
        }
        summary["arms"][arm] = result
        np.savez_compressed(
            str(Path(args.out_dir) / ("pred_%s.npz" % arm)),
            test_idx=test_idx, raw_pred=raw_pred, sorted_pred=sorted_pred, y_test=y_test)
        for name, evaluation in (("raw", "eval_raw"), ("sorted", "eval_sorted")):
            print("\n-- %s : %s predictions --" % (arm, name), flush=True)
            C.print_metrics_table(arm, result[evaluation])
        C.save_json(Path(args.out_dir) / ("metrics_%s.json" % arm), result)

    C.save_json(Path(args.out_dir) / "summary_mlp.json", summary)

    print("\n==================== HEADLINE (raw predictions, mean over orders) ====================")
    print("%-12s %9s %9s %9s %9s %9s" % ("arm", "R2", "R2_log", "MAPE%", "medAPE%", "P90APE%"))
    for arm, result in summary["arms"].items():
        e = result["eval_raw"]
        print("%-12s %9.5f %9.5f %9.2f %9.2f %9.2f"
              % (arm, e["mean_R2"], e["mean_R2_log"], e["mean_MAPE_pct"],
                 e["mean_median_APE_pct"], e["mean_P90_APE_pct"]))
    print("\n[out] %s" % args.out_dir, flush=True)


if __name__ == "__main__":
    main()
