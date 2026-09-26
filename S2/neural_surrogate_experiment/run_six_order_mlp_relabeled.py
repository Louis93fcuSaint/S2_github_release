"""Train six-order MLP surrogates on v4.6 relabels and optional v4.5 data.

The script compares two model families:
1. One masked multi-output MLP that predicts all six positive-forward orders.
2. Six independent single-output MLPs.

Targets are positive damped forward-whirl critical speeds. Missing high-order
labels are represented by NaN and excluded per target from the loss and
metrics. The optional v4.5 dataset is compatible with v4.6: all forward modes
in the relabeled v4.6 modal table have positive log decrement, so its forward
filter uses the same effective target definition.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


N_ORDERS = 6
TARGETS = [f"cs_{i}_rpm" for i in range(1, N_ORDERS + 1)]
RAW_FEATURE_COUNT = 39
AGG_FEATURE_NAMES = [
    "slenderness",
    "log_L",
    "log_OD",
    "disk_span",
    "brg_span",
    "log_k_min",
    "log_k_mean",
    "mass_ratio",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parents[1]
    parser = argparse.ArgumentParser(
        description="Six-order MLP surrogate on relabeled S1 data"
    )
    parser.add_argument(
        "--relabel-dir",
        type=Path,
        default=project_dir
        / "S1"
        / "04_versions" / "v4.6"
        / "relabeled_full_v4.6",
    )
    parser.add_argument(
        "--features-v46",
        type=Path,
        default=project_dir
        / "S1"
        / "02_datasets" / "output_v4.4"
        / "merged_100k"
        / "features_v4.4_100000.csv",
    )
    parser.add_argument(
        "--v45-dir",
        type=Path,
        default=project_dir
        / "S1"
        / "02_datasets" / "rotor2026-9"
        / "rotor2026-9"
        / "output_100k",
    )
    parser.add_argument("--exclude-v45", action="store_true")
    parser.add_argument("--skip-independent", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...], output_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def read_feature_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        columns = next(reader)
        rows = np.asarray(
            [[float(value) for value in row] for row in reader],
            dtype=np.float64,
        )
    return columns, rows


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_engineered_features(
    raw: np.ndarray, columns: list[str]
) -> tuple[np.ndarray, list[str]]:
    def col(name: str, required: bool = True) -> int:
        try:
            return columns.index(name)
        except ValueError:
            if required:
                raise ValueError(f"Missing feature column: {name}")
            return -1

    i_l = col("L")
    i_od = col("OD")
    i_id = col("ID")
    i_nd = col("n_disks")
    i_nb = col("n_bearings")
    disk_od = [col(f"d{i}_OD") for i in range(6)]
    disk_width = [col(f"d{i}_W") for i in range(6)]
    disk_pos = [col(f"d{i}_pos") for i in range(6)]
    bearing_k = [col(f"b{i}_K") for i in range(4)]
    bearing_pos = [col(f"b{i}_pos") for i in range(4)]

    engineered = np.empty((raw.shape[0], raw.shape[1] + len(AGG_FEATURE_NAMES)))
    engineered[:, : raw.shape[1]] = raw
    for row_index, row in enumerate(raw):
        length = row[i_l]
        outer_diameter = row[i_od]
        inner_diameter = row[i_id]
        n_disks = int(round(row[i_nd]))
        n_bearings = int(round(row[i_nb]))

        d_od = np.asarray([row[disk_od[j]] for j in range(n_disks)])
        d_w = np.asarray([row[disk_width[j]] for j in range(n_disks)])
        d_pos = np.asarray([row[disk_pos[j]] for j in range(n_disks)])
        b_k = np.asarray([row[bearing_k[j]] for j in range(n_bearings)])
        b_pos = np.asarray([row[bearing_pos[j]] for j in range(n_bearings)])

        slender = length / max(outer_diameter, 1e-6)
        log_length = np.log(max(length, 1e-6))
        log_outer_diameter = np.log(max(outer_diameter, 1e-6))
        disk_span = (
            float(d_pos.max() - d_pos.min()) / max(length, 1e-6)
            if n_disks >= 2
            else 0.0
        )
        bearing_span = (
            float(b_pos.max() - b_pos.min()) / max(length, 1e-6)
            if n_bearings >= 2
            else 0.0
        )
        log_k_min = np.log(max(float(b_k.min()), 1e4)) if n_bearings else 0.0
        log_k_mean = np.log(max(float(b_k.mean()), 1e4)) if n_bearings else 0.0

        shaft_volume = np.pi * (outer_diameter / 2.0) ** 2 * length
        if inner_diameter > 0:
            shaft_volume -= np.pi * (inner_diameter / 2.0) ** 2 * length
        disk_volume = float(np.sum(np.pi * (d_od / 2.0) ** 2 * d_w))
        mass_ratio = disk_volume / max(shaft_volume, 1e-10)

        engineered[row_index, raw.shape[1] :] = [
            slender,
            log_length,
            log_outer_diameter,
            disk_span,
            bearing_span,
            log_k_min,
            log_k_mean,
            mass_ratio,
        ]

    return engineered, list(columns) + AGG_FEATURE_NAMES


def extract_v46_targets(
    relabel_dir: Path,
) -> tuple[np.ndarray, list[int], list[str]]:
    mode_path = relabel_dir / "relabel_modes_v4.6_all.csv"
    label_path = relabel_dir / "relabeled_v4.6_all.csv"

    forward: dict[int, list[float]] = defaultdict(list)
    with mode_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["whirl_direction"] != "Forward":
                continue
            wd = float(row["wd_rpm"])
            log_dec = float(row["log_dec"])
            if wd > 0 and log_dec > 0:
                forward[int(row["source_row_id"])].append(wd)

    with label_path.open(newline="", encoding="utf-8") as handle:
        label_rows = list(csv.DictReader(handle))

    targets = np.full((len(label_rows), N_ORDERS), np.nan, dtype=np.float64)
    row_ids: list[int] = []
    sources: list[str] = []
    for output_index, row in enumerate(label_rows):
        row_id = int(row["source_row_id"])
        row_ids.append(row_id)
        sources.append("v4.6")
        values = sorted(forward.get(row_id, []))[:N_ORDERS]
        targets[output_index, : len(values)] = values

    # Verify the modal-derived first three orders against the frozen label CSV.
    if label_path.name == "relabeled_v4.6_all.csv":
        for output_index, row in enumerate(label_rows):
            for order_index in range(3):
                value = row.get(TARGETS[order_index], "")
                if value not in ("", None):
                    expected = float(value)
                    actual = targets[output_index, order_index]
                    if not np.isfinite(actual) or abs(actual - expected) > 0.051:
                        raise ValueError(
                            "v4.6 modal-derived cs1-cs3 disagree with label CSV "
                            f"at source_row_id={row['source_row_id']}"
                        )
    return targets, row_ids, sources


def load_v45_targets(v45_dir: Path) -> tuple[np.ndarray, list[int], list[str]]:
    dataset_path = v45_dir / "dataset_v4.5_100000.csv"
    rows = read_dict_rows(dataset_path)
    targets = np.full((len(rows), N_ORDERS), np.nan, dtype=np.float64)
    row_ids: list[int] = []
    sources: list[str] = []
    for index, row in enumerate(rows):
        row_ids.append(index)
        sources.append("v4.5")
        for order_index, target in enumerate(TARGETS):
            value = row.get(target, "")
            if value not in ("", None):
                targets[index, order_index] = float(value)
    return targets, row_ids, sources


def subset_rows(
    x: np.ndarray,
    y: np.ndarray,
    row_ids: list[int],
    sources: list[str],
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    if max_rows <= 0 or max_rows >= x.shape[0]:
        return x, y, row_ids, sources
    indices = np.arange(max_rows)
    return x[indices], y[indices], [row_ids[i] for i in indices], [sources[i] for i in indices]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ape = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10)) * 100.0
    return {
        "n": int(y_true.size),
        "MAE_rpm": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_rpm": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_pct": float(np.mean(ape)),
        "median_APE_pct": float(np.median(ape)),
        "P90_APE_pct": float(np.quantile(ape, 0.90)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def monotonic_violation_rate(pred: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    return float(np.mean(np.any(np.diff(pred, axis=1) <= 0, axis=1)))


def evaluate_predictions(y_true: np.ndarray, pred: np.ndarray) -> dict:
    metrics: dict[str, dict] = {}
    for order_index, target in enumerate(TARGETS):
        valid = np.isfinite(y_true[:, order_index])
        if int(valid.sum()) == 0:
            continue
        metrics[target] = regression_metrics(
            y_true[valid, order_index], pred[valid, order_index]
        )
    mapes = [metrics[target]["MAPE_pct"] for target in metrics]
    r2s = [metrics[target]["R2"] for target in metrics]
    full_six = np.isfinite(y_true).all(axis=1)
    return {
        "per_target": metrics,
        "mean_MAPE_pct": float(np.mean(mapes)) if mapes else None,
        "mean_R2": float(np.mean(r2s)) if r2s else None,
        "monotonic_violation_rate_all_test": monotonic_violation_rate(pred),
        "monotonic_violation_rate_full_six_label_rows": (
            monotonic_violation_rate(pred[full_six])
            if bool(full_six.any())
            else None
        ),
    }


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    squared = (prediction - target) ** 2 * mask
    counts = mask.sum(dim=0)
    present = counts > 0
    if not bool(present.any()):
        return prediction.sum() * 0.0
    per_target = squared.sum(dim=0)[present] / counts[present]
    return per_target.mean()


def evaluate_masked_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            loss = masked_mse(model(x[start:stop]), y[start:stop], mask[start:stop])
            total += float(loss.item()) * (stop - start)
            count += stop - start
    return total / max(count, 1)


def make_balanced_multi_batches(
    fit_indices: np.ndarray,
    target_mask: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    pools = []
    for order_index in range(N_ORDERS):
        valid = fit_indices[target_mask[fit_indices, order_index]]
        if valid.size:
            pools.append(valid)
    if not pools:
        raise ValueError("No labelled training rows")

    per_target = max(1, batch_size // len(pools))
    n_steps = max(1, math.ceil(len(fit_indices) / batch_size))
    batches = []
    for _ in range(n_steps):
        parts = []
        for pool in pools:
            replace = pool.size < per_target
            parts.append(rng.choice(pool, size=per_target, replace=replace))
        batch = np.concatenate(parts)
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def train_multi_model(
    x: np.ndarray,
    log_y: np.ndarray,
    target_mask: np.ndarray,
    fit_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    seed: int,
    hidden: tuple[int, ...],
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> dict:
    set_seed(seed)
    x_scaler = StandardScaler().fit(x[fit_indices])
    means = np.nanmean(log_y[fit_indices], axis=0)
    stds = np.nanstd(log_y[fit_indices], axis=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    normalized = np.zeros_like(log_y, dtype=np.float32)
    column_index = np.broadcast_to(
        np.arange(N_ORDERS), target_mask.shape
    )[target_mask]
    normalized[target_mask] = (
        (log_y[target_mask] - means[column_index]) / stds[column_index]
    )

    train_x = torch.as_tensor(x_scaler.transform(x), dtype=torch.float32)
    train_y = torch.as_tensor(normalized, dtype=torch.float32)
    train_mask = torch.as_tensor(target_mask.astype(np.float32), dtype=torch.float32)

    model = TorchMLP(x.shape[1], tuple(hidden), N_ORDERS)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    history: list[dict] = []
    rng = np.random.default_rng(seed + 7919)
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        batches = make_balanced_multi_batches(
            fit_indices, target_mask, batch_size, rng
        )
        for batch_indices in batches:
            batch_indices = np.asarray(batch_indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_x[batch_indices])
            loss = masked_mse(
                prediction,
                train_y[batch_indices],
                train_mask[batch_indices],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.item()) * batch_indices.size
            seen += batch_indices.size

        train_loss = running_loss / max(seen, 1)
        val_loss = evaluate_masked_loss(
            model,
            train_x[val_indices],
            train_y[val_indices],
            train_mask[val_indices],
            batch_size,
        )
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if val_loss < best_val - 1e-6:
            best_val = float(val_loss)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_normalized = model(train_x[test_indices]).cpu().numpy().astype(np.float64)
    test_log_pred = test_normalized * stds + means
    raw_pred = np.exp(test_log_pred)
    sorted_pred = np.sort(raw_pred, axis=1)

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "fit_seconds": float(time.time() - start_time),
        "test_log_pred": test_log_pred,
        "raw_pred": raw_pred,
        "sorted_pred": sorted_pred,
        "x_scaler": x_scaler,
        "target_means_log": means,
        "target_stds_log": stds,
    }


def train_independent_model(
    x: np.ndarray,
    y_log: np.ndarray,
    order_index: int,
    fit_pool: np.ndarray,
    test_pool: np.ndarray,
    *,
    seed: int,
    hidden: tuple[int, ...],
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    validation_fraction: float,
) -> dict:
    labelled_pool = fit_pool[np.isfinite(y_log[fit_pool, order_index])]
    if labelled_pool.size < 20:
        raise ValueError(
            f"Too few labels for {TARGETS[order_index]}: {labelled_pool.size}"
        )
    fit_indices, val_indices = train_test_split(
        labelled_pool,
        test_size=validation_fraction,
        random_state=seed + 1009 + order_index,
    )
    test_indices = test_pool[np.isfinite(y_log[test_pool, order_index])]

    set_seed(seed + 100 * order_index)
    x_scaler = StandardScaler().fit(x[fit_indices])
    y_scaler = StandardScaler().fit(
        y_log[fit_indices, order_index].reshape(-1, 1)
    )
    train_x = torch.as_tensor(x_scaler.transform(x), dtype=torch.float32)
    train_y = torch.as_tensor(
        y_scaler.transform(y_log[:, order_index].reshape(-1, 1)), dtype=torch.float32
    )
    model = TorchMLP(x.shape[1], tuple(hidden), 1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    history: list[dict] = []
    rng = np.random.default_rng(seed + 104729 + order_index)
    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = rng.permutation(fit_indices)
        running = 0.0
        for start in range(0, permutation.size, batch_size):
            batch = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[batch]), train_y[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += float(loss.item()) * batch.size
        train_loss = running / max(permutation.size, 1)

        model.eval()
        with torch.no_grad():
            val_prediction = model(train_x[val_indices])
            val_loss = float(criterion(val_prediction, train_y[val_indices]).item())
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_loss < best_val - 1e-6:
            best_val = float(val_loss)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_test_log = y_scaler.inverse_transform(
            model(train_x[test_pool]).cpu().numpy()
        )[:, 0].astype(np.float64)
    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "fit_seconds": float(time.time() - start_time),
        "all_test_log": all_test_log,
        "test_indices": test_indices,
        "n_fit": int(fit_indices.size),
        "n_validation": int(val_indices.size),
        "n_test": int(test_indices.size),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }


def ensemble_multi_runs(runs: list[dict], y_test: np.ndarray) -> dict:
    log_prediction = np.mean(
        np.stack([run["test_log_pred"] for run in runs], axis=0), axis=0
    )
    raw = np.exp(log_prediction)
    sorted_prediction = np.sort(raw, axis=1)
    return {
        "raw_metrics": evaluate_predictions(y_test, raw),
        "sorted_metrics": evaluate_predictions(y_test, sorted_prediction),
        "raw_prediction": raw,
        "sorted_prediction": sorted_prediction,
    }


def train_independent_ensemble(
    x: np.ndarray,
    y_log: np.ndarray,
    fit_pool: np.ndarray,
    test_pool: np.ndarray,
    *,
    seeds: list[int],
    hidden: tuple[int, ...],
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    validation_fraction: float,
) -> dict:
    results: dict[str, dict] = {}
    test_log_by_order = {}
    all_runs = []
    for order_index, target in enumerate(TARGETS):
        order_runs = []
        for seed in seeds:
            run = train_independent_model(
                x,
                y_log,
                order_index,
                fit_pool,
                test_pool,
                seed=seed,
                hidden=hidden,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                validation_fraction=validation_fraction,
            )
            prediction = np.exp(run["all_test_log"])
            valid = np.isfinite(y_log[test_pool, order_index])
            metrics = regression_metrics(
                np.exp(y_log[test_pool, order_index][valid]), prediction[valid]
            )
            run["metrics"] = metrics
            run["seed"] = int(seed)
            run["target"] = target
            order_runs.append(run)
            all_runs.append(run)

        test_log_by_order[target] = np.mean(
            np.stack([run["all_test_log"] for run in order_runs], axis=0), axis=0
        )
        valid_test = np.isfinite(y_log[test_pool, order_index])
        results[target] = {
            "per_seed": [
                {
                    "seed": run["seed"],
                    "metrics": run["metrics"],
                    "best_epoch": run["best_epoch"],
                    "best_val_loss": run["best_val_loss"],
                    "fit_seconds": run["fit_seconds"],
                    "n_fit": run["n_fit"],
                    "n_validation": run["n_validation"],
                    "n_test": run["n_test"],
                    "history": run["history"],
                }
                for run in order_runs
            ],
            "ensemble_metrics": regression_metrics(
                np.exp(y_log[test_pool, order_index][valid_test]),
                np.exp(test_log_by_order[target])[valid_test],
            ),
        }

    raw_pred = np.exp(
        np.stack([test_log_by_order[target] for target in TARGETS], axis=1)
    )
    sorted_pred = np.sort(raw_pred, axis=1)
    y_test = np.exp(y_log[test_pool])
    return {
        "target_results": results,
        "raw_metrics": evaluate_predictions(y_test, raw_pred),
        "sorted_metrics": evaluate_predictions(y_test, sorted_pred),
        "raw_prediction": raw_pred,
        "sorted_prediction": sorted_pred,
        "all_runs": all_runs,
    }


def aggregate_history(histories: list[list[dict]]) -> list[dict]:
    max_epoch = max(len(history) for history in histories)
    output = []
    for epoch_index in range(max_epoch):
        train_values = [
            history[epoch_index]["train_loss"]
            for history in histories
            if epoch_index < len(history)
        ]
        val_values = [
            history[epoch_index]["val_loss"]
            for history in histories
            if epoch_index < len(history)
        ]
        output.append(
            {
                "epoch": epoch_index + 1,
                "train_mean": float(np.mean(train_values)),
                "train_std": float(np.std(train_values)),
                "val_mean": float(np.mean(val_values)),
                "val_std": float(np.std(val_values)),
                "n_runs": len(train_values),
            }
        )
    return output


def save_loss_plot(path: Path, histories: list[list[dict]], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = aggregate_history(histories)
    epochs = np.asarray([row["epoch"] for row in aggregate])
    train_mean = np.asarray([row["train_mean"] for row in aggregate])
    val_mean = np.asarray([row["val_mean"] for row in aggregate])
    train_std = np.asarray([row["train_std"] for row in aggregate])
    val_std = np.asarray([row["val_std"] for row in aggregate])

    plt.figure(figsize=(8.2, 5.0))
    plt.plot(epochs, train_mean, color="#0F766E", label="Train")
    plt.fill_between(
        epochs,
        train_mean - train_std,
        train_mean + train_std,
        color="#0F766E",
        alpha=0.16,
    )
    plt.plot(epochs, val_mean, color="#BE123C", label="Validation")
    plt.fill_between(
        epochs,
        val_mean - val_std,
        val_mean + val_std,
        color="#BE123C",
        alpha=0.14,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Masked standardized-log MSE")
    plt.title(title)
    plt.grid(alpha=0.22)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_independent_curves(path: Path, target_results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4))
    for order_index, target in enumerate(TARGETS):
        axis = axes.flat[order_index]
        for seed_run in target_results[target]["per_seed"]:
            axis.plot(
                [row["epoch"] for row in seed_run["history"]],
                [row["val_loss"] for row in seed_run["history"]],
                linewidth=1.1,
                alpha=0.75,
                label=f"seed {seed_run['seed']}",
            )
        axis.set_title(target)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation MSE")
        axis.grid(alpha=0.20)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Independent six-order MLP validation curves")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_prediction_plot(path: Path, y_true: np.ndarray, pred: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.4))
    for order_index, target in enumerate(TARGETS):
        axis = axes.flat[order_index]
        valid = np.isfinite(y_true[:, order_index])
        truth = y_true[valid, order_index]
        prediction = pred[valid, order_index]
        if truth.size:
            metrics = regression_metrics(truth, prediction)
            axis.scatter(truth, prediction, s=4, alpha=0.22, color="#2563EB")
            lower = float(min(truth.min(), prediction.min()))
            upper = float(max(truth.max(), prediction.max()))
            axis.plot([lower, upper], [lower, upper], color="#111827", linewidth=1)
            axis.set_title(
                f"{target}: MAPE={metrics['MAPE_pct']:.2f}%, R2={metrics['R2']:.4f}"
            )
        else:
            axis.set_title(f"{target}: no labels")
        axis.set_xlabel("ROSS RPM")
        axis.set_ylabel("Predicted RPM")
        axis.grid(alpha=0.20)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, results: dict) -> None:
    lines = [
        "# 六阶临界转速 MLP 代理模型结果",
        "",
        "## 数据与协议",
        "",
        f"- 合并样本数：`{results['dataset']['n_samples']}`",
        f"- 输入维度：`{results['dataset']['n_features']}`",
        f"- 训练数据来源：`{', '.join(results['dataset']['sources'])}`",
        f"- 数据划分：`{results['split']['n_fit']}` / `{results['split']['n_validation']}` / `{results['split']['n_test']}`",
        "- 标签：按 `wd_rpm` 排序的前六阶正阻尼正进动临界转速；缺失阶次用掩码排除。",
        "- 目标变换：逐阶 `log(cs)` 标准化；多输出模型对每一阶使用等权掩码 MSE。",
        "",
        "## 各阶标签覆盖",
        "",
        "| 阶次 | 可用标签数 |",
        "| --- | ---: |",
    ]
    for target in TARGETS:
        lines.append(f"| {target} | {results['dataset']['target_counts'][target]} |")

    lines.extend(["", "## 测试集结果", ""])
    for model_name in ("multi_output", "independent_six_networks"):
        if model_name not in results["models"]:
            continue
        for prediction_name in ("raw_metrics", "sorted_metrics"):
            block = results["models"][model_name].get(prediction_name)
            if not block:
                continue
            lines.extend(
                [
                    f"### {model_name} / {prediction_name}",
                    "",
                    "| 阶次 | n | MAE/rpm | MAPE | Median APE | P90 APE | R2 |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for target in TARGETS:
                metric = block["per_target"].get(target)
                if metric is None:
                    lines.append(f"| {target} | 0 | - | - | - | - | - |")
                else:
                    lines.append(
                        f"| {target} | {metric['n']} | {metric['MAE_rpm']:.1f} | "
                        f"{metric['MAPE_pct']:.2f}% | {metric['median_APE_pct']:.2f}% | "
                        f"{metric['P90_APE_pct']:.2f}% | {metric['R2']:.4f} |"
                    )
            lines.append("")

    lines.extend(
        [
            "## 结论",
            "",
            "- `cs1-cs3` 有接近全量的样本标签，可以直接评价。",
            "- `cs4-cs5` 标签量明显减少，`cs6` 只有数百条，六阶结果必须结合标签覆盖量解释。",
            "- 多输出网络共享输入表征；六个独立网络避免不同阶次之间的误差耦合，但高阶模型可训练数据更少。",
            "- 排序后的预测强制满足阶次单调性，原始预测则保留模型自身的排序误差。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    output_dir = args.output_dir or (
        Path(__file__).resolve().parent / "outputs_six_order_mlp_relabeled_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    columns, raw_v46 = read_feature_csv(args.features_v46)
    x_v46, feature_names = build_engineered_features(raw_v46, columns)
    y_v46, ids_v46, source_v46 = extract_v46_targets(args.relabel_dir)
    if x_v46.shape[0] != y_v46.shape[0]:
        raise ValueError("v4.6 feature and label row counts differ")

    x_parts = [x_v46]
    y_parts = [y_v46]
    row_ids = list(ids_v46)
    sources = list(source_v46)
    if not args.exclude_v45:
        columns_v45, raw_v45 = read_feature_csv(
            args.v45_dir / "features_v4.5_100000.csv"
        )
        if columns_v45 != columns:
            raise ValueError("v4.6 and v4.5 feature schemas differ")
        x_v45, _ = build_engineered_features(raw_v45, columns_v45)
        y_v45, ids_v45, source_v45 = load_v45_targets(args.v45_dir)
        x_parts.append(x_v45)
        y_parts.append(y_v45)
        row_ids.extend(ids_v45)
        sources.extend(source_v45)

    x = np.vstack(x_parts)
    y = np.vstack(y_parts)
    x, y, row_ids, sources = subset_rows(
        x, y, row_ids, sources, args.max_rows
    )
    target_mask = np.isfinite(y)
    has_any = target_mask.any(axis=1)
    x = x[has_any]
    y = y[has_any]
    row_ids = [value for value, keep in zip(row_ids, has_any) if keep]
    sources = [value for value, keep in zip(sources, has_any) if keep]
    target_mask = np.isfinite(y)
    log_y = np.full(y.shape, np.nan, dtype=np.float64)
    log_y[target_mask] = np.log(y[target_mask])

    counts = {
        target: int(np.isfinite(y[:, index]).sum())
        for index, target in enumerate(TARGETS)
    }
    source_counts = dict(Counter(sources))

    all_indices = np.arange(x.shape[0])
    train_pool, test_indices = train_test_split(
        all_indices,
        test_size=args.test_size,
        random_state=args.split_seed,
        shuffle=True,
    )
    fit_indices, val_indices = train_test_split(
        train_pool,
        test_size=args.validation_fraction,
        random_state=args.split_seed + 1009,
        shuffle=True,
    )

    results = {
        "experiment": "six-order MLP on v4.6 relabels and optional v4.5 data",
        "dataset": {
            "n_samples": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "feature_names": feature_names,
            "targets": TARGETS,
            "target_counts": counts,
            "source_counts": source_counts,
            "sources": sorted(source_counts),
            "target_definition": "first six positive damped forward-whirl critical speeds",
            "missing_target_policy": "NaN mask; excluded per target from loss and metrics",
        },
        "split": {
            "split_seed": int(args.split_seed),
            "test_size": float(args.test_size),
            "validation_fraction_of_train_pool": float(args.validation_fraction),
            "n_fit": int(fit_indices.size),
            "n_validation": int(val_indices.size),
            "n_test": int(test_indices.size),
        },
        "settings": {
            "seeds": [int(seed) for seed in args.seeds],
            "hidden": [int(width) for width in args.hidden],
            "batch_size": int(args.batch_size),
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "threads": int(args.threads),
            "multi_output_training": "per-target-balanced masked batches",
            "independent_training": "one model per order on that order's labelled rows",
        },
        "models": {},
    }

    multi_runs = []
    for seed in args.seeds:
        print(f"[multi] seed={seed}", flush=True)
        run = train_multi_model(
            x,
            log_y,
            target_mask,
            fit_indices,
            val_indices,
            test_indices,
            seed=seed,
            hidden=tuple(args.hidden),
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        run["seed"] = int(seed)
        multi_runs.append(run)
        torch.save(
            {
                "state_dict": run["model"].state_dict(),
                "config": {
                    "input_dim": int(x.shape[1]),
                    "hidden": [int(width) for width in args.hidden],
                    "output_dim": N_ORDERS,
                },
                "seed": int(seed),
                "target_means_log": run["target_means_log"],
                "target_stds_log": run["target_stds_log"],
            },
            output_dir / f"multi_output_seed{seed}.pt",
        )
        print(
            f"  best_epoch={run['best_epoch']} "
            f"val={run['best_val_loss']:.6f} "
            f"time={run['fit_seconds']:.1f}s",
            flush=True,
        )

    multi_ensemble = ensemble_multi_runs(
        multi_runs, np.exp(log_y[test_indices])
    )
    results["models"]["multi_output"] = {
        "per_seed": [
            {
                "seed": run["seed"],
                "best_epoch": run["best_epoch"],
                "best_val_loss": run["best_val_loss"],
                "fit_seconds": run["fit_seconds"],
                "raw_metrics": evaluate_predictions(
                    np.exp(log_y[test_indices]), run["raw_pred"]
                ),
                "sorted_metrics": evaluate_predictions(
                    np.exp(log_y[test_indices]), run["sorted_pred"]
                ),
            }
            for run in multi_runs
        ],
        "raw_metrics": multi_ensemble["raw_metrics"],
        "sorted_metrics": multi_ensemble["sorted_metrics"],
    }
    (output_dir / "multi_output_history.json").write_text(
        json.dumps([run["history"] for run in multi_runs], indent=2),
        encoding="utf-8",
    )
    (output_dir / "multi_output_linked_history.json").write_text(
        json.dumps(aggregate_history([run["history"] for run in multi_runs]), indent=2),
        encoding="utf-8",
    )
    save_loss_plot(
        output_dir / "multi_output_loss_curves.png",
        [run["history"] for run in multi_runs],
        "Six-order multi-output MLP",
    )
    save_prediction_plot(
        output_dir / "multi_output_predictions_sorted.png",
        np.exp(log_y[test_indices]),
        multi_ensemble["sorted_prediction"],
        "Multi-output MLP ensemble (sorted)",
    )

    if not args.skip_independent:
        print("[independent] six single-output networks", flush=True)
        independent = train_independent_ensemble(
            x,
            log_y,
            train_pool,
            test_indices,
            seeds=args.seeds,
            hidden=tuple(args.hidden),
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            validation_fraction=args.validation_fraction,
        )
        results["models"]["independent_six_networks"] = {
            "target_results": independent["target_results"],
            "raw_metrics": independent["raw_metrics"],
            "sorted_metrics": independent["sorted_metrics"],
        }
        for run in independent["all_runs"]:
            torch.save(
                {
                    "state_dict": run["model"].state_dict(),
                    "config": {
                        "input_dim": int(x.shape[1]),
                        "hidden": [int(width) for width in args.hidden],
                        "output_dim": 1,
                    },
                    "seed": int(run["seed"]),
                    "target": run["target"],
                },
                output_dir
                / f"independent_{run['target']}_seed{run['seed']}.pt",
            )
        save_independent_curves(
            output_dir / "independent_validation_curves.png",
            independent["target_results"],
        )
        save_prediction_plot(
            output_dir / "independent_predictions_sorted.png",
            np.exp(log_y[test_indices]),
            independent["sorted_prediction"],
            "Independent six-network ensemble (sorted)",
        )

    np.savez_compressed(
        output_dir / "test_predictions.npz",
        y_true=np.exp(log_y[test_indices]),
        multi_output_raw=multi_ensemble["raw_prediction"],
        multi_output_sorted=multi_ensemble["sorted_prediction"],
        **(
            {
                "independent_raw": independent["raw_prediction"],
                "independent_sorted": independent["sorted_prediction"],
            }
            if not args.skip_independent
            else {}
        ),
    )
    (output_dir / "results_six_order_mlp.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(output_dir / "六阶代理模型结果.md", results)
    print(f"Saved results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
