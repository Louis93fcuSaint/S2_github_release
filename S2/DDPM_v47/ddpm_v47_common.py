# -*- coding: utf-8 -*-
"""Shared machinery for the v4.7 conditional-DDPM restart (slice nd=3, nb=2).

Everything the generator and every baseline need is defined here, so DDPM / DE /
GA / BO are all scored by the *same* surrogate on the *same* frozen slice:

    load_slice()            -> the 11,894-row nd=3/nb=2 slice of v4.7
    fixed_split()           -> frozen 80/10/10 split (seed 42, stratified by material)
    select_targets()        -> the 9 real target cases T01-T09 taken from the test set
    compact <-> 47 features -> the exact feature recipe the surrogate was trained on
    predict_cs()            -> fast surrogate evaluation (MLP ensemble or LightGBM)
    is_valid_compact()      -> the v4.7 physical constraint filter
    run_ross_on_compact()   -> ground-truth ROSS verification (48 modes, forward whirl)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(HERE))
S1_DIR = os.path.join(PROJECT_DIR, "S1")
S1_V47 = os.path.join(S1_DIR, "04_versions", "v4.7")
S2_DIR = os.path.join(PROJECT_DIR, "S2")
SURROGATE_DIR = os.path.join(S2_DIR, "surrogate_v47")
NEURAL_DIR = os.path.join(PROJECT_DIR, "S2", "neural_surrogate_experiment")

SLICE_CSV = os.path.join(S1_V47, "dataset_v4.7", "dataset_v4.7_all.csv")

SEED = 42
N_DISKS = 3
N_BEARINGS = 2
MATERIAL_ORDER = ["Steel", "Aluminum", "Titanium"]
PRIMARY_CS = ["cs_1_rpm", "cs_2_rpm", "cs_3_rpm"]
ALL_CS = ["cs_%d_rpm" % i for i in range(1, 7)]

# compact design vector: exactly the v4.4/v4.5 layout, which the slice still obeys
COMPACT_COLUMNS = [
    "L", "OD", "ID",
    "d0_OD", "d0_W", "d0_pos",
    "d1_OD", "d1_W", "d1_pos",
    "d2_OD", "d2_W", "d2_pos",
    "b0_K", "b0_C", "b0_pos",
    "b1_K", "b1_C", "b1_pos",
]

_MODULE_CACHE = {}


def _load_module(name, path):
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[name] = module
    return module


def constraint_module():
    return _load_module("s1_constraint_v47_ddpm",
                        os.path.join(S1_V47, "s1_constraint_filter_v4_7.py"))


def ross_module():
    return _load_module("s1_pipeline_v47_ddpm",
                        os.path.join(S1_V47, "s1_run_pipeline_v4_7.py"))


def surrogate_common_module():
    return _load_module("surrogate_v47_common",
                        os.path.join(SURROGATE_DIR, "common.py"))


# --------------------------------------------------------------------------- #
# slice / split / targets
# --------------------------------------------------------------------------- #
def load_slice(path=SLICE_CSV):
    """Return the frozen nd=3/nb=2 slice as a DataFrame (11,894 rows)."""
    import pandas as pd

    frame = pd.read_csv(path)
    mask = ((frame["status"] == "success")
            & (frame["n_disks"] == N_DISKS)
            & (frame["n_bearings"] == N_BEARINGS))
    out = frame.loc[mask].reset_index(drop=True)
    material = np.full(len(out), "", dtype=object)
    for name in MATERIAL_ORDER:
        material[out["mat_%s" % name].to_numpy() > 0.5] = name
    out["material"] = material
    if (material == "").any():
        raise ValueError("slice has rows with no material one-hot set")
    return out


def fixed_split(slice_frame):
    """Frozen stratified 80/10/10 split by material (experiment_seed = 42)."""
    from sklearn.model_selection import train_test_split

    position = np.arange(len(slice_frame))
    material = slice_frame["material"].to_numpy()
    train_all, test_pos = train_test_split(
        position, test_size=0.10, random_state=SEED, stratify=material)
    train_pos, val_pos = train_test_split(
        train_all, test_size=1.0 / 9.0, random_state=SEED,
        stratify=material[train_all])
    return np.sort(train_pos), np.sort(val_pos), np.sort(test_pos)


def select_targets(slice_test):
    """Nine real target cases: cs1 at P20/P50/P80 per material, taken from test."""
    rows = []
    for material in MATERIAL_ORDER:
        group = slice_test[slice_test["material"] == material].sort_values("cs_1_rpm")
        for level, quantile in (("P20", 0.2), ("P50", 0.5), ("P80", 0.8)):
            want = float(group["cs_1_rpm"].quantile(quantile))
            best = group.loc[(group["cs_1_rpm"] - want).abs().idxmin()]
            rows.append({
                "target_id": "T%02d" % (len(rows) + 1),
                "source_row_id": int(best["source_row_id"]),
                "source": str(best["source"]),
                "material": material,
                "level": level,
                "cs_targets": [float(best[c]) for c in ALL_CS],
            })
    return rows


def freeze(out_dir=os.path.join(HERE, "outputs")):
    """Write the frozen split and target cases; safe to re-run (deterministic)."""
    os.makedirs(out_dir, exist_ok=True)
    frame = load_slice()
    train_pos, val_pos, test_pos = fixed_split(frame)
    targets = select_targets(frame.iloc[test_pos])
    payload_split = {
        "experiment_seed": SEED,
        "slice_rows": int(len(frame)),
        "slice_rule": "status==success and n_disks==3 and n_bearings==2",
        "source_csv": SLICE_CSV,
        "train_row_id": [int(v) for v in frame["source_row_id"].to_numpy()[train_pos]],
        "validation_row_id": [int(v) for v in frame["source_row_id"].to_numpy()[val_pos]],
        "test_row_id": [int(v) for v in frame["source_row_id"].to_numpy()[test_pos]],
        "counts": {"train": int(train_pos.size), "validation": int(val_pos.size),
                   "test": int(test_pos.size)},
    }
    save_json(os.path.join(out_dir, "ddpm_v47_split.json"), payload_split)
    save_json(os.path.join(out_dir, "ddpm_v47_targets.json"), targets)
    return frame, (train_pos, val_pos, test_pos), targets


# --------------------------------------------------------------------------- #
# compact design vector <-> model features
# --------------------------------------------------------------------------- #
def sort_compact(x):
    """Sort disk / bearing slots by axial position, keeping geometry aligned."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape[0] != 18:
        raise ValueError("compact x must have 18 dims")
    out = x.copy()
    for base, slots in ((3, [5, 8, 11]), (12, [14, 17])):
        order = np.argsort(x[slots], kind="stable")
        for dst, src in enumerate(order):
            out[base + 3 * dst:base + 3 * dst + 3] = x[base + 3 * src:base + 3 * src + 3]
    return out


def sample_from_compact(x, material):
    """Compact 18-vector -> S1 sample dict (the schema the S1 code speaks)."""
    x = sort_compact(x)
    return {
        "sl": float(x[0]), "od": float(x[1]), "id": float(x[2]),
        "material": material, "bearing_type": "isotropic",
        "nd": N_DISKS,
        "disk_od_list": [float(x[3]), float(x[6]), float(x[9])],
        "disk_w_list": [float(x[4]), float(x[7]), float(x[10])],
        "dp": [float(x[5]), float(x[8]), float(x[11])],
        "nb": N_BEARINGS,
        "bk_list": [float(x[12]), float(x[15])],
        "bc_list": [float(x[13]), float(x[16])],
        "bp": [float(x[14]), float(x[17])],
    }


def material_onehot(material):
    if material not in MATERIAL_ORDER:
        raise ValueError("unknown material: %s" % material)
    return [1.0 if material == m else 0.0 for m in MATERIAL_ORDER]


def condition_vector(cs_values, material):
    cs = [float(v) for v in cs_values]
    return cs + material_onehot(material)


def compact_to_model_rows(compact_rows, materials):
    """(n, 18) compact + materials -> (n, 47) rows in the surrogate's own recipe.

    to_features_v4 is already batch capable, so one call replaces the per-row
    python loop, which was the single biggest cost inside the search loop.
    """
    module = constraint_module()
    engine = surrogate_common_module().mlp_module()
    samples = [sample_from_compact(x, m) for x, m in zip(compact_rows, materials)]
    raw, raw_names = module.to_features_v4(samples)
    raw = np.asarray(raw, dtype=np.float64)
    model_rows, feature_names = engine.build_engineered_features(raw, list(raw_names))
    return np.asarray(model_rows, dtype=np.float64), list(feature_names)


def is_valid_compact(x, material):
    module = constraint_module()
    ok, reasons = module.is_valid_v4(sample_from_compact(x, material))
    return bool(ok), reasons


# --------------------------------------------------------------------------- #
# surrogates
# --------------------------------------------------------------------------- #
def _resolve(relative):
    candidates = [os.path.join(SURROGATE_DIR, relative),
                  os.path.join(S2_DIR, relative),
                  os.path.join(NEURAL_DIR, relative)]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


SURROGATE_FILES = {
    # the frozen best surrogate: 5 seeds x multi-output, MAPE 4.44% on the v4.7
    # test split.  Every method must be scored with this one.
    "mlp_best": _resolve(os.path.join("outputs", "mlp_v47_best", "models")),
    # kept for continuity with the earlier DDPM rounds; weaker, and it does NOT
    # need the log-input transform.
    "mlp_single": _resolve(os.path.join("outputs", "mlp_v47", "models")),
    "mlp_multi": _resolve(os.path.join("outputs", "mlp_v47", "models")),
    "lightgbm": _resolve(os.path.join("surrogate_lightGBM", "outputs_v4.4_100k",
                                      "lightgbm_lhs_v4_6order.pkl")),
}
BEST_SUMMARY = _resolve(os.path.join("outputs", "mlp_v47_best", "summary_mlp.json"))

# Lower clamp for the five logged inputs, equal to their minimum over the frozen
# training split of split_v47.npz (11_baseline_opt.py derives the same numbers at
# run time from the split).  Clamping below the training range is harmless;
# letting log() see an exact 0 is not.
LOG_FLOOR = {
    "ID": 2.145064190701232e-07,
    "b0_K": 1.0e6, "b1_K": 1.0e6, "b2_K": 1.0e6, "b3_K": 1.0e6,
    "brg_span": 4.417531001269041e-03,
    "mass_ratio": 1.4005938319224257e-02,
}


class MLPSurrogate:
    """Per-order MLP ensemble (one network per critical speed, seeds averaged in log)."""

    def __init__(self, model_dir=None, seeds=(42, 43, 44), arm="single", threads=4):
        import torch
        self.torch = torch
        torch.set_num_threads(threads)
        self.dir = model_dir or SURROGATE_FILES["mlp_single"]
        self.models = []
        for order in range(1, 7):
            seed_models = []
            for seed in seeds:
                path = os.path.join(self.dir, "cs_%d_rpm_seed%d.pt" % (order, seed))
                if not os.path.exists(path):
                    continue
                payload = torch.load(path, map_location="cpu", weights_only=False)
                config = payload["config"]
                net = _build_mlp(payload, torch)
                net.eval()
                seed_models.append({
                    "net": net,
                    "x_mean": np.asarray(payload["x_mean"], dtype=np.float64),
                    "x_scale": np.asarray(payload["x_scale"], dtype=np.float64),
                    "y_mean": float(np.asarray(payload["y_mean"]).reshape(-1)[0]),
                    "y_std": float(np.asarray(payload["y_std"]).reshape(-1)[0]),
                })
            if not seed_models:
                raise FileNotFoundError(
                    "no MLP checkpoints for cs_%d_rpm under %s" % (order, self.dir))
            self.models.append(seed_models)
        self.targets = list(ALL_CS)
        self.source = self.dir

    def predict(self, model_rows):
        rows = np.asarray(model_rows, dtype=np.float64)
        out = np.zeros((rows.shape[0], len(self.models)), dtype=np.float64)
        with self.torch.no_grad():
            for index, seed_models in enumerate(self.models):
                logs = []
                for item in seed_models:
                    scaled = (rows - item["x_mean"]) / item["x_scale"]
                    tensor = self.torch.as_tensor(scaled, dtype=self.torch.float32)
                    normalized = item["net"](tensor).cpu().numpy()
                    logs.append(normalized.reshape(-1) * item["y_std"] + item["y_mean"])
                out[:, index] = np.exp(np.mean(np.stack(logs, axis=0), axis=0))
        return out


class MLPBestSurrogate:
    """The frozen 5-seed multi-output MLP (1024-512-256, L1, log target).

    Two things the older per-order checkpoints do not need and this one does:

      * five of the 47 inputs go through log() before z-scoring.  The column
        indices live only in summary_mlp.json (config.log_input_columns); they
        are NOT stored in the .pt payload, so feeding raw features returns
        silent garbage rather than an error.
      * the five seeds aggregate as a geometric mean (mean in log space).  This
        reproduces outputs/mlp_v47_best/pred_multi.npz to 2e-5%.

    `verify()` re-runs that reproduction against the stored test predictions.
    """

    def __init__(self, model_dir=None, summary_path=None,
                 seeds=(42, 43, 44, 45, 46), threads=4):
        import torch

        self.torch = torch
        torch.set_num_threads(threads)
        self.dir = model_dir or SURROGATE_FILES["mlp_best"]
        with open(summary_path or BEST_SUMMARY, encoding="utf-8") as handle:
            summary = json.load(handle)
        self.feature_names = list(summary["feature_names"])
        self.log_columns = [int(i) for i in summary["config"]["log_input_columns"]]
        self.log_floor = np.array(
            [LOG_FLOOR.get(self.feature_names[i], 1e-12) for i in self.log_columns],
            dtype=np.float64)
        self.models = []
        for seed in seeds:
            path = os.path.join(self.dir, "multi_seed%d.pt" % seed)
            if not os.path.exists(path):
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            net = _build_mlp(payload, torch)
            net.eval()
            for parameter in net.parameters():
                parameter.requires_grad_(False)
            self.models.append({
                "net": net,
                "x_mean": np.asarray(payload["x_mean"], dtype=np.float64),
                "x_scale": np.asarray(payload["x_scale"], dtype=np.float64),
                "y_mean": np.asarray(payload["y_mean"], dtype=np.float64).reshape(-1),
                "y_std": np.asarray(payload["y_std"], dtype=np.float64).reshape(-1),
            })
        if not self.models:
            raise FileNotFoundError("no multi_seed*.pt under %s" % self.dir)
        self.targets = list(ALL_CS)
        self.source = "%s (%d seeds, log-inputs)" % (self.dir, len(self.models))

    def prepare(self, model_rows):
        rows = np.array(model_rows, dtype=np.float64, copy=True)
        rows[:, self.log_columns] = np.log(
            np.maximum(rows[:, self.log_columns], self.log_floor))
        return rows

    def predict(self, model_rows):
        rows = self.prepare(model_rows)
        torch = self.torch
        logs = []
        with torch.no_grad():
            for item in self.models:
                scaled = (rows - item["x_mean"]) / item["x_scale"]
                tensor = torch.as_tensor(scaled, dtype=torch.float32)
                out = item["net"](tensor).cpu().numpy().astype(np.float64)
                logs.append(out * item["y_std"] + item["y_mean"])
        return np.exp(np.mean(np.stack(logs, axis=0), axis=0))

    def verify(self):
        """Reproduce the stored test predictions; returns the max gap in percent."""
        split_path = _resolve(os.path.join("outputs", "split_v47.npz"))
        pred_path = _resolve(os.path.join("outputs", "mlp_v47_best", "pred_multi.npz"))
        if not (os.path.exists(split_path) and os.path.exists(pred_path)):
            return None
        engine = surrogate_common_module()
        all_rows = engine.load_dataset(verbose=False)[0]
        test_idx = np.load(split_path)["test_idx"]
        reference = np.load(pred_path)["raw_pred"]
        mine = self.predict(all_rows[test_idx])
        return float(np.max(np.abs(mine - reference) / np.abs(reference)) * 100.0)


def _build_mlp(payload, torch):
    """Rebuild with the *exact* class used for training, so state dict keys match."""
    mlp = surrogate_common_module().mlp_module()
    config = payload["config"]
    net = mlp.TorchMLP(int(config["input_dim"]), tuple(int(w) for w in config["hidden"]),
                       int(config["output_dim"]))
    net.load_state_dict(payload["state_dict"])
    return net


class LightGBMSurrogate:
    """LightGBM pkl fallback (per-order boosters, log target)."""

    def __init__(self, path=None, threads=None):
        import pickle
        self.path = path or SURROGATE_FILES["lightgbm"]
        with open(self.path, "rb") as handle:
            payload = pickle.load(handle)
        models = payload["models"]
        self.targets = list(ALL_CS)
        self.models = [models[name] for name in self.targets]
        self.source = self.path

    def predict(self, model_rows):
        rows = np.asarray(model_rows, dtype=np.float64)
        out = np.zeros((rows.shape[0], len(self.models)), dtype=np.float64)
        for index, booster in enumerate(self.models):
            out[:, index] = np.exp(booster.predict(rows))
        return out


def load_surrogate(kind="mlp_best", **kwargs):
    if kind == "mlp_best":
        return MLPBestSurrogate(**kwargs)
    if kind == "mlp_single":
        return MLPSurrogate(**kwargs)
    if kind == "lightgbm":
        return LightGBMSurrogate(**kwargs)
    raise ValueError("unknown surrogate kind: %s" % kind)


def predict_cs(compact_rows, materials, surrogate):
    model_rows, _ = compact_to_model_rows(compact_rows, materials)
    return surrogate.predict(model_rows)


# --------------------------------------------------------------------------- #
# ground-truth ROSS verification
# --------------------------------------------------------------------------- #
def run_ross_on_compact(x, material, n_base=15, num_modes=None):
    """ROSS ground truth using the *v4.7* semantics: forward whirl, positive damping.

    Fixes the ROSS path to the one the dataset was actually labelled with
    (run_relabel_v4_7.ps1 exports ROSS_FAST_RELABEL=1).  The full
    rotor.run_critical_speed sweep gives the same wd roots but costs 10-100x
    more, because it re-solves a frequency sweep per mode; on the search boxes
    (bearing K up to 4e8 N/m) a single rotor can exceed 20 minutes.
    """
    os.environ.setdefault("ROSS_FAST_RELABEL", "1")
    module = ross_module()
    rotor = module.build_ross_rotor(sample_from_compact(x, material), n_base=n_base)
    forward, _ = module.critical_speed_result(
        rotor, num_modes=num_modes or module.NUM_MODES)
    return [float(v) for v in forward]


def save_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "freeze":
        frame, (tr, va, te), targets = freeze()
        print("slice rows      :", len(frame))
        print("train/val/test  :", tr.size, va.size, te.size)
        print("targets         :", [t["target_id"] for t in targets])
        for t in targets[:3]:
            print("   ", t["target_id"], t["material"], t["level"],
                  ["%.0f" % v for v in t["cs_targets"][:3]])
    else:
        print(__doc__)