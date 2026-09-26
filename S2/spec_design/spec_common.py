# -*- coding: utf-8 -*-
"""Single-order spec-driven design -- shared layer.

Protocol (frozen from `S2/research/一阶临界转速设计需求调研.md`):

  * user gives  material + n_disks + n_bearings + a FIRST-order critical-speed
    spec (lower bound = (1+margin) * n_operating, optional upper bound)
  * we return a batch of valid, diverse designs that meet the spec

Design representation is canonical and family-independent: 6 disk slots and 4
bearing slots, with the slots beyond (nd, nb) pinned to 0 -- the dataset stores
them that way, and the mask is fully determined by (nd, nb).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(HERE))
S1_V47 = os.path.join(PROJECT_DIR, "S1", "04_versions", "v4.7")
POOL_CSV = os.path.join(S1_V47, "dataset_v4.7", "dataset_v4.7_all.csv")

S2_DIR = os.path.join(PROJECT_DIR, "S2")
SURROGATE_DIR = os.path.join(S2_DIR, "surrogate_v47")
NEURAL_DIR = os.path.join(S2_DIR, "neural_surrogate_experiment")

MATERIAL_ORDER = ["Steel", "Aluminum", "Titanium"]
ND_CHOICES = [1, 2, 3, 4, 5, 6]
NB_CHOICES = [2, 3, 4]
MAX_DISKS = 6
MAX_BEARINGS = 4

CANONICAL_COLUMNS = (
    ["L", "OD", "ID"]
    + ["d%d_%s" % (i, k) for i in range(MAX_DISKS) for k in ("OD", "W", "pos")]
    + ["b%d_%s" % (i, k) for i in range(MAX_BEARINGS) for k in ("K", "C", "pos")]
)
DESIGN_DIM = len(CANONICAL_COLUMNS)          # 3 + 18 + 12 = 33

N_OPERATING_RPM = 12000.0
DEFAULT_MARGIN = 0.20
DEFAULT_LOWER_RPM = N_OPERATING_RPM * (1.0 + DEFAULT_MARGIN)      # 14400

CS_COLUMNS = ["cs_%d_rpm" % i for i in range(1, 7)]
ALL_CS = CS_COLUMNS
CS1_COLUMN = "cs_1_rpm"

SEED = 42
_MODULE_CACHE = {}


def _load_module(name, path):
    if name not in _MODULE_CACHE:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULE_CACHE[name] = module
    return _MODULE_CACHE[name]


def constraint_module():
    return _load_module("s1_constraint_v47", os.path.join(S1_V47, "s1_constraint_filter_v4_7.py"))


def ross_module():
    return _load_module("s1_pipeline_v47", os.path.join(S1_V47, "s1_run_pipeline_v4_7.py"))


def mlp_module():
    """The shared feature-engineering module of the six-order surrogate."""
    return _load_module("six_order_mlp_shared",
                        os.path.join(NEURAL_DIR, "run_six_order_mlp_relabeled.py"))


def ddpm_common():
    """The frozen v4.7 DDPM/baseline helpers -- reused for the surrogate loader,
    which already handles the log-input wiring and the 5-seed geometric mean."""
    return _load_module("ddpm_v47_common",
                        os.path.join(S2_DIR, "DDPM_v47", "ddpm_v47_common.py"))


def load_surrogate(kind="mlp_best", **kwargs):
    return ddpm_common().load_surrogate(kind, **kwargs)


def predict_cs(x_batch, materials, nds, nbs, surrogate):
    return surrogate.predict(canonical_to_model_rows(x_batch, materials, nds, nbs))


def run_ross(x, material, nd, nb, n_base=15):
    """Ground truth, on the same ROSS path the dataset labels came from."""
    os.environ.setdefault("ROSS_FAST_RELABEL", "1")
    module = ross_module()
    rotor = module.build_ross_rotor(sample_from_canonical(x, material, nd, nb), n_base=n_base)
    forward, _ = module.critical_speed_result(rotor, num_modes=module.NUM_MODES)
    return [float(v) for v in forward]


# --------------------------------------------------------------------------- #
def material_onehot(material):
    if material not in MATERIAL_ORDER:
        raise ValueError("unknown material: %s" % material)
    return [1.0 if material == m else 0.0 for m in MATERIAL_ORDER]


def family_onehot(nd, nb):
    """nd one-hot (6) + nb one-hot (3) = 9 dims."""
    if nd not in ND_CHOICES or nb not in NB_CHOICES:
        raise ValueError("unsupported family nd=%s nb=%s" % (nd, nb))
    return ([1.0 if nd == v else 0.0 for v in ND_CHOICES]
            + [1.0 if nb == v else 0.0 for v in NB_CHOICES])


DISK_BASE = 3        # canonical layout: L, OD, ID | 6 x (OD, W, pos) | 4 x (K, C, pos)
BRG_BASE = 3 + 3 * MAX_DISKS


def sort_canonical(x, nd=None, nb=None):
    """Sort the *used* disk / bearing slots by axial position, keeping the
    (OD, W, pos) / (K, C, pos) triples intact.  Unused slots stay 0."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape[0] != DESIGN_DIM:
        raise ValueError("canonical x must have %d dims" % DESIGN_DIM)
    out = x.copy()
    for slots, count in (([DISK_BASE + 3 * i for i in range(MAX_DISKS)], nd),
                         ([BRG_BASE + 3 * i for i in range(MAX_BEARINGS)], nb)):
        if count is None or count <= 1:
            continue
        used = slots[:count]
        order = np.argsort([x[s + 2] for s in used], kind="stable")
        for dst, src in enumerate(order):
            out[slots[dst]:slots[dst] + 3] = x[used[src]:used[src] + 3]
    return out


def sample_from_canonical(x, material, nd, nb):
    """Canonical 33-vector -> the S1 sample dict (the schema the S1 code speaks)."""
    x = sort_canonical(x, nd=int(nd), nb=int(nb))
    nd, nb = int(nd), int(nb)
    return {
        "sl": float(x[0]), "od": float(x[1]), "id": float(x[2]),
        "material": material, "bearing_type": "isotropic",
        "nd": nd,
        "disk_od_list": [float(x[DISK_BASE + 3 * i]) for i in range(nd)],
        "disk_w_list": [float(x[DISK_BASE + 3 * i + 1]) for i in range(nd)],
        "dp": [float(x[DISK_BASE + 3 * i + 2]) for i in range(nd)],
        "nb": nb,
        "bk_list": [float(x[BRG_BASE + 3 * i]) for i in range(nb)],
        "bc_list": [float(x[BRG_BASE + 3 * i + 1]) for i in range(nb)],
        "bp": [float(x[BRG_BASE + 3 * i + 2]) for i in range(nb)],
    }


def is_valid(x, material, nd, nb):
    ok, reasons = constraint_module().is_valid_v4(sample_from_canonical(x, material, nd, nb))
    return bool(ok), reasons


def canonical_to_model_rows(x_batch, materials, nds, nbs):
    """(n, 33) + family -> (n, 47) rows in the surrogate's own recipe."""
    samples = [sample_from_canonical(x, m, nd, nb)
               for x, m, nd, nb in zip(x_batch, materials, nds, nbs)]
    raw, raw_names = constraint_module().to_features_v4(samples)
    raw = np.asarray(raw, dtype=np.float64)
    return np.asarray(mlp_module().build_engineered_features(raw, list(raw_names))[0],
                      dtype=np.float64)


def load_pool(path=POOL_CSV, materials=None, families=None):
    """The full v4.7 pool with the canonical design columns and a material column."""
    import pandas as pd

    frame = pd.read_csv(path)
    material = np.full(len(frame), "", dtype=object)
    for name in MATERIAL_ORDER:
        material[frame["mat_%s" % name].to_numpy() > 0.5] = name
    frame = frame.copy()
    frame["material"] = material
    frame = frame[frame["status"] == "success"]
    if materials:
        frame = frame[frame["material"].isin(list(materials))]
    if families:
        wanted = set((int(a), int(b)) for a, b in families)
        keep = np.array([(int(a), int(b)) in wanted
                         for a, b in zip(frame["n_disks"], frame["n_bearings"])])
        frame = frame[keep]
    return frame.reset_index(drop=True)


def spec_bounds(material, pool=None, margin=DEFAULT_MARGIN,
                n_operating=N_OPERATING_RPM, upper=None):
    """(lower, upper) rpm.  upper defaults to the largest cs1 ever realised for
    that material -- so an optimiser cannot win by extrapolating the surrogate."""
    lower = n_operating * (1.0 + margin)
    if upper is not None:
        return lower, float(upper)
    if pool is None:
        pool = load_pool(materials=[material])
    else:
        pool = pool[pool["material"] == material]
    return lower, float(pool["cs_1_rpm"].max())


def spec_ok(cs1, lower, upper):
    cs1 = np.asarray(cs1, dtype=np.float64)
    return (cs1 >= lower) & (cs1 <= upper)


def save_json(path, data):
    import json
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)