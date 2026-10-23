# -*- coding: utf-8 -*-
"""Shared, read-only analysis utilities for the paper_v3 generator study."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
S2 = Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2")
PAPER_V2 = S2 / "paper" / "paper_v2"
FAIR = PAPER_V2 / "revision" / "revision_after_fairness_audit"

DATA_CSV = S2.parent / "S1" / "04_versions" / "v4.7" / "dataset_v4.7" / "dataset_v4.7_all.csv"
CVAE = S2 / "CVAE_baseline"
V2 = S2 / "spec_design_v2"
V2_OUT = V2 / "outputs"
RETRIEVAL = PAPER_V2 / "completer_materials" / "new_experiments" / "retrieval_jitter" / "s42"

SPECS = [
    {"name": "Steel_mid", "material": "Steel", "target": 3600.0},
    {"name": "Steel_high", "material": "Steel", "target": 16000.0},
    {"name": "Aluminum_mid", "material": "Aluminum", "target": 5100.0},
    {"name": "Aluminum_high", "material": "Aluminum", "target": 23000.0},
    {"name": "Titanium_mid", "material": "Titanium", "target": 4300.0},
    {"name": "Titanium_high", "material": "Titanium", "target": 19000.0},
]
SPEC_NAMES = [s["name"] for s in SPECS]
SEEDS = [42, 7, 13]
TOL = 0.05
METHODS = ["ddpm_guided", "ddpm_unguided", "cvae_pure", "cvae_guided"]
METHOD_LABELS = {
    "ddpm_guided": "DDPM guided",
    "ddpm_unguided": "DDPM unguided",
    "cvae_pure": "CVAE unguided",
    "cvae_guided": "CVAE guided",
}

CANONICAL_COLUMNS = (
    ["L", "OD", "ID"]
    + [f"d{i}_{k}" for i in range(6) for k in ("OD", "W", "pos")]
    + [f"b{i}_{k}" for i in range(4) for k in ("K", "C", "pos")]
)
assert len(CANONICAL_COLUMNS) == 33

CHECKPOINTS = {
    "ddpm_interval": V2 / "outputs" / "v2i" / "model.pt",
    "ddpm_point": V2 / "outputs" / "v2m" / "model.pt",
    "cvae": CVAE / "outputs" / "cvae_main" / "model_best.pt",
}


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def spec(name: str) -> dict:
    for row in SPECS:
        if row["name"] == name:
            return dict(row)
    raise KeyError(name)


def band(target: float, tol: float = TOL) -> tuple[float, float]:
    return float(target) * (1.0 - tol), float(target) * (1.0 + tol)


def method_paths(method: str, spec_name: str, seed: int):
    """Return (pool, shortlist, verify_json, verify_csv, passed_csv)."""
    if method == "ddpm_guided":
        if seed == 42:
            pool = V2_OUT / f"nov5k_{spec_name}.csv"
            short = CVAE / "reference_ddpm" / "candidates" / f"ddpm_{spec_name}_s42.csv"
            ver = CVAE / "reference_ddpm" / "verify_norm" / f"ddpm_{spec_name}_s42.json"
            csv = V2_OUT / f"verify_setobj_ours__{spec_name}__top200.csv"
            passed = V2_OUT / f"verify_setobj_ours__{spec_name}__top200__passed.csv"
        else:
            pool = V2_OUT / f"ddpm_pool_{spec_name}_s{seed}.csv"
            short = CVAE / "candidates" / f"ddpm_{spec_name}_s{seed}.csv"
            ver = CVAE / "verify" / f"ddpm_{spec_name}_s{seed}.json"
            csv = CVAE / "verify" / f"verify_ddpm_{spec_name}_s{seed}.csv"
            passed = CVAE / "verify" / f"verify_ddpm_{spec_name}_s{seed}__passed.csv"
    elif method == "ddpm_unguided":
        base = FAIR / "02_ddpm_unguided"
        pool = base / "pools" / f"ddpm_unguided_{spec_name}_s{seed}_pool.csv"
        short = base / "shortlists" / f"ddpm_unguided_{spec_name}_s{seed}.csv"
        ver = base / "ross_verify" / f"ddpm_unguided_{spec_name}_s{seed}.json"
        csv = base / "ross_verify" / f"verify_ddpm_unguided_{spec_name}_s{seed}.csv"
        passed = base / "ross_verify" / f"verify_ddpm_unguided_{spec_name}_s{seed}__passed.csv"
    elif method == "cvae_pure":
        pool = CVAE / "pools" / f"cvae_{spec_name}_s{seed}_pool.csv"
        short = CVAE / "candidates" / f"cvae_{spec_name}_s{seed}.csv"
        ver = CVAE / "verify" / f"cvae_{spec_name}_s{seed}.json"
        csv = CVAE / "verify" / f"verify_cvae_{spec_name}_s{seed}.csv"
        passed = CVAE / "verify" / f"verify_cvae_{spec_name}_s{seed}__passed.csv"
    elif method == "cvae_guided":
        base = FAIR / "03_cvae_guided"
        pool = base / "pools" / f"cvae_guided_{spec_name}_s{seed}_pool.csv"
        short = base / "shortlists" / f"cvae_guided_{spec_name}_s{seed}.csv"
        ver = base / "ross_verify" / f"cvae_guided_{spec_name}_s{seed}.json"
        csv = base / "ross_verify" / f"verify_cvae_guided_{spec_name}_s{seed}.csv"
        passed = base / "ross_verify" / f"verify_cvae_guided_{spec_name}_s{seed}__passed.csv"
    else:
        raise KeyError(method)
    return pool, short, ver, csv, passed


def read_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_raw(columns=None) -> pd.DataFrame:
    usecols = columns if columns is not None else None
    return pd.read_csv(DATA_CSV, usecols=usecols)


def material_series(frame: pd.DataFrame) -> pd.Series:
    out = pd.Series("", index=frame.index, dtype=object)
    for name in ("Steel", "Aluminum", "Titanium"):
        col = f"mat_{name}"
        if col in frame:
            out.loc[frame[col].to_numpy() > 0.5] = name
    return out


def family_allowed(nd, nb, family_text: str | None) -> np.ndarray:
    nd = np.asarray(nd, dtype=int)
    nb = np.asarray(nb, dtype=int)
    if not family_text:
        return np.ones(len(nd), dtype=bool)
    # Syntax used by the source experiments: "3-3x2-2" or "1-2x2-4".
    left, right = family_text.lower().split("x")
    d0, d1 = [int(v) for v in left.split("-")]
    b0, b1 = [int(v) for v in right.split("-")]
    return (nd >= d0) & (nd <= d1) & (nb >= b0) & (nb <= b1)


def p_ref(frame: pd.DataFrame, material: str, constraints: dict[int, tuple[float, float]],
          family_text: str | None = None) -> dict:
    """Empirical support in the finite reference library.

    constraints maps 1-based critical-speed order -> (lower, upper) in rpm.
    The denominator is the number of rows with a finite label for every
    constrained order *after* the material/family filter.  Missing labels are
    reported separately and are not silently treated as failures.
    """
    mat = material_series(frame)
    keep = mat.to_numpy() == material
    if family_text:
        keep &= family_allowed(frame["n_disks"], frame["n_bearings"], family_text)
    sub = frame.loc[keep]
    valid = np.ones(len(sub), dtype=bool)
    for order in constraints:
        col = f"cs_{int(order)}_rpm"
        valid &= pd.to_numeric(sub[col], errors="coerce").notna().to_numpy()
    n_filtered = int(len(sub))
    n_valid = int(valid.sum())
    ok = valid.copy()
    feasible_by_order = {}
    for order, (lo, hi) in constraints.items():
        values = pd.to_numeric(sub[f"cs_{int(order)}_rpm"], errors="coerce").to_numpy()
        inside = (values >= lo) & (values <= hi)
        feasible_by_order[int(order)] = int((valid & inside).sum())
        ok &= inside | ~valid
    n_feasible = int((valid & ok).sum())
    family_counts = sub.loc[valid].groupby(["n_disks", "n_bearings"]).size()
    return {
        "material": material,
        "family_condition": family_text or "free",
        "orders": "|".join(str(k) for k in sorted(constraints)),
        "bounds": json.dumps({str(k): [float(v[0]), float(v[1])] for k, v in sorted(constraints.items())}),
        "n_filtered": n_filtered,
        "n_valid_label": n_valid,
        "n_missing_censored": n_filtered - n_valid,
        "missing_fraction": (n_filtered - n_valid) / n_filtered if n_filtered else float("nan"),
        "n_feasible": n_feasible,
        "p_ref": n_feasible / n_valid if n_valid else float("nan"),
        "support_families": int(len(family_counts)),
        "support_family_counts": ";".join(
            f"{int(a)}d{int(b)}b:{int(c)}" for (a, b), c in family_counts.items()
        ),
        "feasible_by_order": json.dumps(feasible_by_order),
    }


def support_class(n_feasible: int) -> str:
    # Frozen before inspecting any model comparison result.
    if n_feasible <= 0:
        return "zero-observed-support"
    if n_feasible < 100:
        return "low-support"
    if n_feasible < 1000:
        return "medium-support"
    return "high-support"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    out = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        prev = max(prev, val)
        out[idx] = prev
    return out


def bootstrap_mean_ci(values, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def family_entropy(frame: pd.DataFrame) -> float:
    fam = list(zip(frame["n_disks"].astype(int), frame["n_bearings"].astype(int)))
    counts = pd.Series(fam).value_counts().to_numpy(float)
    if len(counts) == 0:
        return float("nan")
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def canonical_array(frame: pd.DataFrame) -> np.ndarray:
    return frame[CANONICAL_COLUMNS].to_numpy(dtype=np.float64)


def active_indices(nd: int, nb: int) -> list[int]:
    idx = [0, 1, 2]
    idx += [3 + 3 * i + j for i in range(int(nd)) for j in range(3)]
    idx += [21 + 3 * i + j for i in range(int(nb)) for j in range(3)]
    return idx


def exact_design_keys(frame: pd.DataFrame, decimals: int = 12) -> list[tuple]:
    """Family-aware exact key: material, topology, and used slots only."""
    keys = []
    for _, row in frame.iterrows():
        nd, nb = int(row["n_disks"]), int(row["n_bearings"])
        idx = active_indices(nd, nb)
        vals = tuple(round(float(row[CANONICAL_COLUMNS[j]]), decimals) for j in idx)
        keys.append((row["material"], nd, nb, vals))
    return keys


def active_scaled_matrix(frame: pd.DataFrame, scales: np.ndarray):
    """Return list of family-aware scaled vectors and their active column ids."""
    mats = []
    for _, row in frame.iterrows():
        nd, nb = int(row["n_disks"]), int(row["n_bearings"])
        idx = active_indices(nd, nb)
        vals = row[CANONICAL_COLUMNS].to_numpy(dtype=float)[idx] / scales[idx]
        mats.append((np.asarray(vals), np.asarray(idx, dtype=int), (nd, nb)))
    return mats


def _pair_distance(a, b):
    return float(np.linalg.norm(a - b))


def same_family_nn_and_pair(frame: pd.DataFrame, scales: np.ndarray,
                            max_points: int = 1200, seed: int = 42) -> tuple[float, float, int]:
    """Median nearest-neighbour and pairwise distance within structural families.

    Distances use only actively used slots, divided by the reference-population
    standard deviation of each slot.  Cross-family pairs are excluded here;
    family coverage is reported separately by entropy.
    """
    if len(frame) > max_points:
        rng = np.random.default_rng(seed)
        frame = frame.iloc[np.sort(rng.choice(len(frame), max_points, replace=False))].copy()
    all_nn = []
    all_pairs = []
    for _, sub in frame.groupby(["n_disks", "n_bearings"]):
        if len(sub) < 2:
            continue
        idx = active_indices(int(sub.iloc[0]["n_disks"]), int(sub.iloc[0]["n_bearings"]))
        z = sub[CANONICAL_COLUMNS].to_numpy(float)[:, idx] / scales[idx]
        # bounded chunking keeps 5,000-row pools manageable
        for start in range(0, len(z), 512):
            q = z[start:start + 512]
            d = np.sqrt(np.maximum(((q * q).sum(1)[:, None]
                                    + (z * z).sum(1)[None, :]
                                    - 2.0 * q @ z.T), 0.0))
            d[np.arange(len(q)), np.arange(start, start + len(q))] = np.inf
            all_nn.extend(d.min(1).tolist())
            if start == 0 and len(z) <= 250:
                iu = np.triu_indices(len(z), 1)
                all_pairs.extend(d[:, :][iu].tolist())
    nn = float(np.median(all_nn)) if all_nn else float("nan")
    pair = float(np.median(all_pairs)) if all_pairs else float("nan")
    return nn, pair, len(frame)


def near_duplicate_fraction(frame: pd.DataFrame, scales: np.ndarray,
                            thresholds=(0.001, 0.005, 0.01, 0.02, 0.05),
                            max_points: int = 1200, seed: int = 42) -> dict:
    if len(frame) > max_points:
        rng = np.random.default_rng(seed)
        frame = frame.iloc[np.sort(rng.choice(len(frame), max_points, replace=False))].copy()
    min_nn = []
    for _, sub in frame.groupby(["n_disks", "n_bearings"]):
        if len(sub) < 2:
            continue
        idx = active_indices(int(sub.iloc[0]["n_disks"]), int(sub.iloc[0]["n_bearings"]))
        z = sub[CANONICAL_COLUMNS].to_numpy(float)[:, idx] / scales[idx]
        for start in range(0, len(z), 512):
            q = z[start:start + 512]
            d = np.sqrt(np.maximum(((q * q).sum(1)[:, None]
                                    + (z * z).sum(1)[None, :]
                                    - 2.0 * q @ z.T), 0.0))
            d[np.arange(len(q)), np.arange(start, start + len(q))] = np.inf
            min_nn.extend(d.min(1).tolist())
    arr = np.asarray(min_nn, dtype=float)
    out = {
        f"near_dup_frac_le_{thr:g}": float(np.mean(arr <= thr)) if len(arr) else float("nan")
        for thr in thresholds
    }
    out.update({"nn_sample_n": int(len(arr)),
                "nn_sample_median": float(np.median(arr)) if len(arr) else float("nan")})
    return out


def nearest_reference_distance(queries: pd.DataFrame, ref: pd.DataFrame,
                               scales: np.ndarray, max_queries: int = 500,
                               seed: int = 42) -> np.ndarray:
    """Exact nearest distance to the full reference library, in standardized units.

    The reference vectors use all 33 canonical dimensions to reproduce the frozen
    project metric.  Candidate subsampling is explicit and seeded.
    """
    if len(queries) > max_queries:
        rng = np.random.default_rng(seed)
        queries = queries.iloc[np.sort(rng.choice(len(queries), max_queries, replace=False))]
    q = canonical_array(queries) / scales
    r = canonical_array(ref) / scales
    out = np.empty(len(q), dtype=float)
    for start in range(0, len(q), 64):
        qq = q[start:start + 64]
        best = np.full(len(qq), np.inf)
        for lo in range(0, len(r), 20000):
            rr = r[lo:lo + 20000]
            d = np.sqrt(np.maximum(((qq * qq).sum(1)[:, None]
                                    + (rr * rr).sum(1)[None, :]
                                    - 2.0 * qq @ rr.T), 0.0))
            best = np.minimum(best, d.min(1))
        out[start:start + len(qq)] = best
    return out


