# -*- coding: utf-8 -*-
"""Shared, isolated utilities for the EAAI pilot study (protocol v1.1).

Read-only with respect to the dataset, checkpoints and historical experiments.
All new outputs are written under this study directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
S2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(STUDY)))))
# The expression above is intentionally replaced below by an explicit anchor check.
S2 = os.path.abspath(os.path.join(STUDY, "..", "..", "..", ".."))
V1_DIR = os.path.join(S2, "spec_design")
V2_DIR = os.path.join(S2, "spec_design_v2")
CVAE_DIR = os.path.join(S2, "CVAE_baseline")
for _p in (V1_DIR, V2_DIR, CVAE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spec_common as S          # noqa: E402
import spec_eval as E            # noqa: E402
import spec_opt as O             # noqa: E402
import spec_interval as SI       # noqa: E402
import latent_design as LD       # noqa: E402
import latent_torch as LT        # noqa: E402
import latent_ddpm_v2 as V2D     # noqa: E402
import cvae_common as C          # noqa: E402

# Importing cvae_common calls os.makedirs on the historical CVAE directories.
# It does not write files.  The pilot never calls its loaders/main().
RUN = os.path.join(STUDY, "pilot_runs")
ENV_DIR = os.path.join(RUN, "00_env")
SPLIT_DIR = os.path.join(RUN, "00_reconstructed_split")
SMOKE_DIR = os.path.join(RUN, "smoke")
POOL_DIR = os.path.join(RUN, "01_pools")
SHORT_DIR = os.path.join(RUN, "02_shortlists")
ROSS_DIR = os.path.join(RUN, "03_ross")
METRIC_DIR = os.path.join(RUN, "04_metrics")
LOG_DIR = os.path.join(RUN, "logs")
for _d in (ENV_DIR, SPLIT_DIR, SMOKE_DIR, POOL_DIR, SHORT_DIR, ROSS_DIR, METRIC_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

DATASET = os.path.join(S2, "..", "S1", "04_versions", "v4.7", "dataset_v4.7",
                       "dataset_v4.7_all.csv")
DATASET = os.path.abspath(DATASET)
DDPM_CKPT = os.path.join(V2_DIR, "outputs", "v2i", "model.pt")
DDPM_POINT_CKPT = os.path.join(V2_DIR, "outputs", "v2m", "model.pt")
CVAE_CKPT = os.path.join(CVAE_DIR, "outputs", "cvae_main", "model_best.pt")
REFERENCE_PDF = os.path.abspath(os.path.join(
    S2, "paper", "paper_v2", "final_output", "eaai", "reference_paper.pdf"))
SPEC_CSV = os.path.join(STUDY, "spec_candidates", "calibration_specs.csv")

SEED = 42
N_VAL = 9990
N_ATTEMPTS = 5000
N_RANDOM = 100
N_SHORT = 100
SAMPLE_STEPS = 25
CFG_GUIDANCE = 1.0
GUIDE_LAMBDA = 0.0
JITTER_FRAC = 0.05
N_ORIGINALS = 1000
H_MAX = 0.70
ROSS_N_BASE = 15
ROSS_NUM_MODES = 48
ROSS_TIMEOUT = 180.0
ROSS_VERSION_EXPECTED = "1.6.1"
ROSS_PROTOCOL = "forward_positive__n_base15__num_modes48__fast_relabel1__rpm"

MATERIALS = list(S.MATERIAL_ORDER)
FAMILIES = list(V2D.v1().FAMILIES)


# --------------------------------------------------------------------------- #
# hashing / json
# --------------------------------------------------------------------------- #
def sha256_file(path: str, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def design_key(x) -> str:
    """Exact identity of the canonical 33-vector (little-endian float64)."""
    arr = np.ascontiguousarray(np.asarray(x, dtype="<f8").reshape(-1))
    return sha256_bytes(arr.tobytes())


def design_key_many(xs) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float64)
    return np.array([design_key(row) for row in xs], dtype=object)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def save_json(path: str, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=json_default)


def load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# frozen specs (read, never rewrite)
# --------------------------------------------------------------------------- #
def load_calibration_specs() -> dict:
    frame = pd.read_csv(SPEC_CSV)
    out = {}
    for _, row in frame.iterrows():
        orders = [int(v) for v in str(row["orders"]).split("|") if str(v).strip()]
        targets = {int(k): float(v) for k, v in json.loads(row["targets"]).items()}
        bands = {int(k): (float(v[0]), float(v[1]))
                 for k, v in json.loads(row["bands"]).items()}
        fam_text = str(row["family"]).strip()
        if fam_text == "free":
            family = None
        else:
            fam_a, fam_b = fam_text.split("-")
            family = (int(fam_a.replace("d", "")), int(fam_b.replace("b", "")))
        out[str(row["slot_id"])] = {
            "spec_id": str(row["spec_id"]), "slot_id": str(row["slot_id"]),
            "role": str(row["role"]), "material": str(row["material"]),
            "family": family, "family_text": fam_text, "orders": orders,
            "kind": str(row["kind"]), "targets": targets, "bands": bands,
            "n_feasible": int(row["n_feasible"]), "n_ref": int(row["n_ref"]),
            "p_ref": float(row["p_ref"]), "support_class": str(row["support_class"]),
            "family_n": int(row["family_n"]), "spec_key": str(row["spec_key"]),
            "freeze_timestamp": str(row["freeze_timestamp"]),
        }
    return out


def band_specs(spec: dict) -> dict:
    return {int(o): tuple(spec["bands"][int(o)]) for o in spec["orders"]}


# --------------------------------------------------------------------------- #
# dataset context and reconstructed split
# --------------------------------------------------------------------------- #
def build_context_split(save: bool = True):
    ctx0 = E.pool_ctx()
    ctx, dropped = V2D.finite_ctx(ctx0)
    order = np.random.default_rng(SEED).permutation(len(ctx["designs"]))
    n_val = max(2000, int(0.05 * len(ctx["designs"])))
    val_idx = order[:n_val].astype(np.int64)
    train_idx = order[n_val:].astype(np.int64)
    meta = {
        "dataset": DATASET, "dataset_sha256": sha256_file(DATASET),
        "pool_rows_status_success": int(len(ctx0["designs"])),
        "finite_rows": int(len(ctx["designs"])), "dropped_missing_cs3": int(dropped),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "seed": SEED, "rule": "default_rng(42).permutation(N); val=first max(2000,5%N); train=rest",
        "status": "RECONSTRUCTED from source code; no historical split index file was found",
        "created": utc_now(),
    }
    if save:
        np.save(os.path.join(SPLIT_DIR, "train_index.npy"), train_idx)
        np.save(os.path.join(SPLIT_DIR, "val_index.npy"), val_idx)
        meta["train_index_sha256"] = sha256_file(os.path.join(SPLIT_DIR, "train_index.npy"))
        meta["val_index_sha256"] = sha256_file(os.path.join(SPLIT_DIR, "val_index.npy"))
        ids = (ctx["pool"]["source"].astype(str) + ":" +
               ctx["pool"]["source_row_id"].astype(str)).to_numpy()
        pd.DataFrame({"row_index": np.arange(len(ids)), "reference_id": ids,
                      "split": np.where(np.isin(np.arange(len(ids)), val_idx), "val", "train")}
                     ).to_csv(os.path.join(SPLIT_DIR, "reconstructed_split.csv"), index=False)
        save_json(os.path.join(SPLIT_DIR, "split_meta.json"), meta)
    return ctx, train_idx, val_idx, meta


def reference_ids(ctx) -> np.ndarray:
    return (ctx["pool"]["source"].astype(str) + ":" +
            ctx["pool"]["source_row_id"].astype(str)).to_numpy()


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def family_sequence(fixed_family, n: int, seed: int = SEED):
    """Common family allocation for DDPM and CVAE.

    For a fixed-family spec every candidate is that family.  For a free-family
    spec both generative models receive the *same* uniform family sequence.
    """
    if fixed_family is not None:
        return [(int(fixed_family[0]), int(fixed_family[1]))] * int(n)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, len(FAMILIES), size=int(n))
    return [FAMILIES[int(i)] for i in idx]


def load_ddpm():
    payload, model, cfg = V2D.load_model("v2i", True)
    return {"payload": payload, "model": model, "cfg": cfg,
            "scalers": payload["scalers"]}


def load_cvae():
    model, payload = C.load_checkpoint(CVAE_CKPT)
    return {"model": model, "payload": payload, "scalers": payload["scalers"]}


def conditions_for(material: str, families, specs: dict, scalers: dict):
    return np.stack([V2D.condition_for(material, fam, specs, scalers, layout="interval")
                     for fam in families]).astype(np.float32)


def sample_ddpm(bundle, material: str, families, specs: dict, seed: int = SEED,
                n: int | None = None):
    n = int(n if n is not None else len(families))
    cond = conditions_for(material, families, specs, bundle["scalers"])
    gen = __import__("torch").Generator().manual_seed(int(seed))
    module = V2D.v1().ddpm()
    betas, alpha_bar = module.cosine_schedule(V2D.v1().T_STEPS)
    u = V2D.v1().sample_latent(
        bundle["model"], __import__("torch").as_tensor(cond, dtype=__import__("torch").float32),
        list(families), n, CFG_GUIDANCE, float(bundle["payload"].get("condition_dropout", 0.0)),
        gen, betas, alpha_bar, LD.LAT_DIM, bundle["payload"].get("predict", "v"),
        steps=SAMPLE_STEPS, guide=None)
    return u.detach().cpu().numpy().astype(np.float64), cond


def sample_cvae(bundle, material: str, families, specs: dict, seed: int = SEED,
                n: int | None = None):
    n = int(n if n is not None else len(families))
    cond = conditions_for(material, families, specs, bundle["scalers"])
    u, z = C.sample_prior(bundle["model"], cond, list(families), n, int(seed))
    return np.asarray(u, dtype=np.float64), cond


def decode_latents(u, families):
    return LT.decode_batch(np.asarray(u, dtype=np.float64), list(families))


def validity(x, material: str, families):
    return O.validity(np.asarray(x, dtype=np.float64), material, list(families))


def predict_six(x, material: str, families, surrogate):
    rows = S.canonical_to_model_rows(
        np.asarray(x, dtype=np.float64), [material] * len(x),
        [int(f[0]) for f in families], [int(f[1]) for f in families])
    return np.asarray(surrogate.predict(rows), dtype=np.float64)[:, :6]


# --------------------------------------------------------------------------- #
# retrieval + jitter (physical, auditable)
# --------------------------------------------------------------------------- #
CONT_COLS = [0, 1, 2] + [3 + 3 * i + k for i in range(S.MAX_DISKS) for k in (0, 1, 2)] \
            + [S.BRG_BASE + 3 * i + k for i in range(S.MAX_BEARINGS) for k in (0, 1, 2)]


def perturb_one(x, family, rng, frac=JITTER_FRAC):
    a, b = int(family[0]), int(family[1])
    out = np.asarray(x, dtype=np.float64).copy()
    cols = [0, 1, 2]
    cols += [3 + 3 * j + k for j in range(a) for k in (0, 1, 2)]
    cols += [S.BRG_BASE + 3 * j + k for j in range(b) for k in (0, 1, 2)]
    for col in cols:
        if out[col] <= 0:
            continue
        out[col] *= 1.0 + rng.uniform(-frac, frac)
    out = S.sort_canonical(out, a, b)
    out = LD.decode(LD.encode(out, family), family)
    return S.sort_canonical(out, a, b)


def retrieval_sources(ctx, train_idx, spec: dict) -> np.ndarray:
    """Training-reference designs whose *stored labels* satisfy every order."""
    material = spec["material"]
    family = spec["family"]
    keep = (ctx["material"] == material)
    if family is not None:
        keep &= (ctx["nd"] == int(family[0])) & (ctx["nb"] == int(family[1]))
    idx = np.where(keep)[0]
    hit = np.ones(len(idx), dtype=bool)
    for order in spec["orders"]:
        lo, hi = spec["bands"][int(order)]
        hit &= (ctx["Y"][idx, int(order) - 1] >= lo) & (ctx["Y"][idx, int(order) - 1] <= hi)
    source = idx[hit]
    train_mask = np.zeros(len(ctx["designs"]), dtype=bool)
    train_mask[train_idx] = True
    return source[train_mask[source]]


def sample_retrieval(ctx, train_idx, spec: dict, surrogate, seed: int = SEED,
                     n_attempts: int = N_ATTEMPTS, n_originals: int = N_ORIGINALS,
                     frac: float = JITTER_FRAC):
    sources = retrieval_sources(ctx, train_idx, spec)
    if len(sources) == 0:
        raise RuntimeError("no training reference source for %s" % spec["slot_id"])
    rng = np.random.default_rng(int(seed) + 100003)
    take = rng.choice(sources, size=int(n_originals), replace=len(sources) < int(n_originals))
    variants = int(math.ceil(float(n_attempts) / len(take)))
    designs, families, origins, changes = [], [], [], []
    for src in take:
        src = int(src)
        fam = (int(ctx["nd"][src]), int(ctx["nb"][src]))
        x0 = ctx["designs"][src]
        for _ in range(variants):
            xp = perturb_one(x0, fam, rng, frac)
            designs.append(xp)
            families.append(fam)
            origins.append(src)
            active = [c for c in CONT_COLS if x0[c] > 0]
            changes.append(float(np.median(np.abs(xp[active] - x0[active]) / x0[active])))
            if len(designs) >= n_attempts:
                break
        if len(designs) >= n_attempts:
            break
    x = np.asarray(designs[:n_attempts], dtype=np.float64)
    fams = families[:n_attempts]
    origins = np.asarray(origins[:n_attempts], dtype=np.int64)
    changes = np.asarray(changes[:n_attempts], dtype=np.float64)
    meta = {
        "retrieval_reference": DATASET,
        "retrieval_reference_sha256": sha256_file(DATASET),
        "reference_subset": "reconstructed training split only",
        "n_reference_feasible": int(len(sources)),
        "n_originals_requested": int(n_originals),
        "n_originals_used": int(len(set(origins.tolist()))),
        "source_selection": "material + fixed family (if any) + all specified orders inside the original user band",
        "source_label_use": "stored reference labels only; no candidate ROSS labels",
        "jitter_space": "active continuous canonical parameters",
        "jitter_scale": "multiplicative 1 + U(-%.3f, %.3f)" % (frac, frac),
        "constraint_handling": "LD.encode -> LD.decode round-trip; decoder enforces H1-H6 by construction",
        "repair_attempts": 0,
        "median_parameter_change": float(np.median(changes)),
        "p95_parameter_change": float(np.percentile(changes, 95)),
        "max_parameter_change": float(np.max(changes)),
    }
    return x, fams, origins, changes, meta


# --------------------------------------------------------------------------- #
# surrogate gate / deterministic shortlist
# --------------------------------------------------------------------------- #
def order_scale(lo: float, hi: float, h_max: float = H_MAX) -> float:
    if hi >= SI.INF_CAP:
        return max(float(lo) * math.exp(float(h_max)), 1e-9)
    return max(float(hi), float(lo) * math.exp(float(h_max)), 1e-9)


def violation_for_order(pred, lo, hi, h_max: float = H_MAX):
    """Non-negative relative violation; zero inside the *original* user band."""
    pred = np.asarray(pred, dtype=np.float64)
    scale = order_scale(lo, hi, h_max)
    below = np.maximum(float(lo) - pred, 0.0) / scale
    above = np.zeros_like(pred) if hi >= SI.INF_CAP else np.maximum(pred - float(hi), 0.0) / scale
    return below + above


def gate_and_score(pred, spec: dict):
    """Return (in_band, joint_nonnegative_violation, per-order violations)."""
    n = len(pred)
    in_band = np.ones(n, dtype=bool)
    score = np.zeros(n, dtype=np.float64)
    per = {}
    for order in spec["orders"]:
        lo, hi = spec["bands"][int(order)]
        v = violation_for_order(pred[:, int(order) - 1], lo, hi)
        per[int(order)] = v
        score += v
        in_band &= (v == 0.0)
    return in_band, score, per


def select_shortlist(dedup_idx, pred, spec: dict, candidate_ids, seed: int,
                     m: int = N_SHORT):
    """Proxy gate + uniform in-band selection; deterministic out-of-band fallback.

    Ties in the fallback are broken by candidate_id (lexicographic), never by
    an unreported index.  The original user band is used, without extra tolerance.
    """
    in_band, score, _ = gate_and_score(pred, spec)
    rng = np.random.default_rng(int(seed))
    inside = np.where(in_band)[0]
    chosen = []
    source = []
    if len(inside) > 0:
        take = min(int(m), len(inside))
        sel = rng.choice(inside, size=take, replace=False)
        chosen.extend(int(i) for i in sel)
        source.extend(["in_band"] * take)
    if len(chosen) < int(m):
        rest = np.array([i for i in range(len(dedup_idx)) if i not in set(chosen)], dtype=int)
        if len(rest) > 0:
            order = np.lexsort((np.asarray(candidate_ids, dtype=object)[rest], score[rest]))
            need = int(m) - len(chosen)
            for i in rest[order[:need]]:
                chosen.append(int(i))
                source.append("fallback")
    chosen_arr = np.asarray(chosen, dtype=int)
    return chosen_arr, np.asarray(source, dtype=object), in_band, score


def exact_unique_indices(keys):
    seen = {}
    keep = []
    for i, key in enumerate(keys):
        if key not in seen:
            seen[key] = i
            keep.append(i)
    return np.asarray(keep, dtype=int), seen


# --------------------------------------------------------------------------- #
# ROSS cache
# --------------------------------------------------------------------------- #
def ross_cache_key(x, material: str, nd: int, nb: int) -> str:
    payload = {
        "design": design_key(x), "material": str(material),
        "nd": int(nd), "nb": int(nb), "units": "rpm",
        "ross_version": ROSS_VERSION_EXPECTED, "n_base": ROSS_N_BASE,
        "num_modes": ROSS_NUM_MODES, "protocol": ROSS_PROTOCOL,
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def ross_cache_path(key: str) -> str:
    return os.path.join(ROSS_DIR, "cache", key + ".json")


def load_ross_cache(key: str):
    path = ross_cache_path(key)
    if os.path.exists(path):
        return load_json(path)
    return None


def save_ross_cache(key: str, record: dict):
    path = ross_cache_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_json(path, record)


# --------------------------------------------------------------------------- #
# environment snapshot
# --------------------------------------------------------------------------- #
def environment_snapshot() -> dict:
    import torch
    try:
        import ross
        ross_version = getattr(ross, "__version__", "unknown")
    except Exception as exc:
        ross_version = "unavailable: %s" % exc
    info = {
        "created": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "ross": ross_version,
        "dataset": DATASET,
        "dataset_sha256": sha256_file(DATASET),
        "checkpoints": {
            "interval_ddpm_v2i": {"path": DDPM_CKPT, "sha256": sha256_file(DDPM_CKPT)},
            "point_ddpm_v2m": {"path": DDPM_POINT_CKPT, "sha256": sha256_file(DDPM_POINT_CKPT)},
            "cvae_main": {"path": CVAE_CKPT, "sha256": sha256_file(CVAE_CKPT)},
        },
        "spec_csv": {"path": SPEC_CSV, "sha256": sha256_file(SPEC_CSV)},
        "reference_pdf": ({"path": REFERENCE_PDF, "sha256": sha256_file(REFERENCE_PDF)}
                          if os.path.exists(REFERENCE_PDF) else None),
        "ross_protocol": ROSS_PROTOCOL,
        "ross_n_base": ROSS_N_BASE,
        "ross_num_modes": ROSS_NUM_MODES,
    }
    return info


# --------------------------------------------------------------------------- #
# output-pollution guard
# --------------------------------------------------------------------------- #
OLD_WATCH = [
    os.path.join(CVAE_DIR, "outputs"),
    os.path.join(CVAE_DIR, "data"),
    os.path.join(CVAE_DIR, "pools"),
    os.path.join(CVAE_DIR, "candidates"),
    os.path.join(CVAE_DIR, "verify"),
    os.path.join(CVAE_DIR, "results"),
    os.path.join(CVAE_DIR, "logs"),
    os.path.join(V2_DIR, "outputs"),
    os.path.join(V1_DIR, "outputs"),
]


def old_output_snapshot() -> dict:
    """Cheap fingerprint of historical output directories.

    The fingerprint changes if any file is added, removed, resized or touched.
    """
    entries = []
    for root in OLD_WATCH:
        if not os.path.isdir(root):
            entries.append([root, "MISSING", 0, 0])
            continue
        count = 0
        total = 0
        newest = 0
        for base, _dirs, files in os.walk(root):
            for name in files:
                path = os.path.join(base, name)
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                count += 1
                total += int(stat.st_size)
                newest = max(newest, int(stat.st_mtime_ns))
        entries.append([root, count, total, newest])
    return {"created": utc_now(), "entries": entries,
            "sha256": sha256_bytes(json.dumps(entries, sort_keys=True).encode("utf-8"))}
