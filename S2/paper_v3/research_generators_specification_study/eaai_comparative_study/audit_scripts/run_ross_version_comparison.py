# -*- coding: utf-8 -*-
"""Paired ROSS 1.6.1 vs 2.3.0 audit for the EAAI pilot.

Selection rules are frozen before any 2.3.0 result is inspected:
  * 2 boundary-enriched designs per pool, selected from existing 1.6.1 labels;
  * 2 uniform random designs per pool, fixed seed;
  * 6 retrieval-source designs per spec, fixed seed.

No TST specification and no full 1200-slot rerun is performed.
"""
from __future__ import annotations

import concurrent.futures as futures
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eaai_common as A

STUDY = A.STUDY
OUT = os.path.join(STUDY, "pilot_runs", "05_ross_version")
os.makedirs(OUT, exist_ok=True)
SPECS = ["CAL02", "CAL05"]
METHODS = ["ddpm", "cvae", "retrieval"]
BOUNDARY_SEED = 20261022
RANDOM_SEED = 20261022
SOURCE_SEED = 20261023
WORKERS = 16
PY161 = sys.executable
PY230 = r"C:\Users\fcu15\ross230_py312\python.exe"
WORKER = os.path.join(A.S2, "spec_design", "_ross_one_canon.py")
TIMEOUT = 240.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pool_lookup():
    frames = []
    for pid in sorted(os.listdir(A.SHORT_DIR)):
        pdir = os.path.join(A.SHORT_DIR, pid)
        if not os.path.isdir(pdir):
            continue
        for fname in ("shortlist.csv", "random_sample.csv"):
            p = os.path.join(pdir, fname)
            if not os.path.exists(p):
                continue
            f = pd.read_csv(p)
            f["pool_id"] = pid
            keep = ["pool_id", "candidate_id"] + A.S.CANONICAL_COLUMNS
            for extra in ("source_ref_id", "source_row"):
                if extra in f:
                    keep.append(extra)
            frames.append(f[keep])
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["pool_id", "candidate_id"]).reset_index(drop=True)


def band_distance(value, lo, hi):
    value = float(value)
    if not np.isfinite(value):
        return np.inf
    width = max(float(hi) - float(lo), 1e-9)
    return min(abs(value - float(lo)), abs(value - float(hi))) / width


def sample_boundary_distance(row, spec):
    vals = []
    for order in spec["orders"]:
        value = row.get("cs%d_ross_rpm" % int(order), np.nan)
        lo, hi = spec["bands"][int(order)]
        vals.append(band_distance(value, lo, hi))
    return float(min(vals)) if vals else np.inf


def pass_spec(forward, spec):
    if forward is None:
        return False
    for order in spec["orders"]:
        o = int(order)
        if len(forward) < o:
            return False
        lo, hi = spec["bands"][o]
        v = float(forward[o - 1])
        if not np.isfinite(v) or v < float(lo) or v > float(hi):
            return False
    return True


def run_version(pyexe, designs):
    """Solve unique designs with one interpreter; return key -> record."""
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["ROSS_FAST_RELABEL"] = "1"

    def one(item):
        key, d = item
        payload = json.dumps({"x": [float(v) for v in d["x"]],
                              "material": d["material"],
                              "nd": int(d["nd"]), "nb": int(d["nb"]),
                              "n_base": 15})
        t0 = time.time()
        try:
            proc = subprocess.run([pyexe, WORKER, payload], capture_output=True,
                                  timeout=TIMEOUT, env=env)
        except subprocess.TimeoutExpired:
            return key, {"status": "timeout", "forward": None,
                         "elapsed_s": time.time() - t0,
                         "error": "timeout>%gs" % TIMEOUT}
        elapsed = time.time() - t0
        lines = (proc.stdout or b"").decode("utf-8", "replace").splitlines()
        result = None
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{"):
                try:
                    result = json.loads(line)
                    break
                except ValueError:
                    continue
        if result is None:
            return key, {"status": "error", "forward": None,
                         "elapsed_s": elapsed,
                         "error": (proc.stderr or b"").decode("utf-8", "replace")[-500:]}
        if result.get("ok"):
            return key, {"status": "ok", "forward": [float(v) for v in result.get("forward", [])],
                         "elapsed_s": elapsed, "error": ""}
        return key, {"status": "error", "forward": None, "elapsed_s": elapsed,
                     "error": str(result.get("error", ""))[-500:]}

    items = sorted(designs.items())
    t0 = time.time()
    with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(one, items))
    return dict(rows), time.time() - t0


def make_design(row):
    return {"x": row[A.S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64),
            "material": str(row["material"]), "nd": int(row["nd"]),
            "nb": int(row["nb"])}


def main():
    specs = A.load_calibration_specs()
    all_slots = pd.read_csv(os.path.join(A.ROSS_DIR, "all_slots.csv"))
    lookup = load_pool_lookup()
    merged = all_slots.merge(lookup, on=["pool_id", "candidate_id"], how="left")
    if merged[A.S.CANONICAL_COLUMNS].isna().any().any():
        raise RuntimeError("canonical lookup failed")

    # Existing 1.6.1 results are the only source used to select boundary-enriched samples.
    designs = {}
    sample_rows = []
    seen = set()
    rng = np.random.default_rng(RANDOM_SEED)

    for spec_id in SPECS:
        spec = specs[spec_id]
        for method in METHODS:
            pid = "%s_%s_s42" % (spec_id, method)
            pool = merged[(merged["pool_id"] == pid) & (merged["status"] == "ok")].copy()
            pool = pool.drop_duplicates("design_key")
            pool = pool[pool["cs1_ross_rpm"].notna()].copy()
            pool["boundary_distance"] = pool.apply(lambda r: sample_boundary_distance(r, spec), axis=1)
            boundary = pool.sort_values(["boundary_distance", "design_key"]).head(2)
            for rank, (_, row) in enumerate(boundary.iterrows(), 1):
                sample_rows.append({
                    "sample_group": "pool_boundary", "spec": spec_id, "method": method,
                    "pool_id": pid, "sample_role": "boundary", "selection_rank": rank,
                    "selection_seed": BOUNDARY_SEED, "candidate_id": row["candidate_id"],
                    "source_ref_id": "", "design_key": row["design_key"],
                    "boundary_distance": float(row["boundary_distance"]),
                })
                if row["design_key"] not in designs:
                    designs[row["design_key"]] = make_design(row)
            candidates = pool[~pool["design_key"].isin(boundary["design_key"])].copy()
            candidates = candidates.sort_values("design_key").reset_index(drop=True)
            take = min(2, len(candidates))
            if take:
                idx = rng.choice(len(candidates), size=take, replace=False)
                for rank, i in enumerate(np.sort(idx), 1):
                    row = candidates.iloc[int(i)]
                    sample_rows.append({
                        "sample_group": "pool_random", "spec": spec_id, "method": method,
                        "pool_id": pid, "sample_role": "uniform_random", "selection_rank": rank,
                        "selection_seed": RANDOM_SEED, "candidate_id": row["candidate_id"],
                        "source_ref_id": "", "design_key": row["design_key"],
                        "boundary_distance": float(row["boundary_distance"]),
                    })
                    if row["design_key"] not in designs:
                        designs[row["design_key"]] = make_design(row)

    # Retrieval sources: select from the frozen retrieval pool, then map to the
    # archived dataset row by source_ref_id.  Selection never uses 2.3.0 output.
    ds_usecols = ["source", "source_row_id", "status", "forward_mode_count",
                  "cs_1_rpm", "cs_2_rpm", "cs_3_rpm", "cs_4_rpm", "cs_5_rpm", "cs_6_rpm",
                  "n_disks", "n_bearings", "mat_Steel", "mat_Aluminum", "mat_Titanium"] + A.S.CANONICAL_COLUMNS
    ds = pd.read_csv(A.DATASET, usecols=ds_usecols)
    ds["source_ref_id"] = ds["source"].astype(str) + ":" + ds["source_row_id"].astype(str)
    if ds["source_ref_id"].duplicated().any():
        raise RuntimeError("duplicate source_ref_id in dataset")
    ds = ds.set_index("source_ref_id", drop=False)
    material = np.full(len(ds), "", dtype=object)
    for name in ("Steel", "Aluminum", "Titanium"):
        material[ds["mat_%s" % name].to_numpy() > 0.5] = name
    ds["material"] = material

    for j, spec_id in enumerate(SPECS):
        spec = specs[spec_id]
        pid = "%s_retrieval_s42" % spec_id
        pool = pd.read_csv(os.path.join(A.POOL_DIR, pid, "pool.csv"))
        refs = sorted(pool["source_ref_id"].dropna().astype(str).unique())
        srng = np.random.default_rng(SOURCE_SEED + j)
        take = min(6, len(refs))
        chosen = sorted(srng.choice(refs, size=take, replace=False).tolist())
        for rank, ref in enumerate(chosen, 1):
            row = ds.loc[ref]
            sample_rows.append({
                "sample_group": "retrieval_source", "spec": spec_id, "method": "retrieval",
                "pool_id": pid, "sample_role": "archived_source", "selection_rank": rank,
                "selection_seed": SOURCE_SEED + j, "candidate_id": "",
                "source_ref_id": ref, "design_key": A.design_key(
                    row[A.S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)),
                "boundary_distance": np.nan,
            })
            key = sample_rows[-1]["design_key"]
            if key not in designs:
                designs[key] = {
                    "x": row[A.S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64),
                    "material": str(row["material"]), "nd": int(row["n_disks"]),
                    "nb": int(row["n_bearings"]),
                }

    samples = pd.DataFrame(sample_rows).drop_duplicates(
        ["sample_group", "spec", "pool_id", "candidate_id", "source_ref_id", "design_key"]
    ).reset_index(drop=True)

    # 1.6.1: reuse existing pilot labels for pool samples; solve only new source designs.
    pool_keys = set(merged["design_key"].dropna().astype(str))
    source_keys = set(samples.loc[samples["sample_group"] == "retrieval_source", "design_key"])
    missing_161 = {k: v for k, v in designs.items() if k in source_keys and k not in pool_keys}
    print("[ross-compare] unique designs=%d, missing 1.6.1 sources=%d, 2.3.0 designs=%d"
          % (len(designs), len(missing_161), len(designs)), flush=True)
    res161_new, wall161 = run_version(PY161, missing_161) if missing_161 else ({}, 0.0)
    res230, wall230 = run_version(PY230, designs)

    # Existing 1.6.1 labels keyed by design_key (pool samples only).
    existing161 = {}
    for _, row in merged[merged["status"] == "ok"].drop_duplicates("design_key").iterrows():
        forward = []
        for o in range(1, 7):
            v = row.get("cs%d_ross_rpm" % o, np.nan)
            if pd.notna(v):
                forward.append(float(v))
        existing161[str(row["design_key"])] = {
            "status": "ok", "forward": forward,
            "elapsed_s": float(row.get("elapsed_s", np.nan)), "error": "",
            "source": "existing_pilot_1.6.1",
        }

    out_rows = []
    for _, s in samples.iterrows():
        key = s["design_key"]
        r161 = existing161.get(key) or res161_new.get(key) or {
            "status": "missing", "forward": None, "elapsed_s": np.nan, "error": "no 1.6.1 result"}
        r230 = res230.get(key, {"status": "missing", "forward": None,
                               "elapsed_s": np.nan, "error": "no 2.3.0 result"})
        f161 = r161.get("forward")
        f230 = r230.get("forward")
        spec = specs[s["spec"]]
        row = {
            **s.to_dict(),
            "material": designs[key]["material"], "nd": designs[key]["nd"],
            "nb": designs[key]["nb"],
            "spec_orders": "|".join(str(o) for o in spec["orders"]),
            "spec_bands_json": json.dumps({str(o): list(spec["bands"][o]) for o in spec["orders"]}),
            "ross161_status": r161.get("status"), "ross161_source": r161.get("source", "new_qa_1.6.1"),
            "ross161_forward_count": (len(f161) if f161 is not None else np.nan),
            "ross161_elapsed_s": r161.get("elapsed_s"),
            "ross230_status": r230.get("status"),
            "ross230_forward_count": (len(f230) if f230 is not None else np.nan),
            "ross230_elapsed_s": r230.get("elapsed_s"),
            "pass161": pass_spec(f161, spec),
            "pass230": pass_spec(f230, spec),
        }
        for o in range(1, 7):
            row["cs%d_161_rpm" % o] = f161[o - 1] if f161 is not None and len(f161) >= o else np.nan
            row["cs%d_230_rpm" % o] = f230[o - 1] if f230 is not None and len(f230) >= o else np.nan
        row["flip_161_230"] = bool(row["pass161"] != row["pass230"])

        archived = None
        if s["sample_group"] == "retrieval_source":
            archived = ds.loc[s["source_ref_id"]]
            row["archived_status"] = archived.get("status", "")
            row["archived_forward_count"] = int(archived.get("forward_mode_count", np.nan))
            af = []
            for o in range(1, 7):
                v = archived.get("cs_%d_rpm" % o, np.nan)
                row["archived_cs%d_rpm" % o] = float(v) if pd.notna(v) else np.nan
                if pd.notna(v):
                    af.append(float(v))
            row["archived_pass"] = pass_spec(af, spec)
            row["flip_archived_161"] = bool(row["archived_pass"] != row["pass161"])
            row["flip_archived_230"] = bool(row["archived_pass"] != row["pass230"])
        else:
            row["archived_status"] = ""
            row["archived_forward_count"] = np.nan
            for o in range(1, 7):
                row["archived_cs%d_rpm" % o] = np.nan
            row["archived_pass"] = None
            row["flip_archived_161"] = None
            row["flip_archived_230"] = None

        for o in spec["orders"]:
            o = int(o)
            a = row["cs%d_161_rpm" % o] if pd.notna(row["cs%d_161_rpm" % o]) else np.nan
            b = row["cs%d_230_rpm" % o] if pd.notna(row["cs%d_230_rpm" % o]) else np.nan
            row["abs_diff_161_230_order%d_rpm" % o] = abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.nan
            row["rel_diff_161_230_order%d_pct" % o] = (100.0 * abs(a - b) / max(abs(b), 1e-12)
                                                        if np.isfinite(a) and np.isfinite(b) else np.nan)
            row["forward_order_match_order%d" % o] = bool(
                np.isfinite(a) and np.isfinite(b) and abs(a - b) <= max(1e-6, 1e-8 * abs(b)))
            if archived is not None:
                av = row["archived_cs%d_rpm" % o] if pd.notna(row["archived_cs%d_rpm" % o]) else np.nan
                row["abs_diff_archived_161_order%d_rpm" % o] = abs(av - a) if np.isfinite(av) and np.isfinite(a) else np.nan
                row["rel_diff_archived_161_order%d_pct" % o] = (100.0 * abs(av - a) / max(abs(av), 1e-12)
                                                                  if np.isfinite(av) and np.isfinite(a) else np.nan)
                row["abs_diff_archived_230_order%d_rpm" % o] = abs(av - b) if np.isfinite(av) and np.isfinite(b) else np.nan
                row["rel_diff_archived_230_order%d_pct" % o] = (100.0 * abs(av - b) / max(abs(av), 1e-12)
                                                                  if np.isfinite(av) and np.isfinite(b) else np.nan)
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    sample_path = os.path.join(OUT, "ross_version_comparison_samples.csv")
    samples.to_csv(sample_path, index=False, encoding="utf-8")
    comp_path = os.path.join(OUT, "ross_version_comparison.csv")
    out.to_csv(comp_path, index=False, encoding="utf-8")
    manifest = {
        "created": A.utc_now(),
        "boundary_seed": BOUNDARY_SEED, "random_seed": RANDOM_SEED, "source_seed": SOURCE_SEED,
        "selection_rules": {
            "pool_boundary": "2 per pool: smallest normalized distance from existing 1.6.1 ROSS value to the nearest user-band boundary, ties by design_key",
            "pool_random": "2 per pool: uniform without replacement from remaining solved and deduplicated pool designs, fixed seed",
            "retrieval_source": "6 per spec: uniform without replacement from unique source_ref_id values in the frozen retrieval pool, fixed seed",
        },
        "unique_designs": len(designs),
        "selected_sample_rows": len(samples),
        "physical_solves_added": {
            "ross_1_6_1": len(missing_161), "ross_2_3_0": len(designs),
            "total": len(missing_161) + len(designs),
        },
        "wall_s": {"ross_1_6_1_new": wall161, "ross_2_3_0": wall230},
        "files": {"samples_sha256": sha256_file(sample_path),
                  "comparison_sha256": sha256_file(comp_path)},
        "env_files": {
            "ross_1_6_1": os.path.join(A.ENV_DIR, "ross_env_161.json"),
            "ross_2_3_0": os.path.join(A.ENV_DIR, "ross_env_230.json"),
        },
    }
    A.save_json(os.path.join(OUT, "ross_version_comparison_manifest.json"), manifest)
    print("[ross-compare] rows=%d physical_added=%d wall161=%.1fs wall230=%.1fs"
          % (len(out), manifest["physical_solves_added"]["total"], wall161, wall230), flush=True)


if __name__ == "__main__":
    main()


