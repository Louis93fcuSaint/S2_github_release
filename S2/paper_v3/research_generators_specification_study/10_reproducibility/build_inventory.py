# -*- coding: utf-8 -*-
"""Stage 0: asset inventory, executability audit and SHA-256 manifest."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pandas as pd

import research_common as C

OUT = C.ROOT / "00_inventory"
OUT.mkdir(parents=True, exist_ok=True)
_hashes = {}


def count_rows(path: Path) -> int:
    if not path.exists():
        return -1
    with open(path, "rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def hash_file(path: Path) -> str:
    key = str(path.resolve())
    if key not in _hashes:
        _hashes[key] = C.sha256(path)
    return _hashes[key]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(C.S2.resolve()))
    except Exception:
        return str(path)


def add_row(rows, group, role, method, spec, seed, path, data_version,
            training_condition, candidate_n=-1, ross_n=-1, notes=""):
    path = Path(path)
    exists = path.exists()
    columns = []
    has_proxy = False
    if exists and path.suffix.lower() == ".csv" and path.stat().st_size < 30_000_000:
        try:
            columns = list(pd.read_csv(path, nrows=0).columns)
            has_proxy = "cs1_pred" in columns or "cs2_pred" in columns
        except Exception:
            pass
    rows.append({
        "group_id": group,
        "asset_role": role,
        "method": method,
        "specification": spec,
        "sampling_seed": seed,
        "path": str(path),
        "path_relative_to_S2": rel(path),
        "exists": exists,
        "data_version": data_version,
        "training_condition": training_condition,
        "rows": count_rows(path) if exists and path.suffix.lower() == ".csv" else -1,
        "candidate_n": candidate_n,
        "ross_submitted_n": ross_n,
        "columns": "|".join(columns),
        "has_proxy_prediction": has_proxy,
        "sha256": hash_file(path) if exists else "",
        "bytes": path.stat().st_size if exists else 0,
        "rerunnable": bool(exists),
        "notes": notes,
    })


rows = []

# 1. Raw reference dataset and model checkpoints
raw_count = count_rows(C.DATA_CSV)
add_row(rows, "reference_data", "raw_ROSS_labeled_design_library", "none", "all", "",
        C.DATA_CSV, "dataset_v4.7", "47 features; status and cs_1..cs_6 labels",
        candidate_n=raw_count, notes="Read-only reference library; not a generated candidate pool.")

for name, path in C.CHECKPOINTS.items():
    condition = {
        "ddpm_interval": "34-D latent DDPM; 21-D interval condition; 3x512 MLP; v-prediction",
        "ddpm_point": "34-D latent DDPM; 18-D point condition; 3x512 MLP; v-prediction",
        "cvae": "34-D latent CVAE; 21-D interval condition; 3x512 encoder/decoder; z=32",
    }[name]
    add_row(rows, "model_checkpoint", name, name, "all", "", path, "v4_or_cvae_main",
            condition, notes="Seeds 7/13/42 are sampling seeds under this checkpoint, not independent training runs.")

# 2. Shared surrogate assets
for path in [
    C.S2 / "surrogate_v47" / "outputs" / "split_v47.npz",
    C.S2 / "surrogate_v47" / "outputs" / "X_v47.npy",
    *[C.S2 / "surrogate_v47" / "outputs" / "mlp_v47_best" / "models" / f"multi_seed{s}.pt" for s in range(42, 47)],
    C.S2 / "surrogate_v47" / "outputs" / "mlp_v47_best" / "summary_mlp.json",
]:
    add_row(rows, "surrogate", "frozen_five_model_ensemble_member" if path.suffix == ".pt" else "surrogate_split_or_summary",
            "surrogate", "all", "", path, "v4.7", "five independently seeded MLPs; log-space ensemble",
            notes="Used by all four generation pipelines for pre-screening.")

# 3. Four-way generator experiment, one row per run
for method in C.METHODS:
    for spec in C.SPEC_NAMES:
        for seed in C.SEEDS:
            pool, short, ver, vcsv, passed = C.method_paths(method, spec, seed)
            payload = C.read_json(ver) if ver.exists() else {}
            delivered = payload.get("deliverable_n", -1)
            submitted = payload.get("n_shortlist", count_rows(short) if short.exists() else -1)
            add_row(rows, "four_way_main", "full_pool", method, spec, seed, pool,
                    "fairness_audit_v4", "same frozen six specifications; 5000 generated",
                    candidate_n=count_rows(pool), ross_n=-1,
                    notes="Full candidate pool; no ROSS labels for all 5000.")
            add_row(rows, "four_way_main", "top200_shortlist", method, spec, seed, short,
                    "fairness_audit_v4", "select_common: proxy in-band first, then farthest-point diversity",
                    candidate_n=count_rows(short), ross_n=submitted,
                    notes="Same selector and 200 ROSS budget for all four pipelines.")
            add_row(rows, "four_way_main", "ROSS_verify_json", method, spec, seed, ver,
                    "fairness_audit_v4", "per-design ROSS 1.6.1 re-computation",
                    candidate_n=submitted, ross_n=submitted,
                    notes=f"deliverable_n={delivered}")
            add_row(rows, "four_way_main", "ROSS_verify_candidates", method, spec, seed, vcsv,
                    "fairness_audit_v4", "per-design ROSS 1.6.1 re-computation",
                    candidate_n=count_rows(vcsv), ross_n=count_rows(vcsv),
                    notes="Includes failures/timeouts; denominator must remain fixed.")
            add_row(rows, "four_way_main", "ROSS_passed_subset", method, spec, seed, passed,
                    "fairness_audit_v4", "cs1_ross within target +/-5%",
                    candidate_n=count_rows(passed), ross_n=count_rows(vcsv),
                    notes=f"passed CSV rows should equal deliverable_n={delivered}.")

# 4. Retrieval + jitter: unified selector and center-ranking arm
for spec in C.SPEC_NAMES:
    s = C.spec(spec)
    pool = C.RETRIEVAL / f"retrieval_jitter_pool__{spec}.csv"
    short = C.RETRIEVAL / f"retrieval_jitter__{spec}.csv"
    ver = C.V2_OUT / f"verify_setobj_jitter__{spec}.json"
    vcsv = C.V2_OUT / f"verify_setobj_jitter__{spec}__top200.csv"
    passed = C.V2_OUT / f"verify_setobj_jitter__{spec}__top200__passed.csv"
    payload = C.read_json(ver) if ver.exists() else {}
    run0 = (payload.get("runs") or [{}])[0]
    add_row(rows, "retrieval_jitter", "jitter_pool", "retrieval_jitter_unified", spec, 42, pool,
            "set_objective_revision", "retrieve in-band reference designs + 5% parameter jitter; seed 42",
            candidate_n=count_rows(pool), notes="Requires a labelled in-band reference library.")
    add_row(rows, "retrieval_jitter", "unified_shortlist", "retrieval_jitter_unified", spec, 42, short,
            "set_objective_revision", "same select_common selector and 200 ROSS budget",
            candidate_n=count_rows(short), ross_n=200)
    add_row(rows, "retrieval_jitter", "ROSS_verify_json", "retrieval_jitter_unified", spec, 42, ver,
            "set_objective_revision", "ROSS 1.6.1; 200 submitted",
            candidate_n=run0.get("n_shortlist", 200), ross_n=run0.get("n_shortlist", 200),
            notes=f"deliverable_n={run0.get('deliverable_n', -1)}")
    add_row(rows, "retrieval_jitter", "ROSS_verify_candidates", "retrieval_jitter_unified", spec, 42, vcsv,
            "set_objective_revision", "ROSS 1.6.1; failures retained",
            candidate_n=count_rows(vcsv), ross_n=count_rows(vcsv))
    add_row(rows, "retrieval_jitter", "ROSS_passed_subset", "retrieval_jitter_unified", spec, 42, passed,
            "set_objective_revision", "passed within target +/-5%",
            candidate_n=count_rows(passed), ross_n=count_rows(vcsv))

    cver = C.V2_OUT / f"verify_jitterphys42__{spec}.json"
    ccsv = C.V2_OUT / f"verify_jitterphys42__{spec}__top200.csv"
    cpass = C.V2_OUT / f"verify_jitterphys42__{spec}__top200__passed.csv"
    cp = C.read_json(cver) if cver.exists() else {}
    cr0 = (cp.get("runs") or [{}])[0]
    add_row(rows, "retrieval_jitter_center", "center_rank_shortlist", "retrieval_jitter_center", spec, 42, ccsv,
            "ross_audit_revision", "known in-band references ranked toward band centre; information advantage",
            candidate_n=count_rows(ccsv), ross_n=count_rows(ccsv),
            notes=f"deliverable_n={cr0.get('deliverable_n', -1)}")
    add_row(rows, "retrieval_jitter_center", "ROSS_passed_subset", "retrieval_jitter_center", spec, 42, cpass,
            "ross_audit_revision", "passed within target +/-5%",
            candidate_n=count_rows(cpass), ross_n=count_rows(ccsv))

# 5. Multi-order / order-difficulty point-DDPM experiments
V2_CONDITIONS = [
    ("cs1", {"1": 3680.8}, "Steel", None),
    ("cs2", {"2": 7021.0}, "Steel", None),
    ("cs3", {"3": 21033.5}, "Steel", None),
    ("cs1cs3", {"1": 3680.8, "3": 18070.4}, "Steel", None),
    ("cs2cs3", {"2": 7021.0, "3": 15857.7}, "Steel", None),
    ("cs1cs2", {"1": 3680.8, "2": 7021.0}, "Steel", None),
    ("cs123", {"1": 3680.8, "2": 7021.0, "3": 17970.5}, "Steel", None),
    ("cs1cs2_big", {"1": 3680.8, "2": 7021.0}, "Steel", None),
    ("cs123_big", {"1": 3680.8, "2": 7021.0, "3": 17970.5}, "Steel", None),
    ("al_cs1cs2", {"1": 3191.1, "2": 6147.6}, "Aluminum", None),
    ("fam32_cs1", {"1": 2974.0}, "Steel", "3-3x2-2"),
    ("fam32_cs12", {"1": 2974.0, "2": 7001.0}, "Steel", "3-3x2-2"),
    ("fam22_cs2", {"2": 8180.0}, "Steel", "2-2x2-2"),
    ("fam12_24_cs12", {"1": 4642.0, "2": 9969.0}, "Steel", "1-2x2-4"),
    ("fam32_cs123", {"1": 2974.0, "2": 7001.0, "3": 20359.0}, "Steel", "3-3x2-2"),
]
for label, targets, material, family in V2_CONDITIONS:
    cand = C.V2_OUT / f"v2_{label}.csv"
    ver = C.V2_OUT / f"verify_{label}.json"
    add_row(rows, "multitorder_point_ddpm", "full_pool", "ddpm_point", label, 42, cand,
            "v2_multiorder", f"18-D point condition; {material}; family={family or 'free'}; 25-step guidance",
            candidate_n=count_rows(cand), notes="Point-conditioned checkpoint; not part of the interval DDPM-vs-CVAE ranking.")
    if ver.exists():
        obj = C.read_json(ver)
        runs = obj.get("runs") or []
        for r in runs:
            add_row(rows, "multitorder_point_ddpm", r.get("label", "ROSS_verify"), "ddpm_point", label, 42,
                    C.V2_OUT / (f"verify_{label}{'__rand200' if 'rand200' in r.get('label','') else '__top200'}.csv"),
                    "v2_multiorder", "top200 and random-200 ROSS controls; +/-1% and +/-5% reported",
                    candidate_n=r.get("n_shortlist", -1), ross_n=r.get("n_shortlist", -1),
                    notes=f"deliverable_n_tol5={r.get('gate_tol5',{}).get('n_passed',-1)}")

# 6. Interval-width and tolerance diagnostics from the interval checkpoint
for label in ["int_narrow_v2i", "int_mid_v2i", "int_wide_v2i", "int_open_v2i",
              "w075", "w100", "w150", "w200", "w250"]:
    cand_name = f"v2_{label}.csv"
    ver_name = f"verify_{label}.json"
    cand = C.V2_OUT / cand_name
    ver = C.V2_OUT / ver_name
    if cand.exists():
        add_row(rows, "interval_width_diagnostics", "full_pool", "ddpm_interval", label, 42, cand,
                "v2_interval_diagnostics", "interval checkpoint; diagnostic sweep",
                candidate_n=count_rows(cand), notes="Used only for width/support trend diagnostics.")
    if ver.exists():
        add_row(rows, "interval_width_diagnostics", "ROSS_verify_json", "ddpm_interval", label, 42, ver,
                "v2_interval_diagnostics", "top-200 ROSS re-computation",
                candidate_n=200, ross_n=200)

# 7. Optimization baselines (already verified under the unified selector)
for method in ["ga", "de", "adam", "bo", "ours", "jitter"]:
    for spec in C.SPEC_NAMES:
        ver = C.V2_OUT / f"verify_setobj_{method}__{spec}.json"
        if ver.exists():
            obj = C.read_json(ver)
            run0 = (obj.get("runs") or [{}])[0]
            add_row(rows, "optimization_baseline", "ROSS_verify_json", method, spec, 42, ver,
                    "set_objective_revision", "same six specifications and 200 ROSS budget",
                    candidate_n=run0.get("n_shortlist", 200), ross_n=run0.get("n_shortlist", 200),
                    notes=f"deliverable_n={run0.get('deliverable_n', -1)}; proxy-call budget differs by method.")

inventory = pd.DataFrame(rows)
inventory.to_csv(OUT / "实验资产总表.csv", index=False, encoding="utf-8-sig")

# Unique hash manifest
manifest = []
for path_str, digest in sorted(_hashes.items()):
    p = Path(path_str)
    manifest.append({"path": path_str, "bytes": p.stat().st_size, "sha256": digest})
pd.DataFrame(manifest).to_csv(OUT / "SHA256SUMS.csv", index=False, encoding="utf-8-sig")

# Dataset status audit
raw = pd.read_csv(C.DATA_CSV, usecols=["status", "cs_1_rpm", "cs_2_rpm", "cs_3_rpm"])
status = raw["status"].value_counts(dropna=False).to_dict()
valid = {f"cs_{k}_finite": int(pd.to_numeric(raw[f"cs_{k}_rpm"], errors="coerce").notna().sum())
         for k in (1, 2, 3)}
missing = {f"cs_{k}_missing": int(len(raw) - v) for k, v in valid.items()}
summary = {
    "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    "root": str(C.ROOT),
    "paper_v2_modified": False,
    "raw_dataset": str(C.DATA_CSV),
    "raw_rows": int(len(raw)),
    "status_counts": {str(k): int(v) for k, v in status.items()},
    "valid_label_counts": valid,
    "missing_label_counts": missing,
    "inventory_rows": int(len(inventory)),
    "unique_files_hashed": int(len(manifest)),
    "checkpoints": {k: {"path": str(v), "sha256": hash_file(v), "bytes": v.stat().st_size}
                    for k, v in C.CHECKPOINTS.items()},
    "sampling_seeds_share_checkpoint": True,
    "seed_interpretation": "seeds 7, 13 and 42 are repeated sampling seeds under one trained checkpoint per model",
    "scientific_boundary": "ROSS results are numerical re-computations, not physical experiments.",
}
with open(OUT / "asset_inventory.json", "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)

# Reuse / new-computation decision table
decisions = pd.DataFrame([
    {"work_item": "six-specification p_ref", "decision": "reuse", "reason": "all ROSS labels are present in dataset_v4.7", "new_ROSS_required": False},
    {"work_item": "DDPM/CVAE four-pipeline comparison", "decision": "reuse", "reason": "72 runs, 5,000 pools, top-200 shortlists and ROSS outputs exist", "new_ROSS_required": False},
    {"work_item": "raw-pool true pass rate", "decision": "cannot estimate exactly", "reason": "only top-200 and selected random-200 controls have ROSS labels", "new_ROSS_required": True},
    {"work_item": "candidate diversity at three layers", "decision": "reuse + derive", "reason": "full pools and shortlists exist; delivered subsets exist", "new_ROSS_required": False},
    {"work_item": "multi-order, fixed-family trend", "decision": "reuse with caveat", "reason": "point-conditioned DDPM v2 experiment, not interval DDPM/CVAE", "new_ROSS_required": False},
    {"work_item": "retrieval+jitter comparison", "decision": "reuse seed 42", "reason": "unified and center-ranking ROSS results exist for six specs", "new_ROSS_required": False},
    {"work_item": "DDPM vs CVAE strict architecture ranking", "decision": "reuse with caveat", "reason": "system-level comparison; guidance mechanisms differ", "new_ROSS_required": False},
    {"work_item": "new support-matched specifications", "decision": "freeze only", "reason": "first batch should not launch large new ROSS runs", "new_ROSS_required": True},
], columns=["work_item", "decision", "reason", "new_ROSS_required"])
decisions.to_csv(OUT / "可复用与新增计算清单.csv", index=False, encoding="utf-8-sig")

print(json.dumps(summary, ensure_ascii=False, indent=2))
