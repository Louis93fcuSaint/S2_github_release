# -*- coding: utf-8 -*-
"""Generate the six fixed-budget raw pools (5000 attempts each).

No retry loop is used to inflate the number of unique valid designs.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eaai_common as A


def pool_id(method: str, slot: str, seed: int = A.SEED) -> str:
    return "%s_%s_s%d" % (slot, method, seed)


def unique_stats(keys, valid):
    keys = np.asarray(keys, dtype=object)
    valid = np.asarray(valid, dtype=bool)
    seen = set()
    representative = np.full(len(keys), -1, dtype=np.int64)
    repeat_of = np.full(len(keys), -1, dtype=np.int64)
    first_seen = {}
    for i, key in enumerate(keys):
        if key in first_seen:
            repeat_of[i] = first_seen[key]
            representative[i] = first_seen[key]
        else:
            first_seen[key] = i
            representative[i] = i
    valid_keys = keys[valid]
    unique_valid = len(set(valid_keys.tolist()))
    all_unique = len(set(keys.tolist()))
    return {
        "n_valid": int(valid.sum()),
        "n_invalid": int((~valid).sum()),
        "n_unique_valid": int(unique_valid),
        "n_unique_all": int(all_unique),
        "n_exact_duplicate_valid_attempts": int(valid.sum() - unique_valid),
        "n_exact_duplicate_all_attempts": int(len(keys) - all_unique),
        "valid_duplicate_rate": float(1.0 - unique_valid / max(int(valid.sum()), 1)),
    }, representative, repeat_of


def write_pool(bundle, ctx, train_idx, spec, method, surrogate, seed=A.SEED):
    pid = pool_id(method, spec["slot_id"], seed)
    out_dir = os.path.join(A.POOL_DIR, pid)
    os.makedirs(out_dir, exist_ok=True)
    t_all = time.time()
    meta = {"pool_id": pid, "method": method, "spec": spec["slot_id"],
            "spec_id": spec["spec_id"], "material": spec["material"],
            "family": spec["family_text"], "orders": spec["orders"],
            "bands": {str(k): list(v) for k, v in spec["bands"].items()},
            "seed": int(seed), "n_attempts": A.N_ATTEMPTS,
            "sample_steps": A.SAMPLE_STEPS if method == "ddpm" else None,
            "cfg_guidance": A.CFG_GUIDANCE if method == "ddpm" else None,
            "guide_lambda": A.GUIDE_LAMBDA,
            "condition_input_active": True,
            "created": A.utc_now()}
    if method == "ddpm":
        families = A.family_sequence(spec["family"], A.N_ATTEMPTS, seed)
        t0 = time.time()
        u, cond = A.sample_ddpm(bundle, spec["material"], families,
                                A.band_specs(spec), seed, A.N_ATTEMPTS)
        x = A.decode_latents(u, families)
        t_gen = time.time() - t0
        extra = {}
    elif method == "cvae":
        families = A.family_sequence(spec["family"], A.N_ATTEMPTS, seed)
        t0 = time.time()
        u, cond = A.sample_cvae(bundle, spec["material"], families,
                                A.band_specs(spec), seed, A.N_ATTEMPTS)
        x = A.decode_latents(u, families)
        t_gen = time.time() - t0
        extra = {}
    elif method == "retrieval":
        t0 = time.time()
        x, families, origins, changes, rmeta = A.sample_retrieval(
            ctx, train_idx, spec, surrogate, seed=seed,
            n_attempts=A.N_ATTEMPTS, n_originals=A.N_ORIGINALS,
            frac=A.JITTER_FRAC)
        u = np.stack([A.LD.encode(row, fam) for row, fam in zip(x, families)])
        t_gen = time.time() - t0
        ref_ids = A.reference_ids(ctx)
        extra = {"retrieval": rmeta,
                 "source_row": origins,
                 "source_ref_id": ref_ids[origins],
                 "jitter_change": changes}
    else:
        raise ValueError(method)

    t1 = time.time()
    valid = A.validity(x, spec["material"], families)
    t_valid = time.time() - t1
    t1 = time.time()
    pred = A.predict_six(x, spec["material"], families, surrogate)
    t_pred = time.time() - t1
    keys = A.design_key_many(x)
    stats, representative, repeat_of = unique_stats(keys, valid)
    in_band, score, per_order = A.gate_and_score(pred, spec)
    t_total = time.time() - t_all

    attempt_id = np.arange(A.N_ATTEMPTS, dtype=np.int64)
    nd = np.array([int(f[0]) for f in families], dtype=np.int16)
    nb = np.array([int(f[1]) for f in families], dtype=np.int16)
    material_arr = np.array([spec["material"]] * A.N_ATTEMPTS)

    np.savez_compressed(
        os.path.join(out_dir, "raw.npz"),
        x=np.asarray(x, dtype=np.float64), u=np.asarray(u, dtype=np.float64),
        pred=np.asarray(pred, dtype=np.float64), nd=nd, nb=nb,
        attempt_id=attempt_id, valid=valid.astype(bool),
        in_band=in_band.astype(bool), violation=score,
        representative=representative, repeat_of=repeat_of,
        material=material_arr,
        design_key=np.asarray(keys, dtype="U64"))

    frame = A.O.make_frame(x, spec["material"], families)
    frame.insert(0, "attempt_id", attempt_id)
    frame["design_key"] = keys
    frame["valid"] = valid
    frame["in_band_proxy"] = in_band
    frame["violation"] = score
    for i in range(6):
        frame["cs%d_pred" % (i + 1)] = pred[:, i]
    if method == "retrieval":
        frame["source_row"] = origins
        frame["source_ref_id"] = extra["source_ref_id"]
        frame["jitter_change"] = changes
    frame.to_csv(os.path.join(out_dir, "pool.csv"), index=False,
                 encoding="utf-8", float_format="%.17g")

    dedup = pd.DataFrame({
        "attempt_id": attempt_id, "design_key": keys, "valid": valid,
        "representative_attempt_id": representative, "repeat_of": repeat_of,
        "is_representative": representative == attempt_id,
    })
    dedup.to_csv(os.path.join(out_dir, "dedup_map.csv"), index=False)

    fam_counts = pd.Series(["%dd-%db" % (int(a), int(b)) for a, b in families]).value_counts()
    unique_idx, _ = A.exact_unique_indices(keys[valid])
    valid_rows = np.where(valid)[0]
    unique_valid_rows = valid_rows[unique_idx]
    meta.update({
        "n_valid": stats["n_valid"], "n_invalid": stats["n_invalid"],
        "n_unique_valid": stats["n_unique_valid"], "n_unique_all": stats["n_unique_all"],
        "n_exact_duplicate_valid_attempts": stats["n_exact_duplicate_valid_attempts"],
        "n_exact_duplicate_all_attempts": stats["n_exact_duplicate_all_attempts"],
        "valid_duplicate_rate": stats["valid_duplicate_rate"],
        "n_proxy_in_band_attempts": int(in_band[valid].sum()),
        "n_proxy_in_band_unique_valid": int(in_band[unique_valid_rows].sum()),
        "proxy_in_band_rate_attempts": float(in_band[valid].mean()) if stats["n_valid"] else None,
        "proxy_in_band_rate_unique_valid": float(in_band[unique_valid_rows].mean())
            if len(unique_valid_rows) else None,
        "median_violation_unique_valid": float(np.median(score[unique_valid_rows]))
            if len(unique_valid_rows) else None,
        "p90_violation_unique_valid": float(np.percentile(score[unique_valid_rows], 90))
            if len(unique_valid_rows) else None,
        "n_nonfinite_pred": int((~np.isfinite(pred)).any(axis=1).sum()),
        "family_counts": {str(k): int(v) for k, v in fam_counts.items()},
        "family_sequence_sha256": A.sha256_bytes("|".join(
            "%d-%d" % (int(f[0]), int(f[1])) for f in families).encode("ascii")),
        "timing": {"generation_s": float(t_gen), "validity_s": float(t_valid),
                   "proxy_s": float(t_pred), "total_s": float(t_total)},
        "sha256": {"x": A.sha256_bytes(np.asarray(x, dtype="<f8").tobytes()),
                   "u": A.sha256_bytes(np.asarray(u, dtype="<f8").tobytes()),
                   "pred": A.sha256_bytes(np.asarray(pred, dtype="<f8").tobytes()),
                   "raw_npz": A.sha256_file(os.path.join(out_dir, "raw.npz")),
                   "pool_csv": A.sha256_file(os.path.join(out_dir, "pool.csv"))},
        "files": {"raw_npz": os.path.join(out_dir, "raw.npz"),
                  "pool_csv": os.path.join(out_dir, "pool.csv"),
                  "dedup_map": os.path.join(out_dir, "dedup_map.csv")},
        **extra,
    })
    A.save_json(os.path.join(out_dir, "manifest.json"), meta)
    return meta, x, families, pred, valid, keys, in_band, score, u


def main():
    t_batch = time.time()
    before = A.old_output_snapshot()
    ctx, train_idx, _val_idx, _ = A.build_context_split(save=False)
    specs = A.load_calibration_specs()
    ddpm = A.load_ddpm()
    cvae = A.load_cvae()
    surrogate = A.S.load_surrogate("mlp_best", threads=16)
    manifests = []
    for slot in ("CAL02", "CAL05"):
        spec = specs[slot]
        for method, bundle in (("ddpm", ddpm), ("cvae", cvae),
                               ("retrieval", None)):
            print("[pool] %s %s" % (slot, method), flush=True)
            meta, *_ = write_pool(bundle, ctx, train_idx, spec, method, surrogate)
            manifests.append(meta)
            print("[pool]   attempts=%d valid=%d unique_valid=%d dup=%d inband_unique=%d %.1fs"
                  % (meta["n_attempts"], meta["n_valid"], meta["n_unique_valid"],
                     meta["n_exact_duplicate_valid_attempts"],
                     meta["n_proxy_in_band_unique_valid"],
                     meta["timing"]["total_s"]), flush=True)
    after = A.old_output_snapshot()
    unchanged = before["sha256"] == after["sha256"]
    A.save_json(os.path.join(A.METRIC_DIR, "old_output_pool_run.json"),
                {"before": before, "after": after, "unchanged": unchanged})
    if not unchanged:
        raise SystemExit("historical output directories changed during pool generation")

    # pilot_run_manifest.csv
    rows = []
    for m in manifests:
        rows.append({
            "pool_id": m["pool_id"], "method": m["method"], "spec": m["spec"],
            "spec_id": m["spec_id"], "material": m["material"], "family": m["family"],
            "orders": "|".join(str(o) for o in m["orders"]),
            "seed": m["seed"], "n_attempts": m["n_attempts"],
            "n_valid": m["n_valid"], "n_invalid": m["n_invalid"],
            "n_unique_valid": m["n_unique_valid"],
            "n_exact_duplicate_valid_attempts": m["n_exact_duplicate_valid_attempts"],
            "valid_duplicate_rate": m["valid_duplicate_rate"],
            "n_proxy_in_band_attempts": m["n_proxy_in_band_attempts"],
            "n_proxy_in_band_unique_valid": m["n_proxy_in_band_unique_valid"],
            "generation_s": m["timing"]["generation_s"],
            "validity_s": m["timing"]["validity_s"],
            "proxy_s": m["timing"]["proxy_s"],
            "total_s": m["timing"]["total_s"],
            "family_sequence_sha256": m["family_sequence_sha256"],
            "x_sha256": m["sha256"]["x"], "u_sha256": m["sha256"]["u"],
            "pred_sha256": m["sha256"]["pred"],
            "raw_npz": m["files"]["raw_npz"], "pool_csv": m["files"]["pool_csv"],
            "dedup_map": m["files"]["dedup_map"],
            "retrieval_reference_feasible": (m.get("retrieval", {}) or {}).get(
                "n_reference_feasible", ""),
            "retrieval_sources_used": (m.get("retrieval", {}) or {}).get(
                "n_originals_used", ""),
        })
    pd.DataFrame(rows).to_csv(os.path.join(A.STUDY, "pilot_run_manifest.csv"),
                              index=False, encoding="utf-8")
    pd.DataFrame(rows).to_csv(os.path.join(A.METRIC_DIR, "pilot_run_manifest.csv"),
                              index=False, encoding="utf-8")

    # pilot_pool_summary.csv
    summary = []
    for m in manifests:
        summary.append({
            "pool_id": m["pool_id"], "method": m["method"], "spec": m["spec"],
            "material": m["material"], "family": m["family"],
            "n_attempts": m["n_attempts"], "n_valid": m["n_valid"],
            "n_invalid": m["n_invalid"], "n_unique_valid": m["n_unique_valid"],
            "n_exact_duplicate_valid_attempts": m["n_exact_duplicate_valid_attempts"],
            "n_exact_duplicate_all_attempts": m["n_exact_duplicate_all_attempts"],
            "valid_duplicate_rate": m["valid_duplicate_rate"],
            "n_proxy_in_band_attempts": m["n_proxy_in_band_attempts"],
            "n_proxy_in_band_unique_valid": m["n_proxy_in_band_unique_valid"],
            "proxy_in_band_rate_unique_valid": m["proxy_in_band_rate_unique_valid"],
            "median_violation_unique_valid": m["median_violation_unique_valid"],
            "p90_violation_unique_valid": m["p90_violation_unique_valid"],
            "n_nonfinite_pred": m["n_nonfinite_pred"],
            "n_families": len(m["family_counts"]),
            "family_counts": ";".join("%s:%d" % (k, v)
                                      for k, v in sorted(m["family_counts"].items())),
            "generation_s": m["timing"]["generation_s"],
            "validity_s": m["timing"]["validity_s"],
            "proxy_s": m["timing"]["proxy_s"],
            "total_s": m["timing"]["total_s"],
        })
    pd.DataFrame(summary).to_csv(os.path.join(A.STUDY, "pilot_pool_summary.csv"),
                                 index=False, encoding="utf-8")
    pd.DataFrame(summary).to_csv(os.path.join(A.METRIC_DIR, "pilot_pool_summary.csv"),
                                 index=False, encoding="utf-8")

    cost = []
    for m in manifests:
        cost.append({
            "pool_id": m["pool_id"], "stage": "generation",
            "wall_s": m["timing"]["generation_s"], "cpu_core_s": "unknown",
            "n_units": m["n_attempts"], "notes": "sample + decode"})
        cost.append({
            "pool_id": m["pool_id"], "stage": "validity",
            "wall_s": m["timing"]["validity_s"], "cpu_core_s": "unknown",
            "n_units": m["n_attempts"], "notes": "H1-H6 checks"})
        cost.append({
            "pool_id": m["pool_id"], "stage": "surrogate",
            "wall_s": m["timing"]["proxy_s"], "cpu_core_s": "unknown",
            "n_units": m["n_attempts"], "notes": "5-seed ensemble, log-space mean"})
        cost.append({
            "pool_id": m["pool_id"], "stage": "pool_total",
            "wall_s": m["timing"]["total_s"], "cpu_core_s": "unknown",
            "n_units": m["n_attempts"], "notes": "excluding shortlist and ROSS"})
    cost.append({"pool_id": "ALL", "stage": "batch_generation",
                 "wall_s": time.time() - t_batch, "cpu_core_s": "unknown",
                 "n_units": len(manifests), "notes": "six pools"})
    pd.DataFrame(cost).to_csv(os.path.join(A.METRIC_DIR, "pilot_cost_breakdown.csv"),
                              index=False, encoding="utf-8")
    pd.DataFrame(cost).to_csv(os.path.join(A.STUDY, "pilot_cost_breakdown.csv"),
                              index=False, encoding="utf-8")
    print("[pool] batch %.1fs -> %s" % (time.time() - t_batch, A.POOL_DIR))


if __name__ == "__main__":
    main()
