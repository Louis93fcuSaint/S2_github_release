# -*- coding: utf-8 -*-
"""Generate the frozen eight-specification TST pools (3 methods x 2 seeds)."""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
EA = os.path.abspath(os.path.join(PROJECT, "..", "research_generators_specification_study", "eaai_comparative_study"))
AUDIT = os.path.join(EA, "audit_scripts")
sys.path.insert(0, AUDIT)

import eaai_common as A  # noqa: E402
import run_eaai_pools as P  # noqa: E402

RUN = os.path.join(PROJECT, "tst_runs")
A.STUDY = RUN
A.SPEC_CSV = os.path.join(EA, "spec_candidates", "test_specs.csv")
A.POOL_DIR = os.path.join(RUN, "01_pools")
A.SHORT_DIR = os.path.join(RUN, "02_shortlists")
A.ROSS_DIR = os.path.join(RUN, "03_ross")
A.METRIC_DIR = os.path.join(RUN, "04_metrics")
A.LOG_DIR = os.path.join(RUN, "logs")
A.N_RANDOM = 100
A.N_SHORT = 200
for d in (A.POOL_DIR, A.SHORT_DIR, A.ROSS_DIR, A.METRIC_DIR, A.LOG_DIR):
    os.makedirs(d, exist_ok=True)

SEEDS = [42, 7]
METHODS = ["ddpm", "cvae", "retrieval"]
SPEC_ORDER = [
    "TST01", "TST02", "TST03", "TST04",
    "TST05", "TST06", "TST07", "TST09",
]


def main():
    t_batch = time.time()
    before = A.old_output_snapshot()
    ctx, train_idx, _val_idx, _ = A.build_context_split(save=False)
    specs = A.load_calibration_specs()
    missing = [s for s in SPEC_ORDER if s not in specs]
    if missing:
        raise SystemExit("missing frozen test specs: %s" % missing)
    if any(s.startswith("TST08") for s in SPEC_ORDER):
        raise SystemExit("TST08 must not be run in the main comparison")
    ddpm = A.load_ddpm()
    cvae = A.load_cvae()
    surrogate = A.S.load_surrogate("mlp_best", threads=16)
    manifests = []
    for slot in SPEC_ORDER:
        spec = specs[slot]
        for seed in SEEDS:
            for method, bundle in (("ddpm", ddpm), ("cvae", cvae), ("retrieval", None)):
                print("[pool] %s seed=%d %s" % (slot, seed, method), flush=True)
                meta, *_ = P.write_pool(bundle, ctx, train_idx, spec, method,
                                        surrogate, seed=seed)
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
        raise SystemExit("historical output directories changed during TST pool generation")

    rows = []
    for m in manifests:
        rows.append({
            "pool_id": m["pool_id"], "method": m["method"], "spec": m["spec"],
            "spec_id": m["spec_id"], "material": m["material"], "family": m["family"],
            "orders": "|".join(str(o) for o in m["orders"]), "seed": m["seed"],
            "n_attempts": m["n_attempts"], "n_valid": m["n_valid"],
            "n_invalid": m["n_invalid"], "n_unique_valid": m["n_unique_valid"],
            "n_exact_duplicate_valid_attempts": m["n_exact_duplicate_valid_attempts"],
            "valid_duplicate_rate": m["valid_duplicate_rate"],
            "n_proxy_in_band_attempts": m["n_proxy_in_band_attempts"],
            "n_proxy_in_band_unique_valid": m["n_proxy_in_band_unique_valid"],
            "generation_s": m["timing"]["generation_s"],
            "validity_s": m["timing"]["validity_s"],
            "proxy_s": m["timing"]["proxy_s"], "total_s": m["timing"]["total_s"],
            "family_sequence_sha256": m["family_sequence_sha256"],
            "x_sha256": m["sha256"]["x"], "u_sha256": m["sha256"]["u"],
            "pred_sha256": m["sha256"]["pred"],
            "raw_npz": m["files"]["raw_npz"], "pool_csv": m["files"]["pool_csv"],
            "retrieval_reference_feasible": (m.get("retrieval", {}) or {}).get(
                "n_reference_feasible", ""),
            "retrieval_sources_used": (m.get("retrieval", {}) or {}).get(
                "n_originals_used", ""),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(PROJECT, "tst_pool_manifest.csv"), index=False,
                    encoding="utf-8")
    manifest.to_csv(os.path.join(A.METRIC_DIR, "tst_pool_manifest.csv"), index=False,
                    encoding="utf-8")
    cost = []
    for m in manifests:
        for stage, key in (("generation", "generation_s"), ("validity", "validity_s"),
                           ("surrogate", "proxy_s"), ("pool_total", "total_s")):
            cost.append({"pool_id": m["pool_id"], "stage": stage,
                         "wall_s": m["timing"][key], "cpu_core_s": "unknown",
                         "n_units": m["n_attempts"], "notes": "TST fixed protocol"})
    cost.append({"pool_id": "ALL", "stage": "batch_generation",
                 "wall_s": time.time() - t_batch, "cpu_core_s": "unknown",
                 "n_units": len(manifests), "notes": "48 TST pools"})
    pd.DataFrame(cost).to_csv(os.path.join(A.METRIC_DIR, "tst_cost_breakdown.csv"),
                              index=False, encoding="utf-8")
    A.save_json(os.path.join(A.METRIC_DIR, "tst_pool_run_meta.json"), {
        "created": A.utc_now(), "spec_order": SPEC_ORDER, "seeds": SEEDS,
        "methods": METHODS, "n_pools": len(manifests),
        "n_attempts_per_pool": A.N_ATTEMPTS,
        "n_random_per_pool": A.N_RANDOM, "n_shortlist_per_pool": A.N_SHORT,
        "batch_wall_s": float(time.time() - t_batch),
        "old_outputs_unchanged": bool(unchanged),
    })
    print("[pool] batch %.1fs; %d pools" % (time.time() - t_batch, len(manifests)))


if __name__ == "__main__":
    main()
