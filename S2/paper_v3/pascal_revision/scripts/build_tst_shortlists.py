# -*- coding: utf-8 -*-
"""Build 100-attempt random samples and 200-design proxy shortlists for TST pools."""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
EA = os.path.abspath(os.path.join(PROJECT, "..", "research_generators_specification_study", "eaai_comparative_study"))
sys.path.insert(0, os.path.join(EA, "audit_scripts"))

import eaai_common as A  # noqa: E402

RUN = os.path.join(PROJECT, "tst_runs")
A.STUDY = RUN
A.SPEC_CSV = os.path.join(EA, "spec_candidates", "test_specs.csv")
A.POOL_DIR = os.path.join(RUN, "01_pools")
A.SHORT_DIR = os.path.join(RUN, "02_shortlists")
A.ROSS_DIR = os.path.join(RUN, "03_ross")
A.METRIC_DIR = os.path.join(RUN, "04_metrics")
A.N_RANDOM = 100
A.N_SHORT = 200
os.makedirs(A.SHORT_DIR, exist_ok=True)

SEEDS = [42, 7]
METHODS = ["ddpm", "cvae", "retrieval"]
SPEC_ORDER = ["TST01", "TST02", "TST03", "TST04", "TST05", "TST06", "TST07", "TST09"]


def pool_id(method, slot, seed):
    return "%s_%s_s%d" % (slot, method, seed)


def selection_seed(slot, method, seed):
    return int(seed) + 1000 * (SPEC_ORDER.index(slot) + 1) + 17 * (METHODS.index(method) + 1)


def frame_from_rows(x, material, families, pred, keys, attempt_ids, pid):
    frame = A.O.make_frame(np.asarray(x, dtype=np.float64), material, families)
    frame.insert(0, "attempt_id", np.asarray(attempt_ids, dtype=np.int64))
    frame.insert(0, "candidate_id", ["%s_a%05d" % (pid, int(a)) for a in attempt_ids])
    frame["design_key"] = keys
    frame["n_disks"] = [int(f[0]) for f in families]
    frame["n_bearings"] = [int(f[1]) for f in families]
    for i in range(6):
        frame["cs%d_pred" % (i + 1)] = np.asarray(pred)[:, i]
    return frame


def main():
    t0 = time.time()
    specs = A.load_calibration_specs()
    rows = []
    for slot in SPEC_ORDER:
        spec = specs[slot]
        for seed in SEEDS:
            for method in METHODS:
                pid = pool_id(method, slot, seed)
                out_dir = os.path.join(A.SHORT_DIR, pid)
                os.makedirs(out_dir, exist_ok=True)
                blob = np.load(os.path.join(A.POOL_DIR, pid, "raw.npz"), allow_pickle=False)
                x, pred = blob["x"], blob["pred"]
                nd, nb = blob["nd"], blob["nb"]
                keys = blob["design_key"].astype(str)
                valid = blob["valid"].astype(bool)
                attempt = blob["attempt_id"].astype(np.int64)
                families = [(int(a), int(b)) for a, b in zip(nd, nb)]
                material = str(blob["material"][0]) if "material" in blob else spec["material"]

                rng = np.random.default_rng(seed + 200003 + 13 * (SPEC_ORDER.index(slot) + 1)
                                            + 29 * (METHODS.index(method) + 1))
                random_ids = np.sort(rng.choice(len(attempt), size=A.N_RANDOM, replace=False))
                random_frame = frame_from_rows(
                    x[random_ids], material, [families[i] for i in random_ids],
                    pred[random_ids], keys[random_ids], attempt[random_ids], pid)
                random_frame.insert(0, "list_type", "random_pool")
                random_frame["selection_source"] = "random_pool"
                random_frame["in_band_proxy"] = blob["in_band"][random_ids]
                random_frame["violation"] = blob["violation"][random_ids]
                random_frame["valid"] = valid[random_ids]
                random_frame["inclusion_probability"] = A.N_RANDOM / float(len(attempt))
                random_frame.to_csv(os.path.join(out_dir, "random_sample.csv"), index=False,
                                    encoding="utf-8", float_format="%.17g")

                unique_idx, _ = A.exact_unique_indices(keys[valid])
                raw_valid_rows = np.where(valid)[0]
                rep_rows = raw_valid_rows[unique_idx]
                cand_ids = np.array(["%s_a%05d" % (pid, int(a)) for a in attempt[rep_rows]],
                                    dtype=object)
                chosen, source, in_band, score = A.select_shortlist(
                    np.arange(len(rep_rows)), pred[rep_rows], spec, cand_ids,
                    selection_seed(slot, method, seed), A.N_SHORT)
                chosen_rows = rep_rows[chosen]
                short_frame = frame_from_rows(
                    x[chosen_rows], material, [families[i] for i in chosen_rows],
                    pred[chosen_rows], keys[chosen_rows], attempt[chosen_rows], pid)
                short_frame.insert(0, "list_type", "shortlist")
                short_frame.insert(1, "selection_rank", np.arange(1, len(chosen_rows) + 1))
                short_frame["selection_source"] = source
                short_frame["in_band_proxy"] = in_band[chosen]
                short_frame["violation"] = score[chosen]
                short_frame["valid"] = True
                short_frame["inclusion_probability"] = np.nan
                short_frame.to_csv(os.path.join(out_dir, "shortlist.csv"), index=False,
                                   encoding="utf-8", float_format="%.17g")

                r_ids = set(int(v) for v in attempt[random_ids])
                s_ids = set(int(v) for v in attempt[chosen_rows])
                r_keys = set(keys[random_ids].tolist())
                s_keys = set(keys[chosen_rows].tolist())
                overlap_ids = sorted(r_ids & s_ids)
                overlap_keys = sorted(r_keys & s_keys)
                pd.DataFrame({"attempt_id": overlap_ids}).to_csv(
                    os.path.join(out_dir, "overlap_by_attempt.csv"), index=False)
                pd.DataFrame({"design_key": overlap_keys}).to_csv(
                    os.path.join(out_dir, "overlap_by_design.csv"), index=False)

                n_short = int(len(short_frame))
                n_in = int((source == "in_band").sum())
                n_fb = int((source == "fallback").sum())
                n_gap = int(A.N_SHORT - n_short)
                manifest = {
                    "pool_id": pid, "spec": slot, "method": method, "seed": seed,
                    "n_random": A.N_RANDOM, "n_shortlist": n_short,
                    "n_shortlist_in_band": n_in, "n_shortlist_fallback": n_fb,
                    "n_shortlist_gap": n_gap,
                    "n_unique_valid_pool": int(len(rep_rows)),
                    "n_pool_in_band_unique_valid": int(in_band.sum()),
                    "shortlist_selection_seed": selection_seed(slot, method, seed),
                    "shortlist_rule": "uniform in-band; out-of-band fallback by (violation, candidate_id)",
                    "random_rule": "uniform over original 5000 attempt IDs, with replacement=False",
                    "random_inclusion_probability": A.N_RANDOM / 5000.0,
                    "n_random_in_band_proxy": int(random_frame["in_band_proxy"].sum()),
                    "n_random_valid": int(random_frame["valid"].sum()),
                    "overlap_attempt_ids": int(len(overlap_ids)),
                    "overlap_design_keys": int(len(overlap_keys)),
                    "shortlist_file": os.path.join(out_dir, "shortlist.csv"),
                    "random_file": os.path.join(out_dir, "random_sample.csv"),
                }
                A.save_json(os.path.join(out_dir, "manifest.json"), manifest)
                rows.append(manifest)
                print("[short] %-26s short=%d in=%d fb=%d gap=%d randomIn=%d overlap=%d/%d"
                      % (pid, n_short, n_in, n_fb, n_gap,
                         manifest["n_random_in_band_proxy"], len(overlap_ids), len(overlap_keys)),
                      flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(PROJECT, "tst_shortlist_summary.csv"), index=False,
                   encoding="utf-8")
    summary.to_csv(os.path.join(A.METRIC_DIR, "tst_shortlist_summary.csv"), index=False,
                   encoding="utf-8")
    A.save_json(os.path.join(A.METRIC_DIR, "tst_shortlist_run_meta.json"),
                {"created": A.utc_now(), "wall_s": float(time.time() - t0),
                 "n_pools": int(len(summary)), "n_shortlist": A.N_SHORT,
                 "n_random": A.N_RANDOM})
    print("[short] done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
