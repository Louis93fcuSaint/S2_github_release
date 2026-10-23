# -*- coding: utf-8 -*-
"""Correct logical-slot cache accounting after the ROSS run.

The solver results are untouched.  Only the per-slot bookkeeping is rebuilt:
the first logical occurrence of a design carries the physical solve; later
occurrences reuse the same numerical result and are marked as cache hits.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eaai_common as A

ORDER = {"random": 0, "shortlist": 1}


def main():
    path = os.path.join(A.ROSS_DIR, "all_slots.csv")
    frame = pd.read_csv(path)
    frame["_order"] = frame["list_type"].map(ORDER).fillna(9)
    frame = frame.sort_values(["pool_id", "_order", "logical_index"]).reset_index(drop=True)
    seen = set()
    cache_hit = []
    elapsed = []
    for _, row in frame.iterrows():
        key = str(row["design_key"])
        if key in seen:
            cache_hit.append(True)
            elapsed.append(0.0)
        else:
            seen.add(key)
            cache_hit.append(False)
            elapsed.append(float(row["elapsed_s"]))
    frame["cache_hit"] = cache_hit
    frame["elapsed_s"] = elapsed
    frame = frame.drop(columns=["_order"])
    frame = frame.sort_values(["pool_id", "list_type", "logical_index"]).reset_index(drop=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    for pid, group in frame.groupby("pool_id", sort=False):
        out_dir = os.path.join(A.ROSS_DIR, pid)
        os.makedirs(out_dir, exist_ok=True)
        group.to_csv(os.path.join(out_dir, "slots.csv"), index=False, encoding="utf-8")

    summary = []
    for pid, group in frame.groupby("pool_id", sort=False):
        for list_type in ("random", "shortlist"):
            g = group[group["list_type"] == list_type]
            summary.append({
                "pool_id": pid, "method": g["method"].iloc[0], "spec": g["spec"].iloc[0],
                "list_type": list_type, "n_logical_slots": int(len(g)),
                "n_invalid_generation": int((~g["valid_generation"]).sum()),
                "n_solved_ok": int((g["status"] == "ok").sum()),
                "n_error": int((g["status"] == "error").sum()),
                "n_timeout": int((g["status"] == "timeout").sum()),
                "n_cache_hit": int(g["cache_hit"].sum()),
                "n_delivered": int(g["delivered"].sum()),
                "delivery_rate": float(g["delivered"].mean()) if len(g) else None,
                "success_solve_rate": float((g["status"] == "ok").mean()) if len(g) else None,
                "median_elapsed_s": float(g["elapsed_s"].median()),
                "sum_elapsed_s": float(g["elapsed_s"].sum()),
            })
    pd.DataFrame(summary).to_csv(os.path.join(A.ROSS_DIR, "ross_pool_summary.csv"),
                                 index=False, encoding="utf-8")

    meta_path = os.path.join(A.ROSS_DIR, "run_meta.json")
    meta = A.load_json(meta_path)
    meta["logical_cache_reuses"] = int(frame["cache_hit"].sum())
    meta["physical_solves"] = int(frame["design_key"].nunique())
    meta["logical_slots"] = int(len(frame))
    A.save_json(meta_path, meta)

    # Cost breakdown: replace the loosely labelled per-pool ROSS rows.
    cost_path = os.path.join(A.METRIC_DIR, "pilot_cost_breakdown.csv")
    cost = pd.read_csv(cost_path)
    cost = cost[~cost["stage"].isin(["ross", "batch_ross", "batch_total"])]
    add = []
    for pid, group in frame.groupby("pool_id", sort=False):
        unique = group.drop_duplicates("design_key")
        add.append({"pool_id": pid, "stage": "ross_solver_elapsed_sum",
                    "wall_s": float(unique["elapsed_s"].sum()), "cpu_core_s": "unknown",
                    "n_units": int(len(unique)),
                    "notes": "sum of per-solve subprocess elapsed; not CPU core-seconds"})
    add.append({"pool_id": "ALL", "stage": "batch_ross_wall",
                "wall_s": float(meta["ross_wall_s"]), "cpu_core_s": "unknown",
                "n_units": int(len(frame)),
                "notes": "16 workers; actual wall clock; excludes QA repeat solves"})
    cost = pd.concat([cost, pd.DataFrame(add)], ignore_index=True)
    cost.to_csv(cost_path, index=False, encoding="utf-8")
    cost.to_csv(os.path.join(A.STUDY, "pilot_cost_breakdown.csv"), index=False,
                encoding="utf-8")

    print("[fix] logical=%d unique=%d reuses=%d" % (len(frame),
                                                    frame["design_key"].nunique(),
                                                    int(frame["cache_hit"].sum())))
    print("[fix] delivery by pool/list:")
    print(pd.DataFrame(summary)[["pool_id", "list_type", "n_delivered", "delivery_rate",
                                 "n_cache_hit"]].to_string(index=False))


if __name__ == "__main__":
    main()
