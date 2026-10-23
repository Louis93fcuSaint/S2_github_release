# -*- coding: utf-8 -*-
"""Run ROSS acceptance for the pilot slots, with a protocol-keyed cache.

Logical slots: 100 random-pool attempts + 100 shortlist designs per pool.
No slot is dropped from the denominator when it is invalid, missing or fails.
"""
from __future__ import annotations

import concurrent.futures as futures
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eaai_common as A
import v2_ross_verify as V2R

WORKERS = 16
METHOD_ORDER = ["ddpm", "cvae", "retrieval"]
SPEC_ORDER = ["CAL02", "CAL05"]


def pool_id(method, slot, seed=A.SEED):
    return "%s_%s_s%d" % (slot, method, seed)


def load_slots():
    slots = []
    for slot in SPEC_ORDER:
        for method in METHOD_ORDER:
            pid = pool_id(method, slot)
            for list_type, fname in (("random", "random_sample.csv"),
                                     ("shortlist", "shortlist.csv")):
                path = os.path.join(A.SHORT_DIR, pid, fname)
                frame = pd.read_csv(path)
                frame = frame.sort_values("attempt_id" if list_type == "random"
                                          else "selection_rank").reset_index(drop=True)
                for i, row in frame.iterrows():
                    slots.append({
                        "pool_id": pid, "method": method, "spec": slot,
                        "list_type": list_type,
                        "logical_index": int(i),
                        "logical_slot_id": "%s__%s__%03d" % (pid, list_type, i),
                        "candidate_id": str(row["candidate_id"]),
                        "attempt_id": int(row["attempt_id"]),
                        "design_key": str(row["design_key"]),
                        "material": slot and A.load_calibration_specs()[slot]["material"],
                        "nd": int(row["n_disks"]), "nb": int(row["n_bearings"]),
                        "valid_generation": bool(row["valid"]),
                        "proxy_in_band": bool(row["in_band_proxy"]),
                        "proxy_violation": float(row["violation"]),
                        "x": row[A.S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64),
                    })
    return slots


def spec_pass(forward, spec):
    if forward is None:
        return False, "no_forward"
    for order in spec["orders"]:
        if len(forward) < int(order):
            return False, "missing_order_%d" % int(order)
        lo, hi = spec["bands"][int(order)]
        value = float(forward[int(order) - 1])
        if not np.isfinite(value) or value < lo or value > hi:
            return False, "out_of_band_order_%d" % int(order)
    return True, "delivered"


def qa_duplicate_check(specs, runner):
    """Solve a few designs twice, outside the 1200 logical-slot accounting."""
    rows = []
    for slot in ("CAL02", "CAL05"):
        pid = pool_id("ddpm", slot)
        frame = pd.read_csv(os.path.join(A.SHORT_DIR, pid, "shortlist.csv")).head(2)
        for _, row in frame.iterrows():
            x = row[A.S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
            material = specs[slot]["material"]
            nd, nb = int(row["n_disks"]), int(row["n_bearings"])
            outs = []
            times = []
            for _ in range(2):
                t0 = time.time()
                res = runner((x, material, nd, nb))
                times.append(time.time() - t0)
                outs.append(res)
            f1 = outs[0].get("forward") or []
            f2 = outs[1].get("forward") or []
            m = min(len(f1), len(f2), 6)
            diff = float(np.max(np.abs(np.array(f1[:m]) - np.array(f2[:m])))) if m else None
            rows.append({
                "spec": slot, "design_key": A.design_key(x),
                "material": material, "nd": nd, "nb": nb,
                "ok_1": bool(outs[0].get("ok")), "ok_2": bool(outs[1].get("ok")),
                "max_abs_diff_rpm": diff, "elapsed_1_s": times[0], "elapsed_2_s": times[1],
                "forward_1": json.dumps(f1[:6]), "forward_2": json.dumps(f2[:6]),
            })
            print("[qa] %s diff=%s %.1fs/%.1fs" % (slot, diff, times[0], times[1]),
                  flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(A.ROSS_DIR, "qa_duplicate_check.csv"), index=False,
                 encoding="utf-8")
    return frame


def main():
    t_all = time.time()
    before = A.old_output_snapshot()
    specs = A.load_calibration_specs()
    slots = load_slots()
    print("[ross] logical slots", len(slots), flush=True)

    lock = threading.Lock()
    stats = {"timeout": 0, "error": 0}
    runner = V2R.make_runner(A.ROSS_TIMEOUT, A.ROSS_N_BASE, stats, lock)
    qa = qa_duplicate_check(specs, runner)

    # Build unique-key jobs.  Invalid generations are not sent to ROSS.
    key_info = {}
    key_slots = {}
    for idx, s in enumerate(slots):
        if not s["valid_generation"]:
            continue
        key = A.ross_cache_key(s["x"], s["material"], s["nd"], s["nb"])
        s["ross_key"] = key
        key_info.setdefault(key, (s["x"], s["material"], s["nd"], s["nb"]))
        key_slots.setdefault(key, []).append(idx)
    print("[ross] unique designs to solve:", len(key_info), flush=True)

    def solve_one(key):
        cached = A.load_ross_cache(key)
        if cached is not None and cached.get("status") == "ok":
            return key, cached, True, 0.0
        x, material, nd, nb = key_info[key]
        t0 = time.time()
        res = runner((x, material, nd, nb))
        elapsed = time.time() - t0
        record = {
            "status": "ok" if (res.get("ok") and res.get("forward")) else
                      ("timeout" if res.get("timeout") else "error"),
            "forward": [float(v) for v in (res.get("forward") or [])],
            "error": res.get("error", ""), "elapsed_s": float(elapsed),
            "ross_version": A.ROSS_VERSION_EXPECTED, "n_base": A.ROSS_N_BASE,
            "num_modes": A.ROSS_NUM_MODES, "protocol": A.ROSS_PROTOCOL,
            "material": material, "nd": int(nd), "nb": int(nb),
            "design_key": A.design_key(x), "created": A.utc_now(),
        }
        if record["status"] == "ok":
            with lock:
                A.save_ross_cache(key, record)
        return key, record, False, elapsed

    results = {}
    solves = []
    t0 = time.time()
    with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for key, record, cache_hit, elapsed in pool.map(solve_one, list(key_info.keys())):
            results[key] = record
            solves.append({
                "ross_key": key, "design_key": record["design_key"],
                "material": record["material"], "nd": record["nd"], "nb": record["nb"],
                "status": record["status"], "cache_hit": cache_hit,
                "elapsed_s": elapsed, "forward_json": json.dumps(record["forward"][:6]),
                "error": record["error"], "created": record["created"],
            })
            if len(solves) % 100 == 0:
                print("[ross] solved %d/%d (%.0fs)" % (len(solves), len(key_info),
                                                       time.time() - t0), flush=True)
    ross_wall = time.time() - t0
    pd.DataFrame(solves).to_csv(os.path.join(A.ROSS_DIR, "solves.csv"), index=False,
                                encoding="utf-8")

    # Assemble logical-slot results.
    rows = []
    for s in slots:
        record = results.get(s.get("ross_key"))
        forward = record.get("forward") if record else None
        if not s["valid_generation"]:
            status = "invalid_generation"
            delivered = False
            reason = "invalid_H1-H6"
            cache_hit = False
            elapsed = 0.0
        elif record is None:
            status = "error"
            delivered = False
            reason = "no_ross_record"
            cache_hit = False
            elapsed = 0.0
        else:
            status = record["status"]
            cache_hit = (status == "ok" and s["ross_key"] in
                         {k for k, v in results.items() if v.get("status") == "ok"}
                         and any(r["ross_key"] == s["ross_key"] and r["cache_hit"]
                                 for r in solves))
            elapsed = 0.0 if cache_hit else float(record.get("elapsed_s", 0.0))
            if status == "ok":
                ok, reason = spec_pass(forward, specs[s["spec"]])
                delivered = bool(ok)
            else:
                delivered = False
                reason = "ross_%s" % status
        row = {
            "logical_slot_id": s["logical_slot_id"], "pool_id": s["pool_id"],
            "method": s["method"], "spec": s["spec"], "list_type": s["list_type"],
            "logical_index": s["logical_index"], "candidate_id": s["candidate_id"],
            "attempt_id": s["attempt_id"], "design_key": s["design_key"],
            "material": s["material"], "nd": s["nd"], "nb": s["nb"],
            "valid_generation": s["valid_generation"],
            "proxy_in_band": s["proxy_in_band"],
            "proxy_violation": s["proxy_violation"],
            "status": status, "cache_hit": bool(cache_hit),
            "elapsed_s": float(elapsed), "delivered": bool(delivered),
            "undelivered_reason": reason,
        }
        for k in range(6):
            row["cs%d_ross_rpm" % (k + 1)] = (float(forward[k])
                                              if forward is not None and len(forward) > k
                                              else np.nan)
        rows.append(row)
    all_slots = pd.DataFrame(rows)
    all_slots.to_csv(os.path.join(A.ROSS_DIR, "all_slots.csv"), index=False,
                     encoding="utf-8")
    for pid in all_slots["pool_id"].unique():
        sub = all_slots[all_slots["pool_id"] == pid].copy()
        out_dir = os.path.join(A.ROSS_DIR, pid)
        os.makedirs(out_dir, exist_ok=True)
        sub.to_csv(os.path.join(out_dir, "slots.csv"), index=False, encoding="utf-8")

    # Pool-level summary.
    summary = []
    for pid, group in all_slots.groupby("pool_id", sort=False):
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
    # Shortlists are always 100; random samples are always 100.  Add explicit gap accounting.
    pd.DataFrame(summary).to_csv(os.path.join(A.ROSS_DIR, "ross_pool_summary.csv"),
                                 index=False, encoding="utf-8")

    # Cost breakdown update.
    cost_path = os.path.join(A.METRIC_DIR, "pilot_cost_breakdown.csv")
    cost = pd.read_csv(cost_path)
    cost = cost[~cost["stage"].isin(["ross", "batch_ross"])]
    add = []
    for pid, group in all_slots.groupby("pool_id", sort=False):
        solved = [r for r in solves if r["ross_key"] in
                  set(all_slots[all_slots["pool_id"] == pid]["design_key"])]
        add.append({"pool_id": pid, "stage": "ross",
                    "wall_s": float(group["elapsed_s"].sum()), "cpu_core_s": "unknown",
                    "n_units": int(len(group)),
                    "notes": "sum of per-slot elapsed; cache hits are zero"})
    add.append({"pool_id": "ALL", "stage": "batch_ross",
                "wall_s": float(ross_wall), "cpu_core_s": "unknown",
                "n_units": int(len(all_slots)),
                "notes": "16 workers; includes cache hits as zero; excludes QA repeats"})
    add.append({"pool_id": "ALL", "stage": "batch_total",
                "wall_s": float(time.time() - t_all), "cpu_core_s": "unknown",
                "n_units": int(len(all_slots)), "notes": "generation + shortlist + ROSS"})
    cost = pd.concat([cost, pd.DataFrame(add)], ignore_index=True)
    cost.to_csv(cost_path, index=False, encoding="utf-8")
    cost.to_csv(os.path.join(A.STUDY, "pilot_cost_breakdown.csv"), index=False,
                encoding="utf-8")

    after = A.old_output_snapshot()
    unchanged = before["sha256"] == after["sha256"]
    A.save_json(os.path.join(A.ROSS_DIR, "run_meta.json"), {
        "logical_slots": int(len(all_slots)), "unique_designs": int(len(key_info)),
        "physical_solves": int(len(solves)), "cache_hits": int(sum(r["cache_hit"] for r in solves)),
        "ross_wall_s": float(ross_wall), "total_wall_s": float(time.time() - t_all),
        "workers": WORKERS, "timeout_s": A.ROSS_TIMEOUT,
        "stats": stats, "old_outputs_unchanged": unchanged,
        "qa_duplicate_check": qa.to_dict("records"),
    })
    print("[ross] logical=%d unique=%d physical=%d cache=%d wall=%.0fs"
          % (len(all_slots), len(key_info), len(solves),
             sum(r["cache_hit"] for r in solves), ross_wall), flush=True)
    if not unchanged:
        raise SystemExit("historical output directories changed during ROSS run")


if __name__ == "__main__":
    main()
