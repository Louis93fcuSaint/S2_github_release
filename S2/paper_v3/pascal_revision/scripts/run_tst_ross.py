# -*- coding: utf-8 -*-
"""ROSS acceptance for the frozen TST protocol (100 random + 200 shortlist per pool)."""
from __future__ import annotations
import concurrent.futures as futures
import json, os, sys, threading, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
EA = PROJECT.parent / "research_generators_specification_study" / "eaai_comparative_study"
sys.path.insert(0, str(EA / "audit_scripts"))
import eaai_common as A
import v2_ross_verify as V2R

A.STUDY = str(PROJECT / "tst_runs")
A.SPEC_CSV = str(EA / "spec_candidates" / "test_specs.csv")
A.POOL_DIR = str(PROJECT / "tst_runs" / "01_pools")
A.SHORT_DIR = str(PROJECT / "tst_runs" / "02_shortlists")
A.ROSS_DIR = str(PROJECT / "tst_runs" / "03_ross")
A.METRIC_DIR = str(PROJECT / "tst_runs" / "04_metrics")
A.LOG_DIR = str(PROJECT / "tst_runs" / "logs")
A.N_RANDOM = 100
A.N_SHORT = 200
WORKERS = int(os.environ.get("TST_ROSS_WORKERS", "16"))
SPEC_FILTER = {s.strip() for s in os.environ.get("TST_SPEC_FILTER", "").split(",") if s.strip()}
SEED_FILTER = {int(s.strip()) for s in os.environ.get("TST_SEED_FILTER", "").split(",") if s.strip()}
METHODS = ["ddpm", "cvae", "retrieval"]


def spec_pass(forward, spec):
    if forward is None:
        return False, "no_forward"
    for order in spec["orders"]:
        if len(forward) < int(order):
            return False, "missing_order_%d" % int(order)
        lo, hi = spec["bands"][int(order)]
        v = float(forward[int(order) - 1])
        if not np.isfinite(v) or v < lo or v > hi:
            return False, "out_of_band_order_%d" % int(order)
    return True, "delivered"


def load_slots():
    specs = A.load_calibration_specs()
    slots = []
    for pool_dir in sorted(Path(A.SHORT_DIR).iterdir()):
        if not pool_dir.is_dir():
            continue
        meta = json.loads((pool_dir / "manifest.json").read_text(encoding="utf-8"))
        pid = meta["pool_id"]; slot = meta["spec"]; method = meta["method"]
        seed = int(meta["seed"])
        if SPEC_FILTER and slot not in SPEC_FILTER:
            continue
        if SEED_FILTER and seed not in SEED_FILTER:
            continue
        spec = specs[slot]
        for list_type, fname, expected in (("random", "random_sample.csv", A.N_RANDOM),
                                           ("shortlist", "shortlist.csv", A.N_SHORT)):
            path = pool_dir / fname
            if not path.exists():
                raise SystemExit("missing %s" % path)
            frame = pd.read_csv(path)
            frame = frame.sort_values("attempt_id" if list_type == "random" else "selection_rank").reset_index(drop=True)
            for i, row in frame.iterrows():
                slots.append({
                    "pool_id": pid, "method": method, "spec": slot, "seed": seed,
                    "list_type": list_type, "logical_index": int(i),
                    "logical_slot_id": "%s__%s__%03d" % (pid, list_type, i),
                    "candidate_id": str(row["candidate_id"]),
                    "attempt_id": int(row["attempt_id"]),
                    "design_key": str(row["design_key"]),
                    "material": spec["material"], "nd": int(row["n_disks"]), "nb": int(row["n_bearings"]),
                    "valid_generation": bool(row["valid"]),
                    "proxy_in_band": bool(row["in_band_proxy"]),
                    "proxy_violation": float(row["violation"]),
                    "x": row[A.S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64),
                    "expected_denominator": expected,
                })
    return slots, specs


def qa_duplicate_check(specs, runner):
    rows = []
    for slot in ("TST01", "TST02"):
        pid = "%s_ddpm_s42" % slot
        frame = pd.read_csv(Path(A.SHORT_DIR) / pid / "shortlist.csv").head(2)
        for _, row in frame.iterrows():
            x = row[A.S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
            material = specs[slot]["material"]
            nd, nb = int(row["n_disks"]), int(row["n_bearings"])
            outs, times = [], []
            for _ in range(2):
                t0 = time.time(); res = runner((x, material, nd, nb)); times.append(time.time()-t0); outs.append(res)
            f1 = outs[0].get("forward") or []; f2 = outs[1].get("forward") or []
            m = min(len(f1), len(f2), 6)
            diff = float(np.max(np.abs(np.array(f1[:m])-np.array(f2[:m])))) if m else None
            rows.append({"spec": slot, "design_key": A.design_key(x), "material": material,
                         "nd": nd, "nb": nb, "ok_1": bool(outs[0].get("ok")), "ok_2": bool(outs[1].get("ok")),
                         "max_abs_diff_rpm": diff, "elapsed_1_s": times[0], "elapsed_2_s": times[1],
                         "forward_1": json.dumps(f1[:6]), "forward_2": json.dumps(f2[:6])})
            print("[qa] %s diff=%s %.1fs/%.1fs" % (slot, diff, times[0], times[1]), flush=True)
    pd.DataFrame(rows).to_csv(Path(A.ROSS_DIR) / "tst_qa_duplicate_check.csv", index=False, encoding="utf-8")
    return rows


def main():
    t_all = time.time()
    Path(A.ROSS_DIR).mkdir(parents=True, exist_ok=True)
    before = A.old_output_snapshot()
    slots, specs = load_slots()
    if not slots:
        raise SystemExit("no TST slots matched TST_SPEC_FILTER/TST_SEED_FILTER")
    print("[ross] logical slots", len(slots), flush=True)
    lock = threading.Lock(); stats = {"timeout": 0, "error": 0}
    runner = V2R.make_runner(A.ROSS_TIMEOUT, A.ROSS_N_BASE, stats, lock)
    qa = qa_duplicate_check(specs, runner)

    key_info, key_slots = {}, {}
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
        t0 = time.time(); res = runner((x, material, nd, nb)); elapsed = time.time()-t0
        record = {"status": "ok" if (res.get("ok") and res.get("forward")) else ("timeout" if res.get("timeout") else "error"),
                  "forward": [float(v) for v in (res.get("forward") or [])], "error": res.get("error", ""),
                  "elapsed_s": float(elapsed), "ross_version": A.ROSS_VERSION_EXPECTED,
                  "n_base": A.ROSS_N_BASE, "num_modes": A.ROSS_NUM_MODES, "protocol": A.ROSS_PROTOCOL,
                  "material": material, "nd": int(nd), "nb": int(nb),
                  "design_key": A.design_key(x), "created": A.utc_now()}
        if record["status"] == "ok":
            with lock: A.save_ross_cache(key, record)
        return key, record, False, elapsed

    results, solves = {}, []
    t0 = time.time()
    with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for key, record, cache_hit, elapsed in pool.map(solve_one, list(key_info.keys())):
            results[key] = record
            solves.append({"ross_key": key, "design_key": record["design_key"], "material": record["material"],
                           "nd": record["nd"], "nb": record["nb"], "status": record["status"],
                           "cache_hit": cache_hit, "elapsed_s": elapsed,
                           "forward_json": json.dumps(record["forward"][:6]), "error": record["error"],
                           "created": record["created"]})
            if len(solves) % 200 == 0:
                print("[ross] solved %d/%d (%.0fs)" % (len(solves), len(key_info), time.time()-t0), flush=True)
    ross_wall = time.time()-t0
    pd.DataFrame(solves).to_csv(Path(A.ROSS_DIR) / "tst_ross_solves.csv", index=False, encoding="utf-8")

    rows = []
    cache_hit_keys = {r["ross_key"] for r in solves if r["cache_hit"]}
    for s in slots:
        record = results.get(s.get("ross_key"))
        forward = record.get("forward") if record else None
        if not s["valid_generation"]:
            status="invalid_generation"; delivered=False; reason="invalid_H1-H6"; cache_hit=False; elapsed=0.0
        elif record is None:
            status="error"; delivered=False; reason="no_ross_record"; cache_hit=False; elapsed=0.0
        else:
            status=record["status"]; cache_hit=s.get("ross_key") in cache_hit_keys
            elapsed=0.0 if cache_hit else float(record.get("elapsed_s",0.0))
            if status=="ok": delivered, reason = spec_pass(forward, specs[s["spec"]])
            else: delivered=False; reason="ross_%s"%status
        row={**{k:v for k,v in s.items() if k!="x"}, "ross_key":s.get("ross_key",""),
             "status":status,"cache_hit":bool(cache_hit),"elapsed_s":float(elapsed),
             "delivered":bool(delivered),"undelivered_reason":reason}
        for k in range(6):
            row["cs%d_ross_rpm"%(k+1)] = float(forward[k]) if forward is not None and len(forward)>k else np.nan
        rows.append(row)
    all_slots=pd.DataFrame(rows)
    all_slots.to_csv(Path(A.ROSS_DIR)/"tst_ross_all_slots.csv",index=False,encoding="utf-8")
    for pid, sub in all_slots.groupby("pool_id"):
        d=Path(A.ROSS_DIR)/pid; d.mkdir(parents=True,exist_ok=True); sub.to_csv(d/"slots.csv",index=False,encoding="utf-8")

    summary=[]
    for pid,g in all_slots.groupby("pool_id",sort=False):
        for lt in ("random","shortlist"):
            h=g[g.list_type==lt]; n=int(len(h)); den=int(h["expected_denominator"].iloc[0]) if n else 0
            summary.append({"pool_id":pid,"method":h["method"].iloc[0],"spec":h["spec"].iloc[0],"list_type":lt,
                "n_logical_slots":n,"denominator":den,"n_invalid_generation":int((~h.valid_generation).sum()),
                "n_solved_ok":int((h.status=="ok").sum()),"n_error":int((h.status=="error").sum()),
                "n_timeout":int((h.status=="timeout").sum()),"n_cache_hit":int(h.cache_hit.sum()),
                "n_delivered":int(h.delivered.sum()),"delivery_rate":float(h.delivered.sum()/den) if den else np.nan,
                "success_solve_rate":float((h.status=="ok").sum()/den) if den else np.nan,
                "median_elapsed_s":float(h.elapsed_s.median()) if n else np.nan,
                "sum_elapsed_s":float(h.elapsed_s.sum())})
    pd.DataFrame(summary).to_csv(Path(A.ROSS_DIR)/"tst_ross_pool_summary.csv",index=False,encoding="utf-8")

    cost_path=Path(A.METRIC_DIR)/"tst_cost_breakdown.csv"
    cost=pd.read_csv(cost_path) if cost_path.exists() else pd.DataFrame(columns=["pool_id","stage","wall_s","cpu_core_s","n_units","notes"])
    cost=cost[~cost.stage.isin(["ross","batch_ross"])]
    add=[]
    for pid,g in all_slots.groupby("pool_id",sort=False):
        add.append({"pool_id":pid,"stage":"ross","wall_s":float(g.elapsed_s.sum()),"cpu_core_s":"unknown",
                    "n_units":int(len(g)),"notes":"sum of per-slot elapsed; cache hits zero"})
    add.append({"pool_id":"ALL","stage":"batch_ross","wall_s":float(ross_wall),"cpu_core_s":"unknown",
                "n_units":int(len(all_slots)),"notes":"%d workers; excludes QA repeats"%WORKERS})
    pd.concat([cost,pd.DataFrame(add)],ignore_index=True).to_csv(cost_path,index=False,encoding="utf-8")

    after=A.old_output_snapshot(); unchanged=before["sha256"]==after["sha256"]
    meta={"logical_slots":int(len(all_slots)),"unique_designs":int(len(key_info)),
          "physical_solves":int(len(solves)),"cache_hits":int(sum(r["cache_hit"] for r in solves)),
          "ross_wall_s":float(ross_wall),"total_wall_s":float(time.time()-t_all),"workers":WORKERS,
          "timeout_s":A.ROSS_TIMEOUT,"stats":stats,"old_outputs_unchanged":bool(unchanged),
          "spec_filter":sorted(SPEC_FILTER),"seed_filter":sorted(SEED_FILTER),
          "qa_duplicate_check":qa,"protocol":"TST_v1_frozen","created":A.utc_now()}
    (Path(A.ROSS_DIR)/"tst_ross_run_meta.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    print("[ross] logical=%d unique=%d physical=%d cache=%d wall=%.0fs"%
          (len(all_slots),len(key_info),len(solves),sum(r["cache_hit"] for r in solves),ross_wall),flush=True)
    if not unchanged: raise SystemExit("historical output directories changed during TST ROSS run")


if __name__=="__main__":
    main()
