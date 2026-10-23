# -*- coding: utf-8 -*-
"""Analyse only the TST pools that are fully solved, using the on-disk ROSS cache.

This does not modify the frozen protocol. It is a monitoring/partial-results tool:
pools with any unsolved slot are excluded from the partial tables and listed separately.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("rtr", HERE / "run_tst_ross.py")
rtr = importlib.util.module_from_spec(spec)
import sys
sys.modules["rtr"] = rtr
spec.loader.exec_module(rtr)
A = rtr.A

OUT = ROOT / "tst_runs" / "03_ross"
slots, specs = rtr.load_slots()

rows = []
for s in slots:
    key = A.ross_cache_key(s["x"], s["material"], s["nd"], s["nb"]) if s["valid_generation"] else None
    rec = A.load_ross_cache(key) if key else None
    if rec is None:
        continue
    ok = rec.get("status") == "ok"
    delivered, reason = (False, "ross_" + rec.get("status", "?"))
    if ok:
        delivered, reason = rtr.spec_pass(rec.get("forward"), specs[s["spec"]])
    rows.append({**{k: v for k, v in s.items() if k != "x"},
                 "status": rec.get("status"), "delivered": bool(delivered),
                 "undelivered_reason": reason})
df = pd.DataFrame(rows)

summary = []
for (pid, lt), g in df.groupby(["pool_id", "list_type"]):
    den = int(g["expected_denominator"].iloc[0])
    complete = len(g) == den
    summary.append({"pool_id": pid, "spec": g["spec"].iloc[0], "method": g["method"].iloc[0],
                    "list_type": lt, "solved": len(g), "denominator": den, "complete": complete,
                    "n_delivered": int(g["delivered"].sum()),
                    "delivery_rate": int(g["delivered"].sum()) / den if den else np.nan,
                    "n_proxy_in_band": int(g["proxy_in_band"].sum())})
sm = pd.DataFrame(summary)
sm.to_csv(OUT / "tst_partial_by_pool.csv", index=False, encoding="utf-8-sig")

complete_pools = sorted(sm[sm.complete].pool_id.unique())
full = sm[sm.complete & (sm.list_type == "shortlist")]
lines = ["# TST partial results (only fully solved pools)\n",
         "Generated from the on-disk ROSS cache. Pools with any unsolved slot are excluded.\n",
         "Complete pools: %d / %d\n" % (len(complete_pools), df.pool_id.nunique())]
if len(full):
    lines.append("| Pool | Method | Shortlist delivery | In-band / 200 |")
    lines.append("|---|---|---:|---:|")
    for _, r in full.sort_values(["spec", "method"]).iterrows():
        lines.append("| %s | %s | %d/%d (%.1f%%) | %d |" % (
            r.spec, r.method, r.n_delivered, r.denominator, 100 * r.delivery_rate, r.n_proxy_in_band))
    piv = full.pivot(index="spec", columns="method", values="delivery_rate")
    lines.append("\n## Method matrix (shortlist, complete pools)\n")
    lines.append("| Spec | DDPM | CVAE | Retrieval |")
    lines.append("|---|---:|---:|---:|")
    for spec_id, row in piv.iterrows():
        lines.append("| %s | %s | %s | %s |" % (
            spec_id,
            "%.1f%%" % (100 * row.get("ddpm", np.nan)) if pd.notna(row.get("ddpm", np.nan)) else "-",
            "%.1f%%" % (100 * row.get("cvae", np.nan)) if pd.notna(row.get("cvae", np.nan)) else "-",
            "%.1f%%" % (100 * row.get("retrieval", np.nan)) if pd.notna(row.get("retrieval", np.nan)) else "-"))
else:
    lines.append("No pool is complete yet.")
(OUT / "tst_partial_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("solved slots", len(df), "of", len(slots))
print("complete pools", len(complete_pools), "of", df.pool_id.nunique())
print((OUT / "tst_partial_summary.md"))
