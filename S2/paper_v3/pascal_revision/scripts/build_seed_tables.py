# -*- coding: utf-8 -*-
"""Per-sampling-seed tables from the reaggregated four-way experiment."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2\paper\paper_v3")
RUNS = BASE / "research_generators_specification_study/03_existing_candidate_analysis/four_way_runs_reaggregated.csv"
SPECS = BASE / "research_generators_specification_study/02_specification_difficulty/frozen_specifications.csv"
OUT = BASE / "pascal_revision"

def wilson(k, n, z=1.959963984540054):
    if n == 0: return (np.nan, np.nan)
    p = k / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n)) / den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return centre-half, centre+half

runs = pd.read_csv(RUNS)
specs = pd.read_csv(SPECS)
specs = specs[specs.spec_family == "frozen_main_six"][["specification_id","constraints_json","p_ref","support_class"]]
specs = specs.rename(columns={"specification_id":"spec","constraints_json":"constraints_user","support_class":"support_class"})
runs = runs.merge(specs, on="spec", how="left")
runs["delivery_rate_pct"] = 100*runs["delivered_n_json"]/runs["ross_submitted_n"]
lo, hi = zip(*[wilson(int(k), int(n)) for k,n in zip(runs.delivered_n_json, runs.ross_submitted_n)])
runs["wilson_low_pct"] = 100*np.array(lo); runs["wilson_high_pct"] = 100*np.array(hi)
runs["level"] = "run_spec_seed"
runs["seed_role"] = "sampling seed on the same trained checkpoint"
runs["denominator"] = "n_jobs"
runs["notes"] = "ROSS-verified; submissions without valid result remain in denominator"

cols = ["level","method","method_label","spec","seed","seed_role","pool_n","shortlist_n",
        "ross_submitted_n","ross_solved_n","ross_timeout_n","ross_error_n","delivered_n_json",
        "delivery_rate_pct","wilson_low_pct","wilson_high_pct","pool_proxy_in_band",
        "shortlist_proxy_in_band","p_ref","support_class","constraints_user","denominator","notes",
        "pool_path","shortlist_path","verify_csv"]
run_tbl = runs[cols].copy()

agg_rows=[]
for (method,label,seed),g in runs.groupby(["method","method_label","seed"],sort=True):
    k=int(g.delivered_n_json.sum()); n=int(g.ross_submitted_n.sum())
    l,h=wilson(k,n)
    agg_rows.append({"level":"method_seed","method":method,"method_label":label,"spec":"ALL6","seed":int(seed),
        "seed_role":"sampling seed on the same trained checkpoint","pool_n":int(g.pool_n.sum()),
        "shortlist_n":int(g.shortlist_n.sum()),"ross_submitted_n":n,"ross_solved_n":int(g.ross_solved_n.sum()),
        "ross_timeout_n":int(g.ross_timeout_n.sum()),"ross_error_n":int(g.ross_error_n.sum()),
        "delivered_n_json":k,"delivery_rate_pct":100*k/n,"wilson_low_pct":100*l,"wilson_high_pct":100*h,
        "pool_proxy_in_band":np.nan,"shortlist_proxy_in_band":np.nan,"p_ref":np.nan,"support_class":"mixed",
        "constraints_user":"six frozen single-order interval specifications","denominator":"n_jobs",
        "notes":"pooled across six specifications; not six independent replicates",
        "n_specs":int(g.spec.nunique()),"macro_spec_rate_pct":g.delivery_rate_pct.mean(),
        "min_spec_rate_pct":g.delivery_rate_pct.min(),"max_spec_rate_pct":g.delivery_rate_pct.max()})
for (method,label),g in runs.groupby(["method","method_label"],sort=True):
    # 18 runs: three sampling seeds x six specs; pooled only descriptively
    k=int(g.delivered_n_json.sum()); n=int(g.ross_submitted_n.sum()); l,h=wilson(k,n)
    agg_rows.append({"level":"method_all_seeds","method":method,"method_label":label,"spec":"ALL6","seed":"42|7|13",
        "seed_role":"three sampling seeds on the same trained checkpoint","pool_n":int(g.pool_n.sum()),
        "shortlist_n":int(g.shortlist_n.sum()),"ross_submitted_n":n,"ross_solved_n":int(g.ross_solved_n.sum()),
        "ross_timeout_n":int(g.ross_timeout_n.sum()),"ross_error_n":int(g.ross_error_n.sum()),
        "delivered_n_json":k,"delivery_rate_pct":100*k/n,"wilson_low_pct":100*l,"wilson_high_pct":100*h,
        "pool_proxy_in_band":np.nan,"shortlist_proxy_in_band":np.nan,"p_ref":np.nan,"support_class":"mixed",
        "constraints_user":"six frozen single-order interval specifications","denominator":"n_jobs",
        "notes":"pooled 3 sampling seeds x 6 specifications; 3600 submissions, not 3600 independent replicates",
        "n_specs":int(g.spec.nunique()),"macro_spec_rate_pct":g.delivery_rate_pct.mean(),
        "min_spec_rate_pct":g.delivery_rate_pct.min(),"max_spec_rate_pct":g.delivery_rate_pct.max()})
agg=pd.DataFrame(agg_rows)
allcols=sorted(set(run_tbl.columns)|set(agg.columns))
out=pd.concat([run_tbl.reindex(columns=allcols),agg.reindex(columns=allcols)],ignore_index=True)
out.to_csv(OUT/"results_by_sampling_seed.csv",index=False,encoding="utf-8")
print(out[out.level!="run_spec_seed"].to_string(index=False))
