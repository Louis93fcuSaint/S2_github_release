# -*- coding: utf-8 -*-
"""Consolidate ROSS-valid design diversity for the Pascal meeting."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2\paper\paper_v3")
DIV = BASE / "research_generators_specification_study/05_valid_diversity"
RUNS = BASE / "research_generators_specification_study/03_existing_candidate_analysis/four_way_runs_reaggregated.csv"
OUT = BASE / "pascal_revision/ross_valid_diversity.csv"

det = pd.read_csv(DIV / "diversity_by_run_layer.csv")
eq = pd.read_csv(DIV / "diversity_equalized_summary.csv")
runs = pd.read_csv(RUNS)

det = det[det.layer == "ROSS_delivered"].copy()
det = det.merge(runs[["method","spec","seed","delivered_n_json","ross_submitted_n"]], on=["method","spec","seed"], how="left")
det["delivery_rate_pct"] = 100*det.delivered_n_json/det.ross_submitted_n
det["level"] = "run_method_spec_seed"
det["notes"] = "diversity computed only on ROSS-valid designs"

keep = ["level","method","method_label","spec","seed","n","n_exact_unique","exact_unique_rate",
        "family_count","family_entropy","family_effective_count","dominant_family_share",
        "median_nn_same_family","median_pair_same_family","distance_sample_n",
        "novelty_median_to_reference","novelty_p10_to_reference","novelty_sample_n",
        "near_dup_frac_le_0.001","near_dup_frac_le_0.005","near_dup_frac_le_0.01",
        "near_dup_frac_le_0.02","near_dup_frac_le_0.05","nn_sample_n","nn_sample_median",
        "delivered_n_json","ross_submitted_n","delivery_rate_pct","notes"]
run_tbl = det[keep].copy()

agg_rows=[]
for (method,label,spec),g in det.groupby(["method","method_label","spec"],sort=True):
    agg_rows.append({
        "level":"method_spec","method":method,"method_label":label,"spec":spec,"seed":"42|7|13",
        "n":float(g.n.mean()),"n_exact_unique":float(g.n_exact_unique.mean()),
        "exact_unique_rate":float(g.exact_unique_rate.mean()),
        "family_count":float(g.family_count.mean()),"family_entropy":float(g.family_entropy.mean()),
        "family_effective_count":float(g.family_effective_count.mean()),
        "dominant_family_share":float(g.dominant_family_share.mean()),
        "median_nn_same_family":float(g.median_nn_same_family.mean()),
        "median_pair_same_family":float(g.median_pair_same_family.mean()),
        "distance_sample_n":float(g.distance_sample_n.mean()),
        "novelty_median_to_reference":float(g.novelty_median_to_reference.mean()),
        "novelty_p10_to_reference":float(g.novelty_p10_to_reference.mean()),
        "novelty_sample_n":float(g.novelty_sample_n.mean()),
        "near_dup_frac_le_0.001":float(g["near_dup_frac_le_0.001"].mean()),
        "near_dup_frac_le_0.005":float(g["near_dup_frac_le_0.005"].mean()),
        "near_dup_frac_le_0.01":float(g["near_dup_frac_le_0.01"].mean()),
        "near_dup_frac_le_0.02":float(g["near_dup_frac_le_0.02"].mean()),
        "near_dup_frac_le_0.05":float(g["near_dup_frac_le_0.05"].mean()),
        "nn_sample_n":float(g.nn_sample_n.mean()),"nn_sample_median":float(g.nn_sample_median.mean()),
        "delivered_n_json":int(g.delivered_n_json.sum()),
        "ross_submitted_n":int(g.ross_submitted_n.sum()),
        "delivery_rate_pct":100*g.delivered_n_json.sum()/g.ross_submitted_n.sum(),
        "notes":"mean over three sampling seeds; n is mean delivered count",
    })
for (method,label),g in det.groupby(["method","method_label"],sort=True):
    agg_rows.append({
        "level":"method_all","method":method,"method_label":label,"spec":"ALL6","seed":"42|7|13",
        "n":float(g.n.mean()),"n_exact_unique":float(g.n_exact_unique.mean()),
        "exact_unique_rate":float(g.exact_unique_rate.mean()),
        "family_count":float(g.family_count.mean()),"family_entropy":float(g.family_entropy.mean()),
        "family_effective_count":float(g.family_effective_count.mean()),
        "dominant_family_share":float(g.dominant_family_share.mean()),
        "median_nn_same_family":float(g.median_nn_same_family.mean()),
        "median_pair_same_family":float(g.median_pair_same_family.mean()),
        "distance_sample_n":float(g.distance_sample_n.mean()),
        "novelty_median_to_reference":float(g.novelty_median_to_reference.mean()),
        "novelty_p10_to_reference":float(g.novelty_p10_to_reference.mean()),
        "novelty_sample_n":float(g.novelty_sample_n.mean()),
        "near_dup_frac_le_0.001":float(g["near_dup_frac_le_0.001"].mean()),
        "near_dup_frac_le_0.005":float(g["near_dup_frac_le_0.005"].mean()),
        "near_dup_frac_le_0.01":float(g["near_dup_frac_le_0.01"].mean()),
        "near_dup_frac_le_0.02":float(g["near_dup_frac_le_0.02"].mean()),
        "near_dup_frac_le_0.05":float(g["near_dup_frac_le_0.05"].mean()),
        "nn_sample_n":float(g.nn_sample_n.mean()),"nn_sample_median":float(g.nn_sample_median.mean()),
        "delivered_n_json":int(g.delivered_n_json.sum()),
        "ross_submitted_n":int(g.ross_submitted_n.sum()),
        "delivery_rate_pct":100*g.delivered_n_json.sum()/g.ross_submitted_n.sum(),
        "notes":"pooled over six specs and three sampling seeds; diversity means are descriptive",
    })
agg=pd.DataFrame(agg_rows)

eqr = eq[eq.layer == "ROSS_delivered"].copy()
eqr["level"]="method_spec_equalized"
eqr["method_label"]=eqr.method.map(dict(zip(det.method,det.method_label)))
eqr["seed"]="42|7|13"
eqr["notes"]="equal sample size within specification; 100 repeated subsamples"
eqr = eqr.rename(columns={"n_equalized":"n"})
allcols=sorted(set(run_tbl.columns)|set(agg.columns)|set(eqr.columns))
out=pd.concat([run_tbl.reindex(columns=allcols),agg.reindex(columns=allcols),eqr.reindex(columns=allcols)],ignore_index=True)
out.to_csv(OUT,index=False,encoding="utf-8")
print(out[out.level=="method_all"][["method","n","delivery_rate_pct","family_entropy","median_nn_same_family","median_pair_same_family","novelty_median_to_reference","near_dup_frac_le_0.01"]].to_string(index=False))

