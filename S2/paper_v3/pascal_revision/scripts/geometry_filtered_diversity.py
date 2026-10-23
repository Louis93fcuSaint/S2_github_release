# -*- coding: utf-8 -*-
"""Post-hoc geometry filter on ROSS-valid shortlists: retention, families, diversity."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

BASE=Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2\paper\paper_v3")
sys.path.insert(0,str(BASE/"research_generators_specification_study/10_reproducibility"))
sys.path.insert(0,str(BASE/"pascal_revision/scripts"))
import research_common as C
from geometry_audit import geometry_metrics

RUNS=BASE/"research_generators_specification_study/03_existing_candidate_analysis/four_way_runs_reaggregated.csv"
DIV=BASE/"research_generators_specification_study/05_valid_diversity"
OUT=BASE/"pascal_revision/results"
scales=np.load(DIV/"reference_scales.npy")
runs=pd.read_csv(RUNS)

def entropy(counts):
    p=np.asarray(counts,float); p=p[p>0]/p.sum()
    return float(-(p*np.log(p)).sum()), float(np.exp(-(p*np.log(p)).sum()))

rows=[]
for rec in runs.itertuples(index=False):
    ver=Path(str(rec.verify_csv)); pas=Path(str(rec.passed_csv))
    if not ver.exists() or not pas.exists(): continue
    df=pd.read_csv(ver); geo=geometry_metrics(df)
    p=pd.read_csv(pas); delivered=df["rank"].astype(int).isin(set(p["rank"].astype(int))).to_numpy()
    keep=delivered & geo["pass_edge_10mm"].to_numpy()
    sub=df.loc[keep].reset_index(drop=True)
    fam=sub.groupby(["material","n_disks","n_bearings"]).size() if len(sub) else pd.Series(dtype=int)
    ent,eff=entropy(fam.values) if len(fam) else (np.nan,np.nan)
    nn,pair,n=C.same_family_nn_and_pair(sub,scales,max_points=1000,seed=42) if len(sub)>=2 else (np.nan,np.nan,len(sub))
    # equal-size resampling: sample the delivered set down to the survivor count
    n_eq=int(keep.sum())
    nn_eq=[]; pair_eq=[]
    if n_eq>=2 and delivered.sum()>=n_eq:
        rng=np.random.default_rng(20261022+int(rec.seed)+sum(map(ord,rec.method+rec.spec)))
        for rep in range(20):
            idx=rng.choice(np.where(delivered)[0],size=n_eq,replace=False)
            a,pv,_=C.same_family_nn_and_pair(df.iloc[np.sort(idx)],scales,max_points=n_eq,seed=rep)
            nn_eq.append(a); pair_eq.append(pv)
    rows.append({
        "method":rec.method,"method_label":rec.method_label,"spec":rec.spec,"seed":int(rec.seed),
        "n_delivered":int(delivered.sum()),"n_delivered_and_edge10":n_eq,
        "retention_given_delivered":n_eq/max(int(delivered.sum()),1),
        "family_count_before":int(df.loc[delivered].groupby(["material","n_disks","n_bearings"]).ngroups) if delivered.any() else 0,
        "family_count_after":int(len(fam)),
        "family_entropy_before":entropy(df.loc[delivered].groupby(["material","n_disks","n_bearings"]).size().values)[0] if delivered.any() else np.nan,
        "family_entropy_after":ent,
        "family_effective_count_after":eff,
        "nn_after":nn,"pair_after":pair,"distance_n_after":n,
        "nn_equalized_before":float(np.mean(nn_eq)) if nn_eq else np.nan,
        "pair_equalized_before":float(np.mean(pair_eq)) if pair_eq else np.nan,
        "n_equalized":n_eq,"n_resamples":len(nn_eq),
    })
det=pd.DataFrame(rows)
det.to_csv(OUT/"geometry_filtered_diversity_by_run.csv",index=False,encoding="utf-8")
summary=det.groupby(["method","method_label"],sort=True).agg(
    runs=("method","size"),delivered=("n_delivered","sum"),
    delivered_and_edge10=("n_delivered_and_edge10","sum"),
    mean_retention=("retention_given_delivered","mean"),
    mean_family_entropy_before=("family_entropy_before","mean"),
    mean_family_entropy_after=("family_entropy_after","mean"),
    mean_nn_before=("nn_equalized_before","mean"),mean_nn_after=("nn_after","mean"),
    mean_pair_before=("pair_equalized_before","mean"),mean_pair_after=("pair_after","mean"),
).reset_index()
summary["pooled_retention"]=summary.delivered_and_edge10/summary.delivered
summary.to_csv(OUT/"geometry_filtered_diversity_summary.csv",index=False,encoding="utf-8")
print(summary.to_string(index=False))
