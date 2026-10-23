# -*- coding: utf-8 -*-
"""Verifiable interval/joint comparison table for the Pascal meeting."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd

V2 = Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2\spec_design_v2\outputs")
OUT = Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2\paper\paper_v3\pascal_revision\interval_joint_comparison.csv")
EA = Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2\paper\paper_v3\research_generators_specification_study\eaai_comparative_study")

def wilson(k,n,z=1.959963984540054):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return 100*(c-h),100*(c+h)

def load(name):
    return json.loads((V2/(name+".json")).read_text(encoding="utf-8"))

def row(group, model, layout, condition_form, spec_text, source_json, label, gate_key, n_expected=200):
    j=load(source_json)
    r=[x for x in j["runs"] if x["label"]==label][0]
    g=r.get(gate_key)
    if g is None:
        raise SystemExit("missing %s in %s/%s" % (gate_key, source_json, label))
    k=int(g["n_passed"]); n=int(r["n_shortlist"])
    lo,hi=wilson(k,n)
    return {
        "experiment_group":group,"model_or_method":model,"condition_layout":layout,
        "condition_form":condition_form,"specification":spec_text,"candidate_pool":int(j.get("n_batch",5000)),
        "shortlist_n":n,"ross_solved_n":int(r["n_solved"]),"delivered_n":k,
        "delivery_rate_pct":100*k/n,"wilson_low_pct":lo,"wilson_high_pct":hi,
        "acceptance_rule":("exact user band, tol=0" if gate_key=="gate_tol0" else "point target +/-5%"),
        "selector":"surrogate top-200 (historical selector)","evidence_json":str(V2/(source_json+".json")),
        "evidence_csv":r.get("csv"),"status":"verified_existing_result",
        "comparability_note":"",
    }

rows=[]
rows.append(row("point_vs_interval_specialisation","point-conditioned DDPM (v2m)","point","point target +/-5%","cs1 = 3680.8 rpm","verify_cs1","cs1__top200","gate_tol5"))
rows.append(row("point_vs_interval_specialisation","interval-conditioned DDPM (v2i)","interval","point input +/-5%","cs1 = 3680.8 rpm","verify_pt_v2i","pt_v2i__top200","gate_tol5"))
# tol1 is reported in the same run under a different field
j=load("verify_cs1"); r=[x for x in j["runs"] if x["label"]=="cs1__top200"][0]
k=int(r["gate_tol1"]["n_passed"]); lo,hi=wilson(k,200)
rows.append({"experiment_group":"point_vs_interval_specialisation","model_or_method":"point-conditioned DDPM (v2m)",
 "condition_layout":"point","condition_form":"point target +/-1%","specification":"cs1 = 3680.8 rpm",
 "candidate_pool":5000,"shortlist_n":200,"ross_solved_n":int(r["n_solved"]),"delivered_n":k,
 "delivery_rate_pct":100*k/200,"wilson_low_pct":lo,"wilson_high_pct":hi,"acceptance_rule":"point target +/-1%",
 "selector":"surrogate top-200 (historical selector)","evidence_json":str(V2/"verify_cs1.json"),
 "evidence_csv":r.get("csv"),"status":"verified_existing_result","comparability_note":""})
# v2i point target is in verify_cs1? no: int_mid_v2i is 2600-5200, not point. The paper's 186/200 point input is likely another run. Keep only verified interval runs and note.
for name,form,spec in [
 ("verify_int_narrow_v2i","single-order interval (narrow)","cs1 in [3500, 3870] rpm"),
 ("verify_int_mid_v2i","single-order interval (medium)","cs1 in [2600, 5200] rpm"),
 ("verify_int_wide_v2i","single-order interval (wide)","cs1 in [1800, 7500] rpm"),
 ("verify_int_open_v2i","one-sided lower bound","cs1 >= 3681 rpm"),
 ("verify_int_narrow_v2m","single-order interval (narrow)","cs1 in [3500, 3870] rpm"),
 ("verify_int_mid_v2m","single-order interval (medium)","cs1 in [2600, 5200] rpm"),
 ("verify_int_wide_v2m","single-order interval (wide)","cs1 in [1800, 7500] rpm"),
 ("verify_int_open_v2m","one-sided lower bound","cs1 >= 3681 rpm"),
]:
    label=name.replace("verify_","")+"__top200"
    model="interval-conditioned DDPM (v2i)" if name.endswith("v2i") else "point-layout DDPM (v2m, band-centre conditioning)"
    layout="interval" if name.endswith("v2i") else "point"
    rows.append(row("single_order_interval",model,layout,form,spec,name,label,"gate_tol0"))
for name,form,spec,label in [
 ("verify_joint_cs12","two-order joint interval","cs1 in [3000,4500], cs2 in [6000,9000] rpm","joint_cs12__top200"),
 ("verify_joint_cs123","three-order joint interval","cs1 in [3000,4500], cs2 in [6000,9000], cs3 in [14000,22000] rpm","joint_cs123__top200"),
 ("verify_cs1cs2","two-order joint point +/-5%","cs1 = 3680.8, cs2 = 7021.0 rpm","cs1cs2__top200"),
 ("verify_cs1cs2_big","two-order joint point +/-5% (20k pool)","cs1 = 3680.8, cs2 = 7021.0 rpm","cs1cs2_big__top200"),
 ("verify_cs123","three-order joint point +/-5%","cs1 = 3680.8, cs2 = 7021.0, cs3 = 17970.5 rpm","cs123__top200"),
 ("verify_cs123_big","three-order joint point +/-5% (20k pool)","cs1 = 3680.8, cs2 = 7021.0, cs3 = 17970.5 rpm","cs123_big__top200"),
]:
    grp="multi_order_joint"
    rows.append(row(grp,"interval-conditioned DDPM (v2i)" if name.startswith("verify_joint") else "point-layout DDPM (v2m)",
                    "interval" if name.startswith("verify_joint") else "point",form,spec,name,label,
                    "gate_tol0" if name.startswith("verify_joint") else "gate_tol5"))

# TST planned rows
specs=pd.read_csv(EA/"spec_candidates"/"test_specs.csv")
for s in specs.itertuples():
    if s.slot_id=="TST08": continue
    rows.append({"experiment_group":"planned_TST","model_or_method":"DDPM / CVAE / retrieval+jitter (three methods)",
      "condition_layout":"interval for DDPM, model-specific for CVAE/retrieval","condition_form":s.kind,
      "specification":"%s | %s | %s | orders %s" % (s.slot_id,s.material,s.family,s.orders),
      "candidate_pool":5000,"shortlist_n":200,"ross_solved_n":np.nan,"delivered_n":np.nan,
      "delivery_rate_pct":np.nan,"wilson_low_pct":np.nan,"wilson_high_pct":np.nan,
      "acceptance_rule":"user band, tol=0","selector":"common surrogate gate + uniform in-band",
      "evidence_json":str(EA/"spec_candidates"/"test_specs.csv"),"evidence_csv":"",
      "status":"frozen_not_run","comparability_note":"TST protocol frozen; no ROSS results yet. Reference-library support n_feasible=%d." % int(s.n_feasible)})

df=pd.DataFrame(rows)
df["comparability_note"]=df.apply(lambda r: r.comparability_note or (
    "Same checkpoint and historical top-200 selector; compare only within the same condition form. Not a CVAE/retrieval head-to-head."
    if r.experiment_group in ("single_order_interval","multi_order_joint") else
    "Both checkpoints are separately trained; the point-layout run conditions on the band centre and is not an interval-conditioned model."),axis=1)
df.to_csv(OUT,index=False,encoding="utf-8")
print(df[df.status=="verified_existing_result"][["experiment_group","model_or_method","condition_form","delivered_n","shortlist_n","delivery_rate_pct"]].to_string(index=False))


