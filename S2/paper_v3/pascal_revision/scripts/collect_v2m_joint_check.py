# -*- coding: utf-8 -*-
"""Collect the same-specification v2m vs v2i joint-interval comparison."""
import json
from pathlib import Path
import pandas as pd

V2 = Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2\spec_design_v2\outputs")
OUT = Path(__file__).resolve().parents[1] / "results" / "v2m_joint_interval_check.csv"
ROWS = [
    ("two-order joint interval", "cs1 in [3000,4500], cs2 in [6000,9000]",
     V2 / "verify_joint_cs12.json", V2 / "verify_joint_cs12_v2mcentre.json"),
    ("three-order joint interval", "cs1 in [3000,4500], cs2 in [6000,9000], cs3 in [14000,22000]",
     V2 / "verify_joint_cs123.json", V2 / "verify_joint_cs123_v2mcentre.json"),
]
rows = []
for group, spec, pi, pm in ROWS:
    for model, path, note in (
        ("v2i (interval condition)", pi, "interval checkpoint, interval target"),
        ("v2m (band-centre condition)", pm, "point checkpoint, interval replaced by its geometric centre"),
    ):
        if not path.exists():
            rows.append(dict(experiment_group=group, specification=spec, model=model,
                             candidate_pool=5000, shortlist_n=200, delivered_n=None,
                             delivery_rate_pct=None, note=note + "; MISSING " + path.name))
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        r = d["runs"][0]
        rows.append(dict(experiment_group=group, specification=spec, model=model,
                         candidate_pool=int(d["n_batch"]), shortlist_n=int(r["n_shortlist"]),
                         delivered_n=int(r["gate_tol0"]["n_passed"]),
                         delivery_rate_pct=100.0 * r["gate_tol0"]["n_passed"] / r["n_shortlist"],
                         note=note))
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(df.to_string(index=False))
print("wrote", OUT)
