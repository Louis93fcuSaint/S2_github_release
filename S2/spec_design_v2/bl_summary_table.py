# -*- coding: utf-8 -*-
"""Build the single-source §6.3 table (S2, new approach).

Why this file exists
--------------------
`06_report.py` (spec_design/) reads its proxy table from
`spec_design/outputs/baseline_metrics.json`, which was written on 2026-09-23 by
the OLD canonical-space baseline suite (`spec_baselines.py`).  The ROSS truth
column of the same report was produced on 2026-09-26 from the candidate files of
the LATER latent-space suite (`spec_baselines_latent.py`, 2026-09-25).  The two
halves of that table therefore come from two different experiments, and their
proxy numbers disagree (DE 0.764 vs 1.000, and so on).

This script rebuilds the table from ONE source per column, so the paper can cite
a consistent set:

  proxy / novelty   spec_design/outputs/latent_baselines/baseline_metrics.json
                    (the metrics of exactly the candidate files that were solved)
  ROSS truth        spec_design_v2/outputs/bl_truth/bl_truth_summary.json
                    (self-ranked top-20 by the frozen surrogate, and an
                     unbiased random 20, both solved through the v2 toolchain)

Output: `outputs/bl_truth/table_6_3.md` and `.csv`.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V1_OUT = os.path.join(os.path.dirname(HERE), "spec_design", "outputs")
BL_METRICS = os.path.join(V1_OUT, "latent_baselines", "baseline_metrics.json")
TRUTH = os.path.join(HERE, "outputs", "bl_truth", "bl_truth_summary.json")
OUT_DIR = os.path.join(HERE, "outputs", "bl_truth")

ORDER = ["ga", "adam", "de", "retrieval_jitter", "bo", "random_latent", "retrieval"]
LABEL = {"ga": "GA", "adam": "Adam", "de": "DE", "retrieval_jitter": "retrieval+jitter",
         "bo": "BO", "random_latent": "random (latent 箱)", "retrieval": "retrieval (检索)"}


def mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    with open(BL_METRICS, encoding="utf-8") as fh:
        base = json.load(fh)
    with open(TRUTH, encoding="utf-8") as fh:
        truth = json.load(fh)

    specs = sorted(set(k.split("|")[0] for k in base))
    by_method = {}
    for row in truth["rows"]:
        by_method.setdefault(row["method"], []).append(row)

    out = []
    for m in ORDER:
        if not any(k.endswith("|" + m) for k in base):
            continue
        rows = by_method.get(m, [])
        rec = {
            "method": m,
            "proxy_in_band": mean([base[s + "|" + m]["spec_rate_proxy"] for s in specs]),
            "novelty": mean([base[s + "|" + m]["novelty_min_median"] for s in specs]),
            "joint": mean([base[s + "|" + m]["spec_and_novel_1.0"] for s in specs]),
            "truth_top20": mean([r["top20"]["truth_in_band_5"] for r in rows]),
            "truth_rand20": mean([r.get("rand20", {}).get("truth_in_band_5") for r in rows]),
            "n_specs_truth": len(rows),
        }
        out.append(rec)

    lines = ["| 方法 | 代理在带 | 新颖度 | 联合判据(在带且新颖≥1.0) | 真值 top-20 在带 | 真值 random-20 在带 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in out:
        lines.append("| %s | %.1f %% | %.2f | %.1f %% | %.3f | %.3f |"
                     % (LABEL.get(r["method"], r["method"]), 100 * r["proxy_in_band"],
                        r["novelty"], 100 * r["joint"], r["truth_top20"], r["truth_rand20"]))
    text = "\n".join(lines) + "\n"

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "table_6_3.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(OUT_DIR, "table_6_3.csv"), "w", encoding="utf-8") as fh:
        fh.write("method,proxy_in_band,novelty,joint,truth_top20,truth_rand20,n_specs_truth\n")
        for r in out:
            fh.write("%s,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n"
                     % (r["method"], r["proxy_in_band"], r["novelty"], r["joint"],
                        r["truth_top20"], r["truth_rand20"], r["n_specs_truth"]))
    print(text)
    print("[out] %s" % os.path.join(OUT_DIR, "table_6_3.md"))


if __name__ == "__main__":
    main()
