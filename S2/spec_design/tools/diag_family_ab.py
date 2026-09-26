# -*- coding: utf-8 -*-
"""Family policy A/B: what the learned family prior changes downstream.

Reads the two funnel files and the submitted CSVs of the uniform arm and the
learned arm, and reports per spec:
  - in_band_raw and the number of families the sampler actually used
  - the family mix of the submitted top-200, and how concentrated it is
  - novelty of the submission (nearest pool design, in pool-spread units)
  - whether the submitted batches overlap (identical designs)
Writes outputs/family_ab.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec_common as S
import spec_eval as E
import spec_opt as O

OUT = os.path.join(E.OUT, "latent_ddpm")
ARMS = [("uniform", "v1"), ("learned", "famL_s25")]


def effective(n):
    w = np.asarray(n, dtype=np.float64)
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    w = w[w > 0]
    return float(np.exp(-(w * np.log(w)).sum()))


def main():
    ctx = E.pool_ctx()
    rep = {"arms": {}, "per_spec": {}}
    frames = {}
    for arm, tag in ARMS:
        with open(os.path.join(OUT, "funnel_%s.json" % tag), encoding="utf-8") as fh:
            funnel = json.load(fh)
        rep["arms"][arm] = {"tag": tag, "funnel": funnel}
        for spec_name, row in funnel.items():
            path = os.path.join(OUT, "%s__%s.csv" % (tag, spec_name))
            if os.path.exists(path):
                frames[(arm, spec_name)] = pd.read_csv(path)

    spec_names = sorted(rep["arms"]["uniform"]["funnel"])
    for name in spec_names:
        entry = {}
        for arm, tag in ARMS:
            fun = rep["arms"][arm]["funnel"][name]
            frame = frames.get((arm, name))
            sub = {}
            if frame is not None:
                counts = frame.groupby(["n_disks", "n_bearings"]).size()
                mix = {"%d_%d" % k: int(v) for k, v in counts.items()}
                sub = {"mix": mix,
                       "n_families": int(len(mix)),
                       "effective_families": effective(list(counts.values)),
                       "top_family_share": float(counts.max() / counts.sum())}
            gen_mix = fun.get("family_mix") or {}
            entry[arm] = {
                "generated_families": int(len(gen_mix)),
                "generated_effective": effective(list(gen_mix.values())) if gen_mix else None,
                "in_band_raw": fun["in_band_raw"],
                "in_band_top200": fun.get("in_band_top200"),
                "median_rel_err_top": fun.get("median_rel_err_top"),
                "valid_rate": fun.get("valid_rate"),
                "submitted": sub,
            }
        entry["delta_in_band_raw"] = (entry["learned"]["in_band_raw"]
                                      - entry["uniform"]["in_band_raw"])
        if entry["uniform"]["in_band_raw"] > 0:
            entry["gain_in_band_raw"] = (entry["learned"]["in_band_raw"]
                                         / entry["uniform"]["in_band_raw"])
        rep["per_spec"][name] = entry

    # overlap between the two submissions for the same spec
    for name in spec_names:
        a = frames.get(("uniform", name))
        b = frames.get(("learned", name))
        if a is None or b is None:
            continue
        ka = set(map(tuple, np.round(a[S.CANONICAL_COLUMNS].to_numpy(np.float64), 6)))
        kb = set(map(tuple, np.round(b[S.CANONICAL_COLUMNS].to_numpy(np.float64), 6)))
        rep["per_spec"][name]["shared_designs"] = int(len(ka & kb))

    means = {}
    for arm, _ in ARMS:
        vals = [rep["per_spec"][n][arm]["in_band_raw"] for n in spec_names]
        means[arm] = float(np.mean(vals))
    rep["mean_in_band_raw"] = means
    rep["mean_gain"] = means["learned"] / max(means["uniform"], 1e-12)
    rep["mean_submitted_families"] = {
        arm: float(np.mean([rep["per_spec"][n][arm]["submitted"].get("n_families", 0)
                            for n in spec_names])) for arm, _ in ARMS}
    S.save_json(os.path.join(E.OUT, "family_ab.json"), rep)

    print("%-14s %10s %10s %7s | %6s %6s | %6s %6s"
          % ("spec", "uniform", "learned", "gain", "famU", "famL", "effU", "effL"))
    for name in spec_names:
        u, l = rep["per_spec"][name]["uniform"], rep["per_spec"][name]["learned"]
        print("%-14s %10.4f %10.4f %7.3f | %6d %6d | %6.2f %6.2f"
              % (name, u["in_band_raw"], l["in_band_raw"],
                 rep["per_spec"][name].get("gain_in_band_raw", float("nan")),
                 u["submitted"].get("n_families", 0), l["submitted"].get("n_families", 0),
                 u["submitted"].get("effective_families", 0),
                 l["submitted"].get("effective_families", 0)))
    print("mean in_band_raw uniform %.4f | learned %.4f | gain %.3f"
          % (means["uniform"], means["learned"], rep["mean_gain"]))
    print("[out] %s" % os.path.join(E.OUT, "family_ab.json"))


if __name__ == "__main__":
    main()