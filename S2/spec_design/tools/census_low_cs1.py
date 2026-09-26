# -*- coding: utf-8 -*-
"""How many low-cs1 labels are solver artefacts?

Two populations are tested with the twice-solved reproducibility rule:

  * every dataset row whose stored cs1 is below 400 rpm (172 of 199811),
  * the three ROSS-verified stage-B rotors whose truth came back below 200 rpm.

Writes outputs/cs1_census.json.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec_common as S
import spec_eval as E
import ross_guard as G

OUT_JSON = os.path.join(E.OUT, "cs1_census.json")
THRESHOLD = 400.0


def main():
    ctx = E.pool_ctx()
    pool = ctx["pool"]
    cs1 = ctx["Y"][:, 0]
    idx = np.where(cs1 < THRESHOLD)[0]
    print("dataset rows with cs1 < %.0f rpm: %d of %d" % (THRESHOLD, len(idx), len(cs1)))
    jobs = [(pool.loc[i, S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64),
             str(pool.loc[i, "material"]), int(pool.loc[i, "n_disks"]),
             int(pool.loc[i, "n_bearings"]), float(cs1[i])) for i in idx]
    started = time.time()
    results = G.probe_many(jobs, workers=12)
    print("dataset census done in %.0f s" % (time.time() - started), flush=True)

    records = []
    for i, res in zip(idx, results):
        records.append({"i": int(i), "material": str(pool.loc[i, "material"]),
                        "nd": int(pool.loc[i, "n_disks"]),
                        "nb": int(pool.loc[i, "n_bearings"]),
                        "stored": float(cs1[i]), "first_a": res["first_a"],
                        "first_b": res["first_b"], "trusted": res["trusted"],
                        "truth": res["truth"]})
    flagged = [r for r in records if r["trusted"] is False]
    undecided = [r for r in records if r["trusted"] is None]
    print("  trusted (both solves agree) : %d" % sum(1 for r in records if r["trusted"]))
    print("  artefacts (do not agree)    : %d" % len(flagged))
    print("  undecided (a solve failed)  : %d" % len(undecided))
    for r in flagged[:25]:
        print("    row %6d %-9s stored %8.2f -> %s / %s"
              % (r["i"], r["material"], r["stored"],
                 "None" if r["first_a"] is None else "%.2f" % r["first_a"],
                 "None" if r["first_b"] is None else "%.2f" % r["first_b"]))

    store = json.load(open(os.path.join(E.OUT, "uncertainty_stageB.json"), encoding="utf-8"))
    stage_b = []
    for key, entry in store.items():
        for r in entry["rotors"]:
            if r["truth"] and G.needs_probe(r["truth"]):
                frame_path = None
                for folder in ("latent_baselines", "latent_ddpm"):
                    candidate = os.path.join(E.OUT, folder, "%s__%s.csv" % (entry["method"],
                                                                            entry["spec"]))
                    if os.path.exists(candidate):
                        frame_path = candidate
                        break
                import pandas as pd
                frame = pd.read_csv(frame_path)
                x = frame.loc[r["index"], S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
                stage_b.append({"key": key, "index": r["index"], "stored": r["truth"],
                                "job": (x, r["material"], r["nd"], r["nb"], r["truth"])})
    print("\nstage-B rotors needing the second solve: %d" % len(stage_b))
    res_b = G.probe_many([item["job"] for item in stage_b], workers=12)
    for item, res in zip(stage_b, res_b):
        item.pop("job")
        item.update({k: res[k] for k in ("first_a", "first_b", "trusted", "truth")})
        print("  %-24s index %3d stored %8.3f -> a %s b %s | %s"
              % (item["key"], item["index"], item["stored"],
                 "None" if res["first_a"] is None else "%.3f" % res["first_a"],
                 "None" if res["first_b"] is None else "%.3f" % res["first_b"],
                 "trusted" if res["trusted"] else "ARTEFACT -> %s" % res["truth"]))

    S.save_json(OUT_JSON, {"threshold_rpm": THRESHOLD, "n_dataset_rows": int(len(idx)),
                           "n_trusted": int(sum(1 for r in records if r["trusted"])),
                           "n_artefact": len(flagged), "n_undecided": len(undecided),
                           "dataset": records, "stage_b": stage_b})
    print("[out] %s" % OUT_JSON)


if __name__ == "__main__":
    main()