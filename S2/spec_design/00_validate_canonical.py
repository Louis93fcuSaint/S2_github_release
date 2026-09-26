# -*- coding: utf-8 -*-
"""Feasibility check for the single-order spec-driven protocol.

Three things must hold before any model is trained:

1. The canonical 33-vector round-trips: dataset row -> canonical -> S1 sample ->
   47 surrogate features must reproduce the row's own features exactly, for
   EVERY family (nd 1-6, nb 2-4), not just the old (3, 2) slice.
2. The frozen surrogate is accurate on every family (it was trained on all of
   them, but nobody has checked the per-family numbers for this use).
3. The spec (cs1 >= 14400, <= material ceiling) is actually reachable, and ROSS
   runs on a non-(3,2) family through the same fast path.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_common as S

OUT = os.path.join(S.HERE, "outputs")


def main():
    report = {}
    pool = S.load_pool()
    raw_names = list(S.constraint_module().to_features_v4(
        [S.sample_from_canonical(np.zeros(S.DESIGN_DIM), "Steel", 3, 3)])[1])
    print("pool rows %d | raw features %d | canonical dim %d"
          % (len(pool), len(raw_names), S.DESIGN_DIM))

    surrogate = S.load_surrogate("mlp_best", threads=6)
    print("surrogate: %s\n" % surrogate.source)

    rng = np.random.default_rng(0)
    per_family = {}
    worst_roundtrip = 0.0
    print("%-12s %-7s %-9s %-9s %-9s %-9s %-8s" %
          ("family", "n", "rt_max", "valid%", "MAPE1%", "med1%", "spec%"))
    design_cols = S.CANONICAL_COLUMNS
    for nd in S.ND_CHOICES:
        for nb in S.NB_CHOICES:
            subset = pool[(pool["n_disks"] == nd) & (pool["n_bearings"] == nb)]
            picks = subset.iloc[rng.choice(len(subset), size=25, replace=False)]
            x = picks[design_cols].to_numpy(dtype=np.float64)
            materials = picks["material"].tolist()
            nds = [nd] * len(picks)
            nbs = [nb] * len(picks)

            rows = S.canonical_to_model_rows(x, materials, nds, nbs)
            ref, _ = S.mlp_module().build_engineered_features(
                picks[raw_names].to_numpy(dtype=np.float64), raw_names)
            rt = float(np.abs(rows - np.asarray(ref, dtype=np.float64)).max())
            worst_roundtrip = max(worst_roundtrip, rt)

            ok = sum(1 for xi, m in zip(x, materials) if S.is_valid(xi, m, nd, nb)[0])
            pred = surrogate.predict(rows)
            truth = picks[S.ALL_CS].to_numpy(dtype=np.float64)
            keep = np.isfinite(truth) & (truth > 0) & np.isfinite(pred)
            ape = np.abs(pred[keep] - truth[keep]) / truth[keep] * 100.0
            spec = float((subset["cs_1_rpm"] >= S.DEFAULT_LOWER_RPM).mean() * 100)

            per_family["%d_%d" % (nd, nb)] = {
                "n": int(len(subset)), "roundtrip_max_abs": rt,
                "constraint_valid_rate": ok / len(picks),
                "MAPE_pct_all_orders": float(ape.mean()),
                "median_APE_pct_cs1": float(np.median(
                    np.abs(pred[:, 0] - truth[:, 0]) / truth[:, 0] * 100.0)),
                "spec_pass_pct_cs1_ge_14400": spec,
                "cs1_max_rpm": float(subset["cs_1_rpm"].max()),
            }
            print("%-12s %-7d %-9.2e %-9.3f %-9.2f %-9.2f %-8.2f"
                  % ("(%d,%d)" % (nd, nb), len(subset), rt, ok / len(picks),
                     ape.mean(), per_family["%d_%d" % (nd, nb)]["median_APE_pct_cs1"], spec))

    ceilings = {}
    for material in S.MATERIAL_ORDER:
        lo, hi = S.spec_bounds(material, pool=pool)
        ceilings[material] = {"lower": lo, "upper": hi,
                              "n_in_band": int(((pool["material"] == material)
                                                & S.spec_ok(pool["cs_1_rpm"], lo, hi)).sum())}
    print("\nspec band per material (cs1 in [%.0f, ceiling]):" % S.DEFAULT_LOWER_RPM)
    for material, info in ceilings.items():
        print("  %-9s ceiling %8.0f rpm   real designs in band: %d"
              % (material, info["upper"], info["n_in_band"]))

    nd, nb, material = 3, 3, "Steel"
    subset = pool[(pool["n_disks"] == nd) & (pool["n_bearings"] == nb)
                  & (pool["material"] == material)]
    xi = subset[design_cols].to_numpy(dtype=np.float64)[0]
    started = time.time()
    forward = S.run_ross(xi, material, nd, nb)
    seconds = time.time() - started
    print("\nROSS on a (%d,%d)/%s rotor: %.1f s, %d forward modes, first3 %s"
          % (nd, nb, material, seconds, len(forward), [round(v) for v in forward[:3]]))

    report = {"canonical_dim": S.DESIGN_DIM, "pool_rows": int(len(pool)),
              "roundtrip_worst_max_abs": worst_roundtrip,
              "per_family": per_family, "spec_band": ceilings,
              "ross_seconds_one_rotor": seconds,
              "ross_forward_modes": len(forward)}
    S.save_json(os.path.join(OUT, "validated_canonical.json"), report)
    print("\n[out] %s" % os.path.join(OUT, "validated_canonical.json"))
    print("[verdict] round-trip worst max abs diff = %.2e -> %s"
          % (worst_roundtrip, "OK" if worst_roundtrip < 1e-8 else "FAIL"))


if __name__ == "__main__":
    main()