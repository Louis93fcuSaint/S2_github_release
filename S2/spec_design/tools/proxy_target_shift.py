# -*- coding: utf-8 -*-
"""Does the shift the ROSS samples reveal actually buy anything?

Every verified shortlist that stored its per-design proxy and truth values is
re-ranked twice: once with the raw surrogate, once with a log-space offset fitted
out of fold on the other folds.  The comparison is paired by construction, the
same designs and the same truths, so only the ranking changes.  If the offset
buys nothing, the honest thing is to say so and stop trying to calibrate.

Writes outputs/proxy_calibration_report.json["target_shift"].
"""
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec_common as S
import spec_eval as E
import spec_opt as O
import proxy_calibration as C

KS = [10, 20, 50]
FOLDS = 5


def in_band(cs1, spec):
    if spec.get("mode") == "lower":
        return (cs1 >= float(spec["lower"])) & (cs1 <= float(spec["upper"]))
    return np.abs(cs1 - float(spec["target"])) / float(spec["target"]) <= float(spec["tol"])


def score(cs1, spec):
    return O.objective_from_cs1(cs1, spec, None)


def main():
    ctx = E.pool_ctx()
    specs = {sp["name"]: sp for sp in E.specs(ctx, with_adhoc=True)}
    with io.open(os.path.join(E.OUT, "ross_verify_rand.json"), encoding="utf-8") as handle:
        verify = json.load(handle)
    report_path = os.path.join(E.OUT, "proxy_calibration_report.json")
    report = {}
    if os.path.exists(report_path):
        with io.open(report_path, encoding="utf-8") as handle:
            report = json.load(handle)

    per_arm = {}
    print("%-12s %-16s %-8s %-8s %-8s %-8s %s"
          % ("method", "spec", "claim", "truth", "shifted", "delta%", "top10 / top50 raw -> shifted"))
    for key in sorted(verify):
        entry = verify[key]
        if key == "_calibration" or "truth_cs1" not in entry:
            continue
        spec = specs.get(entry["spec"])
        if spec is None or len(entry["truth_cs1"]) < 20:
            continue
        truth = np.asarray(entry["truth_cs1"], dtype=np.float64)
        pred = np.asarray(entry["pred_cs1"], dtype=np.float64)
        if not (np.isfinite(truth).all() and np.isfinite(pred).all()):
            continue
        delta = C.shift_cross_fit(truth, pred, folds=FOLDS)
        hit = in_band(truth, spec)
        raw = C.topk_rate(score(pred, spec), hit, KS)
        shifted = C.topk_rate(score(pred * np.exp(delta), spec), hit, KS)
        per_arm[key] = {
            "method": entry["method"], "spec": entry["spec"],
            "mode": spec.get("mode", "band"), "n": int(len(truth)),
            "aim_rpm": float(spec.get("aim") or spec.get("target")),
            "band_rpm": [float(spec["lower"]), float(spec["upper"])],
            "subset_truth_rate": float(hit.mean()),
            "subset_proxy_claim": float(in_band(pred, spec).mean()),
            "delta_log_median": C.shift_fit(truth, pred),
            "delta_oof_range": [float(delta.min()), float(delta.max())],
            "raw": raw, "shifted": shifted,
            "gain_top10": shifted["top10"] - raw["top10"],
            "gain_top20": shifted["top20"] - raw["top20"],
            "gain_top50": shifted["top50"] - raw["top50"],
        }
        print("%-12s %-16s %-8.3f %-8.3f %-8.3f %-8.2f %.2f->%.2f / %.2f->%.2f"
              % (entry["method"], entry["spec"],
                 per_arm[key]["subset_proxy_claim"], hit.mean(), shifted["top20"],
                 100.0 * per_arm[key]["delta_log_median"],
                 raw["top10"], shifted["top10"], raw["top50"], shifted["top50"]))

    gains = [v["gain_top20"] for v in per_arm.values()]
    report["target_shift"] = {
        "folds": FOLDS, "ks": KS, "per_arm": per_arm,
        "gain_top20_mean": float(np.mean(gains)) if gains else None,
        "n_arms_helped": int(sum(1 for g in gains if g > 0.0)),
        "n_arms": len(gains),
        "method": "delta = median log(truth/proxy), fitted out of fold; rank by the "
                  "shifted score on the same designs",
    }
    S.save_json(report_path, report)
    print("\n[out] %s" % report_path)
    print("mean top20 gain over %d arms: %s"
          % (len(gains), "n/a" if not gains else "%+.3f" % np.mean(gains)))


if __name__ == "__main__":
    main()