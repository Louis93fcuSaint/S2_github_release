# -*- coding: utf-8 -*-
"""Score the uncertainty margin on the stored stage-B evidence.

Reads outputs/uncertainty_stageB.json (per-rotor proxy, seed spread, novelty and
ROSS truth) and answers:

  1. how optimistic is the proxy, and does the margin make its in-band claim
     honest?  (claimed rate, true rate, precision of the claim)
  2. at a fixed ROSS budget of k rotors per batch, do the rotors the margin
     keeps hit the band more often than the proxy's own pick?
  3. is the stage-A constant q90(|log err| / spread) = 2.04 still valid out of
     distribution, or does it have to be re-fit?

The protocol band is two-sided (point target +- 5 %), so "in band" always means
lower <= cs1 <= upper and the honest interval is intersected with both edges.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec_common as S
import spec_eval as E
import uncertainty_signals as U

STORE = os.path.join(E.OUT, "uncertainty_stageB.json")
STAGE_A = os.path.join(E.OUT, "uncertainty_stageA.json")
CALIB = os.path.join(E.OUT, "uncertainty_calibration.json")
OUT_JSON = os.path.join(E.OUT, "uncertainty_stageB_analysis.json")
KS = [5, 10, 20, 40]
DEGENERATE_RPM = 200.0        # a cs1 this low is a solver artefact, not a design


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(a)),
                             np.argsort(np.argsort(b)))[0, 1])


CENSUS = os.path.join(E.OUT, "cs1_census.json")


def load_rows():
    with open(STORE, encoding="utf-8") as handle:
        store = json.load(handle)
    untrusted = set()
    if os.path.exists(CENSUS):
        with open(CENSUS, encoding="utf-8") as handle:
            for item in json.load(handle).get("stage_b", []):
                if item.get("trusted") is False:
                    untrusted.add((item["key"], item["index"]))
    out = []
    for key, entry in store.items():
        for rotor in entry["rotors"]:
            if not rotor["truth"] or (key, rotor["index"]) in untrusted:
                continue
            row = dict(rotor)
            row["method"] = entry["method"]
            row["spec"] = entry["spec"]
            row["lower"] = entry["band"][0]
            row["upper"] = entry["band"][1]
            row["target"] = 0.5 * (entry["band"][0] + entry["band"][1])
            out.append(row)
    return store, out


def batch_block(ok, q):
    """Metrics for one batch of verified rotors.

    Two screens are scored side by side: a single constant margin q on the seed
    spread, and the novelty-scaled bar err90(dF) calibrated in stage A.  The
    second is the one that should transfer, because it is narrow inside the
    manifold and wide outside it.
    """
    lower = np.array([r["lower"] for r in ok])
    upper = np.array([r["upper"] for r in ok])
    target = np.array([r["target"] for r in ok])
    proxy = np.array([r["proxy"] for r in ok])
    spread = np.array([r["spread"] for r in ok])
    dF = np.array([r["dF"] for r in ok])
    truth = np.array([r["truth"] for r in ok])
    lo = proxy / np.exp(q * spread)
    hi = proxy * np.exp(q * spread)
    bar = np.array([r["err90_dF"] for r in ok])
    lo_b = proxy / np.exp(bar)
    hi_b = proxy * np.exp(bar)
    in_band = (truth >= lower) & (truth <= upper)
    say = (proxy >= lower) & (proxy <= upper)
    keep = (lo >= lower) & (hi <= upper)
    keep_b = (lo_b >= lower) & (hi_b <= upper)
    ape = np.abs(proxy - truth) / truth
    ok_rpm = truth > DEGENERATE_RPM
    rel_t = np.abs(truth - target) / target
    rec = {
        "n": int(len(ok)),
        "proxy_say": float(say.mean()), "truth_in": float(in_band.mean()),
        "optimism": float(say.mean() - in_band.mean()),
        "precision_proxy": float(in_band[say].mean()) if say.any() else float("nan"),
        "margin_keep": float(keep.mean()),
        "precision_margin": float(in_band[keep].mean()) if keep.any() else float("nan"),
        "recall_margin": float(keep[in_band].mean()) if in_band.any() else float("nan"),
        "keep_and_band": float((keep & in_band).mean()),
        "bar_median": float(np.median(bar)),
        "margin_keep_dF": float(keep_b.mean()),
        "precision_dF": float(in_band[keep_b].mean()) if keep_b.any() else float("nan"),
        "recall_dF": float(keep_b[in_band].mean()) if in_band.any() else float("nan"),
        "MAPE": float(ape.mean() * 100), "median_APE": float(np.median(ape) * 100),
        "MAPE_trim": float(ape[ok_rpm].mean() * 100) if ok_rpm.any() else float("nan"),
        "n_degenerate": int((~ok_rpm).sum()),
        "hit_true_5": float((rel_t <= 0.05).mean()),
        "spearman_proxy": spearman(proxy, truth), "spearman_margin": spearman(lo, truth),
        "spearman_dF": spearman(dF, truth), "spearman_spread": spearman(spread, truth),
    }
    for k in KS:
        take = min(k, len(ok))
        order = np.argsort(-(-np.abs(proxy - target)), kind="stable")[:take]
        pscore = -(np.abs(proxy - target) + q * spread * target)
        porder = np.argsort(-pscore, kind="stable")[:take]
        rec["top%d_proxy" % k] = float(in_band[order].mean())
        rec["top%d_margin" % k] = float(in_band[porder].mean())
        bscore = -(np.abs(proxy - target) + bar * target)
        border = np.argsort(-bscore, kind="stable")[:take]
        rec["top%d_dF" % k] = float(in_band[border].mean())
        rec["top%d_rand" % k] = float(in_band.mean())
        rec["top%d_n" % k] = int(take)
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, default=None,
                        help="margin constant; default re-fits q90(|log err|/spread) on stage B")
    args = parser.parse_args()

    with open(CALIB, encoding="utf-8") as handle:
        calib = json.load(handle)
    store, rows = load_rows()
    q_a = float(calib["q90_err_over_spread"])
    proxy = np.array([r["proxy"] for r in rows])
    spread = np.array([r["spread"] for r in rows])
    truth = np.array([r["truth"] for r in rows])
    err = np.abs(np.log(proxy / truth))
    bars = U.err90_from_bins(calib, np.array([r["dF"] for r in rows]))
    for row, bar in zip(rows, bars):
        row["err90_dF"] = float(bar)
    ratio = err / np.maximum(spread, 1e-12)
    q_b = float(np.quantile(ratio, 0.9))
    q = args.q or q_b
    print("rotors %d over %d batches | q90(|log err|/spread): stage A %.2f, stage B %.2f, using %.2f"
          % (len(rows), len(store), q_a, q_b, q))
    print("interval coverage: stage-A constant %.3f | stage-B constant %.3f | stage-A dF bar %.3f"
          % (float((err <= q_a * spread).mean()), float((err <= q_b * spread).mean()),
             float((err <= bars).mean())))

    grouped = {}
    for row in rows:
        grouped.setdefault("%s|%s" % (row["method"], row["spec"]), []).append(row)
    blocks = {key: batch_block(grouped[key], q) for key in sorted(grouped)
              if len(grouped[key]) >= 5}
    keys = sorted(blocks)

    print("\n1. how optimistic is the proxy, and does the margin make the claim honest?")
    print("   %-16s %-14s %6s %6s %8s %10s %10s %7s"
          % ("method", "spec", "say", "truth", "optimism", "prec_proxy", "prec_marg", "keep"))
    for key in keys:
        b = blocks[key]
        print("   %-16s %-14s %6.3f %6.3f %8.3f %10.3f %10.3f %7.3f"
              % (store[key]["method"], store[key]["spec"], b["proxy_say"], b["truth_in"],
                 b["optimism"], b["precision_proxy"], b["precision_margin"], b["margin_keep"]))
    print("   (say = share the proxy calls in band, truth = share that really is,")
    print("    prec_proxy / prec_marg = share truly in band among those called / kept)")

    methods = sorted(set(r["method"] for r in rows))

    def agg(field, sel):
        vals = [blocks[k][field] for k in sel]
        vals = [v for v in vals if v == v]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n   averaged over specs")
    print("   %-16s %6s %6s %8s %7s %7s %8s %8s %6s %8s"
          % ("method", "say", "truth", "optimism", "precraw", "precsp", "keepsp",
             "precdF", "keepdF", "recall"))
    by_method = {}
    fields = ("proxy_say", "truth_in", "optimism", "precision_proxy", "precision_margin",
              "margin_keep", "recall_margin", "MAPE", "MAPE_trim", "median_APE",
              "hit_true_5", "spearman_proxy", "spearman_margin", "spearman_dF",
              "spearman_spread", "n", "n_degenerate", "margin_keep_dF",
              "precision_dF", "recall_dF", "bar_median", "keep_and_band")
    for method in methods:
        sel = [k for k in keys if k.startswith(method + "|")]
        if not sel:
            continue
        by_method[method] = {f: agg(f, sel) for f in fields}
        b = by_method[method]
        print("   %-16s %6.3f %6.3f %8.3f %7.3f %7.3f %8.3f %8.3f %6.3f %7.3f %8.3f"
              % (method, b["proxy_say"], b["truth_in"], b["optimism"], b["precision_proxy"],
                 b["precision_margin"], b["margin_keep"], b["precision_dF"],
                 b["margin_keep_dF"], b["recall_dF"], b["MAPE_trim"]))

    print("\n2. fixed ROSS budget per batch: truth in band among the k sent")
    print("   %-5s %12s %12s %12s %12s" % ("k", "proxy pick", "spread pick", "dF pick",
                                            "random"))
    print("   %-5s %12s %12s %12s %12s" % ("", "(protocol)", "(pessimistic)",
                                            "(pessimistic)", "(baseline)"))
    budget = {}
    for k in KS:
        r = {"proxy": agg("top%d_proxy" % k, keys), "margin": agg("top%d_margin" % k, keys),
             "dF": agg("top%d_dF" % k, keys),
             "rand": agg("top%d_rand" % k, keys),
             "mean_take": agg("top%d_n" % k, keys)}
        budget[str(k)] = r
        print("   %-5d %12.3f %12.3f %12.3f %12.3f"
              % (k, r["proxy"], r["margin"], r["dF"], r["rand"]))

    print("\n3. within-batch ranking quality (spearman against the true cs1)")
    print("   %-16s %11s %11s %11s %11s" % ("method", "sp(proxy)", "sp(margin)", "sp(dF)",
                                             "sp(spread)"))
    ranking = {}
    for method in methods:
        sel = [k for k in keys if k.startswith(method + "|")]
        if not sel:
            continue
        ranking[method] = {f: agg(f, sel) for f in ("spearman_proxy", "spearman_margin",
                                                    "spearman_dF", "spearman_spread")}
        r = ranking[method]
        print("   %-16s %11.3f %11.3f %11.3f %11.3f"
              % (method, r["spearman_proxy"], r["spearman_margin"], r["spearman_dF"],
                 r["spearman_spread"]))

    print("\n   degenerate truths (cs1 <= %.0f rpm, a solver artefact rather than a design):"
          % DEGENERATE_RPM)
    for method in methods:
        if method in by_method and by_method[method]["n_degenerate"]:
            print("     %-16s %.2f per batch of %.0f" % (method, by_method[method]["n_degenerate"],
                                                         by_method[method]["n"]))

    S.save_json(OUT_JSON, {
        "q_stage_a": q_a, "q_stage_b": q_b, "q_used": q,
        "coverage_stage_a_const": float((err <= q_a * spread).mean()),
        "coverage_refit": float((err <= q_b * spread).mean()),
        "n_rotors": len(rows), "n_batches": len(blocks),
        "blocks": blocks, "by_method": by_method, "budget": budget, "ranking": ranking})
    print("\n[out] %s" % OUT_JSON)


if __name__ == "__main__":
    main()