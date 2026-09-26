# -*- coding: utf-8 -*-
"""Evaluate the two calibrations on the stored evidence.

Part 1  the honest bar.  A smooth, novelty-aware 0.9 quantile replaces the
        binned constant; the tau knob is exposed as a measured
        precision/recall curve instead of a hard-coded 2.94.
Part 2  the bias layer.  log truth = log proxy + g(signals), validated
        leave-one-batch-out, and scored on the metric that was still broken:
        the ranking inside one batch.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec_common as S
import spec_eval as E
import proxy_calibration as P

STAGE_A = os.path.join(E.OUT, "uncertainty_stageA.npz")
STORE = os.path.join(E.OUT, "uncertainty_stageB.json")
CENSUS = os.path.join(E.OUT, "cs1_census.json")
OUT_JSON = os.path.join(E.OUT, "proxy_calibration_report.json")
KS = [5, 10, 20]
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9]


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(a)),
                             np.argsort(np.argsort(b)))[0, 1])


def load_batches():
    with open(STORE, encoding="utf-8") as handle:
        store = json.load(handle)
    untrusted = set()
    if os.path.exists(CENSUS):
        with open(CENSUS, encoding="utf-8") as handle:
            census = json.load(handle)
        for item in census.get("stage_b", []):
            if item.get("trusted") is False:
                untrusted.add((item["key"], item["index"]))
    batches = {}
    for key, entry in store.items():
        keep = []
        for rotor in entry["rotors"]:
            if not rotor["truth"] or (key, rotor["index"]) in untrusted:
                continue
            row = dict(rotor)
            row["method"] = entry["method"]
            row["spec"] = entry["spec"]
            row["lower"] = entry["band"][0]
            row["upper"] = entry["band"][1]
            row["target"] = 0.5 * (entry["band"][0] + entry["band"][1])
            row["key"] = key
            keep.append(row)
        if keep:
            batches[key] = keep
    return batches, len(untrusted)


def flat(batches, keys=None):
    out = []
    for key in (keys if keys is not None else sorted(batches)):
        out.extend(batches[key])
    return out


def band_metrics(rows, factor):
    """factor is the multiplicative half-width on the rpm scale.

    Coverage is computed here, from the same rows and the same factor, so the
    log-space error and the rpm-space interval can never be compared by mistake.
    """
    proxy = np.array([r["proxy"] for r in rows])
    lower = np.array([r["lower"] for r in rows])
    upper = np.array([r["upper"] for r in rows])
    truth = np.array([r["truth"] for r in rows])
    in_band = (truth >= lower) & (truth <= upper)
    keep = (proxy / factor >= lower) & (proxy * factor <= upper)
    err = np.abs(np.log(proxy / truth))
    return {"keep": float(keep.mean()) if len(rows) else float("nan"),
            "precision": float(in_band[keep].mean()) if keep.any() else float("nan"),
            "recall": float(keep[in_band].mean()) if in_band.any() else float("nan"),
            "truth_in": float(in_band.mean()) if len(rows) else float("nan"),
            "coverage": float((err <= np.log(factor)).mean()) if len(rows) else float("nan")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=0.9)
    args = parser.parse_args()

    A = np.load(STAGE_A)
    batches, n_untrusted = load_batches()
    rows = flat(batches)
    keys = sorted(batches)
    print("stage A rows %d | stage B rows %d over %d batches | untrusted labels dropped %d"
          % (len(A["err"]), len(rows), len(keys), n_untrusted))
    err_a, spread_a, dF_a = A["err"], A["spread"], A["dF"]

    # ---------------- part 1: the honest bar ----------------
    print("\n1a. smooth bar fitted on the test split (tau = %.2f)" % args.tau)
    calib_a = P.fit_bar(spread_a, dF_a, err_a, tau=args.tau)
    print("    h = exp(%.3f + %.3f log(spread) + %.3f log(dF))   median h %.4f  coverage %.3f"
          % (calib_a["coef"][0], calib_a["coef"][1], calib_a["coef"][2],
             calib_a["median_half"], calib_a["coverage_fit"]))
    n = len(err_a)
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    fit_half, check_half = order[:n // 2], order[n // 2:]
    half = P.fit_bar(spread_a[fit_half], dF_a[fit_half], err_a[fit_half], tau=args.tau)
    print("    split-half on the test split: coverage on the unseen half %.3f"
          % P.coverage(err_a[check_half], P.bar_predict(half, spread_a[check_half], dF_a[check_half])))

    # the split has to interleave *within* each method: batches sort grouped by
    # method, so a plain even/odd split puts all of de/ga/retrieval in one half and
    # the calibration set no longer represents the validation set.
    by_method = {}
    for key in keys:
        by_method.setdefault(key.split("|")[0], []).append(key)
    even, odd = [], []
    for m_index, method in enumerate(sorted(by_method)):
        for i, key in enumerate(sorted(by_method[method])):
            (even if (i + m_index) % 2 == 0 else odd).append(key)
    cal_rows, val_rows = flat(batches, even), flat(batches, odd)
    print("    calibration batches %d (%d rotors) | held-out batches %d (%d rotors)"
          % (len(even), len(cal_rows), len(odd), len(val_rows)))

    def spread_dF(rows_):
        return (np.array([r["spread"] for r in rows_]), np.array([r["dF"] for r in rows_]))

    err_c = np.abs(np.log(np.array([r["proxy"] for r in cal_rows])
                          / np.array([r["truth"] for r in cal_rows])))
    err_v = np.abs(np.log(np.array([r["proxy"] for r in val_rows])
                          / np.array([r["truth"] for r in val_rows])))
    s_c, f_c = spread_dF(cal_rows)
    s_v, f_v = spread_dF(val_rows)

    q_in = 2.04
    q_refit = float(np.quantile(err_c / s_c, args.tau))
    bar_a = P.fit_bar(spread_a, dF_a, err_a, tau=args.tau)
    bar_refit = P.fit_bar(s_c, f_c, err_c, tau=args.tau)
    print("    refit bar h = exp(%.3f + %.3f log(spread) + %.3f log(dF))  median h %.4f  coverage %.3f"
          % (bar_refit["coef"][0], bar_refit["coef"][1], bar_refit["coef"][2],
             bar_refit["median_half"], bar_refit["coverage_fit"]))
    variants = {
        "constant %.2f (test split)" % q_in:
            (np.exp(q_in * s_v), None),
        "constant refit %.2f" % q_refit:
            (np.exp(q_refit * s_v), None),
        "smooth bar (test split)":
            (P.bar_factor(bar_a, s_v, f_v), (bar_a, bar_refit)),
        "smooth bar refit":
            (P.bar_factor(bar_refit, s_v, f_v), None),
        "smooth bar test+refit":
            (np.maximum(P.bar_factor(bar_a, s_v, f_v), P.bar_factor(bar_refit, s_v, f_v)),
             None),
    }
    print("\n1b. on the held-out batches: coverage of the interval, and what the screen keeps")
    print("    %-30s %8s %8s %9s %9s" % ("rule", "cover", "keep", "prec", "recall"))
    order_report = []
    for name, (half_v, _) in variants.items():
        met = band_metrics(val_rows, half_v)
        print("    %-30s %8.3f %8.3f %9.3f %9.3f"
              % (name, met["coverage"], met["keep"], met["precision"], met["recall"]))
        order_report.append((name, met, met["coverage"]))

    chosen = P.bar_predict(bar_refit, s_v, f_v)
    print("\n1c. per method under the re-fitted smooth bar (held-out batches)")
    print("    %-16s %7s %7s %9s %9s" % ("method", "keep", "recall", "prec", "truth"))
    per_method = {}
    for method in sorted(set(r["method"] for r in val_rows)):
        sel = [r for r in val_rows if r["method"] == method]
        half_m = P.bar_factor(bar_refit, np.array([r["spread"] for r in sel]),
                              np.array([r["dF"] for r in sel]))
        met = band_metrics(sel, half_m)
        per_method[method] = met
        print("    %-16s %7.3f %7.3f %9.3f %9.3f"
              % (method, met["keep"], met["recall"], met["precision"], met["truth_in"]))

    print("\n1d. the tau knob is the conservative/aggressive dial (held-out batches)")
    print("    %-6s %8s %8s %9s %9s %9s" % ("tau", "cover", "keep", "prec", "recall",
                                             "keep_retr"))
    tau_curve = {}
    retr = [r for r in val_rows if r["method"] == "retrieval"]
    for tau in TAUS:
        calib_t = P.fit_bar(s_c, f_c, err_c, tau=tau)
        half_t = P.bar_factor(calib_t, s_v, f_v)
        met = band_metrics(val_rows, half_t)
        half_r = P.bar_factor(calib_t, np.array([r["spread"] for r in retr]),
                              np.array([r["dF"] for r in retr]))
        met_r = band_metrics(retr, half_r)
        tau_curve[str(tau)] = {"coverage": float(met["coverage"]),
                               "keep": met["keep"], "precision": met["precision"],
                               "recall": met["recall"], "keep_retrieval": met_r["keep"],
                               "precision_retrieval": met_r["precision"],
                               "coef": calib_t["coef"]}
        print("    %-6.2f %8.3f %8.3f %9.3f %9.3f %9.3f"
              % (tau, tau_curve[str(tau)]["coverage"], met["keep"], met["precision"],
                 met["recall"], met_r["keep"]))

    # ---------------- part 2: the bias layer ----------------
    print("\n2a. bias layer, leave-one-batch-out")
    global_fit = P.fit_bias(rows)
    print("    in-sample R2 %.3f rmse %.3f | coef: %s"
          % (global_fit["r2"], global_fit["rmse"],
             " ".join("%s=%.3f" % (nm, b) for nm, b in zip(P.BIAS_NAMES, global_fit["coef"]))))
    corrected = np.full(len(rows), np.nan)
    index_of = {id(r): i for i, r in enumerate(rows)}
    for held in keys:
        train = [r for r in rows if r["key"] != held]
        calib = P.fit_bias(train)
        for r in batches[held]:
            corrected[index_of[id(r)]] = float(P.bias_correct(calib, [r])[0])
    resid_model = np.array([np.log(r["truth"] / r["proxy"]) for r in rows])
    pred_resid = np.log(corrected / np.array([r["proxy"] for r in rows]))
    r2_out = 1.0 - np.var(resid_model - pred_resid) / np.var(resid_model)
    print("    leave-one-batch-out R2 of the residual model %.3f" % r2_out)

    def mape(values):
        truth = np.array([r["truth"] for r in rows])
        return float(np.mean(np.abs(values - truth) / truth) * 100)

    raw = np.array([r["proxy"] for r in rows])
    print("    MAPE raw %.2f%% -> corrected %.2f%% | median APE raw %.2f%% -> corrected %.2f%%"
          % (mape(raw), mape(corrected),
             float(np.median(np.abs(raw - np.array([r["truth"] for r in rows]))
                             / np.array([r["truth"] for r in rows])) * 100),
             float(np.median(np.abs(corrected - np.array([r["truth"] for r in rows]))
                             / np.array([r["truth"] for r in rows])) * 100)))

    print("\n2b. within-batch ranking (spearman against the true cs1, mean over batches)")
    print("    %-16s %11s %11s %11s" % ("method", "sp(proxy)", "sp(corrected)", "delta"))
    ranking = {}
    for method in sorted(set(r["method"] for r in rows)):
        raw_s, cor_s = [], []
        for key in keys:
            if not key.startswith(method + "|"):
                continue
            sel = batches[key]
            truth = np.array([r["truth"] for r in sel])
            raw_s.append(spearman(np.array([r["proxy"] for r in sel]), truth))
            cor_s.append(spearman(np.array([corrected[index_of[id(r)]] for r in sel]), truth))
        if not raw_s:
            continue
        ranking[method] = {"sp_proxy": float(np.mean(raw_s)), "sp_corrected": float(np.mean(cor_s)),
                           "delta": float(np.mean(cor_s) - np.mean(raw_s))}
        print("    %-16s %11.3f %11.3f %+11.3f"
              % (method, ranking[method]["sp_proxy"], ranking[method]["sp_corrected"],
                 ranking[method]["delta"]))

    print("\n2c. fixed ROSS budget, truth in band among the k sent (held-out batches only)")
    print("    %-5s %12s %12s %12s" % ("k", "proxy pick", "corrected", "screen"))
    screen_half = np.exp(q_refit * np.array([r["spread"] for r in rows]))
    budget = {}
    val_keys = [k for k in keys if k in odd]
    for k in KS:
        acc = {"proxy": [], "corrected": [], "screen": []}
        for key in val_keys:
            sel = batches[key]
            truth = np.array([r["truth"] for r in sel])
            lower = np.array([r["lower"] for r in sel])
            upper = np.array([r["upper"] for r in sel])
            target = np.array([r["target"] for r in sel])
            in_band = (truth >= lower) & (truth <= upper)
            take = min(k, len(sel))
            idx = np.arange(len(sel))
            cand = {
                "proxy": np.argsort(-(-np.abs(np.array([r["proxy"] for r in sel]) - target)))[:take],
                "corrected": np.argsort(-(-np.abs(np.array([corrected[index_of[id(r)]] for r in sel])
                                                  - target)))[:take],
            }
            spread_sel = np.array([r["spread"] for r in sel])
            half_sel = P.bar_factor(bar_refit, spread_sel, np.array([r["dF"] for r in sel]))
            score = -np.abs(np.array([corrected[index_of[id(r)]] for r in sel]) - target) \
                - half_sel * target
            cand["screen"] = np.argsort(-score)[:take]
            for name, pick in cand.items():
                acc[name].append(float(in_band[pick].mean()))
        budget[str(k)] = {name: float(np.mean(v)) for name, v in acc.items()}
        print("    %-5d %12.3f %12.3f %12.3f"
              % (k, budget[str(k)]["proxy"], budget[str(k)]["corrected"], budget[str(k)]["screen"]))

    print("\n2d. gated correction: fit and apply the bias layer only where the batch is")
    print("    out of the manifold (mean novelty or mean spread above the test split p90)")
    gate_dF = float(np.quantile(dF_a, 0.90))
    gate_spread = float(np.quantile(spread_a, 0.90))
    print("    gates: dF > %.2f  or  spread > %.4f" % (gate_dF, gate_spread))
    gated = {}
    for key, sel in batches.items():
        gated[key] = bool(np.mean([r["dF"] for r in sel]) > gate_dF
                          or np.mean([r["spread"] for r in sel]) > gate_spread)
    for key in keys:
        print("      %-24s mean dF %5.2f  mean spread %.4f  %s"
              % (key, float(np.mean([r["dF"] for r in batches[key]])),
                 float(np.mean([r["spread"] for r in batches[key]])),
                 "GATED" if gated[key] else "-"))
    gate_rows = [r for r in rows if gated[r["key"]]]
    corrected_g = np.full(len(rows), np.nan)
    for held in [k for k in keys if gated[k]]:
        train = [r for r in gate_rows if r["key"] != held]
        calib_g = P.fit_bias(train)
        for r in batches[held]:
            corrected_g[index_of[id(r)]] = float(P.bias_correct(calib_g, [r])[0])
    pick = np.where(np.isfinite(corrected_g), corrected_g, raw)
    print("    gated batches %d of %d | rotors %d" % (sum(gated.values()), len(keys), len(gate_rows)))

    print("\n2e. within-batch ranking, four rankers")
    print("    proxy  = |proxy - target|, the protocol's own pick")
    print("    lcb    = proxy / bar, the pessimistic end of the honest interval")
    print("    screen = |proxy - target| + bar * target, closeness penalised by uncertainty")
    print("    %-16s %10s %10s %10s %10s" % ("method", "sp(proxy)", "sp(lcb)", "sp(screen)",
                                              "sp(gated)"))
    ranking_gated = {}
    for method in sorted(set(r["method"] for r in rows)):
        a, b, c, d = [], [], [], []
        for key, sel in batches.items():
            if not key.startswith(method + "|"):
                continue
            truth = np.array([r["truth"] for r in sel])
            target = np.array([r["target"] for r in sel])
            proxy_sel = np.array([r["proxy"] for r in sel])
            half = P.bar_factor(bar_refit, np.array([r["spread"] for r in sel]),
                                np.array([r["dF"] for r in sel]))
            a.append(spearman(proxy_sel, truth))
            b.append(spearman(proxy_sel / half, truth))
            c.append(spearman(-np.abs(proxy_sel - target) - half * target, truth))
            d.append(spearman(np.array([pick[index_of[id(r)]] for r in sel]), truth))
        ranking_gated[method] = {"sp_proxy": float(np.mean(a)), "sp_lcb": float(np.mean(b)),
                                 "sp_screen": float(np.mean(c)), "sp_gated": float(np.mean(d)),
                                 "gated": float(np.mean([gated[k] for k in keys
                                                         if k.startswith(method + "|")]))}
        print("    %-16s %10.3f %10.3f %10.3f %10.3f"
              % (method, ranking_gated[method]["sp_proxy"], ranking_gated[method]["sp_lcb"],
                 ranking_gated[method]["sp_screen"], ranking_gated[method]["sp_gated"]))
    for field in ("sp_proxy", "sp_lcb", "sp_screen", "sp_gated"):
        print("    overall mean %-10s %.3f" % (field, float(np.mean(
            [v[field] for v in ranking_gated.values()]))))

    truth_all = np.array([r["truth"] for r in rows])
    print("    accuracy on gated batches: median APE proxy %.2f%% -> gated %.2f%%"
          % (float(np.median(np.abs(raw - truth_all)[np.isfinite(corrected_g)] / truth_all[np.isfinite(corrected_g)]) * 100),
             float(np.median(np.abs(corrected_g - truth_all)[np.isfinite(corrected_g)] / truth_all[np.isfinite(corrected_g)]) * 100)))

    print("\n2f. fixed ROSS budget with the gated ranker (held-out batches only)")
    print("    %-5s %12s %12s %12s" % ("k", "proxy pick", "gated pick", "screen"))
    budget_gated = {}
    for k in KS:
        acc = {"proxy": [], "gated": [], "screen": []}
        for key in val_keys:
            sel = batches[key]
            truth = np.array([r["truth"] for r in sel])
            lower = np.array([r["lower"] for r in sel])
            upper = np.array([r["upper"] for r in sel])
            target = np.array([r["target"] for r in sel])
            in_band = (truth >= lower) & (truth <= upper)
            take = min(k, len(sel))
            score_raw = -np.abs(np.array([r["proxy"] for r in sel]) - target)
            score_gated = -np.abs(np.array([pick[index_of[id(r)]] for r in sel]) - target)
            half_sel = P.bar_factor(bar_refit, np.array([r["spread"] for r in sel]),
                                    np.array([r["dF"] for r in sel]))
            score_screen = score_gated - half_sel * target
            acc["proxy"].append(float(in_band[np.argsort(-score_raw)[:take]].mean()))
            acc["gated"].append(float(in_band[np.argsort(-score_gated)[:take]].mean()))
            acc["screen"].append(float(in_band[np.argsort(-score_screen)[:take]].mean()))
        budget_gated[str(k)] = {name: float(np.mean(v)) for name, v in acc.items()}
        for name in acc:
            budget_gated[str(k)]["paired_%s_minus_proxy" % name] = [
                float(a - b) for a, b in zip(acc[name], acc["proxy"])]
        print("    %-5d %12.3f %12.3f %12.3f"
              % (k, budget_gated[str(k)]["proxy"], budget_gated[str(k)]["gated"],
                 budget_gated[str(k)]["screen"]))
    print("    paired over the %d held-out batches (mean +/- se, t):" % len(val_keys))
    for k in KS:
        for name in ("screen", "gated"):
            d = np.array(budget_gated[str(k)]["paired_%s_minus_proxy" % name])
            se = d.std(ddof=1) / np.sqrt(len(d))
            print("      k=%-3d %-7s - proxy  %+.4f +/- %.4f  t %+.2f"
                  % (k, name, d.mean(), se, d.mean() / se if se else float("nan")))

    S.save_json(OUT_JSON, {
        "n_untrusted_dropped": n_untrusted, "tau_used": args.tau,
        "bar_test_split": bar_a, "bar_refit": bar_refit, "q_refit": q_refit,
        "held_out_variants": {name: {"keep": met["keep"], "precision": met["precision"],
                                     "recall": met["recall"], "coverage": cov}
                              for name, met, cov in order_report},
        "per_method_screen": per_method, "tau_curve": tau_curve,
        "bias": {"global": global_fit, "lobo_r2": float(r2_out),
                 "MAPE_raw": mape(raw), "MAPE_corrected": mape(corrected)},
        "ranking": ranking, "budget": budget,
        "gate": {"dF": gate_dF, "spread": gate_spread,
                 "gated_batches": [k for k in keys if gated[k]]},
        "ranking_gated": ranking_gated, "budget_gated": budget_gated})
    print("\n[out] %s" % OUT_JSON)


if __name__ == "__main__":
    main()