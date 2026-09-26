# -*- coding: utf-8 -*-
"""Stage A -- how informative are the three signals where the truth is free?

Run on the surrogate's own held-out test split (39961 rotors the 5-seed net has
never seen).  Writes outputs/uncertainty_calibration.json, which stage B then
transfers to the generated submissions.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec_common as S
import spec_eval as E
import uncertainty_signals as U

OUT_JSON = os.path.join(E.OUT, "uncertainty_stageA.json")


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)),
                             np.argsort(np.argsort(b)))[0, 1])


def main():
    ctx = E.pool_ctx()
    sur = S.load_surrogate("mlp_best", threads=8)
    bank = U.SignalBank(ctx, sur)
    rows = bank.test_pool
    print("test rows %d (dropped %d, the one rejected_cs1 rotor)"
          % (len(rows), bank.n_dropped_test))

    sig = bank.score(ctx["X"][rows], ctx["designs"][rows])
    mu = sig["log_mean"]
    sd = sig["log_spread"]
    truth_rpm = ctx["Y"][rows, 0]
    err = np.abs(mu - np.log(np.maximum(truth_rpm, 1.0)))
    keep = np.isfinite(err) & np.isfinite(sd) & (truth_rpm > 100)
    err, sd, mu = err[keep], sd[keep], mu[keep]
    truth_rpm = truth_rpm[keep]
    dF, dD, box = sig["dF"][keep], sig["dD"][keep], sig["box"][keep]
    print("scored %d | median |log err| %.4f (= %.2f%%) | mean %.4f"
          % (len(err), np.median(err), (np.exp(np.median(err)) - 1) * 100, err.mean()))

    print("\nsignal -> |log err|")
    print("  %-8s %8s %10s %10s %10s %10s %10s"
          % ("signal", "median", "pearson", "spearman", "medianAPE", "p90APE", "over5%"))
    summary = {}
    for name, v in (("spread", sd), ("dF", dF), ("dD", dD), ("box", box)):
        summary[name] = {"pearson": float(np.corrcoef(np.log(v + 1e-12), err)[0, 1]),
                         "spearman": spearman(v, err)}
        print("  %-8s %8.4f %10.3f %10.3f %9.2f%% %9.2f%% %9.1f%%"
              % (name, np.median(v), summary[name]["pearson"], summary[name]["spearman"],
                 (np.exp(np.median(err)) - 1) * 100, (np.exp(np.quantile(err, 0.9)) - 1) * 100,
                 (np.exp(err) - 1 > 0.05).mean() * 100))

    calib = U.fit_calibration(err, sd, dF)
    print("\nnovelty-scaled error bar (q90 of |log err| by dF quintile)")
    for row in calib["dF_bins"]:
        print("  dF %5.2f..%5.2f  n=%6d  q90 %5.2f%%  median %5.2f%%"
              % (row["dF_lo"], row["dF_hi"], row["n"],
                 (np.exp(row["q90_log_err"]) - 1) * 100,
                 (np.exp(row["median_log_err"]) - 1) * 100))
    print("single-parameter form: q90(|log err| / spread) = %.2f   median = %.2f"
          % (calib["q90_err_over_spread"], calib["median_err_over_spread"]))

    A = np.column_stack([np.ones_like(err), np.log(sd + 1e-12), np.log(dF + 1e-12),
                         np.log(dD + 1e-12), box])
    target = np.log(err + 1e-12)
    tot = target.var()
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    resid = target - A @ coef
    print("\nlog|err| ~ 1 + log(s) + log(dF) + log(dD) + box : R2 %.3f" % (1 - resid.var() / tot))
    fits = {}
    for label, cols in (("spread", [0, 1]), ("dF", [0, 2]), ("dD", [0, 3]), ("box", [0, 4]),
                        ("spread+dF", [0, 1, 2]), ("all four", [0, 1, 2, 3, 4])):
        c, *_ = np.linalg.lstsq(A[:, cols], target, rcond=None)
        r = target - A[:, cols] @ c
        fits[label] = float(1 - r.var() / tot)
        print("  %-12s R2 %.3f" % (label, fits[label]))

    lower = S.DEFAULT_LOWER_RPM
    in_band = truth_rpm >= lower
    say_band = np.exp(mu) >= lower
    fp = say_band & ~in_band
    print("\nband screen cs1 >= %.0f : proxy yes %.4f | truth yes %.4f | FP %.4f"
          % (lower, say_band.mean(), in_band.mean(), fp.mean()))
    print("  of the rotors the proxy calls in-band, %.2f%% are not (n=%d)"
          % (fp.sum() / max(say_band.sum(), 1) * 100, int(say_band.sum())))
    conserv = say_band & (mu - calib["q90_err_over_spread"] * sd >= np.log(lower))
    print("  adding the spread-based 90%% margin: proxy yes %.4f | FP %.4f | recall of true in-band %.3f"
          % (conserv.mean(), (conserv & ~in_band).mean(),
             (conserv & in_band).sum() / max(in_band.sum(), 1)))

    np.savez_compressed(
        os.path.join(E.OUT, "uncertainty_stageA.npz"),
        spread=sd, dF=dF, dD=dD, box=box, err=err, mu=mu, truth_rpm=truth_rpm,
        rows=np.asarray(rows, dtype=np.int64))

    S.save_json(OUT_JSON, {"n_rows": int(len(err)), "signals": summary, "fits": fits,
                           "coef": [float(v) for v in coef], "r2_all": float(1 - resid.var() / tot),
                           "median_log_err": float(np.median(err)),
                           "band": {"lower": lower, "proxy_yes": float(say_band.mean()),
                                    "truth_yes": float(in_band.mean()),
                                    "false_positive": float(fp.mean()),
                                    "conservative_yes": float(conserv.mean()),
                                    "conservative_fp": float((conserv & ~in_band).mean())}})
    S.save_json(U.CALIB_JSON, calib)
    print("\n[out] %s\n[out] %s" % (OUT_JSON, U.CALIB_JSON))


if __name__ == "__main__":
    main()