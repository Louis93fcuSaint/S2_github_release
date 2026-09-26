"""Funnel check: does surrogate accuracy actually matter for the DDPM stage?

The pipeline is  DDPM generates candidates -> surrogate ranks them -> ROSS
simulates the top 50.  The surrogate never has to be *right*, it has to *rank*.
We replay that selection step on the frozen v4.7 test split, where the ground
truth is known, and ask the only question that matters:

    if the surrogate decides who gets simulated, how much of the attainable
    optimum actually survives the funnel?

Four surrogates spanning the accuracy ladder we built (7.18% -> 5.92% -> 5.50%
-> 4.44% MAPE) are compared, so the funnel's sensitivity to surrogate accuracy
becomes a number instead of an opinion.

Two families of objective are used because they stress different things:
  * max_cs1 / max_cs6  -- "push one critical speed as far as possible"
  * match_*            -- "hit a specified target cs vector" (the DDPM use case)
The match objectives come in an L2 (mean squared log error) and an L-inf
(worst-order log error) flavour; the L-inf one is the honest one for a spec,
because a design is only as good as its worst order.

Caveat that matters: the pool here is the LHS-sampled dataset distribution.
DDPM candidates may live elsewhere, so these numbers are an in-distribution
upper bound on how well the ranking can work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

SURROGATES = [
    ("A v4.7 baseline 256-128 3seed MSE 200ep",
     OUT / "mlp_v47" / "pred_multi.npz", OUT / "mlp_v47" / "summary_mlp.json"),
    ("B +L1 +capacity 1024-512-256 1seed",
     OUT / "tune" / "xl_1024" / "pred_multi.npz", OUT / "tune" / "xl_1024" / "summary_mlp.json"),
    ("C +log-inputs 1seed",
     OUT / "tune" / "login_big" / "pred_multi.npz", OUT / "tune" / "login_big" / "summary_mlp.json"),
    ("D +log-inputs 5seed ensemble",
     OUT / "mlp_v47_best" / "pred_multi.npz", OUT / "mlp_v47_best" / "summary_mlp.json"),
]

TOP_TRUE = 50          # ROSS budget: only the best 50 by surrogate get simulated
SELECT_K = 200         # surrogate shortlist size
RECALL_KS = [50, 100, 200, 500, 1000]
N_MATCH_TARGETS = 8


def load_pred(path):
    with np.load(str(path)) as handle:
        return handle["test_idx"], handle["y_test"], handle["raw_pred"]


def quote_mape(summary_path):
    try:
        with open(str(summary_path), "r", encoding="utf-8") as handle:
            return float(json.load(handle)["arms"]["multi"]["eval_raw"]["mean_MAPE_pct"])
    except Exception:
        return float("nan")


def l2_scores(y, p, target):
    """Negative mean squared log error against one target cs vector."""
    log_t = np.log(np.maximum(target, 1e-12))
    return (-np.mean((np.log(np.maximum(y, 1e-12)) - log_t) ** 2, axis=1),
            -np.mean((np.log(np.maximum(p, 1e-12)) - log_t) ** 2, axis=1))


def linf_scores(y, p, target):
    """Negative worst-order log error: a design is as good as its worst order."""
    log_t = np.log(np.maximum(target, 1e-12))
    dev_y = np.abs(np.log(np.maximum(y, 1e-12)) - log_t).max(axis=1)
    dev_p = np.abs(np.log(np.maximum(p, 1e-12)) - log_t).max(axis=1)
    return -dev_y, -dev_p, dev_y


def ranking_metrics(true_score, pred_score):
    n = true_score.size
    true_top = np.argsort(-true_score)[:TOP_TRUE]
    selected = np.argsort(-pred_score)[:SELECT_K]
    hit = np.intersect1d(selected, true_top).size

    best_selected = selected[np.argmax(true_score[selected])]
    rank_of_best = int(np.sum(true_score > true_score[best_selected])) + 1
    spread = true_score.max() - true_score.mean()
    captured = ((true_score[selected].max() - true_score.mean()) / spread
                if spread > 0 else float("nan"))

    out = {
        "recall@%d" % SELECT_K: hit / float(TOP_TRUE),
        "best_found_rank_in_pool": rank_of_best,
        "capture@%d" % SELECT_K: float(captured),
        "spearman": float(spearmanr(pred_score, true_score).correlation),
        "random_recall@%d" % SELECT_K: SELECT_K / float(n),
    }
    for k in RECALL_KS:
        chosen = np.argsort(-pred_score)[:k]
        out["recall@%d" % k] = np.intersect1d(chosen, true_top).size / float(TOP_TRUE)
    return out


def average_metrics(metrics_list):
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT / "rank_check"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = None
    report = {"select_k": SELECT_K, "top_true": TOP_TRUE, "surrogates": {}}
    curves = {}

    for label, pred_path, summary_path in SURROGATES:
        if not pred_path.exists():
            print("[skip] %s (missing %s)" % (label, pred_path))
            continue
        test_idx, y_test, pred = load_pred(pred_path)
        if reference is None:
            reference = (test_idx, y_test)
        else:
            assert np.array_equal(reference[0], test_idx), "split mismatch for %s" % label

        complete = np.isfinite(y_test).all(axis=1)
        y, p = y_test[complete], pred[complete]
        print("[%s] pool=%d rows (complete six-order labels)" % (label, y.shape[0]))

        entry = {"mape_pct": quote_mape(summary_path), "pool_rows": int(y.shape[0]),
                 "spearman_per_order": {}, "objectives": {}, "target_region": {}}

        for order in range(y.shape[1]):
            entry["spearman_per_order"]["cs_%d_rpm" % (order + 1)] = float(
                spearmanr(p[:, order], y[:, order]).correlation)

        entry["objectives"]["max_cs1"] = ranking_metrics(y[:, 0], p[:, 0])
        entry["objectives"]["max_cs6"] = ranking_metrics(y[:, 5], p[:, 5])

        rng = np.random.RandomState(0)
        picks = rng.choice(y.shape[0], size=N_MATCH_TARGETS, replace=False)
        for name, targets in (("match_l2_random_targets", [y[i] for i in picks]),
                              ("match_l2_median_target", [np.median(y, axis=0)]),
                              ("match_linf_random_targets", [y[i] for i in picks])):
            is_linf = "linf" in name
            metrics_list, spread_list = [], []
            for target in targets:
                if is_linf:
                    true_s, pred_s, dev = linf_scores(y, p, target)
                    spread_list.append(float(np.sort(dev)[TOP_TRUE - 1] * 100.0))
                else:
                    true_s, pred_s = l2_scores(y, p, target)
                metrics_list.append(ranking_metrics(true_s, pred_s))
            entry["objectives"][name] = average_metrics(metrics_list)
            if spread_list:
                # how tight the winning region is: |log| deviation of the true
                # top-50 under the L-inf objective, in per cent
                entry["target_region"]["linf_true_top50_dev_pct"] = float(np.mean(spread_list))

        report["surrogates"][label] = entry
        curves[label] = entry["objectives"]["match_l2_random_targets"]

    with open(str(out_dir / "rank_check.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    labels = [l for l, _, _ in SURROGATES if l in report["surrogates"]]

    print("\n=========== funnel: %d shortlisted, top %d simulated ===========" % (SELECT_K, TOP_TRUE))
    print("%-40s %6s %8s %10s %10s %9s" % ("surrogate", "MAPE%", "spear", "recall@200", "capture@200", "best#"))
    for label in labels:
        e = report["surrogates"][label]
        m = e["objectives"]["match_l2_random_targets"]
        print("%-40s %6.2f %8.4f %10.3f %10.3f %9d" % (
            label, e["mape_pct"], float(np.mean(list(e["spearman_per_order"].values()))),
            m["recall@200"], m["capture@200"], m["best_found_rank_in_pool"]))

    for name in ("match_l2_random_targets", "match_linf_random_targets", "max_cs1", "max_cs6"):
        print("\nobjective = %s" % name)
        print("%-40s %8s %8s %8s %8s %8s %10s" % (
            "surrogate", "K=50", "K=100", "K=200", "K=500", "K=1000", "best#"))
        for label in labels:
            m = report["surrogates"][label]["objectives"][name]
            print("%-40s %8.3f %8.3f %8.3f %8.3f %8.3f %10d" % (
                label, m["recall@50"], m["recall@100"], m["recall@200"],
                m["recall@500"], m["recall@1000"], m["best_found_rank_in_pool"]))

    print("\nper-order Spearman (rank fidelity per critical speed)")
    print("%-40s %8s %8s %8s %8s %8s %8s" % ("surrogate", "cs1", "cs2", "cs3", "cs4", "cs5", "cs6"))
    for label in labels:
        sp = report["surrogates"][label]["spearman_per_order"]
        print("%-40s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f" % tuple(
            [label] + [sp["cs_%d_rpm" % (i + 1)] for i in range(6)]))

    tight = [report["surrogates"][l]["target_region"].get("linf_true_top50_dev_pct")
             for l in labels]
    if tight and tight[0] is not None:
        print("\nwinning region tightness (L-inf objective): the whole true top-50 sits within +/- %.2f%% of the target, measured on each design's worst order" % tight[0])
    print("random-selection reference recall@200 = %.4f" %
          (SELECT_K / float(report["surrogates"][labels[0]]["pool_rows"])))
    print("[out] %s" % (out_dir / "rank_check.json"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir = OUT / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for label in labels:
            m = report["surrogates"][label]["objectives"]["match_l2_random_targets"]
            axes[0].plot(RECALL_KS, [m["recall@%d" % k] for k in RECALL_KS],
                         marker="o", label="%s  (MAPE %.2f%%)" % (label[:1], report["surrogates"][label]["mape_pct"]))
        axes[0].axhline(1.0, color="grey", ls=":", lw=1)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("shortlist size K offered to ROSS")
        axes[0].set_ylabel("recall@K of the true top-50")
        axes[0].set_title("Funnel: how many true top-50 survive the shortlist")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=7)

        xpos = np.arange(len(labels))
        axes[1].bar(xpos, [report["surrogates"][l]["objectives"]["match_l2_random_targets"]["recall@200"]
                           for l in labels], width=0.38, label="recall@200")
        axes[1].bar(xpos + 0.4, [float(np.mean(list(report["surrogates"][l]["spearman_per_order"].values())))
                                 for l in labels], width=0.38, label="mean Spearman")
        axes[1].set_xticks(xpos + 0.2)
        axes[1].set_xticklabels([l[:1] for l in labels])
        axes[1].set_ylim(0.90, 1.01)
        axes[1].set_title("Ranking quality vs surrogate accuracy")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3, axis="y")
        fig.tight_layout()
        out_png = fig_dir / "04_funnel_recall.png"
        fig.savefig(str(out_png), dpi=140)
        print("[fig] %s" % out_png)
    except Exception as exc:
        print("[fig] skipped: %s" % exc)


if __name__ == "__main__":
    main()
