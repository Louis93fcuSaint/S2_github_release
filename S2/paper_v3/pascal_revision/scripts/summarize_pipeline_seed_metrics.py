# -*- coding: utf-8 -*-
"""Summarise delivery and ROSS-valid diversity by pipeline and sampling seed."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLING = ROOT / "results_by_sampling_seed.csv"
DIVERSITY = ROOT / "ross_valid_diversity.csv"
OUT = ROOT / "results" / "pipeline_seed_summary.csv"


def main() -> None:
    runs = pd.read_csv(SAMPLING)
    runs = runs[runs["level"] == "run_spec_seed"].copy()
    if len(runs) != 72:
        raise SystemExit("expected 72 run/spec/seed rows, found %d" % len(runs))
    div = pd.read_csv(DIVERSITY)
    # The exported source CSV has a malformed denominator field. The frozen
    # protocol uses 200 submissions per specification; verify that invariant
    # rather than silently renormalising on a different column.
    if not runs["ross_submitted_n"].between(0, 200).all():
        raise SystemExit("unexpected number of ROSS submissions per run")
    runs["denominator"] = 200

    delivery = (
        runs.groupby(["method_label", "seed"])
        .agg(
            n_delivered=("delivered_n_json", "sum"),
            denominator=("denominator", "sum"),
            n_specs=("spec", "nunique"),
        )
        .reset_index()
    )
    delivery["delivery_rate_pct"] = 100 * delivery.n_delivered / delivery.denominator
    diversity = (
        div.groupby(["method_label", "seed"])
        .agg(
            family_entropy=("family_entropy", "mean"),
            median_nn_same_family=("median_nn_same_family", "mean"),
            median_pair_same_family=("median_pair_same_family", "mean"),
            novelty_median_to_reference=("novelty_median_to_reference", "mean"),
            n_specs=("spec", "nunique"),
        )
        .reset_index()
    )
    by_seed = delivery.merge(
        diversity, on=["method_label", "seed", "n_specs"], how="left"
    )
    by_seed.to_csv(OUT.with_name("pipeline_seed_metrics.csv"), index=False, encoding="utf-8-sig")

    metrics = [
        "delivery_rate_pct",
        "family_entropy",
        "median_nn_same_family",
        "median_pair_same_family",
        "novelty_median_to_reference",
    ]
    summary = (
        by_seed.groupby("method_label")[metrics]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(x) for x in col if x]).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(by_seed.round(3).to_string(index=False))
    print("\nwrote", OUT.with_name("pipeline_seed_metrics.csv"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
