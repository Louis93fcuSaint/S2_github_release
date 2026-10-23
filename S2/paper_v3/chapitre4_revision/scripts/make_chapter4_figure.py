# -*- coding: utf-8 -*-
"""Build the Chapter 4 TST summary figure from frozen result tables."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PASCAL = ROOT.parents[0] / "pascal_revision"
OUT = ROOT / "figures"

SPECS = ["TST01", "TST02", "TST04", "TST07", "TST09"]
METHODS = ["ddpm", "cvae", "retrieval"]
LABELS = {"ddpm": "DDPM", "cvae": "CVAE", "retrieval": "Retrieval+jitter"}
COLORS = {"ddpm": "#2F5D8A", "cvae": "#D17A22", "retrieval": "#4C9A6B"}


def main():
    raw = pd.read_csv(PASCAL / "tst_runs" / "03_ross" / "tst_ross_pool_summary.csv")
    raw = raw[(raw["list_type"] == "random") & raw["spec"].isin(SPECS)].copy()
    raw["rate"] = raw["n_delivered"] / raw["denominator"]

    geo = pd.read_csv(ROOT / "tst_geometry_acceptance.csv")
    short = geo[geo["row_type"] == "specification"].copy()
    short["ross_rate"] = short["ross_pass"] / short["submitted"]
    short["joint_rate"] = short["ross_and_geometry_pass"] / short["submitted"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.8,
        "legend.fontsize": 7.9,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.75), sharey=True)
    x = np.arange(len(SPECS))
    width = 0.24

    panels = [
        ("raw", "(a) Uniform sample, n = 100", raw, "rate"),
        ("short", "(b) Shortlist, n = 200", short, "ross_rate"),
        ("joint", "(c) Joint dynamic-geometric, n = 200", short, "joint_rate"),
    ]
    for ax, (key, title, frame, col) in zip(axes, panels):
        for j, method in enumerate(METHODS):
            if key == "raw":
                rows = frame[frame["method"] == method].set_index("spec").reindex(SPECS)
            else:
                rows = frame[frame["method"] == method].set_index("specification").reindex(SPECS)
            vals = rows[col].to_numpy(dtype=float) * 100.0
            bars = ax.bar(x + (j - 1) * width, vals, width=width,
                          color=COLORS[method], edgecolor="white", linewidth=0.45,
                          label=LABELS[method])
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 2.0,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=5.5,
                        rotation=90, clip_on=False)
        ax.set_title(title, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(SPECS)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Acceptance rate (%)" if key == "raw" else "")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.89, bottom=0.22, wspace=0.10)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_ch4_tst.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_ch4_tst.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
