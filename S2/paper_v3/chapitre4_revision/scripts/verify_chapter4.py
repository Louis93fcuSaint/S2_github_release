# -*- coding: utf-8 -*-
"""Recompute the numbers quoted in Chapter 4 and check them against sources.

Run from anywhere:
    python scripts/verify_chapter4.py

The script is read-only. It reads the chapter sources, the two statistics
produced for this revision, and the archived validation and case-C records.
It prints PASS/FAIL lines and exits non-zero if any check fails.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent
SRC = PAPER / "pascal_revision"

TEX = ROOT / "chapter4_revised.tex"
ZH = ROOT / "chapter4_revised_zh.md"
GEOM = ROOT / "tst_geometry_acceptance.csv"
DIV = ROOT / "accepted_set_diversity.csv"
CV_MD = SRC / "final_surrogate_validation.md"
CASE_C = SRC / "case_C_audit.md"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(label)


def pct(num: float, den: float) -> float:
    return 100.0 * num / den


# ---------------------------------------------------------------- geometry --
geom = pd.read_csv(GEOM)
spec = geom[geom["row_type"] == "specification"]
totals = spec.groupby("method")[["ross_pass", "geometry_pass", "ross_and_geometry_pass"]].sum()

expected = {
    "ddpm": (592, 336, 196),
    "cvae": (678, 342, 235),
    "retrieval": (832, 323, 270),
}
for method, (ross, edge, joint) in expected.items():
    row = totals.loc[method]
    check(f"geometry totals {method}",
          (row.ross_pass, row.geometry_pass, row.ross_and_geometry_pass) == (ross, edge, joint),
          f"{row.ross_pass}/{row.geometry_pass}/{row.ross_and_geometry_pass}")

check("joint macro rates 19.6/23.5/27.0",
      all(round(pct(totals.loc[m, "ross_and_geometry_pass"], 1000), 1) == v
          for m, v in (("ddpm", 19.6), ("cvae", 23.5), ("retrieval", 27.0))))
check("edge macro rates 33.6/34.2/32.3",
      all(round(pct(totals.loc[m, "geometry_pass"], 1000), 1) == v
          for m, v in (("ddpm", 33.6), ("cvae", 34.2), ("retrieval", 32.3))))
check("geometry retention given ROSS 33.1/34.7/32.5",
      all(round(pct(totals.loc[m, "ross_and_geometry_pass"], totals.loc[m, "ross_pass"]), 1) == v
          for m, v in (("ddpm", 33.1), ("cvae", 34.7), ("retrieval", 32.5))))
check("all submitted shortlists are 200 per cell",
      (spec["submitted"] == 200).all(), f"cells={len(spec)}")

quoted = {
    ("TST01", "ddpm"): (192, 67, 64),
    ("TST02", "cvae"): (189, 89, 80),
    ("TST04", "retrieval"): (160, 16, 12),
    ("TST07", "cvae"): (16, 72, 5),
    ("TST09", "ddpm"): (136, 77, 46),
}
for (s, m), values in quoted.items():
    row = spec[(spec.specification == s) & (spec.method == m)].iloc[0]
    check(f"geometry cell {s}/{m}",
          (row.ross_pass, row.geometry_pass, row.ross_and_geometry_pass) == values,
          f"{row.ross_pass}/{row.geometry_pass}/{row.ross_and_geometry_pass}")

# --------------------------------------------------------------- diversity --
div = pd.read_csv(DIV)


def drow(s: str, m: str) -> pd.Series:
    return div[(div.specification == s) & (div.method == m)].iloc[0]


dchecks = [
    ("TST01 ddpm families", drow("TST01", "ddpm").n_families, 14),
    ("TST01 ddpm entropy", round(drow("TST01", "ddpm").family_entropy, 3), 2.472),
    ("TST01 ddpm within-family NN", round(drow("TST01", "ddpm").median_nn_same_family_active, 3), 3.771),
    ("TST04 retrieval within-family NN", round(drow("TST04", "retrieval").median_nn_same_family_active, 3), 0.462),
    ("TST04 ddpm within-family NN", round(drow("TST04", "ddpm").median_nn_same_family_active, 3), 3.507),
    ("TST07 retrieval accepted", drow("TST07", "retrieval").n_accepted, 63),
    ("TST07 retrieval dominant share", round(drow("TST07", "retrieval").dominant_family_share, 3), 0.444),
    ("TST09 cvae entropy", round(drow("TST09", "cvae").family_entropy, 3), 2.520),
]
for label, got, want in dchecks:
    check(label, got == want, f"{got} vs {want}")

check("TST07 has no common-sample distance",
      pd.isna(drow("TST07", "cvae").median_nn_same_family_active))

# ------------------------------------------------------ surrogate five-fold --
cv_text = CV_MD.read_text(encoding="utf-8")
for value in ("6.044", "4.579", "2.539", "10.437", "11.857", "0.177"):
    check(f"CV source reports {value}", value in cv_text)

deg = pd.read_csv(SRC / "results" / "surrogate_cv_final" / "final_surrogate_cv_degenerate_excluded.csv")
fold_mean = deg["ensemble_mean_MAPE_pct"].mean()
check("adjusted fold mean reproduces 4.579", round(fold_mean, 3) == 4.579, f"{fold_mean:.4f}")
order_means = [round(deg[f"MAPE_cs_{i}_rpm"].mean(), 3) for i in range(1, 7)]
check("adjusted per-order means 2.977/4.052/4.377/4.886/5.412/5.773",
      order_means == [2.977, 4.052, 4.377, 4.886, 5.412, 5.773], str(order_means))

# Fold-3 degenerate labels dominate the APE sum, not the absolute-error sum.
npz = SRC / "results" / "surrogate_cv_final" / "fold3" / "pred_ensemble.npz"
try:
    import numpy as np

    z = np.load(npz, allow_pickle=True)
    yt, yp = z["y_true"], z["y_pred"]
    ape = np.abs((yp - yt) / yt)
    mask = yt[:, 0] < 100
    share = ape[mask, 0].sum() / ape[:, 0].sum()
    check("two degenerate labels account for 94.0% of fold-3 cs1 APE",
          mask.sum() == 2 and round(100 * share, 1) == 94.0, f"{100 * share:.2f}%")
    abs_share = np.abs(yp[mask, 0] - yt[mask, 0]).sum() / np.abs(yp[:, 0] - yt[:, 0]).sum()
    check("degenerate labels are not 94% of absolute error",
          abs_share < 0.01, f"{100 * abs_share:.2f}% of absolute error")
except ImportError:
    print("SKIP  fold-3 APE share (numpy unavailable)")

# ------------------------------------------------------------------ case C --
case_text = CASE_C.read_text(encoding="utf-8")
for value in ("54.538", "3.418", "13.418", "1576.93", "4156.59", "7607.86"):
    check(f"case-C source reports {value}", value in case_text)

# ------------------------------------------------------- chapter text audit --
tex = TEX.read_text(encoding="utf-8")
must_appear = [
    "6.044\\%", "4.579\\%", "2.539\\%", "10.436\\%",
    "59.2\\% (592/1{,}000)", "67.8\\% (678/1{,}000)", "83.2\\% (832/1{,}000)",
    "14.0\\% (70/500)", "31.4\\% (157/500)", "59.8\\% (299/500)",
    "19.6\\% (196/1{,}000)", "23.5\\% (235/1{,}000)", "27.0\\% (270/1{,}000)",
    "4{,}277", "165", "94.0\\%",
]
for token in must_appear:
    check(f"chapter contains {token}", token in tex)

check("sec:cost label present", "\\label{sec:cost}" in tex)
check("no top-200 wording", "top-200" not in tex.lower())
check("H4 not called a full assembly check",
      "not a complete assembly" in tex or "not a complete assembly check" in tex)

# ------------------------------------------------------------- word counts --
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def normalize_latex(s: str) -> str:
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    return s


old_raw = "\n\n".join((SRC / "manuscript" / f).read_text(encoding="utf-8")
                      for f in ("sec04a.tex", "sec04b.tex"))
old_words = len(WORD.findall(normalize_latex(old_raw)))
new_words = len(WORD.findall(normalize_latex(tex)))
ratio = 100 * new_words / old_words
print(f"\nEnglish alphabetic words (tables included): old={old_words}, new={new_words}, "
      f"ratio={ratio:.1f}%")
check("compression within 55-65%", 55.0 <= ratio <= 65.0, f"{ratio:.1f}%")

zh = ZH.read_text(encoding="utf-8")
cjk = len(re.findall(r"[\u4e00-\u9fff]", zh))
print(f"Chinese CJK characters: {cjk}")

print()
if failures:
    print(f"{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("All checks passed.")
