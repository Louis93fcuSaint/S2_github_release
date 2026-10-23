# -*- coding: utf-8 -*-
"""Normalise the DDPM set-objective reference results to the CVAE JSON schema.

The DDPM files copied from spec_design_v2 store the record of one batch under
``runs[0]``.  The CVAE pipeline writes the same fields at the top level.  This
script only re-arranges keys; no number is recomputed or altered.
"""
from __future__ import annotations

import glob
import json
import os

import cvae_common as C

SRC_VERIFY = os.path.join(C.HERE, "reference_ddpm", "verify")
DST_VERIFY = os.path.join(C.HERE, "reference_ddpm", "verify_norm")
os.makedirs(DST_VERIFY, exist_ok=True)

TOP_KEYS = ("tag", "orders", "bands", "tol", "n_shortlist", "n_solved", "n_timeout",
            "n_error", "seconds", "proxy", "truth", "spearman_joint_proxy_vs_true",
            "spearman_order1", "gate_tol5", "csv", "deliverable_tol",
            "deliverable_n", "deliverable")


def main():
    written = 0
    for path in sorted(glob.glob(os.path.join(SRC_VERIFY, "*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        runs = raw.get("runs") or []
        if not runs:
            print("[skip] no runs: %s" % path)
            continue
        source = dict(runs[0])
        out = {key: source.get(key) for key in TOP_KEYS if key in source}
        out["method"] = "ddpm"
        out["spec"] = os.path.basename(path)[len("ddpm_"):-len("_s42.json")]
        out["material"] = out["spec"].split("_")[0]
        out["seed"] = 42
        out["label"] = source.get("label", os.path.basename(path)[:-5])
        out["seconds_wall"] = source.get("seconds")
        out["source_json"] = os.path.abspath(path)
        out["normalised_from"] = "runs[0]"
        name = os.path.basename(path)
        with open(os.path.join(DST_VERIFY, name), "w", encoding="utf-8") as handle:
            json.dump(out, handle, ensure_ascii=False, indent=1)
        written += 1
        print("[ok] %s  deliverable_n=%s" % (name, out.get("deliverable_n")))
    print("[done] %d file(s) -> %s" % (written, DST_VERIFY))


if __name__ == "__main__":
    main()
