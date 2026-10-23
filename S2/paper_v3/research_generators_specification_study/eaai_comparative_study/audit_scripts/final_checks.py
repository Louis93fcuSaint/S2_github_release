# -*- coding: utf-8 -*-
"""Final integrity checks for the v1.1 pilot."""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eaai_common as A


def main():
    env = A.load_json(os.path.join(A.ENV_DIR, "environment.json"))
    now = A.old_output_snapshot()
    before = env["old_output_snapshot_before"]
    checks = {}

    checks["old_outputs_unchanged"] = (now["sha256"] == before["sha256"])
    checks["dataset_sha256_unchanged"] = (A.sha256_file(A.DATASET) == env["dataset_sha256"])
    checks["ddpm_v2i_sha256_unchanged"] = (A.sha256_file(A.DDPM_CKPT) ==
                                            env["checkpoints"]["interval_ddpm_v2i"]["sha256"])
    checks["cvae_sha256_unchanged"] = (A.sha256_file(A.CVAE_CKPT) ==
                                       env["checkpoints"]["cvae_main"]["sha256"])
    sums = pd.read_csv(os.path.join(A.STUDY, "spec_candidates", "SHA256SUMS.csv"))
    checks["spec_files_unchanged"] = all(
        A.sha256_file(os.path.join(A.STUDY, "spec_candidates", row["file"])) == row["sha256"]
        for _, row in sums.iterrows())

    manifest = pd.read_csv(os.path.join(A.STUDY, "pilot_run_manifest.csv"))
    checks["manifest_has_six_pools"] = (len(manifest) == 6)
    checks["only_cal_specs"] = set(manifest["spec"]) == {"CAL02", "CAL05"}
    checks["no_tst_spec"] = not any(str(v).startswith("TST") for v in manifest["spec"])

    short = pd.read_csv(os.path.join(A.STUDY, "pilot_shortlist_summary.csv"))
    checks["shortlist_denominator_100"] = bool((short["n_shortlist"] == 100).all())
    checks["random_denominator_100"] = bool((short["n_random"] == 100).all())

    ross = pd.read_csv(os.path.join(A.ROSS_DIR, "all_slots.csv"))
    checks["logical_slots_1200"] = (len(ross) == 1200)
    checks["no_ross_error_or_timeout"] = bool(
        ((ross["status"] == "error") | (ross["status"] == "timeout")).sum() == 0)
    checks["logical_cache_reuses_19"] = bool(int(ross["cache_hit"].sum()) == 19)

    required = [
        "protocol_v1_1.md", "protocol_changes_v1_1.md",
        "third_generator_evidence_correction.md",
        "pilot_run_manifest.csv", "pilot_pool_summary.csv",
        "pilot_shortlist_summary.csv", "pilot_valid_diversity.csv",
        "pilot_cost_breakdown.csv", "pilot_report.md",
    ]
    checks["required_files_present"] = all(
        os.path.exists(os.path.join(A.STUDY, f)) and os.path.getsize(os.path.join(A.STUDY, f)) > 0
        for f in required)

    lines = ["# Final integrity checks", "",
             "- Created: %s" % A.utc_now(), "",
             "| Check | Result |", "|---|---|"]
    for key, value in checks.items():
        lines.append("| %s | %s |" % (key, "PASS" if value else "FAIL"))
    lines += ["", "## Old output fingerprint", "",
              "- Before: `%s`" % before["sha256"],
              "- After: `%s`" % now["sha256"], ""]
    text = "\n".join(lines)
    with open(os.path.join(A.METRIC_DIR, "final_checks.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    if not all(checks.values()):
        raise SystemExit("final checks failed")


if __name__ == "__main__":
    main()
