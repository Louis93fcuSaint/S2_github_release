# -*- coding: utf-8 -*-
"""Freeze the environment, reconstructed split and source-file hashes."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eaai_common as A


def main():
    env = A.environment_snapshot()
    env["old_output_snapshot_before"] = A.old_output_snapshot()
    env["source_hashes"] = {
        name: A.sha256_file(path) for name, path in {
            "spec_common.py": os.path.join(A.V1_DIR, "spec_common.py"),
            "spec_opt.py": os.path.join(A.V1_DIR, "spec_opt.py"),
            "spec_interval.py": os.path.join(A.V2_DIR, "spec_interval.py"),
            "latent_design.py": os.path.join(A.V1_DIR, "latent_design.py"),
            "latent_torch.py": os.path.join(A.V1_DIR, "latent_torch.py"),
            "05_latent_ddpm.py": os.path.join(A.V1_DIR, "05_latent_ddpm.py"),
            "latent_ddpm_v2.py": os.path.join(A.V2_DIR, "latent_ddpm_v2.py"),
            "v2_ross_verify.py": os.path.join(A.V2_DIR, "v2_ross_verify.py"),
            "cvae_common.py": os.path.join(A.CVAE_DIR, "cvae_common.py"),
            "train_cvae.py": os.path.join(A.CVAE_DIR, "train_cvae.py"),
            "_ross_one_canon.py": os.path.join(A.V1_DIR, "_ross_one_canon.py"),
            "eaai_common.py": os.path.join(HERE, "eaai_common.py"),
        }.items()
    }
    payload = torch.load(A.DDPM_CKPT, map_location="cpu", weights_only=False)
    env["ddpm_v2i_config"] = payload["config"]
    env["ddpm_v2i_scalers"] = payload["scalers"]
    env["ddpm_v2i_predict"] = payload.get("predict")
    env["ddpm_v2i_condition_dropout"] = payload.get("condition_dropout")
    cmodel, cpayload = __import__("cvae_common").load_checkpoint(A.CVAE_CKPT)
    env["cvae_config"] = cpayload["config"]
    env["cvae_epoch"] = int(cpayload.get("epoch", -1))
    env["cvae_val_loss"] = float(cpayload.get("val_loss", float("nan")))
    env["cvae_uses_ema"] = bool(cpayload.get("ema_state_dict") is not None)

    ctx, train_idx, val_idx, split_meta = A.build_context_split(save=True)
    env["split"] = split_meta
    env["specs"] = A.load_calibration_specs()
    A.save_json(os.path.join(A.ENV_DIR, "environment.json"), env)

    # Verify the frozen spec hashes against the recorded SHA256SUMS.csv.
    sums_path = os.path.join(A.STUDY, "spec_candidates", "SHA256SUMS.csv")
    sums = pd.read_csv(sums_path)
    checks = []
    for _, row in sums.iterrows():
        name = str(row["file"])
        expected = str(row["sha256"])
        expected_bytes = int(row["bytes"])
        path = os.path.join(A.STUDY, "spec_candidates", name)
        actual = A.sha256_file(path) if os.path.exists(path) else "MISSING"
        actual_bytes = os.path.getsize(path) if os.path.exists(path) else -1
        checks.append({"file": name, "expected": expected, "actual": actual,
                       "expected_bytes": expected_bytes, "actual_bytes": actual_bytes,
                       "match": bool(expected == actual and expected_bytes == actual_bytes)})
    env["spec_hash_checks"] = checks
    A.save_json(os.path.join(A.ENV_DIR, "environment.json"), env)

    lines = [
        "# Environment and interface snapshot",
        "",
        "- Created: %s" % env["created"],
        "- Python: `%s`" % env["python"].splitlines()[0],
        "- Executable: `%s`" % env["python_executable"],
        "- Platform: `%s`" % env["platform"],
        "- Logical CPUs: %s" % env["cpu_count_logical"],
        "- PyTorch: `%s`" % env["torch"],
        "- NumPy: `%s`" % env["numpy"],
        "- pandas: `%s`" % env["pandas"],
        "- ROSS: `%s`" % env["ross"],
        "",
        "## Frozen data",
        "",
        "- Dataset: `%s`" % env["dataset"],
        "- Dataset SHA-256: `%s`" % env["dataset_sha256"],
        "- Rows: status=success %d, finite cs1-cs3 %d, dropped %d" % (
            env["split"]["pool_rows_status_success"], env["split"]["finite_rows"],
            env["split"]["dropped_missing_cs3"]),
        "- Reconstructed split: train %d / val %d (seed %d)" % (
            env["split"]["n_train"], env["split"]["n_val"], env["split"]["seed"]),
        "- Split status: %s" % env["split"]["status"],
        "",
        "## Checkpoints",
        "",
        "| Name | SHA-256 |",
        "|---|---|",
    ]
    for name, item in env["checkpoints"].items():
        lines.append("| %s | `%s` |" % (name, item["sha256"]))
    lines += [
        "",
        "## DDPM v2i",
        "",
        "```json",
        __import__("json").dumps({"config": env["ddpm_v2i_config"],
                                   "predict": env["ddpm_v2i_predict"],
                                   "condition_dropout": env["ddpm_v2i_condition_dropout"],
                                   "scalers": env["ddpm_v2i_scalers"]}, indent=2),
        "```",
        "",
        "## CVAE",
        "",
        "```json",
        __import__("json").dumps({"config": env["cvae_config"], "epoch": env["cvae_epoch"],
                                   "val_loss": env["cvae_val_loss"],
                                   "uses_ema": env["cvae_uses_ema"]}, indent=2),
        "```",
        "",
        "## Spec hash checks",
        "",
        "| File | Match |",
        "|---|---|",
    ]
    for row in checks:
        lines.append("| %s | %s |" % (row["file"], "OK" if row["match"] else "MISMATCH"))
    lines += [
        "",
        "## Import side effects",
        "",
        "`cvae_common.py` calls `os.makedirs` for the historical CVAE output directories at import time.",
        "The directories already existed. The pilot does not call `cvae_common.load_training_arrays` or any `main()`.",
        "`v2_ross_verify.py` writes only through `report()`; the pilot calls its `make_runner()` only.",
        "",
        "Historical output fingerprint before the pilot: `%s`" % env["old_output_snapshot_before"]["sha256"],
        "",
    ]
    with open(os.path.join(A.ENV_DIR, "environment_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("[env] wrote", os.path.join(A.ENV_DIR, "environment.json"))
    print("[env] split", len(train_idx), len(val_idx))
    print("[env] spec hashes all match:", all(c["match"] for c in checks))


if __name__ == "__main__":
    main()
