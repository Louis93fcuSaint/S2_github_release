# -*- coding: utf-8 -*-
"""Read-only asset inventory for the EAAI comparative study.

The script records path, role, size, SHA-256 and a reuse verdict.  It does not
open model weights beyond hashing the file bytes, and it never writes outside
the new study directory.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
S2 = Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2")
S1 = Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S1")
STUDY = S2 / "paper" / "paper_v3" / "research_generators_specification_study"


def asset(category, name, path, purpose, reuse, note=""):
    return {"category": category, "name": name, "path": str(path),
            "purpose": purpose, "reuse_status": reuse, "note": note}


ASSETS = [
    asset("data", "v4.7 reference dataset",
          S1 / "04_versions/v4.7/dataset_v4.7/dataset_v4.7_all.csv",
          "ROSS labels, material/family metadata, support counting",
          "read-only core asset",
          "199,815 rows; status=success 199,811; 3 rows missing cs3; cs labels in rpm"),
    asset("data", "surrogate split indices",
          S2 / "surrogate_v47/outputs/split_v47.npz",
          "canonical 80/20 stratified train/test split for the surrogate",
          "reusable if protocol matches",
          "saved indices; stratified by source|n_disks|n_bearings, seed 42"),
    asset("surrogate", "surrogate run script", S2 / "surrogate_v47/run_mlp.py",
          "feature construction, fit/validation/test split, MLP training",
          "read-only reference", "validation split is carved from train only"),
    asset("surrogate", "surrogate common utilities", S2 / "surrogate_v47/common.py",
          "47-feature construction, metrics, split persistence", "read-only reference",
          "MAPE is mean APE over rows with finite positive labels"),
    asset("surrogate", "surrogate report", S2 / "surrogate_v47/代理模型结果_v4.7.md",
          "reported 4.44% test MAPE and per-order errors", "citation source",
          "test-set protocol; model selection evidence must be checked separately"),
    asset("surrogate", "surrogate 5-seed ensemble",
          S2 / "surrogate_v47/outputs/mlp_v47_best/models",
          "frozen 47->1024->512->256 multi-output surrogate, five seeds",
          "reuse for screening",
          "log-space geometric mean; seed files multi_seed42..46.pt"),
    asset("surrogate", "surrogate metrics", S2 / "surrogate_v47/outputs/mlp_v47_best/metrics_multi.json",
          "per-order and pooled test metrics", "citation source", "frozen evaluation output"),
    asset("ddpm", "point-conditioned DDPM checkpoint",
          S2 / "spec_design_v2/outputs/v2m/model.pt",
          "18-D point-conditioned latent DDPM", "reuse only for point-spec experiments",
          "independent checkpoint; do not merge with interval model"),
    asset("ddpm", "point-conditioned DDPM log",
          S2 / "spec_design_v2/outputs/v2m/training_log.json",
          "training configuration and loss history", "citation source",
          "200 epochs, 189,818 train / 9,990 val, hidden 512x3, 1,188 s"),
    asset("ddpm", "interval-conditioned DDPM checkpoint",
          S2 / "spec_design_v2/outputs/v2i/model.pt",
          "21-D interval-conditioned latent DDPM", "reuse for the comparative study",
          "independent checkpoint; supports point, two-sided and lower-bounded one-sided bands"),
    asset("ddpm", "interval-conditioned DDPM log",
          S2 / "spec_design_v2/outputs/v2i/training_log.json",
          "training configuration and loss history", "citation source",
          "200 epochs, 189,818 train / 9,990 val, hidden 512x3, 1,190 s"),
    asset("ddpm", "DDPM training/sampling code", S2 / "spec_design_v2/latent_ddpm_v2.py",
          "condition construction, v-prediction denoiser, guidance", "read-only reference",
          "T_STEPS=1000, cosine schedule, 3x512 SiLU MLP, time_dim 64"),
    asset("representation", "latent encoder/decoder", S2 / "spec_design/latent_design.py",
          "34-D latent <-> 33-D canonical design; H1-H6 by construction", "reuse",
          "H4 is axial centre distance >=10 mm only"),
    asset("representation", "torch decoder", S2 / "spec_design/latent_torch.py",
          "differentiable decoder used by guidance", "reuse", "no neural decoder"),
    asset("ddpm", "v1 DDPM primitives", S2 / "spec_design/05_latent_ddpm.py",
          "denoiser, reverse chain, v-prediction recovery", "reuse", "T_STEPS=1000"),
    asset("cvae", "CVAE checkpoint", S2 / "CVAE_baseline/outputs/cvae_main/model_best.pt",
          "21-D conditional VAE baseline", "reuse for the comparative study",
          "z=32, beta=0.001, 1,252,258 params, best epoch 195"),
    asset("cvae", "CVAE training log", S2 / "CVAE_baseline/outputs/cvae_main/training_log.json",
          "training configuration, loss history", "citation source",
          "200 epochs, 1,675 s, 189,818 train / 9,990 val"),
    asset("cvae", "CVAE common code", S2 / "CVAE_baseline/cvae_common.py",
          "data loading, condition builder, selector, guidance", "read-only reference",
          "guidance is 25 z-space gradient steps and only uses cs1"),
    asset("cvae", "CVAE training script", S2 / "CVAE_baseline/train_cvae.py",
          "CVAE training and checkpoint selection", "read-only reference",
          "model selection starts after beta warm-up"),
    asset("pipeline", "DDPM pool generation", S2 / "CVAE_baseline/run_ddpm_pools.py",
          "5,000-candidate pool generation and selector call", "read-only reference",
          "guidance 1.0, lambda 0.05, 25 sample steps"),
    asset("pipeline", "CVAE pool generation", S2 / "CVAE_baseline/run_cvae_pools.py",
          "5,000-candidate pool generation and selector call", "read-only reference",
          "guide lambda/steps configurable; default 25 steps"),
    asset("pipeline", "ROSS verifier", S2 / "spec_design_v2/v2_ross_verify.py",
          "per-design ROSS recomputation, timeouts, band judging", "reuse with cache-key check",
          "subprocess per design; timeout 180 s; n_base=15"),
    asset("pipeline", "ROSS model builder", S2 / "spec_design/spec_common.py",
          "canonical design -> ROSS rotor -> forward critical speeds", "reuse",
          "build_ross_rotor(n_base=15); positive forward modes only"),
    asset("pipeline", "interval spec utilities", S2 / "spec_design_v2/spec_interval.py",
          "parse/effective band/deviation", "reuse",
          "inference caps log width at h_max; judging uses original band"),
    asset("pipeline", "optimisation baselines", S2 / "spec_design/spec_baselines_latent.py",
          "GA/DE/Adam/BO/random/retrieval baselines", "read-only reference",
          "existing objective is centre-distance, not multi-solution set quality"),
    asset("results", "four-pipeline summary",
          STUDY / "03_existing_candidate_analysis/four_way_summary_reaggregated.csv",
          "headline delivery rates", "reuse as exploratory evidence",
          "DDPM guided 73.17%, DDPM unguided 75.53%, CVAE unguided 76.50%, CVAE guided 79.92%"),
    asset("results", "four-pipeline by-run table",
          STUDY / "03_existing_candidate_analysis/four_way_runs_reaggregated.csv",
          "per method x spec x seed counts and timings", "reuse with protocol caveats",
          "72 runs; 14,400 ROSS submissions"),
    asset("results", "72-pool inventory",
          STUDY / "11_batch2_pilot/results/pool_inventory_72.csv",
          "complete 5,000-candidate pools and paths", "reuse",
          "72/72 pools exist, 5,000 rows, canonical columns present"),
    asset("results", "selector A/B/C report",
          STUDY / "11_selector_ablation/06_reports/代理门控与多样性选择器消融报告.md",
          "FPS vs in-band random vs margin ranking", "post-hoc sensitivity only",
          "A 73.2/79.9, B 86.5/89.8, C 89.8/84.8 for DDPM/CVAE guided"),
    asset("results", "retrieval comparison",
          STUDY / "06_retrieval_comparison/retrieval_vs_generators_by_spec.csv",
          "retrieval+jitter unified and centre ranking", "reuse seed 42",
          "unified 79.92%, centre 93.0%; centre uses extra label information"),
    asset("results", "pool-level ROSS sample",
          STUDY / "11_batch2_pilot/四流程池级ROSS抽样结果_汇总.csv",
          "50-per-pool unbiased ROSS estimates", "reuse as pilot evidence",
          "3,600 samples, 3,595 solved, 5 errors; not a full-pool census"),
    asset("spec", "frozen rare-spec list",
          STUDY / "11_batch2_pilot/极稀缺规格冻结清单.csv",
          "zero/low-support specifications frozen before generation", "reuse only as exploratory",
          "R01-R03 zero-observed; already probed, not unseen"),
    asset("protocol", "first-batch frozen protocol",
          STUDY / "01_protocol/实验协议冻结说明.md",
          "definitions, seeds, metrics, statistical unit", "reuse definitions",
          "six specs already used; not a confirmatory test set"),
    asset("environment", "requirements", S2 / "requirements_v4.7.txt",
          "Python dependency versions", "citation source", "pandas/numpy/torch/scikit-learn"),
    asset("study", "calibration specs", HERE / "spec_candidates/calibration_specs.csv",
          "new calibration specifications frozen from reference support", "new output",
          "six specs; support counted before any generator run"),
    asset("study", "test specs", HERE / "spec_candidates/test_specs.csv",
          "new held-out specifications frozen from reference support", "new output",
          "nine specs: eight non-zero main specs plus one zero-observed exploratory spec"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rows = []
    for item in ASSETS:
        path = Path(item["path"])
        exists = path.exists()
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file())
            size = sum(p.stat().st_size for p in files)
            digest = hashlib.sha256(
                "".join(sha256(p) for p in files).encode("ascii")).hexdigest()
            kind = "directory (%d files)" % len(files)
        elif exists:
            size = path.stat().st_size
            digest = sha256(path)
            kind = "file"
        else:
            size, digest, kind = 0, "", "missing"
        rows.append({**item, "exists": exists, "kind": kind, "bytes": size,
                     "sha256": digest, "sha256_16": digest[:16]})
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "00_asset_inventory.csv", index=False, encoding="utf-8-sig")
    print("assets=%d existing=%d missing=%d"
          % (len(frame), int(frame["exists"].sum()), int((~frame["exists"]).sum())))
    print(frame[["category", "name", "exists", "sha256_16"]].to_string(index=False))


if __name__ == "__main__":
    main()
