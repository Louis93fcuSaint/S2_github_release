# -*- coding: utf-8 -*-
"""Freeze the TST protocol before any generation or ROSS run."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

PASCAL = Path(r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2\paper\paper_v3\pascal_revision")
EA = PASCAL.parent / "research_generators_specification_study" / "eaai_comparative_study"
AUDIT = EA / "audit_scripts"
sys.path.insert(0, str(AUDIT))
import eaai_common as A

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

spec_csv = EA / "spec_candidates" / "test_specs.csv"
specs = pd.read_csv(spec_csv)
included = [s for s in specs.slot_id.tolist() if s != "TST08"]
excluded = specs[specs.slot_id == "TST08"].to_dict("records")
scripts = {
    "run_tst_pools.py": PASCAL / "scripts" / "run_tst_pools.py",
    "build_tst_shortlists.py": PASCAL / "scripts" / "build_tst_shortlists.py",
    "run_tst_ross.py": PASCAL / "scripts" / "run_tst_ross.py",
    "freeze_tst_protocol.py": PASCAL / "scripts" / "freeze_tst_protocol.py",
}
payload = {
    "protocol_id": "TST_v1_frozen",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": "frozen_before_generation",
    "study_root": str(PASCAL),
    "spec_file": str(spec_csv),
    "spec_file_sha256": sha256(spec_csv),
    "specifications": {
        s.slot_id: {
            "spec_id": s.spec_id, "material": s.material, "family": s.family,
            "orders": s.orders, "kind": s.kind, "targets": json.loads(s.targets),
            "bands": json.loads(s.bands), "n_feasible": int(s.n_feasible),
            "support_class": getattr(s, "support_class", None),
        } for s in specs.itertuples() if s.slot_id in included
    },
    "excluded_specifications": excluded,
    "exclusion_reason": "TST08 has zero feasible reference designs and is an exploratory zero-support case, not part of the main comparison.",
    "methods": ["interval_conditioned_ddpm_unguided", "cvae_unguided", "retrieval_plus_jitter"],
    "checkpoints": {
        "interval_ddpm_v2i": {"path": A.DDPM_CKPT, "sha256": sha256(Path(A.DDPM_CKPT))},
        "cvae_main": {"path": A.CVAE_CKPT, "sha256": sha256(Path(A.CVAE_CKPT))},
    },
    "point_conditioned_checkpoint_excluded": {
        "path": A.DDPM_POINT_CKPT, "sha256": sha256(Path(A.DDPM_POINT_CKPT)),
        "reason": "Point-conditioned model is a separate checkpoint and is not used in this TST interval/joint comparison."
    },
    "seeds": {"sampling": [42, 7], "note": "sampling seeds, not independent training repeats"},
    "n_attempts_per_pool": 5000,
    "n_random_per_pool": 100,
    "n_shortlist_per_pool": 200,
    "n_pools": 8 * 3 * 2,
    "n_attempts_total": 8 * 3 * 2 * 5000,
    "n_logical_ross_slots": 8 * 3 * 2 * 300,
    "selection": {
        "random": "uniform over the original 5000 attempt IDs without replacement",
        "shortlist": "common surrogate gate; uniform in-band; out-of-band fallback by (violation, candidate_id)",
        "no_historical_fps": True,
        "user_band_only": True,
        "extra_tolerance": 0.0,
        "dedup": "exact design key on active canonical slots; representative is the first valid attempt",
        "gap_policy": "shortlist shorter than 200 is retained; missing slots count as undelivered in the fixed 200 denominator",
    },
    "ross": {
        "required_version": "1.6.1",
        "python": r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2\neural_surrogate_experiment\.venv_torch\Scripts\python.exe",
        "failures_count_in_denominator": True,
        "timeouts_count_in_denominator": True,
        "cache_scope": str(PASCAL / "tst_runs" / "03_ross"),
        "cache_rule": "reuse only when design parameters, material, units, solver version and acceptance protocol match exactly",
    },
    "output_dirs": {
        "pools": str(PASCAL / "tst_runs" / "01_pools"),
        "shortlists": str(PASCAL / "tst_runs" / "02_shortlists"),
        "ross": str(PASCAL / "tst_runs" / "03_ross"),
        "metrics": str(PASCAL / "tst_runs" / "04_metrics"),
    },
    "scripts": {k: {"path": str(v), "sha256": sha256(v)} for k, v in scripts.items() if v.exists()},
    "freeze_note": "No TST pool existed at freeze time. Any later change requires a new protocol id and a change log.",
}
out = PASCAL / "tst_runs" / "00_protocol" / "tst_protocol_frozen.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(out)
print("included", included)
print("n_pools", payload["n_pools"], "attempts", payload["n_attempts_total"], "ross", payload["n_logical_ross_slots"])
