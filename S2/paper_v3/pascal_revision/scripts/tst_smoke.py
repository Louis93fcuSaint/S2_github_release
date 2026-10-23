# -*- coding: utf-8 -*-
"""32-candidate TST smoke test: DDPM, CVAE and retrieval on representative specs."""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
EA = PROJECT.parent / "research_generators_specification_study" / "eaai_comparative_study"
sys.path.insert(0, str(EA / "audit_scripts"))
import eaai_common as A
import run_eaai_pools as P

A.STUDY = str(PROJECT / "tst_runs")
A.SPEC_CSV = str(EA / "spec_candidates" / "test_specs.csv")
A.POOL_DIR = str(PROJECT / "tst_runs" / "00_protocol" / "smoke" / "pools")
A.LOG_DIR = str(PROJECT / "tst_runs" / "logs")
A.N_ATTEMPTS = 32
A.N_RANDOM = 8
A.N_SHORT = 16
os.makedirs(A.POOL_DIR, exist_ok=True)

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as fh:
        for b in iter(lambda: fh.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def main():
    ctx, train_idx, _, _ = A.build_context_split(save=False)
    specs = A.load_calibration_specs()
    ddpm, cvae = A.load_ddpm(), A.load_cvae()
    surrogate = A.S.load_surrogate("mlp_best", threads=8)
    rows=[]
    for slot, method, bundle in [("TST01","ddpm",ddpm),("TST02","cvae",cvae),("TST04","retrieval",None)]:
        spec=specs[slot]
        meta, *_ = P.write_pool(bundle, ctx, train_idx, spec, method, surrogate, seed=42)
        d=Path(A.POOL_DIR)/meta["pool_id"]
        blob=np.load(d/"raw.npz", allow_pickle=False)
        x,pred=blob["x"],blob["pred"]
        valid=blob["valid"].astype(bool)
        rows.append({
            "slot":slot,"method":method,"pool_id":meta["pool_id"],
            "n_attempts":int(len(x)),"n_valid":int(valid.sum()),
            "n_unique_valid":int(meta["n_unique_valid"]),
            "n_proxy_in_band":int(meta["n_proxy_in_band_attempts"]),
            "pred_all_finite":bool(np.isfinite(pred).all()),
            "x_all_finite":bool(np.isfinite(x).all()),
            "material":meta["material"],"family":meta["family"],
            "orders":"|".join(str(o) for o in meta["orders"]),
            "raw_npz":str(d/"raw.npz"),"pool_csv":meta["files"]["pool_csv"],
            "raw_sha256":sha256_file(d/"raw.npz"),
        })
        # reproducibility: regenerate with same seed and compare the decoded design array
        meta2, *_ = P.write_pool(bundle, ctx, train_idx, spec, method, surrogate, seed=42)
        b2=np.load(Path(A.POOL_DIR)/meta2["pool_id"]/"raw.npz", allow_pickle=False)
        rows[-1]["reproducible_same_seed"]=bool(np.array_equal(x,b2["x"]))
        rows[-1]["reproducibility_meta"]=meta2["sha256"]["x"]==meta["sha256"]["x"]
        print(rows[-1], flush=True)
    out=PROJECT/"tst_runs"/"00_protocol"/"tst_smoke_summary.csv"
    pd.DataFrame(rows).to_csv(out,index=False,encoding="utf-8")
    print(out)

if __name__=="__main__":
    main()
