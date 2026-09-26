# -*- coding: utf-8 -*-
"""Compare the random repair with the structured projection on the same pool.

The pool is fixed: the raw output of one trained arm for one spec, so the two
repairs see identical designs and only the repair differs.
"""
import importlib.util
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_common as S
import spec_eval as E
import spec_opt as O

HERE = os.path.dirname(os.path.abspath(__file__))
TAGS = [("armA", 1.0), ("armB", 1.5), ("armB", 3.0)]
SPECS = ["Steel_mid", "Steel_high", "Aluminum_high", "Titanium_mid"]
N = 1500


def step04():
    path = os.path.join(HERE, "04_ddpm.py")
    spec = importlib.util.spec_from_file_location("step04_ddpm2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["step04_ddpm2"] = module
    spec.loader.exec_module(module)
    return module


def main():
    started = time.time()
    torch.set_num_threads(8)
    D = step04()
    ctx = E.pool_ctx()
    surrogate = S.load_surrogate("mlp_best", threads=8)
    betas, alpha_bar = D.ddpm().cosine_schedule(D.T_STEPS)
    box = O.Box(ctx)
    spec_by_name = {sp["name"]: sp for sp in E.specs(ctx)}
    cs = ctx["pool"]["cs_1_rpm"].to_numpy(dtype=float)
    mat = ctx["material"]
    report, lines = {}, []
    for tag, g in TAGS:
        module = D.ddpm()
        payload = torch.load(os.path.join(D.OUT_DIR, tag, "model.pt"), map_location="cpu")
        config = payload["config"]
        model = module.Denoiser(design_dim=config["design_dim"],
                                condition_dim=config["condition_dim"],
                                hidden=tuple(config["hidden"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        scalers = payload["scalers"]
        dropout = float(payload.get("condition_dropout", 0.0))
        for name in SPECS:
            spec = spec_by_name[name]
            rng = np.random.default_rng(2024)
            fams = [D.FAMILIES[i % len(D.FAMILIES)]
                    for i in rng.integers(0, len(D.FAMILIES), N)]
            cond = np.stack([D.condition_for(spec["material"], f, float(spec["target"]),
                                            scalers) for f in fams])
            gen = torch.Generator().manual_seed(11)
            with torch.no_grad():
                unit = module.sample_tensor(
                    model, torch.as_tensor(cond, dtype=torch.float32), N, g, dropout,
                    gen, betas, alpha_bar, config["design_dim"]).numpy()
            raw = np.stack([D.decode(u, f, scalers) for u, f in zip(unit, fams)])
            target = float(spec["target"])
            band = lambda v: np.abs(v - target) / target <= spec["tol"]

            def predict(x):
                if not len(x):
                    return np.zeros(0)
                return surrogate.predict(S.canonical_to_model_rows(
                    x, [spec["material"]] * len(x), [f[0] for f in fams],
                    [f[1] for f in fams]))[:, 0]

            p_raw = predict(raw)
            old, _ = O.repair(raw, spec["material"], fams, np.random.default_rng(5))
            new, stats = O.repair_structured(raw, spec["material"], fams,
                                             box=box, rng=np.random.default_rng(5))
            ok_old = O.validity(old, spec["material"], fams)
            ok_new = O.validity(new, spec["material"], fams)
            p_old, p_new = predict(old), predict(new)
            sel = mat == spec["material"]
            base = float((np.abs(cs[sel] - target) / target <= spec["tol"]).mean())
            entry = {
                "arm": tag, "guidance": g, "spec": name,
                "in_band_raw": float(band(p_raw).mean()), "base_rate": base,
                "valid_raw": float(O.validity(raw, spec["material"], fams).mean()),
                "old_shift_pct": float(np.median(np.abs(p_old - p_raw)) / target * 100),
                "new_shift_pct": float(np.median(np.abs(p_new - p_raw)) / target * 100),
                "old_yield": float((ok_old & band(p_old)).mean()),
                "new_yield": float((ok_new & band(p_new)).mean()),
                "old_valid": float(ok_old.mean()), "new_valid": float(ok_new.mean()),
                "stats": stats,
            }
            report["%s|g%.1f|%s" % (tag, g, name)] = entry
            lines.append("%-5s g%.1f %-14s raw %.3f | old shift %5.1f%% yield %.4f | "
                         "new shift %5.1f%% yield %.4f | fallback %d"
                         % (tag, g, name, entry["in_band_raw"], entry["old_shift_pct"],
                            entry["old_yield"], entry["new_shift_pct"],
                            entry["new_yield"], stats["fallback"]))
            print(lines[-1], flush=True)
    out = os.path.join(E.OUT, "repair_compare.json")
    S.save_json(out, report)
    print("[out] %s (%.0f s)" % (out, time.time() - started))


if __name__ == "__main__":
    main()