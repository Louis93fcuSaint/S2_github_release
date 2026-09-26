# -*- coding: utf-8 -*-
"""Step 03 -- where does the target conditioning get lost?

Two suspects, separated without ROSS:

  1. the reverse process itself: on how many raw samples is cs1 near the target
     the network was asked for?
  2. the constraint repair: it moves the design after the fact, so it can move
     cs1 away from the target too.

On top of that it sweeps the classifier-free guidance weight, which is the only
conditioning knob that does not need a retrain.
"""
import importlib.util
import io
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
TAGS = ["armA", "armB"]
GUIDANCE = [1.0, 2.0, 4.0]
SPECS = ["Steel_mid", "Steel_high", "Titanium_mid", "Titanium_high"]
N = 1500


def step04():
    path = os.path.join(HERE, "04_ddpm.py")
    spec = importlib.util.spec_from_file_location("step04_ddpm", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["step04_ddpm"] = module
    spec.loader.exec_module(module)
    return module


def load_arm(D, tag):
    module = D.ddpm()
    payload = torch.load(os.path.join(D.OUT_DIR, tag, "model.pt"), map_location="cpu")
    config = payload["config"]
    model = module.Denoiser(design_dim=config["design_dim"],
                            condition_dim=config["condition_dim"],
                            hidden=tuple(config["hidden"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return module, model, payload


def main():
    started = time.time()
    torch.set_num_threads(8)
    D = step04()
    ctx = E.pool_ctx()
    surrogate = S.load_surrogate("mlp_best", threads=8)
    betas, alpha_bar = D.ddpm().cosine_schedule(D.T_STEPS)
    spec_by_name = {sp["name"]: sp for sp in E.specs(ctx)}
    cs = ctx["pool"]["cs_1_rpm"].to_numpy(dtype=float)
    mat = ctx["material"]

    def in_band(pred, spec):
        return np.abs(pred - float(spec["target"])) / float(spec["target"]) <= spec["tol"]

    report, lines = {}, []
    for tag in TAGS:
        module, model, payload = load_arm(D, tag)
        scalers = payload["scalers"]
        dropout = float(payload.get("condition_dropout", 0.0))
        for g in GUIDANCE:
            for name in SPECS:
                spec = spec_by_name[name]
                rng = np.random.default_rng(1234)
                fams = [D.FAMILIES[i % len(D.FAMILIES)]
                        for i in rng.integers(0, len(D.FAMILIES), N)]
                targets = np.full(N, float(spec["target"]))
                cond = np.stack([D.condition_for(spec["material"], f, t, scalers)
                                 for f, t in zip(fams, targets)])
                gen = torch.Generator().manual_seed(7)
                with torch.no_grad():
                    unit = module.sample_tensor(
                        model, torch.as_tensor(cond, dtype=torch.float32), N, g,
                        dropout, gen, betas, alpha_bar, payload["config"]["design_dim"]).numpy()
                raw = np.stack([D.decode(u, f, scalers) for u, f in zip(unit, fams)])

                def predict(x):
                    if not len(x):
                        return np.zeros(0)
                    return surrogate.predict(S.canonical_to_model_rows(
                        x, [spec["material"]] * len(x),
                        [f[0] for f in fams], [f[1] for f in fams]))[:, 0]

                ok_before = O.validity(raw, spec["material"], fams)
                pred_before = predict(raw)
                fixed, need = O.repair(raw, spec["material"], fams, rng)
                ok_after = O.validity(fixed, spec["material"], fams)
                pred_after = predict(fixed)

                bad = ~ok_before
                sel = mat == spec["material"]
                base = float((np.abs(cs[sel] - float(spec["target"]))
                              / float(spec["target"]) <= spec["tol"]).mean())
                entry = {
                    "arm": tag, "guidance": g, "spec": name, "n": N,
                    "in_band_raw": float(in_band(pred_before, spec).mean()),
                    "base_rate": base,
                    "valid_before": float(ok_before.mean()),
                    "repair_share": float(need),
                    "in_band_raw_on_repaired": float(in_band(pred_before, spec)[bad].mean())
                    if bad.any() else 0.0,
                    "in_band_repaired": float(in_band(pred_after, spec)[bad].mean())
                    if bad.any() else 0.0,
                    "cs1_shift_pct": float(np.median(
                        np.abs(pred_after[bad] - pred_before[bad])
                        / float(spec["target"])) * 100) if bad.any() else 0.0,
                    "yield": float((ok_after & in_band(pred_after, spec)).mean()),
                    "valid_after": float(ok_after.mean()),
                }
                report["%s|g%.1f|%s" % (tag, g, name)] = entry
                lines.append("%-5s g%.1f %-14s base %.3f | raw in-band %.3f (x%.1f) | "
                             "repaired %.3f | yield %.3f | shift %.2f%%"
                             % (tag, g, name, base, entry["in_band_raw"],
                                entry["in_band_raw"] / max(base, 1e-9),
                                entry["in_band_repaired"],
                                entry["yield"], entry["cs1_shift_pct"]))
                print(lines[-1], flush=True)

    out = os.path.join(E.OUT, "cond_diag.json")
    S.save_json(out, report)
    print("[out] %s (%.0f s)" % (out, time.time() - started))


if __name__ == "__main__":
    main()