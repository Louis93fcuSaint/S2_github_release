# -*- coding: utf-8 -*-
"""Step 04 -- target-conditioned DDPM over the 33-dim canonical design.

The condition is exactly what the user of the tool can supply:

    [ material one-hot (3) | family one-hot (9) | z(log cs1 target) (1) ]

so p(x | material, n_disks, n_bearings, desired first critical speed).  The
band then enters only at sampling time, by asking for a target inside it.

The reverse process is the frozen, already-debugged one from
DDPM_v47/02_ddpm.py (cosine schedule, per-step betas), which is imported
rather than re-derived.
"""
import argparse
import importlib.util
import json
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
OUT_DIR = os.path.join(E.OUT, "ddpm")
T_STEPS = 1000
FAMILIES = [(a, b) for a in S.ND_CHOICES for b in S.NB_CHOICES]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DD = None


def ddpm():
    global DD
    if DD is None:
        DD = _load(os.path.join(S.S2_DIR, "DDPM_v47", "02_ddpm.py"), "ddpm_train_v47")
    return DD


# --------------------------------------------------------------------------- #
def build_arrays(ctx):
    """Normalised design X in [0,1]^33 and the condition Y."""
    design = ctx["designs"]
    lo = design.min(axis=0)
    hi = design.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    X = ((design - lo) / span).astype(np.float32)
    log_cs = np.log(np.maximum(ctx["Y"][:, 0], 1.0))
    cs_mean, cs_std = float(log_cs.mean()), float(log_cs.std())
    z = ((log_cs - cs_mean) / cs_std).astype(np.float64)
    onehot = np.array([S.material_onehot(m) + S.family_onehot(int(a), int(b))
                       for m, a, b in zip(ctx["material"], ctx["nd"], ctx["nb"])],
                      dtype=np.float64)
    Y = np.concatenate([onehot, z[:, None]], axis=1).astype(np.float32)
    scalers = {"design_min": lo.tolist(), "design_span": span.tolist(),
               "cs_mean": cs_mean, "cs_std": cs_std}
    return X, Y, scalers


def condition_for(material, family, cs_target, scalers):
    z = (np.log(max(cs_target, 1.0)) - scalers["cs_mean"]) / scalers["cs_std"]
    return np.array(S.material_onehot(material)
                    + S.family_onehot(int(family[0]), int(family[1])) + [z],
                    dtype=np.float64)


def decode(unit, family, scalers):
    lo = np.asarray(scalers["design_min"])
    span = np.asarray(scalers["design_span"])
    x = lo + np.clip(unit, 0.0, 1.0) * span
    used = O.Box._used(int(family[0]), int(family[1]))
    mask = np.zeros(S.DESIGN_DIM, dtype=bool)
    mask[used] = True
    x = np.where(mask, x, 0.0)
    return S.sort_canonical(x, int(family[0]), int(family[1]))


# --------------------------------------------------------------------------- #
def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    ctx = E.pool_ctx()
    X, Y, scalers = build_arrays(ctx)
    print("train set %s  condition dim %d" % (X.shape, Y.shape[1]), flush=True)

    module = ddpm()
    betas, alpha_bar = module.cosine_schedule(T_STEPS)
    betas_t = torch.as_tensor(betas, dtype=torch.float32)
    abar_t = torch.as_tensor(alpha_bar, dtype=torch.float32)

    hidden = tuple(args.hidden)
    model = module.Denoiser(design_dim=X.shape[1], condition_dim=Y.shape[1],
                            hidden=hidden)
    print("denoiser params %.2fM  hidden %s"
          % (sum(p.numel() for p in model.parameters()) / 1e6, hidden), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    x_tensor = torch.as_tensor(X)
    y_tensor = torch.as_tensor(Y)
    generator = torch.Generator().manual_seed(args.seed)
    out_dir = os.path.join(OUT_DIR, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    history, best = [], float("inf")
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(x_tensor.shape[0], generator=generator)
        running, seen = 0.0, 0
        for start in range(0, order.numel(), args.batch_size):
            index = order[start:start + args.batch_size]
            xb, yb = x_tensor[index], y_tensor[index]
            t = torch.randint(0, T_STEPS, (xb.shape[0],), generator=generator)
            noise = torch.randn(xb.shape, generator=generator)
            a = abar_t[t].unsqueeze(1)
            if args.condition_dropout > 0:
                drop = (torch.rand(xb.shape[0], generator=generator)
                        < args.condition_dropout)
                yb = yb.clone()
                yb[drop] = 0.0
            loss = torch.nn.functional.mse_loss(
                model(a.sqrt() * xb + (1.0 - a).sqrt() * noise, t, yb), noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item()) * xb.shape[0]
            seen += xb.shape[0]
        scheduler.step()
        epoch_loss = running / max(seen, 1)
        history.append({"epoch": epoch, "train_loss": epoch_loss})
        if epoch_loss < best:
            best = epoch_loss
            torch.save({"state_dict": model.state_dict(),
                        "config": {"design_dim": int(X.shape[1]),
                                   "condition_dim": int(Y.shape[1]),
                                   "hidden": [int(w) for w in hidden]},
                        "scalers": scalers, "tag": args.tag,
                        "condition_dropout": float(args.condition_dropout)},
                       os.path.join(out_dir, "model.pt"))
        if epoch % 5 == 0 or epoch == args.epochs:
            print("epoch %3d | train %.5f | %.0fs"
                  % (epoch, epoch_loss, time.time() - started), flush=True)

    S.save_json(os.path.join(out_dir, "training_log.json"),
                {"tag": args.tag, "epochs": args.epochs, "hidden": list(hidden),
                 "batch_size": args.batch_size, "n_train": int(X.shape[0]),
                 "condition_dropout": float(args.condition_dropout),
                 "best_train_loss": best, "seconds": time.time() - started,
                 "history": history})
    print("[train] done in %.0fs" % (time.time() - started), flush=True)


# --------------------------------------------------------------------------- #
def sample(args):
    torch.set_num_threads(args.threads)
    ctx = E.pool_ctx()
    module = ddpm()
    betas, alpha_bar = module.cosine_schedule(T_STEPS)
    model_dir = os.path.join(OUT_DIR, args.tag)
    payload = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    scalers = payload["scalers"]
    config = payload["config"]
    model = module.Denoiser(design_dim=config["design_dim"],
                            condition_dim=config["condition_dim"],
                            hidden=tuple(config["hidden"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    dropout = float(payload.get("condition_dropout", 0.0))
    generator = torch.Generator().manual_seed(args.seed)
    surrogate = S.load_surrogate("mlp_best", threads=args.threads)

    os.makedirs(os.path.join(OUT_DIR, args.tag), exist_ok=True)
    specs = E.specs(ctx)
    if args.specs:
        wanted = set(args.specs.split(","))
        specs = [sp for sp in specs if sp["name"] in wanted]

    funnel = {}
    for spec in specs:
        rng = np.random.default_rng(abs(hash(spec["name"])) % 1000 + args.seed)
        n = args.n_generate
        fams = [FAMILIES[i % len(FAMILIES)] for i in rng.integers(0, len(FAMILIES), n)]
        if spec.get("target"):
            targets = np.full(n, float(spec["target"]))
        else:
            targets = np.exp(rng.uniform(np.log(spec["lower"]),
                                         np.log(spec["upper"]), n))
        cond = np.stack([condition_for(spec["material"], f, t, scalers)
                         for f, t in zip(fams, targets)])
        with torch.no_grad():
            unit = module.sample_tensor(
                model, torch.as_tensor(cond, dtype=torch.float32), n, args.guidance,
                dropout, generator, betas, alpha_bar, config["design_dim"]).numpy()
        x = np.stack([decode(u, f, scalers) for u, f in zip(unit, fams)])
        raw_fams = list(fams)
        valid = O.validity(x, spec["material"], fams)
        x, repair_rate = O.repair(x, spec["material"], fams, rng)
        valid = O.validity(x, spec["material"], fams)
        keep = np.where(valid)[0]
        x, fams = x[keep], [fams[i] for i in keep]
        cs1 = surrogate.predict(S.canonical_to_model_rows(
            x, [spec["material"]] * len(x), [f[0] for f in fams],
            [f[1] for f in fams]))[:, 0]
        score = O.objective_from_cs1(cs1, spec, None)
        order = np.argsort(-score)[:args.n_submit]
        x, fams = x[order], [fams[i] for i in order]
        funnel[spec["name"]] = {
            "generated": int(n), "repair_rate": repair_rate,
            "valid": int(len(keep)), "submitted": int(len(x)),
            "raw_spec_rate_proxy": float(np.mean(
                (cs1 >= spec["lower"]) & (cs1 <= spec["upper"]))),
        }
        frame = O.make_frame(x, spec["material"], fams)
        path = os.path.join(OUT_DIR, "%s__%s.csv" % (args.tag, spec["name"]))
        frame.to_csv(path, index=False, encoding="utf-8")
        print("[sample] %-14s generated %d -> valid %d -> submitted %d | raw proxy "
              "in-band %.3f" % (spec["name"], n, len(keep), len(x),
                                funnel[spec["name"]]["raw_spec_rate_proxy"]), flush=True)
    S.save_json(os.path.join(OUT_DIR, "funnel_%s.json" % args.tag), funnel)
    print("[sample] %s" % os.path.join(OUT_DIR, "funnel_%s.json" % args.tag), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "sample"])
    parser.add_argument("--tag", default="armA")
    parser.add_argument("--specs", default="")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--condition-dropout", type=float, default=0.0)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--n-generate", type=int, default=1500)
    parser.add_argument("--n-submit", type=int, default=200)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else:
        sample(args)


if __name__ == "__main__":
    main()
