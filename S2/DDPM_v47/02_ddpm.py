# -*- coding: utf-8 -*-
"""Conditional DDPM over the 18-dim design vector (slice nd=3, nb=2).

One implementation, two frozen configurations:

  v1  python 02_ddpm.py train --condition-dropout 0.0 --tag v1
      python 02_ddpm.py sample --tag v1 --guidance 1.0
  v2  python 02_ddpm.py train --condition-dropout 0.1 --tag v2
      python 02_ddpm.py sample --tag v2 --guidance 1.5

The design vector lives in [0,1]^18 (per-dim min/max taken from the training
split).  The condition is log(cs targets) standardised, concatenated with the
material one-hot.  Nothing else is hidden: no retrieval anchor, no trust gate,
no hybrid portfolio -- this is the plain conditional generator.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddpm_v47_common as D

T_STEPS = 1000
CS_COLUMNS = D.PRIMARY_CS


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_arrays(frame, positions, cs_columns=CS_COLUMNS):
    """Return normalised design X in [0,1]^18, condition Y, and the bounds."""
    design = frame[D.COMPACT_COLUMNS].to_numpy(dtype=np.float64)[positions]
    cs = frame[cs_columns].to_numpy(dtype=np.float64)[positions]
    if not np.isfinite(cs).all():
        raise ValueError("condition targets contain NaN")
    materials = frame["material"].to_numpy()[positions]
    return design, cs, materials


def fit_scalers(design, cs, materials):
    lo = design.min(axis=0)
    hi = design.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    log_cs = np.log(cs)
    cs_mean = log_cs.mean(axis=0)
    cs_std = np.where(log_cs.std(axis=0) < 1e-12, 1.0, log_cs.std(axis=0))
    return {
        "design_min": lo.tolist(), "design_span": span.tolist(),
        "cs_mean": cs_mean.tolist(), "cs_std": cs_std.tolist(),
        "cs_columns": list(CS_COLUMNS),
    }


def encode(design, cs, materials, scalers):
    lo = np.asarray(scalers["design_min"])
    span = np.asarray(scalers["design_span"])
    x = (design - lo) / span
    log_cs = np.log(cs)
    z = (log_cs - np.asarray(scalers["cs_mean"])) / np.asarray(scalers["cs_std"])
    onehot = np.asarray([D.material_onehot(m) for m in materials], dtype=np.float64)
    return x.astype(np.float32), np.concatenate([z, onehot], axis=1).astype(np.float32)


def decode(design_unit, scalers):
    lo = np.asarray(scalers["design_min"])
    span = np.asarray(scalers["design_span"])
    return lo + np.clip(design_unit, 0.0, 1.0) * span


# --------------------------------------------------------------------------- #
# diffusion schedule
# --------------------------------------------------------------------------- #
def cosine_schedule(t_steps=T_STEPS, s=0.008):
    """Nichol & Dhariwal cosine schedule.

    Returns PER-STEP betas and the CUMULATIVE alpha_bar.  Getting this wrong is
    silent: feeding the cumulative value back as a per-step beta makes every
    reverse step amplify by ~1/sqrt(1-beta) and the chain overflows to NaN.
    """
    steps = np.arange(t_steps + 1, dtype=np.float64)
    f = np.cos((steps / t_steps + s) / (1.0 + s) * math.pi / 2.0) ** 2
    f = f / f[0]
    betas = np.clip(1.0 - f[1:] / f[:-1], 1e-4, 0.9999)
    alpha_bar = np.cumprod(1.0 - betas)
    return betas, alpha_bar


# --------------------------------------------------------------------------- #
# denoiser
# --------------------------------------------------------------------------- #
class Denoiser(nn.Module):
    def __init__(self, design_dim=18, condition_dim=6, hidden=(256, 256, 256)):
        super().__init__()
        self.time_dim = 64
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, hidden[0]), nn.SiLU(),
            nn.Linear(hidden[0], hidden[0]), nn.SiLU())
        layers = []
        previous = design_dim + condition_dim + hidden[0]
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.SiLU()])
            previous = width
        layers.append(nn.Linear(previous, design_dim))
        self.body = nn.Sequential(*layers)

    def time_embedding(self, t, device):
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000.0)
                           * torch.arange(half, device=device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    def forward(self, x, t, condition):
        embedding = self.time_mlp(self.time_embedding(t, x.device))
        return self.body(torch.cat([x, condition, embedding], dim=1))


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    frame = D.load_slice()
    train_pos, val_pos, _ = D.fixed_split(frame)
    if args.max_rows and args.max_rows < train_pos.size:
        train_pos = train_pos[: args.max_rows]
        val_pos = val_pos[: max(64, args.max_rows // 8)]

    design, cs, materials = build_arrays(frame, train_pos)
    scalers = fit_scalers(design, cs, materials)
    x_all, y_all = encode(design, cs, materials, scalers)
    v_design, v_cs, v_mat = build_arrays(frame, val_pos)
    x_val, y_val = encode(v_design, v_cs, v_mat, scalers)

    betas, alpha_bar = cosine_schedule()
    betas_t = torch.as_tensor(betas, dtype=torch.float32)
    abar_t = torch.as_tensor(alpha_bar, dtype=torch.float32)

    device = torch.device("cpu")
    model = Denoiser(design_dim=x_all.shape[1], condition_dim=y_all.shape[1],
                     hidden=tuple(args.hidden)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    x_tensor = torch.as_tensor(x_all)
    y_tensor = torch.as_tensor(y_all)
    x_val_t = torch.as_tensor(x_val)
    y_val_t = torch.as_tensor(y_val)
    generator = torch.Generator().manual_seed(args.seed)

    history, best = [], float("inf")
    out_dir = os.path.join(D.HERE, "outputs", args.tag)
    os.makedirs(out_dir, exist_ok=True)
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
            x_t = a.sqrt() * xb + (1.0 - a).sqrt() * noise
            if args.condition_dropout > 0:
                drop = (torch.rand(xb.shape[0], generator=generator)
                        < args.condition_dropout)
                yb = yb.clone()
                yb[drop] = 0.0
            loss = nn.functional.mse_loss(model(x_t, t, yb), noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item()) * xb.shape[0]
            seen += xb.shape[0]
        scheduler.step()

        model.eval()
        with torch.no_grad():
            t = torch.randint(0, T_STEPS, (x_val_t.shape[0],), generator=generator)
            a = abar_t[t].unsqueeze(1)
            noise = torch.randn(x_val_t.shape, generator=generator)
            val_loss = float(nn.functional.mse_loss(
                model(a.sqrt() * x_val_t + (1.0 - a).sqrt() * noise, t, y_val_t),
                noise).item())
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1),
                        "val_loss": val_loss,
                        "learning_rate": float(optimizer.param_groups[0]["lr"])})
        if val_loss < best:
            best = val_loss
            torch.save({"state_dict": model.state_dict(),
                        "config": {"design_dim": int(x_all.shape[1]),
                                   "condition_dim": int(y_all.shape[1]),
                                   "hidden": [int(w) for w in args.hidden]},
                        "scalers": scalers,
                        "tag": args.tag,
                        "condition_dropout": float(args.condition_dropout)},
                       os.path.join(out_dir, "model.pt"))
        if args.verbose or epoch % args.log_every == 0 or epoch == args.epochs:
            print("epoch %4d | train %.5f | val %.5f | lr %.2e | %.1fs"
                  % (epoch, history[-1]["train_loss"], val_loss,
                     history[-1]["learning_rate"], time.time() - started), flush=True)

    D.save_json(os.path.join(out_dir, "training_log.json"),
                {"tag": args.tag, "epochs": args.epochs,
                 "condition_dropout": float(args.condition_dropout),
                 "hidden": list(args.hidden), "batch_size": args.batch_size,
                 "learning_rate": args.learning_rate,
                 "n_train": int(train_pos.size), "n_val": int(val_pos.size),
                 "best_val_loss": best, "seconds": time.time() - started,
                 "history": history})
    print("[train] best val loss %.5f -> %s" % (best, os.path.join(out_dir, "model.pt")),
          flush=True)


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_tensor(model, condition, n, guidance, condition_dropout, generator,
                  betas, alpha_bar, design_dim):
    """Shared reverse process; guidance=1 disables classifier-free guidance."""
    device = torch.device("cpu")
    model.eval()
    x = torch.randn((n, design_dim), generator=generator)
    betas_t = torch.as_tensor(betas, dtype=torch.float32)
    abar_t = torch.as_tensor(alpha_bar, dtype=torch.float32)
    zero = torch.zeros_like(condition)
    for step in range(T_STEPS - 1, -1, -1):
        t = torch.full((n,), step, dtype=torch.long)
        eps_cond = model(x, t, condition)
        if guidance != 1.0 and condition_dropout > 0:
            eps_uncond = model(x, t, zero)
            eps = eps_uncond + guidance * (eps_cond - eps_uncond)
        else:
            eps = eps_cond
        a = abar_t[step]
        a_prev = abar_t[step - 1] if step > 0 else torch.tensor(1.0)
        beta = betas_t[step]
        mean = (x - beta / (1.0 - a).sqrt() * eps) / (1.0 - beta).sqrt()
        if step > 0:
            sigma = (beta * (1.0 - a_prev) / (1.0 - a)).sqrt()
            x = mean + sigma * torch.randn(x.shape, generator=generator)
        else:
            x = mean
    return x


def sample(args):
    torch.set_num_threads(args.threads)
    stem = args.out_tag or args.tag
    model_dir = os.path.join(D.HERE, "outputs", args.tag)
    out_dir = os.path.join(D.HERE, "outputs", stem)
    os.makedirs(out_dir, exist_ok=True)
    payload = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    scalers = payload["scalers"]
    config = payload["config"]
    model = Denoiser(design_dim=config["design_dim"],
                     condition_dim=config["condition_dim"],
                     hidden=tuple(config["hidden"]))
    print("[sample] model %s" % model_dir)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    condition_dropout = float(payload.get("condition_dropout", 0.0))
    betas, alpha_bar = cosine_schedule()

    frame = D.load_slice()
    _, _, test_pos = D.fixed_split(frame)
    targets = D.select_targets(frame.iloc[test_pos])
    if args.targets:
        wanted = set(args.targets)
        targets = [t for t in targets if t["target_id"] in wanted]

    generator = torch.Generator().manual_seed(args.seed)
    cs_std = np.asarray(scalers["cs_std"])
    cs_mean = np.asarray(scalers["cs_mean"])

    rows, audits = [], []
    for target in targets:
        cs = np.asarray(target["cs_targets"][: len(scalers["cs_columns"])])
        z = (np.log(cs) - cs_mean) / cs_std
        cond = np.concatenate([z, D.material_onehot(target["material"])])
        condition = torch.as_tensor(np.tile(cond, (args.n_per_target, 1)),
                                    dtype=torch.float32)
        unit = sample_tensor(model, condition, args.n_per_target, args.guidance,
                             condition_dropout, generator, betas, alpha_bar,
                             config["design_dim"]).numpy()
        design = decode(unit, scalers)
        valid_count = 0
        for design_row in design:
            ok, _ = D.is_valid_compact(design_row, target["material"])
            valid_count += int(ok)
            rows.append({
                "target_id": target["target_id"], "material": target["material"],
                "valid": int(ok),
                **{name: float(value)
                   for name, value in zip(D.COMPACT_COLUMNS, design_row)},
            })
        audits.append({"target_id": target["target_id"],
                       "raw_constraint_pass_rate": valid_count / args.n_per_target,
                       "n_generated": args.n_per_target})
        print("[sample] %s %s raw_valid=%.3f"
              % (target["target_id"], target["material"],
                 valid_count / args.n_per_target), flush=True)

    import pandas as pd
    csv_path = os.path.join(out_dir, "candidates_%s.csv" % stem)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    D.save_json(os.path.join(out_dir, "sampling_audit_%s.json" % stem),
                {"tag": args.tag, "out_tag": stem, "guidance": args.guidance,
                 "condition_dropout": condition_dropout,
                 "n_per_target": args.n_per_target, "per_target": audits})
    print("[sample] wrote %s (%d rows)" % (csv_path, len(rows)), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("train", "sample"):
        node = sub.add_parser(name)
        node.add_argument("--tag", default="v1")
        node.add_argument("--threads", type=int, default=4)
        node.add_argument("--seed", type=int, default=D.SEED)

    tr = sub.choices["train"]
    tr.add_argument("--epochs", type=int, default=400)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--learning-rate", type=float, default=2e-4)
    tr.add_argument("--weight-decay", type=float, default=1e-6)
    tr.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    tr.add_argument("--condition-dropout", type=float, default=0.0)
    tr.add_argument("--max-rows", type=int, default=0)
    tr.add_argument("--log-every", type=int, default=20)
    tr.add_argument("--verbose", action="store_true")

    sp = sub.choices["sample"]
    sp.add_argument("--out-tag", default=None)
    sp.add_argument("--guidance", type=float, default=1.0)
    sp.add_argument("--n-per-target", type=int, default=200)
    sp.add_argument("--targets", nargs="+", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "train":
        train(arguments)
    else:
        sample(arguments)