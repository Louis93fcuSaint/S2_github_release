# -*- coding: utf-8 -*-
"""Train the conditional-VAE baseline on the same latent/condition arrays as v2i."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import cvae_common as C


def parse_hidden(text: str):
    return tuple(int(v) for v in str(text).replace("x", ",").split(",") if v.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-tag", default="cvae")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--free-bits", type=float, default=0.02)
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--hidden", default="512,512,512")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=35)
    args = parser.parse_args()

    torch.set_num_threads(int(args.threads))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    hidden = parse_hidden(args.hidden)
    U, Y, nd, nb, material, scalers = C.load_training_arrays(seed=args.seed)

    rng = np.random.default_rng(int(args.seed))
    order = rng.permutation(len(U))
    n_val = max(2000, int(0.05 * len(U)))
    val_idx, train_idx = order[:n_val], order[n_val:]
    print("[cvae-train] train=%d val=%d U=%s Y=%s hidden=%s z=%d beta=%.3f"
          % (len(train_idx), len(val_idx), U.shape, Y.shape, hidden, args.z_dim, args.beta),
          flush=True)

    x_all = torch.as_tensor(U[train_idx], dtype=torch.float32)
    y_all = torch.as_tensor(Y[train_idx], dtype=torch.float32)
    w_all = torch.as_tensor(C.family_masks([(nd[i], nb[i]) for i in train_idx]),
                            dtype=torch.float32)
    x_val = torch.as_tensor(U[val_idx], dtype=torch.float32)
    y_val = torch.as_tensor(Y[val_idx], dtype=torch.float32)
    w_val = torch.as_tensor(C.family_masks([(nd[i], nb[i]) for i in val_idx]),
                            dtype=torch.float32)

    model = C.ConditionalVAE(u_dim=U.shape[1], c_dim=Y.shape[1],
                             z_dim=args.z_dim, hidden=hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print("[cvae-train] params %.3f M" % (n_params / 1e6), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    gen = torch.Generator().manual_seed(int(args.seed) + 2026)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    ema_decay = 0.999

    out_dir = os.path.join(C.OUT_DIR, args.out_tag)
    os.makedirs(out_dir, exist_ok=True)
    history = []
    best = float("inf")
    bad = 0
    started = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        perm = torch.randperm(x_all.shape[0], generator=gen)
        running, seen = 0.0, 0
        beta_eff = float(args.beta) * min(1.0, epoch / max(int(args.warmup_epochs), 1))
        for start in range(0, perm.numel(), int(args.batch_size)):
            idx = perm[start:start + int(args.batch_size)]
            xb, yb, wb = x_all[idx], y_all[idx], w_all[idx]
            loss, recon, kl = C.cvae_loss(model, xb, yb, wb, beta_eff,
                                          free_bits=args.free_bits, generator=gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                for key, value in model.state_dict().items():
                    ema[key].mul_(ema_decay).add_(value.detach().float(), alpha=1.0 - ema_decay)
            running += float(loss.item()) * xb.shape[0]
            seen += xb.shape[0]
        sched.step()

        model.eval()
        with torch.no_grad():
            v_loss, v_recon, v_kl = C.cvae_loss(model, x_val, y_val, w_val,
                                                beta_eff, free_bits=args.free_bits,
                                                generator=gen)
        row = {"epoch": epoch, "beta": beta_eff,
               "train_loss": running / max(seen, 1),
               "val_loss": float(v_loss.item()),
               "val_recon_mse": float(v_recon.item()),
               "val_kl": float(v_kl.item())}
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print("[cvae-train] %03d beta %.4f train %.5f val %.5f recon %.5f kl %.3f"
                  % (epoch, beta_eff, row["train_loss"], row["val_loss"],
                     row["val_recon_mse"], row["val_kl"]), flush=True)
        # During the KL warm-up the total validation loss is not comparable
        # across epochs: a nearly zero beta trivially wins.  Model selection
        # therefore starts only after beta has reached its final value.
        if epoch >= int(args.warmup_epochs):
            if row["val_loss"] < best - 1e-6:
                best = row["val_loss"]
                bad = 0
                C.save_checkpoint(
                    os.path.join(out_dir, "model_best.pt"), model,
                    extra={"ema_state_dict": ema, "epoch": epoch, "val_loss": best,
                           "scalers": scalers, "history": history,
                           "train_args": vars(args)})
            else:
                bad += 1
        C.save_checkpoint(
            os.path.join(out_dir, "model_last.pt"), model,
            extra={"ema_state_dict": ema, "epoch": epoch, "val_loss": best,
                   "scalers": scalers, "history": history,
                   "train_args": vars(args)})
        if bad >= int(args.patience):
            print("[cvae-train] early stop at epoch %d (best %.6f)" % (epoch, best),
                  flush=True)
            break

    log = {"tag": args.out_tag, "epochs_run": len(history), "best_val_loss": best,
           "seconds": time.time() - started, "params": n_params,
           "hidden": list(hidden), "z_dim": args.z_dim, "beta": args.beta,
           "free_bits": args.free_bits, "history": history,
           "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
           "train_args": vars(args)}
    with open(os.path.join(out_dir, "training_log.json"), "w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2, ensure_ascii=False)
    print("[cvae-train] wrote %s | best %.6f | %.0fs"
          % (out_dir, best, time.time() - started), flush=True)


if __name__ == "__main__":
    main()

