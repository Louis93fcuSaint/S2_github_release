# -*- coding: utf-8 -*-
"""v2 -- multi-order conditioned latent DDPM.

v1 conditions on ONE number: the requested first critical speed.

  condition = [ material one-hot (3) | family one-hot (9) | z(log cs1) ]      13 dims

v2 lets the user pin any subset of {cs1, cs2, cs3}, alone or combined, and the
condition carries a value AND a mask per order:

  condition = [ material one-hot (3) | family one-hot (9)
                | z1 m1 z2 m2 z3 m3 ]                                          18 dims

  z_j  z-scored log(cs_j) target, 0 when the order is not requested
  m_j  1 when the order IS requested

The mask is the whole point.  Training draws a random subset of orders for every
rotor, so ONE network learns

  p(latent | cs1),  p(latent | cs2),  p(latent | cs1, cs2),  p(latent | cs1, cs2, cs3)

and every other non-empty subset.  At sampling time the user hands in whichever
subset they care about and switches the rest off by setting m_j = 0.  No retrain
per combination, and the cs1-only arm degrades gracefully into the v1 behaviour.

Everything else is unchanged from v1 on purpose: same latent space, same decoder,
same frozen six-order surrogate, same cosine schedule, same reverse chain
(imported from 05_latent_ddpm.py so there is one implementation, not two).
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(os.path.dirname(HERE), "spec_design")
for path in (V1_DIR, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import spec_common as S          # noqa: E402
import spec_eval as E            # noqa: E402
import spec_opt as O             # noqa: E402
import latent_design as LD       # noqa: E402
import latent_torch as LT        # noqa: E402
import spec_interval as SI       # noqa: E402

OUT_DIR = os.path.join(HERE, "outputs")
ORDERS = (1, 2, 3)
N_ORD = len(ORDERS)
CS_COLS = ["cs_%d_rpm" % i for i in range(1, 7)]

_V1 = None


def v1():
    """The frozen v1 module: model, schedule and reverse chain live there."""
    global _V1
    if _V1 is None:
        path = os.path.join(V1_DIR, "05_latent_ddpm.py")
        spec = importlib.util.spec_from_file_location("v1_latent_ddpm", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["v1_latent_ddpm"] = module
        spec.loader.exec_module(module)
        _V1 = module
    return _V1


# --------------------------------------------------------------------------- #
# condition
# --------------------------------------------------------------------------- #
def order_scalers(ctx):
    """Per-order z-scoring of log(cs_j) over the training pool.

    Per order, not pooled: cs1 spans 3.6 decades and cs3 only 2.9, so one shared
    mean would make the cs3 channel a near-constant and waste the mask.
    """
    cs = ctx["Y"][:, [o - 1 for o in ORDERS]]
    log_cs = np.log(np.maximum(cs, 1.0))
    return {"mean": log_cs.mean(axis=0).tolist(),
            "std": np.maximum(log_cs.std(axis=0), 1e-9).tolist()}


def finite_ctx(ctx, orders=ORDERS):
    """Drop the rows whose requested orders carry no label.

    The v4.7 pool is censored in the tail: 3 rotors have no cs3, 81 no cs4, 811
    no cs5 and 6 849 no cs6 (ROSS refused to certify the mode).  v1 never noticed
    because it only ever read cs1; conditioning on cs2/cs3 turns those blanks
    into NaN in Z and kills the loss on the very first batch.
    """
    cs = ctx["Y"][:, [o - 1 for o in orders]]
    keep = np.isfinite(cs).all(axis=1)
    if keep.all():
        return ctx, 0
    out = dict(ctx)
    for key in ("X", "Y", "designs", "nd", "nb", "material"):
        value = out.get(key)
        if value is not None and len(value) == len(keep):
            out[key] = value[keep]
    out["pool"] = ctx["pool"][keep].reset_index(drop=True)
    out["scales"] = np.maximum(out["designs"].std(axis=0), 1e-9)
    return out, int((~keep).sum())


def sample_masks(n, policy, seed):
    """(n, 3) 0/1 mask over the orders, drawn ONCE per rotor and reused all epoch.

    Redrawing every epoch would turn the condition itself into noise.  `mixed`
    weights single-order conditions highest so the cs1-only behaviour -- the one
    the v1 protocol is scored on -- stays sharp.
    """
    rng = np.random.default_rng(int(seed) + 12345)
    mask = np.zeros((n, N_ORD), dtype=np.float64)
    if policy == "all3":
        mask[:] = 1.0
        return mask
    if policy == "single":
        mask[np.arange(n), rng.integers(0, N_ORD, size=n)] = 1.0
        return mask
    if policy != "mixed":
        raise ValueError("unknown mask policy %r" % policy)
    k = rng.choice([1, 2, 3], size=n, p=[0.45, 0.35, 0.20])
    for i in range(n):
        for j in rng.choice(N_ORD, size=int(k[i]), replace=False):
            mask[i, j] = 1.0
    return mask


def build_arrays(ctx, mask_policy="mixed", seed=0, layout="point",
                 h_max=0.7, point_prob=0.40, open_prob=0.25):
    """Latent codes + conditions.

    layout="point"    [ onehot | z1 z2 z3 m1 m2 m3 ]                    18 dims
    layout="interval" [ onehot | lo1 lo2 lo3 hi1 hi2 hi3 m1 m2 m3 ]     21 dims

    In the interval layout every rotor is handed a *range* rather than a value:
    with probability `point_prob` the range collapses on its own cs (a point
    spec), with `open_prob` it is one-sided (cs >= own, the "margin only" spec),
    otherwise it is a two-sided range of a random log width up to `h_max` with
    the rotor placed at a random relative position inside it.  The range always
    contains the rotor's own value, so the conditional is honest.
    """
    design, nd, nb = ctx["designs"], ctx["nd"], ctx["nb"]
    U = np.zeros((len(design), LD.LAT_DIM), dtype=np.float32)
    for i in range(len(design)):
        U[i] = LD.encode(design[i], (int(nd[i]), int(nb[i])))
    scalers = order_scalers(ctx)
    cs = ctx["Y"][:, [o - 1 for o in ORDERS]]
    log_cs = np.log(np.maximum(cs, 1.0))
    mean = np.asarray(scalers["mean"])
    std = np.asarray(scalers["std"])
    mask = sample_masks(len(design), mask_policy, seed)
    onehot = np.array([S.material_onehot(m) + S.family_onehot(int(a), int(b))
                       for m, a, b in zip(ctx["material"], nd, nb)], dtype=np.float64)
    if layout == "interval":
        rng = np.random.default_rng(int(seed) + 777)
        shape = rng.random((len(design), N_ORD))
        width = rng.uniform(0.05, float(h_max), size=(len(design), N_ORD))
        pos = rng.random((len(design), N_ORD))
        log_lo = log_cs - pos * width
        log_hi = log_cs + (1.0 - pos) * width
        exact = shape < float(point_prob)
        log_lo = np.where(exact, log_cs, log_lo)
        log_hi = np.where(exact, log_cs, log_hi)
        opened = (shape >= float(point_prob)) & (shape < float(point_prob) + float(open_prob))
        log_lo = np.where(opened, log_cs, log_lo)
        log_hi = np.where(opened, log_cs + width, log_hi)
        Y = np.concatenate([onehot,
                            ((log_lo - mean) / std) * mask,
                            ((log_hi - mean) / std) * mask,
                            mask], axis=1)
    elif layout == "point":
        Y = np.concatenate([onehot, ((log_cs - mean) / std) * mask, mask], axis=1)
    else:
        raise ValueError("unknown layout %r" % layout)
    Y = Y.astype(np.float32)
    if not np.isfinite(Y).all():
        raise RuntimeError("non-finite condition: an order in ORDERS has a "
                           "missing label on some rotor; run finite_ctx()")
    scalers.update({"orders": list(ORDERS), "mask_policy": mask_policy,
                    "layout": layout, "h_max": float(h_max),
                    "point_prob": float(point_prob),
                    "vector": ("[material 3 | family 9 | lo1-3 hi1-3 m1-3]"
                               if layout == "interval" else
                               "[material 3 | family 9 | z1-3 m1-3]"),
                    "condition_dim": int(Y.shape[1])})
    return U, Y, scalers


def condition_for(material, family, specs, scalers, layout=None, relax=0.0):
    """specs: {order: (lo, hi)}.  Orders absent from the dict are switched off.

    The band is used exactly as the user wrote it (relax=0); `relax` exists so a
    caller can deliberately ask for a widened band.  Judging uses the unwidened
    band -- conditioning and scoring must not drift apart silently.
    """
    if layout is None:
        layout = scalers.get("layout", "point")
    mean = np.asarray(scalers["mean"])
    std = np.asarray(scalers["std"])
    h_max = float(scalers.get("h_max", 0.7))
    low, high, ms = [], [], []
    for i, order in enumerate(ORDERS):
        band = specs.get(order)
        if band is None:
            low.append(0.0)
            high.append(0.0)
            ms.append(0.0)
            continue
        lo = max(float(band[0]) * (1.0 - relax), 1.0)
        hi = max(float(band[1]) * (1.0 + relax), lo)
        if layout == "interval":
            lo, hi = SI.effective(lo, hi, h_max)
            low.append((math.log(lo) - mean[i]) / std[i])
            high.append((math.log(hi) - mean[i]) / std[i])
        else:
            low.append((math.log(SI.centre(lo, hi)) - mean[i]) / std[i])
            high.append(0.0)
        ms.append(1.0)
    base = S.material_onehot(material) + S.family_onehot(int(family[0]), int(family[1]))
    if layout == "interval":
        return np.array(base + low + high + ms, dtype=np.float64)
    return np.array(base + low + ms, dtype=np.float64)


# --------------------------------------------------------------------------- #
# differentiable multi-order surrogate (for the guidance step)
# --------------------------------------------------------------------------- #
class SmoothSurrogateMulti(O.SmoothSurrogate):
    """SmoothSurrogate predicts cs1 only; v2 needs any order."""

    def cs(self, x, material, nd, nb, order):
        torch = self.torch
        feats = self._features(x, material, nd, nb)
        k = int(order) - 1
        logs = []
        for item in self.s.models:
            rows = feats.clone()
            rows[:, self.s.log_columns] = torch.log(torch.clamp(
                rows[:, self.s.log_columns],
                min=torch.as_tensor(self.s.log_floor, dtype=rows.dtype)))
            scaled = (rows - torch.as_tensor(item["x_mean"], dtype=rows.dtype)) \
                / torch.as_tensor(item["x_scale"], dtype=rows.dtype)
            dtype = next(item["net"].parameters()).dtype
            out = item["net"](scaled.to(dtype))
            logs.append(out[:, k] * float(item["y_std"][k]) + float(item["y_mean"][k]))
        return torch.exp(torch.stack(logs, dim=0).mean(dim=0))

    def cs1(self, x, material, nd, nb):
        return self.cs(x, material, nd, nb, 1)


def make_guide(surrogate, families, material, specs, strength):
    """Push each x0 estimate towards the user's band while the chain runs.

    The loss is a hinge in log space: it is exactly the old point loss when the
    band is a point, and it is flat inside a real interval -- which is what we
    want, because inside the interval there is nothing to correct, and pushing
    towards the centre would undo the width conditioning we just trained.
    """
    smooth = SmoothSurrogateMulti(surrogate, torch)
    groups = {}
    for i, fam in enumerate(families):
        groups.setdefault((int(fam[0]), int(fam[1])), []).append(i)
    index = {fam: torch.as_tensor(idx, dtype=torch.long) for fam, idx in groups.items()}
    bands = {int(o): (math.log(max(float(b[0]), 1.0)), math.log(max(float(b[1]), 1.0)))
             for o, b in specs.items()}
    # one step of the joint descent must not take three times the step of the
    # single-order case just because three orders are pinned: normalise by count.
    step_scale = float(strength) / max(len(bands), 1)

    def step(x0, a):
        with torch.enable_grad():
            for fam, idx in index.items():
                u = x0[idx].detach().clone().to(torch.float64).requires_grad_(True)
                x = LT.decode_torch(torch, u, fam)
                loss = torch.zeros((), dtype=torch.float64)
                for order, (log_lo, log_hi) in bands.items():
                    value = torch.log(torch.clamp(
                        smooth.cs(x, material, fam[0], fam[1], order), min=1.0))
                    loss = loss + 0.5 * (
                        torch.clamp(log_lo - value, min=0.0) ** 2
                        + torch.clamp(value - log_hi, min=0.0) ** 2).sum()
                grad = torch.autograd.grad(loss, u)[0]
                x0[idx] = (u - (step_scale * a) * grad).detach().to(x0.dtype)
        return x0

    return step


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ctx, dropped = finite_ctx(E.pool_ctx())
    if dropped:
        print("[v2] dropped %d rotors with a missing cs1/cs2/cs3 label"
              % dropped, flush=True)
    started = time.time()
    U, Y, scalers = build_arrays(ctx, args.mask_policy, args.seed, args.layout,
                                 args.h_max, args.point_prob)
    print("latent set %s  condition %s  mask %s  (%.0f s)"
          % (U.shape, Y.shape, args.mask_policy, time.time() - started), flush=True)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(U))
    n_val = max(2000, int(args.val_frac * len(U)))
    val_idx, train_idx = order[:n_val], order[n_val:]
    print("train %d  val %d" % (len(train_idx), len(val_idx)), flush=True)

    module = v1().ddpm()
    betas, alpha_bar = module.cosine_schedule(v1().T_STEPS)
    abar_t = torch.as_tensor(alpha_bar, dtype=torch.float32)
    hidden = tuple(args.hidden)
    model = v1().build_denoiser(module, LD.LAT_DIM, int(Y.shape[1]), hidden, args.time_dim)
    print("denoiser params %.2fM hidden %s time_dim %d"
          % (sum(p.numel() for p in model.parameters()) / 1e6, hidden, args.time_dim),
          flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    ema = ({k: t.detach().clone().float() for k, t in model.state_dict().items()}
           if args.ema else None)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    masks = v1().MASKS
    x_all = torch.as_tensor(U[train_idx])
    y_all = torch.as_tensor(Y[train_idx])
    w_all = torch.as_tensor(np.stack([masks[(int(a), int(b))]
                                      for a, b in zip(ctx["nd"][train_idx],
                                                      ctx["nb"][train_idx])]))
    x_val = torch.as_tensor(U[val_idx])
    y_val = torch.as_tensor(Y[val_idx])
    w_val = torch.as_tensor(np.stack([masks[(int(a), int(b))]
                                      for a, b in zip(ctx["nd"][val_idx],
                                                      ctx["nb"][val_idx])]))
    generator = torch.Generator().manual_seed(args.seed)
    out_dir = os.path.join(OUT_DIR, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    def masked_mse(pred, target, weight):
        return ((pred - target) ** 2 * weight).sum() / weight.sum()

    def target_of(a, noise, clean):
        if args.predict == "v":
            return a.sqrt() * noise - (1.0 - a).sqrt() * clean
        if args.predict == "x0":
            return clean
        return noise

    if args.min_snr and float(args.min_snr) > 0:
        snr = abar_t / (1.0 - abar_t)
        snr_coef = torch.clamp(snr, max=float(args.min_snr)) / snr
    else:
        snr_coef = torch.ones_like(abar_t)

    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(x_all.shape[0], generator=generator)
        running, seen = 0.0, 0
        for start in range(0, perm.numel(), args.batch_size):
            index = perm[start:start + args.batch_size]
            xb, yb, wb = x_all[index], y_all[index], w_all[index]
            t = torch.randint(0, v1().T_STEPS, (xb.shape[0],), generator=generator)
            noise = torch.randn(xb.shape, generator=generator)
            a = abar_t[t].unsqueeze(1)
            x_t = (a.sqrt() * xb + (1.0 - a).sqrt() * noise) * wb
            if args.condition_dropout > 0:
                drop = (torch.rand(xb.shape[0], generator=generator) < args.condition_dropout)
                yb = yb.clone()
                yb[drop] = 0.0
            cw = snr_coef[t].unsqueeze(1) * wb
            loss = masked_mse(model(x_t, t, yb), target_of(a, noise, xb), cw)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                v1()._ema_update(ema, model, args.ema_decay)
            running += float(loss.item()) * xb.shape[0]
            seen += xb.shape[0]
        scheduler.step()

        model.eval()
        with torch.no_grad():
            t = torch.randint(0, v1().T_STEPS, (x_val.shape[0],), generator=generator)
            a = abar_t[t].unsqueeze(1)
            noise = torch.randn(x_val.shape, generator=generator)
            val = float(masked_mse(
                model((a.sqrt() * x_val + (1.0 - a).sqrt() * noise) * w_val, t, y_val),
                target_of(a, noise, x_val),
                snr_coef[t].unsqueeze(1) * w_val).item())
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_loss": val})
        if val < best:
            best = val
            torch.save({"state_dict": model.state_dict(), "ema_state_dict": ema,
                        "config": {"design_dim": LD.LAT_DIM, "condition_dim": int(Y.shape[1]),
                                   "hidden": [int(v) for v in hidden],
                                   "time_dim": int(args.time_dim),
                                   "layout": args.layout, "h_max": float(args.h_max)},
                        "scalers": scalers, "tag": args.tag, "latent": True,
                        "predict": args.predict, "ema_decay": float(args.ema_decay),
                        "condition_dropout": float(args.condition_dropout),
                        "cond_encoding": "multi_order_mask", "mask_policy": args.mask_policy},
                       os.path.join(out_dir, "model.pt"))
        if epoch % args.log_every == 0 or epoch == args.epochs:
            print("epoch %4d | train %.5f | val %.5f | lr %.2e | %.0fs"
                  % (epoch, history[-1]["train_loss"], val,
                     float(optimizer.param_groups[0]["lr"]), time.time() - started), flush=True)
    S.save_json(os.path.join(out_dir, "training_log.json"),
                {"tag": args.tag, "epochs": args.epochs, "hidden": list(hidden),
                 "mask_policy": args.mask_policy, "layout": args.layout,
                 "condition_dim": int(Y.shape[1]),
                 "batch_size": args.batch_size, "n_train": int(len(train_idx)),
                 "n_val": int(len(val_idx)), "best_val_loss": best,
                 "seconds": time.time() - started, "history": history})
    print("[train] best val %.5f -> %s" % (best, os.path.join(out_dir, "model.pt")), flush=True)


# --------------------------------------------------------------------------- #
# sample
# --------------------------------------------------------------------------- #
def parse_targets(text):
    """Kept for compatibility -- interval parsing lives in spec_interval."""
    return SI.parse_specs(text)


def family_set(text, count, rng):
    if text:
        return v1().parse_families(text)
    return list(v1().FAMILIES)


def load_model(tag, use_ema):
    path = os.path.join(OUT_DIR, tag, "model.pt")
    if not os.path.exists(path):
        raise SystemExit("no checkpoint at %s" % path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    module = v1().ddpm()
    model = v1().build_denoiser(module, cfg["design_dim"], cfg["condition_dim"],
                                cfg["hidden"], cfg["time_dim"])
    state = payload["ema_state_dict"] if (use_ema and payload.get("ema_state_dict")) \
        else payload["state_dict"]
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.eval()
    return payload, model, cfg


def sample(args):
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    torch_gen = torch.Generator().manual_seed(args.seed)
    payload, model, cfg = load_model(args.tag, args.use_ema)
    scalers = payload["scalers"]
    layout = cfg.get("layout") or scalers.get("layout") or "point"
    predict = payload.get("predict", "eps")
    dropout = float(payload.get("condition_dropout", 0.0))
    specs = SI.parse_specs(args.targets)
    for order in specs:
        if order not in ORDERS:
            raise SystemExit("v2 handles orders %s only" % (list(ORDERS),))
    if not specs:
        raise SystemExit("--targets is required, e.g. --targets 1=5000 "
                         "or --targets \"1=[10000,12000]\"")

    families = family_set(args.fam_allowed, args.n_generate, rng)
    fams = [families[i] for i in rng.integers(0, len(families), size=args.n_generate)]
    cond = np.stack([condition_for(args.material, f, specs, scalers, layout)
                     for f in fams])
    print("[sample] tag %s | layout %s | orders %s | bands %s | relax +-%.2f%% | "
          "material %s | families %d"
          % (args.tag, layout, sorted(specs),
             {o: SI.band_text(*specs[o]) for o in sorted(specs)},
             100.0 * args.tol, args.material, len(families)), flush=True)

    guide = None
    if args.guide_lambda and float(args.guide_lambda) > 0:
        guide = make_guide(S.load_surrogate(args.surrogate), fams, args.material,
                           specs, float(args.guide_lambda))

    module = v1().ddpm()
    betas, alpha_bar = module.cosine_schedule(v1().T_STEPS)
    u = v1().sample_latent(model, torch.as_tensor(cond, dtype=torch.float32), fams,
                           args.n_generate, args.guidance, dropout, torch_gen,
                           betas, alpha_bar, LD.LAT_DIM, predict,
                           steps=(args.sample_steps or None), guide=guide)
    x = LT.decode_batch(u.detach().numpy().astype(np.float64), fams)
    valid = O.validity(x, args.material, fams)

    surrogate = S.load_surrogate(args.surrogate)
    pred = surrogate.predict(S.canonical_to_model_rows(
        x, [args.material] * len(x), [f[0] for f in fams], [f[1] for f in fams]))

    n_generated = len(fams)
    if args.fam_filter:
        allowed = v1().parse_families(args.fam_filter)
        keep = np.array([(int(f[0]), int(f[1])) in allowed for f in fams])
        n_kept = int(keep.sum())
        print("[sample] post-filter %s: kept %d/%d generated (%.1f%%)"
              % (args.fam_filter, n_kept, n_generated,
                 100.0 * n_kept / max(n_generated, 1)), flush=True)
        if n_kept < args.n_submit:
            print("[sample] note: fewer survivors than --n-submit (%d)" % n_kept,
                  flush=True)
        x = x[keep]
        valid = valid[keep]
        pred = pred[keep]
        fams = [f for f, k in zip(fams, keep) if k]

    dev = np.zeros((len(x), N_ORD))
    hit = np.ones(len(x), dtype=bool)
    per_order = {}
    for i, order in enumerate(ORDERS):
        if order not in specs:
            continue
        lo, hi = specs[order]
        dev[:, i] = SI.deviation(pred[:, order - 1], lo, hi)
        good = SI.inside(pred[:, order - 1], lo, hi, args.tol)
        hit &= good
        per_order[order] = float(good.mean())
    active = np.array([1.0 if o in specs else 0.0 for o in ORDERS])
    score = np.sqrt(((dev ** 2) * active).sum(axis=1))     # joint deviation
    rank = np.argsort(score)

    print("[sample] validity %.4f | joint in-band (proxy) %.4f | per-order %s"
          % (valid.mean(), float(hit.mean()),
             {o: round(v, 4) for o, v in sorted(per_order.items())}), flush=True)
    for k in sorted({int(args.n_generate), int(args.n_submit)}):
        k = min(k, len(x))
        top = rank[:k]
        print("[sample]   top %-6d proxy joint in-band %.4f | median joint dev %.4f"
              % (k, float(hit[top].mean()), float(np.median(score[top]))), flush=True)

    order_keep = rank[:args.n_submit]
    xs = x[order_keep]
    fams_s = [fams[i] for i in order_keep]
    frame = O.make_frame(xs, args.material, fams_s)
    for order in ORDERS:
        frame["cs%d_pred" % order] = np.round(pred[order_keep, order - 1], 2)
        if order in specs:
            lo, hi = specs[order]
            frame["cs%d_lo" % order] = round(lo, 1)
            frame["cs%d_hi" % order] = round(hi, 1)
            frame["cs%d_dev_pct" % order] = np.round(
                100.0 * dev[order_keep, ORDERS.index(order)], 3)
    frame["joint_err"] = np.round(score[order_keep], 5)
    frame.insert(0, "rank", np.arange(len(order_keep)))
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "%s.csv" % args.out_tag)
    frame.to_csv(out_csv, index=False, encoding="utf-8")
    S.save_json(os.path.join(OUT_DIR, "funnel_%s.json" % args.out_tag),
                {"tag": args.tag, "out_tag": args.out_tag, "layout": layout,
                 "bands": {str(o): list(specs[o]) for o in sorted(specs)},
                 "relax": float(args.tol), "material": args.material,
                 "family": args.fam_allowed or "free",
                 "fam_filter": args.fam_filter or "",
                 "n_generated": int(n_generated),
                 "n_kept_after_filter": int(len(fams)),
                 "n_generate": int(args.n_generate), "n_submit": int(args.n_submit),
                 "validity": float(valid.mean()),
                 "proxy_joint_in_band_raw": float(hit.mean()),
                 "proxy_joint_in_band_top": float(hit[order_keep].mean()),
                 "proxy_per_order_top": {str(o): float(
                     SI.inside(pred[order_keep, o - 1], specs[o][0], specs[o][1],
                               args.tol).mean()) for o in sorted(specs)},
                 "median_joint_dev_top": float(np.median(score[order_keep])),
                 "csv": out_csv})
    print("[sample] wrote %s" % out_csv, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "sample"])
    parser.add_argument("--tag", default="v2m")
    parser.add_argument("--out-tag", default="v2run")
    parser.add_argument("--targets", default="",
                        help="1=5000 (point) | \"1=[10000,12000]\" (interval) | "
                             "\"1=[14400,+]\" (one-sided); comma-separated")
    parser.add_argument("--layout", choices=["point", "interval"], default="point",
                        help="condition on a value, or on the two edges of a band")
    parser.add_argument("--h-max", type=float, default=0.70,
                        help="widest band the denoiser is asked to fill, in log units "
                             "(0.70 ~ a 4x ratio)")
    parser.add_argument("--point-prob", type=float, default=0.40,
                        help="training share of exact-point bands in interval layout")
    parser.add_argument("--material", default="Steel", choices=list(S.MATERIAL_ORDER))
    parser.add_argument("--tol", type=float, default=0.05,
                        help="judging relaxation around the band (fraction)")
    parser.add_argument("--fam-allowed", default="")
    parser.add_argument("--fam-filter", default="",
                        help="post-generation family filter, same syntax as "
                             "--fam-allowed (e.g. 3-3x2-2).  The batch is drawn "
                             "free and then only these families are kept, which is "
                             "the 'filter afterwards' alternative to conditioning "
                             "on the family before sampling")
    parser.add_argument("--mask-policy", choices=["mixed", "all3", "single"], default="mixed")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--condition-dropout", type=float, default=0.10)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict", choices=["eps", "v", "x0"], default="v")
    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--use-ema", type=int, default=1)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--min-snr", type=float, default=0.0)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--guide-lambda", type=float, default=0.0)
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument("--n-generate", type=int, default=5000)
    parser.add_argument("--n-submit", type=int, default=200)
    parser.add_argument("--surrogate", default="mlp_best")
    args = parser.parse_args()
    (train if args.mode == "train" else sample)(args)


if __name__ == "__main__":
    main()
