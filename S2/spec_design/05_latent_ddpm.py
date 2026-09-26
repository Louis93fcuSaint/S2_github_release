# -*- coding: utf-8 -*-
"""Step 05 -- target-conditioned DDPM inside the constraint-free latent box.

The generator no longer emits a design; it emits the 34 independent latent
coordinates of latent_design.py, and the decoder maps them onto the admissible
set by construction.  Same network, same reverse process as the frozen
DDPM_v47 module: only the space the model lives in changed.

  condition = [ material one-hot (3) | family one-hot (9) | z(log cs1 target) ]

Slots that the family does not use are pinned to 0 and excluded from the loss.
"""
import argparse
import importlib.util
import math
import os
import zlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_common as S
import spec_eval as E
import spec_opt as O
import latent_design as LD
import latent_torch as LT
import family_policy as FP

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(E.OUT, "latent_ddpm")
T_STEPS = 1000
FAMILIES = [(a, b) for a in S.ND_CHOICES for b in S.NB_CHOICES]
MASKS = {f: LD.used_mask(f).astype(np.float32) for f in FAMILIES}


def expand_span(text, choices):
    """'3' -> [3]; '3-3' -> [3]; '2-4' -> [2, 3, 4]; '3-' / '' -> whole axis.

    A leading axis letter is tolerated, so 'd3' and 'b2' are read as 3 and 2.
    """
    text = text.strip().lower()
    while text and not (text[0].isdigit() or text[0] == "-"):
        text = text[1:]
    if not text:
        return list(choices)
    if "-" not in text:
        return [int(text)]
    lo, _, hi = text.partition("-")
    lo = int(lo) if lo.strip() else min(choices)
    hi = int(hi) if hi.strip() else max(choices)
    if hi < lo:
        raise ValueError("empty range %r" % text)
    return list(range(lo, hi + 1))


def parse_families(text):
    """'d3_b2' / '3x2' / '3:2' / '3-3/2-4' -> [(n_disks, n_bearings)]; '' -> None.

    Both axes take a range: '3-5x2-4' spans 3..5 disks and 2..4 bearings,
    '3x2-4' fixes the disks and restricts the bearings, '3-/2' and '3x' open the
    end of an axis.  '/' is accepted as the axis separator in place of 'x'.
    """
    if not text:
        return None
    out = []
    for token in text.replace(",", " ").split():
        for sep in ("x", "X", "/", "_", ":"):
            if sep in token:
                a, b = token.split(sep)
                break
        else:
            raise ValueError("family token %r, expected dDxB such as 3x2" % token)
        nds = [v for v in expand_span(a, S.ND_CHOICES) if v in S.ND_CHOICES]
        nbs = [v for v in expand_span(b, S.NB_CHOICES) if v in S.NB_CHOICES]
        if not nds or not nbs:
            raise ValueError("family token %r selects no valid family" % token)
        out += [(x, y) for x in nds for y in nbs]
    seen = []
    for family in out:
        if family not in seen:
            seen.append(family)
    return seen


def family_weights(spec, args, policy):
    """None means the historical uniform draw over the 18 families.

    learned : w_f proportional to P_f(cs1 in band), fitted on the pool.
    pinned  : the user fixes or restricts the family, no band preference.
    learned + --fam-allowed restricts the band weights to that subset.
    """
    mode = args.fam_policy
    allowed = parse_families(args.fam_allowed)
    if mode == "uniform":
        if not allowed:
            return None
        w = np.full(len(allowed), 1.0 / len(allowed))
        return {"families": [tuple(f) for f in allowed], "w": w.tolist(),
                "effective": float(len(allowed)), "raw_mass": w.tolist(),
                "mode": "uniform-band"}
    if mode == "pinned":
        if not allowed:
            raise ValueError("--fam-policy pinned needs --fam-allowed")
        w = np.full(len(allowed), 1.0 / len(allowed))
        return {"families": [tuple(f) for f in allowed], "w": w.tolist(),
                "effective": float(len(allowed)), "raw_mass": w.tolist(),
                "mode": mode}
    if mode == "learned":
        row = FP.weights(policy, spec["material"], spec["lower"],
                         spec.get("upper"), allowed=allowed,
                         floor_frac=float(args.fam_floor))
        row["mode"] = mode
        return row
    raise ValueError("unknown family policy %s" % mode)


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


TARGET_FREQS = (0.5, 1.0, 2.0, 4.0)


def target_features(z, encoding="z"):
    """How the requested first critical speed enters the condition.

    "z"       one z-scored log target, which is what every earlier arm used.
    "fourier" the same z plus sin/cos at four frequencies.  The requested speed
              spans three decades, and a 1 % band around it is a very narrow
              slice of that range; a ReLU MLP reading a single scalar resolves
              it with piecewise-constant slopes, while a multi-scale basis gives
              the net structure at both the whole-range and the local scale.
    """
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    if encoding == "z":
        return z[:, None]
    if encoding == "fourier":
        parts = [z[:, None]]
        for w in TARGET_FREQS:
            parts.append(np.sin(w * z)[:, None])
            parts.append(np.cos(w * z)[:, None])
        return np.concatenate(parts, axis=1)
    raise ValueError("unknown target encoding %r" % encoding)


def build_arrays(ctx, encoding="z"):
    """Latent U in [0,1]^34 (invalid-for-family slots pinned to 0) and condition Y."""
    design, nd, nb = ctx["designs"], ctx["nd"], ctx["nb"]
    U = np.zeros((len(design), LD.LAT_DIM), dtype=np.float32)
    for i in range(len(design)):
        fam = (int(nd[i]), int(nb[i]))
        U[i] = LD.encode(design[i], fam)
    log_cs = np.log(np.maximum(ctx["Y"][:, 0], 1.0))
    cs_mean, cs_std = float(log_cs.mean()), float(log_cs.std())
    z = (log_cs - cs_mean) / cs_std
    onehot = np.array([S.material_onehot(m) + S.family_onehot(int(a), int(b))
                       for m, a, b in zip(ctx["material"], nd, nb)], dtype=np.float64)
    Y = np.concatenate([onehot, target_features(z, encoding)],
                       axis=1).astype(np.float32)
    return U, Y, {"cs_mean": cs_mean, "cs_std": cs_std}


def condition_for(material, family, cs_target, scalers, encoding="z"):
    z = (np.log(max(cs_target, 1.0)) - scalers["cs_mean"]) / scalers["cs_std"]
    feats = target_features(z, encoding)[0]
    return np.array(S.material_onehot(material)
                    + S.family_onehot(int(family[0]), int(family[1]))
                    + list(feats), dtype=np.float64)


def _mask_tensor(families, device=None):
    return torch.as_tensor(np.stack([MASKS[f] for f in families]))


@torch.no_grad()
def sample_latent(model, condition, families, n, guidance, dropout, generator,
                  betas, alpha_bar, dim, predict="eps", steps=None, guide=None):
    """Reverse chain with the family-invalid coordinates pinned to 0.

    predict="eps" is the classic DDPM parameterisation; predict="v" is the
    rotated target v = sqrt(a) eps - sqrt(1-a) x0, whose conditioning stays
    sane at small t, where the fine structure that decides cs1 lives; predict
    ="x0" asks the denoiser to emit the clean latent directly, so nothing is
    divided by sqrt(a) at the small t where cs1 is decided.

    guide, when given, is a callable (x0, a) -> x0 applied after every x0
    estimate; see make_guide.

    steps < len(betas) walks a strided sub-schedule with the same endpoints; the
    jump then uses the exact ratio alpha = abar[t] / abar[t_prev].
    """
    model.eval()
    mask = _mask_tensor(families)
    x = torch.randn((n, dim), generator=generator) * mask
    betas_t = torch.as_tensor(betas, dtype=torch.float32)
    abar_t = torch.as_tensor(alpha_bar, dtype=torch.float32)
    zero = torch.zeros_like(condition)
    full = int(steps or len(betas)) >= len(betas)
    index = (list(range(len(betas))) if full else sorted(set(
        np.linspace(0, len(betas) - 1, int(steps)).round().astype(int).tolist())))
    for k in range(len(index) - 1, -1, -1):
        step = index[k]
        t = torch.full((n,), step, dtype=torch.long)
        out_cond = model(x, t, condition)
        if guidance != 1.0 and dropout > 0:
            out_uncond = model(x, t, zero)
            out = out_uncond + guidance * (out_cond - out_uncond)
        else:
            out = out_cond
        a = abar_t[step]
        a_prev = abar_t[index[k - 1]] if k > 0 else torch.tensor(1.0)
        beta = betas_t[step] if full else (1.0 - a / a_prev)
        alpha = 1.0 - beta
        if predict == "v":
            x0 = a.sqrt() * x - (1.0 - a).sqrt() * out
        elif predict == "x0":
            x0 = out
        else:
            x0 = (x - (1.0 - a).sqrt() * out) / a.sqrt()
        if guide is not None:
            x0 = guide(x0, a)
        if k > 0:
            mean = (a_prev.sqrt() * beta / (1.0 - a)) * x0 \
                + (alpha.sqrt() * (1.0 - a_prev) / (1.0 - a)) * x
            sigma = (beta * (1.0 - a_prev) / (1.0 - a)).sqrt()
            x = mean + sigma * torch.randn(x.shape, generator=generator)
        else:
            x = x0
        x = x * mask
    return x


def make_guide(surrogate, families, material, aim, strength):
    """Likelihood-ascent step on every predicted x0, per family.

    A scalar target in the condition is a weak constraint: the denoiser learns
    the whole conditional spread instead of the slice that hits the number, and
    retraining is the expensive way to fix that.  The frozen MLP surrogate is
    already differentiable (`spec_opt.SmoothSurrogate`, the mirror the `adam`
    baseline optimises through), and it is the same model the selection stage
    ranks with, so its gradient can be spent directly inside the reverse chain.

    The step nudges x0 down the surrogate's log-cs1 error, scaled by a = abar[t]
    so that a badly estimated x0 at high noise is corrected less than a sharp
    one near the end.
    """
    smooth = O.SmoothSurrogate(surrogate, torch)
    groups = {}
    for i, fam in enumerate(families):
        groups.setdefault((int(fam[0]), int(fam[1])), []).append(i)
    index = {fam: torch.as_tensor(idx, dtype=torch.long)
             for fam, idx in groups.items()}
    log_target = math.log(max(float(aim), 1.0))

    def step(x0, a):
        with torch.enable_grad():
            for fam, idx in index.items():
                u = x0[idx].detach().clone().to(torch.float64).requires_grad_(True)
                x = LT.decode_torch(torch, u, fam)
                cs1 = smooth.cs1(x, material, fam[0], fam[1])
                loss = 0.5 * ((torch.log(torch.clamp(cs1, min=1.0))
                               - log_target) ** 2).sum()
                grad = torch.autograd.grad(loss, u)[0]
                x0[idx] = (u - (float(strength) * a) * grad).detach().to(x0.dtype)
        return x0

    return step


class TimeDenoiser(torch.nn.Module):
    """The frozen Denoiser MLP with a configurable time-embedding width.

    time_dim 64 reproduces DDPM_v47 byte for byte; wider embeddings give the
    network finer resolution in t, which is the only input that tells it which
    noise level it is denoising.
    """

    def __init__(self, design_dim=18, condition_dim=6, hidden=(256, 256, 256),
                 time_dim=64):
        super().__init__()
        self.time_dim = int(time_dim)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.time_dim, hidden[0]), torch.nn.SiLU(),
            torch.nn.Linear(hidden[0], hidden[0]), torch.nn.SiLU())
        layers = []
        previous = design_dim + condition_dim + hidden[0]
        for width in hidden:
            layers.extend([torch.nn.Linear(previous, width), torch.nn.SiLU()])
            previous = width
        layers.append(torch.nn.Linear(previous, design_dim))
        self.body = torch.nn.Sequential(*layers)

    def time_embedding(self, t, device):
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000.0)
                           * torch.arange(half, device=device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    def forward(self, x, t, condition):
        embedding = self.time_mlp(self.time_embedding(t, x.device))
        return self.body(torch.cat([x, condition, embedding], dim=1))


def build_denoiser(module, design_dim, condition_dim, hidden, time_dim):
    if int(time_dim) == 64:
        return module.Denoiser(design_dim=design_dim, condition_dim=condition_dim,
                               hidden=tuple(hidden))
    return TimeDenoiser(design_dim=design_dim, condition_dim=condition_dim,
                        hidden=tuple(hidden), time_dim=int(time_dim))


def _ema_update(ema, model, decay):
    """In-place exponential moving average of every floating-point tensor."""
    with torch.no_grad():
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                ema[key].mul_(decay).add_(value.detach().float(), alpha=1.0 - decay)
            else:
                ema[key].copy_(value.detach())


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ctx = E.pool_ctx()
    started = time.time()
    U, Y, scalers = build_arrays(ctx, args.cond_encoding)
    print("latent set %s  condition %s  (%.0f s)" % (U.shape, Y.shape, time.time() - started),
          flush=True)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(U))
    n_val = max(2000, int(args.val_frac * len(U)))
    val_idx, train_idx = order[:n_val], order[n_val:]
    print("train %d  val %d" % (len(train_idx), len(val_idx)), flush=True)

    module = ddpm()
    betas, alpha_bar = module.cosine_schedule(T_STEPS)
    betas_t = torch.as_tensor(betas, dtype=torch.float32)
    abar_t = torch.as_tensor(alpha_bar, dtype=torch.float32)
    hidden = tuple(args.hidden)
    model = build_denoiser(module, LD.LAT_DIM, Y.shape[1], hidden, args.time_dim)
    print("denoiser params %.2fM hidden %s time_dim %d" % (
        sum(p.numel() for p in model.parameters()) / 1e6, hidden, args.time_dim), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    ema = ({k: v.detach().clone().float() for k, v in model.state_dict().items()}
           if args.ema else None)
    print("predict %s | ema %s (decay %.4f)" % (args.predict, args.ema, args.ema_decay),
          flush=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    x_all = torch.as_tensor(U[train_idx]); y_all = torch.as_tensor(Y[train_idx])
    w_all = torch.as_tensor(np.stack([MASKS[(int(a), int(b))]
                                      for a, b in zip(ctx["nd"][train_idx],
                                                      ctx["nb"][train_idx])]))
    x_val = torch.as_tensor(U[val_idx]); y_val = torch.as_tensor(Y[val_idx])
    w_val = torch.as_tensor(np.stack([MASKS[(int(a), int(b))]
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
            t = torch.randint(0, T_STEPS, (xb.shape[0],), generator=generator)
            noise = torch.randn(xb.shape, generator=generator)
            a = abar_t[t].unsqueeze(1)
            x_t = (a.sqrt() * xb + (1.0 - a).sqrt() * noise) * wb
            if args.condition_dropout > 0:
                drop = (torch.rand(xb.shape[0], generator=generator)
                        < args.condition_dropout)
                yb = yb.clone(); yb[drop] = 0.0
            cw = snr_coef[t].unsqueeze(1) * wb
            loss = masked_mse(model(x_t, t, yb), target_of(a, noise, xb), cw)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                _ema_update(ema, model, args.ema_decay)
            running += float(loss.item()) * xb.shape[0]
            seen += xb.shape[0]
        scheduler.step()

        model.eval()
        with torch.no_grad():
            t = torch.randint(0, T_STEPS, (x_val.shape[0],), generator=generator)
            a = abar_t[t].unsqueeze(1)
            noise = torch.randn(x_val.shape, generator=generator)
            val = float(masked_mse(
                model((a.sqrt() * x_val + (1.0 - a).sqrt() * noise) * w_val, t, y_val),
                target_of(a, noise, x_val),
                snr_coef[t].unsqueeze(1) * w_val).item())
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_loss": val})
        if val < best:
            best = val
            torch.save({"state_dict": model.state_dict(),
                        "ema_state_dict": ema,
                        "config": {"design_dim": LD.LAT_DIM, "condition_dim": int(Y.shape[1]),
                                   "hidden": [int(v) for v in hidden],
                                   "time_dim": int(args.time_dim)},
                        "scalers": scalers, "tag": args.tag, "latent": True,
                        "predict": args.predict, "ema_decay": float(args.ema_decay),
                        "condition_dropout": float(args.condition_dropout),
                        "cond_encoding": args.cond_encoding},
                       os.path.join(out_dir, "model.pt"))
        if epoch % args.log_every == 0 or epoch == args.epochs:
            print("epoch %4d | train %.5f | val %.5f | lr %.2e | %.0fs"
                  % (epoch, history[-1]["train_loss"], val,
                     float(optimizer.param_groups[0]["lr"]), time.time() - started),
                  flush=True)
    S.save_json(os.path.join(out_dir, "training_log.json"),
                {"tag": args.tag, "epochs": args.epochs, "hidden": list(hidden),
                 "batch_size": args.batch_size, "n_train": int(len(train_idx)),
                 "n_val": int(len(val_idx)), "best_val_loss": best,
                 "condition_dropout": float(args.condition_dropout),
                 "predict": args.predict, "ema": bool(args.ema),
                 "ema_decay": float(args.ema_decay), "time_dim": int(args.time_dim),
                 "min_snr": float(args.min_snr), "sample_steps_default": 0,
                 "seconds": time.time() - started, "history": history})
    print("[train] best val %.5f -> %s" % (best, os.path.join(out_dir, "model.pt")), flush=True)


def sample(args):
    torch.set_num_threads(args.threads)
    out_tag = args.out_tag or args.tag
    ctx = E.pool_ctx()
    module = ddpm()
    payload = torch.load(os.path.join(OUT_DIR, args.tag, "model.pt"), map_location="cpu")
    cfg = payload["config"]; scalers = payload["scalers"]
    model = build_denoiser(module, cfg["design_dim"], cfg["condition_dim"],
                           tuple(cfg["hidden"]), cfg.get("time_dim", 64))
    model.load_state_dict(payload["state_dict"]); model.eval()
    predict = payload.get("predict", "eps")
    used_ema = bool(args.use_ema and payload.get("ema_state_dict"))
    if used_ema:
        model.load_state_dict(payload["ema_state_dict"])
    dropout = float(payload.get("condition_dropout", 0.0))
    print("[sample] tag %s -> out %s | predict %s | ema %s | guidance %.2f | "
          "steps %s | guide-lambda %.3g"
          % (args.tag, out_tag, predict, used_ema, args.guidance,
             args.sample_steps or "full", float(args.guide_lambda)), flush=True)
    betas, alpha_bar = module.cosine_schedule(T_STEPS)
    generator = torch.Generator().manual_seed(args.seed)
    surrogate = S.load_surrogate("mlp_best", threads=args.threads)
    os.makedirs(OUT_DIR, exist_ok=True)

    specs = E.specs(ctx, with_adhoc=True)
    if args.target:
        if not args.material:
            raise SystemExit("--target needs --material (Steel / Aluminum / Titanium)")
        spec = E.adhoc_spec(args.material, args.target, args.tol,
                            upper=(args.target_upper or None),
                            name=(args.spec_name or None), mode=args.spec_mode)
        if spec["mode"] == "lower" and spec.get("upper") is None:
            spec["upper"] = float(ctx["upper"][args.material])
        specs = [E.add_adhoc_spec(spec)]
        if spec["mode"] == "lower":
            print("[sample] ad-hoc requirement %s cs1 >= %.0f rpm (aim %.0f, "
                  "ceiling %.0f) -> spec %s"
                  % (spec["material"], spec["lower"], spec["aim"], spec["upper"],
                     spec["name"]), flush=True)
        else:
            print("[sample] ad-hoc target %s %.0f rpm +-%.1f%% -> spec %s"
                  % (spec["material"], spec["target"], 100.0 * spec["tol"],
                     spec["name"]), flush=True)
    policy = FP.load(ctx=ctx) if args.fam_policy == "learned" else None
    if args.specs and not args.target:
        wanted = set(args.specs.split(","))
        specs = [sp for sp in specs if sp["name"] in wanted]
    funnel = {}
    for spec in specs:
        # crc32, not hash(): str hash is salted per process, so the per-spec
        # family draw used to differ between runs and between arms.
        rng = np.random.default_rng(zlib.crc32(spec["name"].encode("utf-8"))
                                    % 100000 + args.seed)
        n = args.n_generate
        frow = family_weights(spec, args, policy)
        if frow is None:
            fams = [FAMILIES[i % len(FAMILIES)]
                    for i in rng.integers(0, len(FAMILIES), n)]
        else:
            fams = FP.sample(frow, n, rng)
        targets = np.full(n, float(spec["target"]))
        # the encoding lives next to the weights, not in cfg: a checkpoint
        # trained on a wider condition must be sampled with the same one.
        encoding = payload.get("cond_encoding", cfg.get("cond_encoding", "z"))
        if len(condition_for(spec["material"], fams[0], float(spec["target"]),
                             scalers, encoding)) != int(cfg["condition_dim"]):
            raise ValueError("checkpoint %s wants a %d-dim condition but encoding "
                             "%r gives %d dims"
                             % (args.tag, int(cfg["condition_dim"]), encoding,
                                len(condition_for(spec["material"], fams[0],
                                                  float(spec["target"]), scalers,
                                                  encoding))))
        cond = np.stack([condition_for(spec["material"], f, t, scalers, encoding)
                         for f, t in zip(fams, targets)])
        guide = None
        if float(args.guide_lambda) > 0:
            guide = make_guide(surrogate, fams, spec["material"],
                               spec.get("aim") or spec.get("target"),
                               float(args.guide_lambda))
        with torch.no_grad():
            u = sample_latent(model, torch.as_tensor(cond, dtype=torch.float32), fams, n,
                              args.guidance, dropout, generator, betas, alpha_bar,
                              LD.LAT_DIM, predict=predict,
                              steps=(args.sample_steps or None), guide=guide).numpy()
        x = np.stack([LD.decode(u[i], fams[i]) for i in range(n)])
        valid = O.validity(x, spec["material"], fams)
        cs1 = surrogate.predict(S.canonical_to_model_rows(
            x, [spec["material"]] * n, [f[0] for f in fams], [f[1] for f in fams]))[:, 0]
        if spec.get("mode") == "lower":
            in_band = (cs1 >= float(spec["lower"])) & (cs1 <= float(spec["upper"]))
        else:
            in_band = np.abs(cs1 - float(spec["target"])) / float(spec["target"]) <= float(spec["tol"])
        score = O.objective_from_cs1(cs1, spec, None)
        order = np.argsort(-score)[:args.n_submit]
        xs, fams_s = x[order], [fams[i] for i in order]
        fam_mix = {}
        for nd_i, nb_i in fams:
            key = "%d_%d" % (int(nd_i), int(nb_i))
            fam_mix[key] = fam_mix.get(key, 0) + 1
        # Real designs the pool already holds in this band and family band: zero
        # means the shortlist is extrapolation, whatever the surrogate claims.
        in_band_pool = ((ctx["material"] == spec["material"])
                        & (ctx["Y"][:, 0] >= float(spec["lower"]))
                        & (ctx["Y"][:, 0] <= float(spec["upper"])))
        if frow is not None:
            allowed_set = set(frow["families"])
            band_ok = np.array([(int(a), int(b)) in allowed_set
                                for a, b in zip(ctx["nd"], ctx["nb"])])
            in_band_pool = in_band_pool & band_ok
        support = int(in_band_pool.sum())
        funnel[spec["name"]] = {
            "generated": int(n), "valid_rate": float(valid.mean()),
            "mode": spec.get("mode", "band"),
            "family_mix": fam_mix,
            "family_band": (None if frow is None
                            else ["%d_%d" % (int(f[0]), int(f[1]))
                                  for f in frow["families"]]),
            "support_pool_band_family": support,
            "family_weights": (None if frow is None else frow["w"]),
            "family_effective": (None if frow is None else frow["effective"]),
            "in_band_raw": float(in_band.mean()),
            "in_band_top%d" % args.n_submit: float(in_band[order].mean()),
            "median_rel_err_top": float(np.median(np.abs(cs1[order] - float(spec["target"]))
                                                  / float(spec["target"]))),
            "submitted": int(len(xs)),
        }
        # the delivered file has to explain itself: without the prediction next to
        # the geometry, nobody can tell what the shortlist was ranked on.
        frame = O.make_frame(xs, spec["material"], fams_s)
        frame.insert(3, "cs1_pred", np.round(cs1[order], 2))
        aim = spec.get("aim") or spec.get("target")
        if aim:
            frame.insert(4, "rel_err_pct", np.round(
                100.0 * (cs1[order] - float(aim)) / float(aim), 3))
        frame.to_csv(os.path.join(OUT_DIR, "%s__%s.csv" % (out_tag, spec["name"])),
                     index=False, encoding="utf-8")
        if args.dump_batch:
            # Drawn after the family and submit draws, so the batch itself is
            # unchanged; the file is what an unbiased sample of the batch looks
            # like, and it is verified with --select rand.
            pick = rng.choice(n, size=min(int(args.dump_batch), n), replace=False)
            pool = O.make_frame(x[pick], spec["material"], [fams[i] for i in pick])
            pool.insert(3, "cs1_pred", np.round(cs1[pick], 2))
            pool.to_csv(os.path.join(
                OUT_DIR, "%s_pool__%s.csv" % (out_tag, spec["name"])),
                index=False, encoding="utf-8")
            print("[sample] pool sample of %d written to %s_pool__%s.csv"
                  % (len(pick), out_tag, spec["name"]), flush=True)
        print("[sample] %-14s valid %.4f | in-band raw %.4f -> top%d %.4f | med rel err "
              "%.4f | fam %s | pool support %d"
              % (spec["name"], funnel[spec["name"]]["valid_rate"],
                 funnel[spec["name"]]["in_band_raw"], args.n_submit,
                 funnel[spec["name"]]["in_band_top%d" % args.n_submit],
                 funnel[spec["name"]]["median_rel_err_top"],
                 "all 18" if frow is None else "%d" % len(frow["families"]), support),
              flush=True)
    S.save_json(os.path.join(OUT_DIR, "funnel_%s.json" % out_tag), funnel)
    S.save_json(os.path.join(OUT_DIR, "meta_%s.json" % out_tag),
                {"tag": args.tag, "out_tag": out_tag, "predict": predict, "used_ema": used_ema,
                 "fam_policy": args.fam_policy, "fam_allowed": args.fam_allowed,
                 "fam_floor": float(args.fam_floor),
                 "guidance": float(args.guidance),
                 "guide_lambda": float(args.guide_lambda),
                 "sample_steps": int(args.sample_steps or 0),
                 "n_generate": int(args.n_generate), "n_submit": int(args.n_submit),
                 "seed": int(args.seed), "model": os.path.join(OUT_DIR, args.tag, "model.pt")})
    print("[sample] %s" % os.path.join(OUT_DIR, "funnel_%s.json" % out_tag), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "sample"])
    parser.add_argument("--tag", default="armL")
    parser.add_argument("--out-tag", default="")
    parser.add_argument("--specs", default="")
    parser.add_argument("--spec-mode", dest="spec_mode", choices=["band", "lower"],
                        default="band",
                        help="band: accept cs1 inside target +- tol; lower: accept "
                             "cs1 at or above target, tol becomes the aiming margin")
    parser.add_argument("--target", type=float, default=0.0,
                        help="ad-hoc cs1 target in rpm; replaces the frozen spec "
                             "list, requires --material")
    parser.add_argument("--material", default="",
                        choices=[""] + list(S.MATERIAL_ORDER))
    parser.add_argument("--tol", type=float, default=0.05,
                        help="relative band of an ad-hoc target, "
                             "e.g. 0.05 = +-5 percent")
    parser.add_argument("--target-upper", type=float, default=0.0,
                        help="override the upper band edge of an ad-hoc target")
    parser.add_argument("--spec-name", default="",
                        help="name of the ad-hoc spec, default <material>_<rpm>")
    parser.add_argument("--fam-policy", choices=["uniform", "learned", "pinned"],
                        default="uniform")
    parser.add_argument("--fam-allowed", default="",
                        help="restrict the (disks, bearings) family, e.g. 3x2, "
                             "3-3/2-4, 2-4x2-3; empty means no restriction")
    parser.add_argument("--fam-floor", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--condition-dropout", type=float, default=0.10)
    parser.add_argument("--cond-encoding", choices=["z", "fourier"], default="z",
                        help="how the requested first critical speed is encoded in "
                             "the condition; fourier adds sin/cos at four frequencies")
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--n-generate", type=int, default=1500)
    parser.add_argument("--n-submit", type=int, default=200)
    parser.add_argument("--dump-batch", type=int, default=0,
                        help="also write a random sample of this many designs from "
                             "the whole generated batch, so the batch's own in-band "
                             "rate can be measured without the ranking in between")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict", choices=["eps", "v", "x0"], default="eps")
    parser.add_argument("--guide-lambda", type=float, default=0.0,
                        help="strength of the differentiable-surrogate step "
                             "applied to every predicted x0 at sampling time")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--use-ema", type=int, default=1)
    parser.add_argument("--sample-steps", type=int, default=0)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--min-snr", type=float, default=0.0)
    args = parser.parse_args()
    (train if args.mode == "train" else sample)(args)


if __name__ == "__main__":
    main()