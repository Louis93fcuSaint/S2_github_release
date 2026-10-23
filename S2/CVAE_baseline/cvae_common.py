# -*- coding: utf-8 -*-
"""Shared CVAE baseline utilities.

The baseline deliberately reuses the DDPM paper stack:
  * the same 34-D feasibility-preserving latent space (latent_design.py);
  * the same physical decoder g_theta (latent_torch.py);
  * the same 21-D interval condition vector (latent_ddpm_v2.build_arrays);
  * the same six frozen specifications, 5 000 candidates, 200-design
    pre-screening shortlist and ROSS re-computation protocol.

Only the generative model changes: a conditional VAE q(z|u,c) p(u|z,c)
instead of a conditional DDPM.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
S2 = os.path.dirname(HERE)
V1_DIR = os.path.join(S2, "spec_design")
V2_DIR = os.path.join(S2, "spec_design_v2")
for _path in (V1_DIR, V2_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import spec_common as S          # noqa: E402
import spec_eval as E            # noqa: E402
import spec_opt as O             # noqa: E402
import spec_interval as SI       # noqa: E402
import latent_design as LD       # noqa: E402
import latent_torch as LT        # noqa: E402
import latent_ddpm_v2 as V2D     # noqa: E402

OUT_DIR = os.path.join(HERE, "outputs")
DATA_DIR = os.path.join(HERE, "data")
POOL_DIR = os.path.join(HERE, "pools")
CAND_DIR = os.path.join(HERE, "candidates")
VERIFY_DIR = os.path.join(HERE, "verify")
RESULT_DIR = os.path.join(HERE, "results")
LOG_DIR = os.path.join(HERE, "logs")

for _path in (OUT_DIR, DATA_DIR, POOL_DIR, CAND_DIR, VERIFY_DIR, RESULT_DIR, LOG_DIR):
    os.makedirs(_path, exist_ok=True)

SPECS = [
    {"name": "Steel_mid", "material": "Steel", "target": 3600.0},
    {"name": "Steel_high", "material": "Steel", "target": 16000.0},
    {"name": "Aluminum_mid", "material": "Aluminum", "target": 5100.0},
    {"name": "Aluminum_high", "material": "Aluminum", "target": 23000.0},
    {"name": "Titanium_mid", "material": "Titanium", "target": 4300.0},
    {"name": "Titanium_high", "material": "Titanium", "target": 19000.0},
]
SEEDS = [42, 7, 13]
N_GENERATE = 5000
N_SUBMIT = 200
TOL = 0.05
GUIDE_LAMBDA = 0.05
GUIDE_STEPS = 25


def spec_by_name(name: str) -> dict:
    for spec in SPECS:
        if spec["name"] == name:
            return dict(spec)
    raise KeyError(name)


def frozen_spec(spec: dict) -> dict:
    """Return the paper's frozen point target with its +-5% judging band."""
    return {
        "name": spec["name"], "material": spec["material"], "target": float(spec["target"]),
        "tol": TOL,
        "lower": float(spec["target"]) * (1.0 - TOL),
        "upper": float(spec["target"]) * (1.0 + TOL),
    }


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_training_arrays(layout: str = "interval", mask_policy: str = "mixed",
                         seed: int = 42, cache: bool = True):
    """Build or load the exact latent/condition arrays used by the DDPM."""
    tag = "%s_%s_s%d" % (layout, mask_policy, seed)
    npz_path = os.path.join(DATA_DIR, "arrays_%s.npz" % tag)
    json_path = os.path.join(DATA_DIR, "arrays_%s.json" % tag)
    if cache and os.path.exists(npz_path) and os.path.exists(json_path):
        blob = np.load(npz_path, allow_pickle=False)
        with open(json_path, encoding="utf-8") as handle:
            scalers = json.load(handle)
        nd = blob["nd"].astype(int)
        nb = blob["nb"].astype(int)
        material = blob["material"].astype(str)
        return (blob["U"].astype(np.float32), blob["Y"].astype(np.float32),
                nd, nb, material, scalers)
    ctx, dropped = V2D.finite_ctx(E.pool_ctx())
    U, Y, scalers = V2D.build_arrays(ctx, mask_policy=mask_policy, seed=seed,
                                     layout=layout)
    nd = ctx["nd"].astype(int)
    nb = ctx["nb"].astype(int)
    material = ctx["material"].astype(str)
    if cache:
        np.savez_compressed(npz_path, U=U, Y=Y, nd=nd, nb=nb, material=material)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(scalers, handle, indent=2, ensure_ascii=False)
    print("[cvae-data] dropped=%d U=%s Y=%s" % (dropped, U.shape, Y.shape), flush=True)
    return U, Y, nd, nb, material, scalers


def family_masks(families: Sequence[tuple[int, int]]) -> np.ndarray:
    return np.stack([LD.used_mask((int(a), int(b))) for a, b in families]).astype(np.float32)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def _mlp(in_dim: int, hidden: Iterable[int], out_dim: int) -> nn.Sequential:
    layers = []
    last = int(in_dim)
    for width in hidden:
        width = int(width)
        layers += [nn.Linear(last, width), nn.LayerNorm(width), nn.SiLU()]
        last = width
    layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)


class ConditionalVAE(nn.Module):
    """CVAE in the same rotor-parameter latent space as the DDPM.

    Encoder: q(z | u, c).  Decoder: p(u | z, c), with a sigmoid output so that
    the output can be passed straight to the frozen physical decoder g_theta.
    Unused family slots are masked to zero in both the loss and the sample.
    """

    def __init__(self, u_dim: int = LD.LAT_DIM, c_dim: int = 21,
                 z_dim: int = 32, hidden: Sequence[int] = (512, 512, 512)):
        super().__init__()
        self.u_dim = int(u_dim)
        self.c_dim = int(c_dim)
        self.z_dim = int(z_dim)
        self.hidden = tuple(int(v) for v in hidden)
        self.encoder = _mlp(self.u_dim + self.c_dim, self.hidden, 2 * self.z_dim)
        # A conditional prior p(z|c) is part of the CVAE, not an extra model.
        # It keeps prior sampling aligned with the conditional posterior and is
        # essential here because the condition (a target band) is much more
        # informative than an unconditional N(0, I).
        self.prior = _mlp(self.c_dim, (256, 256), 2 * self.z_dim)
        self.decoder = _mlp(self.z_dim + self.c_dim, self.hidden, self.u_dim)

    def encode(self, u: torch.Tensor, c: torch.Tensor):
        h = self.encoder(torch.cat([u, c], dim=1))
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, torch.clamp(logvar, min=-12.0, max=8.0)

    def prior_params(self, c: torch.Tensor):
        h = self.prior(c)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, torch.clamp(logvar, min=-12.0, max=8.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor,
                       generator: torch.Generator | None = None):
        eps = torch.randn(mu.shape, generator=generator, dtype=mu.dtype)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z: torch.Tensor, c: torch.Tensor,
               mask: torch.Tensor | None = None):
        u = torch.sigmoid(self.decoder(torch.cat([z, c], dim=1)))
        if mask is not None:
            u = u * mask
        return u

    def forward(self, u: torch.Tensor, c: torch.Tensor, mask: torch.Tensor,
                generator: torch.Generator | None = None):
        mu, logvar = self.encode(u, c)
        z = self.reparameterize(mu, logvar, generator)
        u_hat = self.decode(z, c, mask)
        return u_hat, mu, logvar


def cvae_loss(model: ConditionalVAE, u: torch.Tensor, c: torch.Tensor,
              mask: torch.Tensor, beta: float, free_bits: float = 0.0,
              generator: torch.Generator | None = None):
    u_hat, mu_q, logvar_q = model(u, c, mask, generator)
    mu_p, logvar_p = model.prior_params(c)
    recon = (((u_hat - u) ** 2) * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    kl_dim = 0.5 * (logvar_p - logvar_q
                    + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2))
                    / torch.clamp(torch.exp(logvar_p), min=1e-8) - 1.0)
    kl = torch.clamp(kl_dim, min=float(free_bits)).sum(dim=1).mean()
    loss = recon + float(beta) * kl
    return loss, recon, kl


def save_checkpoint(path: str, model: ConditionalVAE, extra: dict | None = None):
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": {"u_dim": model.u_dim, "c_dim": model.c_dim,
                   "z_dim": model.z_dim, "hidden": list(model.hidden)},
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = payload["config"]
    model = ConditionalVAE(cfg["u_dim"], cfg["c_dim"], cfg["z_dim"], cfg["hidden"])
    state = payload.get("ema_state_dict") or payload["state_dict"]
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.eval()
    return model, payload


# --------------------------------------------------------------------------- #
# conditions and sampling
# --------------------------------------------------------------------------- #
def make_conditions(material: str, families: Sequence[tuple[int, int]],
                    target: float, scalers: dict) -> np.ndarray:
    specs = {1: (float(target), float(target))}
    return np.stack([
        V2D.condition_for(material, fam, specs, scalers, layout="interval")
        for fam in families
    ]).astype(np.float32)


def draw_families(n: int, rng: np.random.Generator,
                  allowed: Sequence[tuple[int, int]] | None = None):
    pool = list(allowed) if allowed else list(V2D.v1().FAMILIES)
    idx = rng.integers(0, len(pool), size=int(n))
    return [pool[int(i)] for i in idx]


def sample_prior(model: ConditionalVAE, conditions: np.ndarray,
                 families: Sequence[tuple[int, int]], n: int, seed: int):
    gen = torch.Generator().manual_seed(int(seed) + 9001)
    c = torch.as_tensor(conditions, dtype=torch.float32)
    mask = torch.as_tensor(family_masks(families), dtype=torch.float32)
    with torch.no_grad():
        mu_p, logvar_p = model.prior_params(c)
        eps = torch.randn((int(n), model.z_dim), generator=gen, dtype=torch.float32)
        z = mu_p + torch.exp(0.5 * logvar_p) * eps
        u = model.decode(z, c, mask)
    return u.cpu().numpy(), z.cpu().numpy()


def guided_prior(model: ConditionalVAE, conditions: np.ndarray,
                 families: Sequence[tuple[int, int]], material: str,
                 target: float, surrogate, strength: float = GUIDE_LAMBDA,
                 steps: int = GUIDE_STEPS, seed: int = 42):
    """Apply the same hinge surrogate guidance as the DDPM, but in CVAE z-space.

    Each of the 25 guidance steps costs one surrogate evaluation, matching the
    25-step reverse chain used by the DDPM.  The schedule sums to approximately
    the same effective strength as the DDPM's alpha-weighted guidance.
    """
    u_raw, z_np = sample_prior(model, conditions, families, len(families), seed)
    if not strength or strength <= 0:
        return u_raw, z_np, {"guide_fired": 0, "saturated_frac": float(np.mean((u_raw <= 1e-6) | (u_raw >= 1 - 1e-6)))}
    c = torch.as_tensor(conditions, dtype=torch.float32)
    mask = torch.as_tensor(family_masks(families), dtype=torch.float32)
    z = torch.as_tensor(z_np, dtype=torch.float32).detach().clone()
    smooth = V2D.SmoothSurrogateMulti(surrogate, torch)
    log_lo = math.log(max(float(target), 1.0))
    log_hi = log_lo
    groups = {}
    for i, fam in enumerate(families):
        groups.setdefault((int(fam[0]), int(fam[1])), []).append(i)
    for k in range(int(steps)):
        z = z.detach().clone().requires_grad_(True)
        total = torch.zeros((), dtype=torch.float32)
        for fam, idx in groups.items():
            index = torch.as_tensor(idx, dtype=torch.long)
            u = model.decode(z[index], c[index], mask[index])
            x = LT.decode_torch(torch, u, fam)
            value = torch.log(torch.clamp(
                smooth.cs(x, material, fam[0], fam[1], 1), min=1.0))
            # Sum, not mean: every sample must receive the same per-sample
            # gradient as in the DDPM guidance.  A mean over the family would
            # scale the step by 1/n_family and make the strength batch-size
            # dependent (at 5 000 samples guidance would be ~25x too weak).
            total = total + (torch.clamp(log_lo - value, min=0.0) ** 2
                             + torch.clamp(value - log_hi, min=0.0) ** 2).sum()
        grad = torch.autograd.grad(total, z)[0]
        a = float(k + 1) / max(float(steps), 1.0)
        z = (z - (float(strength) * a) * grad).detach()
    with torch.no_grad():
        u = model.decode(z, c, mask).cpu().numpy()
    diag = {"guide_fired": int(steps),
            "saturated_frac": float(np.mean((u <= 1e-6) | (u >= 1 - 1e-6)))}
    return u, z.detach().cpu().numpy(), diag


def decode_latents(u: np.ndarray, families: Sequence[tuple[int, int]]):
    return LT.decode_batch(np.asarray(u, dtype=np.float64), list(families))


def predict_cs1(x: np.ndarray, material: str,
                families: Sequence[tuple[int, int]], surrogate):
    nd = [int(f[0]) for f in families]
    nb = [int(f[1]) for f in families]
    rows = S.canonical_to_model_rows(x, [material] * len(x), nd, nb)
    pred = np.asarray(surrogate.predict(rows), dtype=np.float64)
    return pred[:, 0]


# --------------------------------------------------------------------------- #
# the exact common selector used in the set-objective comparison
# --------------------------------------------------------------------------- #
def select_common(x: np.ndarray, families: Sequence[tuple[int, int]],
                  pred_cs1: np.ndarray, spec: dict, scales: np.ndarray,
                  m: int = N_SUBMIT):
    """In-band first, then farthest-point diversity in standardised geometry.

    This is a byte-for-byte equivalent of select_common() in the existing
    set_objective_suite.py; it is copied here so the baseline folder is
    self-contained.
    """
    dev = np.abs(SI.deviation(pred_cs1, spec["lower"], spec["upper"]))
    inside = dev <= 1e-12
    z = np.asarray(x, dtype=np.float64) / np.asarray(scales, dtype=np.float64)
    chosen = []
    if inside.any():
        first = int(np.where(inside)[0][np.argmin(dev[inside])])
        chosen.append(first)
        while len(chosen) < min(int(m), int(inside.sum())):
            rest = np.where(inside)[0]
            rest = rest[~np.isin(rest, chosen)]
            if len(rest) == 0:
                break
            d = np.sqrt(((z[rest][:, None, :] - z[np.asarray(chosen)][None, :, :]) ** 2).sum(axis=2))
            chosen.append(int(rest[np.argmax(d.min(axis=1))]))
    if len(chosen) < int(m):
        rest = np.setdiff1d(np.arange(len(x)), np.asarray(chosen), assume_unique=False)
        order = rest[np.argsort(dev[rest])]
        chosen.extend(int(i) for i in order[:int(m) - len(chosen)])
    chosen = np.asarray(chosen[:int(m)], dtype=int)
    chosen = chosen[np.argsort(dev[chosen], kind="stable")]
    frame = pd.DataFrame(np.asarray(x)[chosen], columns=S.CANONICAL_COLUMNS)
    frame.insert(0, "material", spec["material"])
    frame.insert(1, "n_disks", [int(families[i][0]) for i in chosen])
    frame.insert(2, "n_bearings", [int(families[i][1]) for i in chosen])
    frame["cs1_pred"] = np.asarray(pred_cs1)[chosen]
    frame["rank"] = np.arange(1, len(chosen) + 1)
    return frame, {
        "pool_in_band_proxy": int(inside.sum()),
        "selected_in_band_proxy": int((dev[chosen] <= 1e-12).sum()),
        "selected": int(len(chosen)),
    }


def make_pool_frame(x: np.ndarray, families: Sequence[tuple[int, int]],
                    pred_cs1: np.ndarray, spec: dict):
    frame = O.make_frame(np.asarray(x), spec["material"], list(families))
    frame["cs1_pred"] = np.asarray(pred_cs1, dtype=np.float64)
    dev = SI.deviation(np.asarray(pred_cs1), spec["lower"], spec["upper"])
    frame["cs1_lo"] = float(spec["lower"])
    frame["cs1_hi"] = float(spec["upper"])
    frame["cs1_dev_pct"] = 100.0 * dev
    frame["joint_err"] = np.abs(dev)
    frame["rank"] = np.argsort(np.abs(dev), kind="stable") + 1
    return frame.iloc[np.argsort(np.abs(dev), kind="stable")].reset_index(drop=True)


def save_json(path: str, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


