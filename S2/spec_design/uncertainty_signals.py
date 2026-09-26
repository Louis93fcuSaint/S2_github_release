# -*- coding: utf-8 -*-
"""Cheap per-candidate uncertainty signals for the frozen MLP surrogate.

Why this exists.  The proxy is honest inside the data manifold: on the held-out
test split it calls 10.2 % of the pool in-band while the truth is 10.3 %.  It is
optimistic outside it: on differential-evolution submissions it calls 100 %
in-band while the truth is 34.8 %.  A screening rule that cannot tell those two
regimes apart is what is left of the "proxy shortlist, ROSS verifies" pipeline,
so every candidate is scored with three signals that are free to compute:

    spread : 5-seed disagreement, std of the predicted log cs1
    dF     : distance to the nearest *training* design in the surrogate's own
             47-dim feature space, in units of the training per-column std
    dD     : distance to the nearest *training* design in the 33-dim canonical
             design space, in units of the pool per-column spread
    box    : how far outside the training min/max box the candidate sits, in
             units of the training per-column std (0 when inside)

Reference arrays are built once from the training rows and reused, so scoring a
batch of candidates is a handful of matrix products.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_common as S
import spec_eval as E

SPLIT = os.path.join(S.SURROGATE_DIR, "outputs", "split_v47.npz")
CALIB_JSON = os.path.join(E.OUT, "uncertainty_calibration.json")
REF_SIZE = 20000
BLOCK = 1024


def mapped_indices():
    """split_v47.npz row numbers index the raw CSV; the pool drops the four
    status != 'success' rows, so a positional copy is wrong from the first
    dropped row onwards.  This returns pool-row indices instead."""
    import pandas as pd

    status = pd.read_csv(S.POOL_CSV, usecols=["status"])["status"].to_numpy()
    csv_pos = np.where(status == "success")[0]
    blob = np.load(SPLIT)
    train = blob["train_idx"]
    test = blob["test_idx"]
    train = train[np.isin(train, csv_pos)]
    dropped = int((~np.isin(test, csv_pos)).sum())
    keep = test[np.isin(test, csv_pos)]
    order = np.searchsorted(csv_pos, keep)
    assert np.all(csv_pos[order] == keep)
    return np.searchsorted(csv_pos, train), order, dropped


def nearest(query, ref, scales=None):
    """Distance from every query row to its nearest reference row."""
    query = np.asarray(query, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if scales is not None:
        query = query / scales
        ref = ref / scales
    ref_sq = (ref * ref).sum(axis=1)
    out = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), BLOCK):
        q = query[start:start + BLOCK]
        best = np.full(len(q), np.inf)
        for lo in range(0, len(ref), REF_SIZE):
            r = ref[lo:lo + REF_SIZE]
            d = (q * q).sum(axis=1)[:, None] + ref_sq[lo:lo + REF_SIZE][None, :] \
                - 2.0 * (q @ r.T)
            chunk = np.sqrt(np.maximum(d, 0.0)).min(axis=1)
            best = np.minimum(best, chunk)
        out[start:start + BLOCK] = best
    return out


def seed_log_cs1(surrogate, model_rows):
    """Per-seed log cs1, shape (seeds, n).

    prepare() puts the five logged inputs through log(); skipping it feeds raw
    features to a network trained on logged ones and returns silent garbage.
    """
    prepared = surrogate.prepare(model_rows)
    torch = surrogate.torch
    out = []
    with torch.no_grad():
        for item in surrogate.models:
            scaled = (prepared - item["x_mean"]) / item["x_scale"]
            tensor = torch.as_tensor(scaled, dtype=torch.float32)
            pred = item["net"](tensor).cpu().numpy().astype(np.float64)
            out.append((pred * item["y_std"] + item["y_mean"])[:, 0])
    return np.stack(out, axis=0)


class SignalBank:
    """Training-set reference for the three distance signals."""

    def __init__(self, ctx, surrogate, seed=0, ref_size=REF_SIZE):
        self.ctx = ctx
        self.surrogate = surrogate
        train_pool, self.test_pool, self.n_dropped_test = mapped_indices()
        self.train_pool = train_pool
        rng = np.random.default_rng(seed)
        self.ref_rows = train_pool[rng.choice(len(train_pool),
                                              min(ref_size, len(train_pool)),
                                              replace=False)]
        prepared = surrogate.prepare(ctx["X"][self.ref_rows])
        self.f_scale = np.maximum(surrogate.prepare(ctx["X"][train_pool]).std(axis=0), 1e-9)
        self.f_lo = surrogate.prepare(ctx["X"][train_pool]).min(axis=0)
        self.f_hi = surrogate.prepare(ctx["X"][train_pool]).max(axis=0)
        self.f_ref = prepared / self.f_scale
        self.d_ref = ctx["designs"][self.ref_rows]

    def score(self, model_rows, designs):
        """model_rows: (n, 47) surrogate features; designs: (n, 33) canonical."""
        model_rows = np.asarray(model_rows, dtype=np.float64)
        designs = np.asarray(designs, dtype=np.float64)
        logs = seed_log_cs1(self.surrogate, model_rows)
        prepared = self.surrogate.prepare(model_rows)
        box = np.maximum.reduce([
            (prepared - self.f_hi) / self.f_scale,
            (self.f_lo - prepared) / self.f_scale,
            np.zeros_like(prepared)]).max(axis=1)
        return {
            "log_mean": logs.mean(axis=0),
            "log_spread": logs.std(axis=0, ddof=1),
            "dF": nearest(prepared / self.f_scale, self.f_ref),
            "dD": nearest(designs, self.d_ref, self.ctx["scales"]),
            "box": box,
        }


def fit_calibration(err, spread, dF, edges=(0.2, 0.4, 0.6, 0.8)):
    """Novelty-scaled error bar: the 90th percentile of |log err| per dF bin.

    Also fits the single-parameter form err90(s) = c90 * s, which is what the
    screening rule uses when only the seed spread is available.
    """
    cut = np.quantile(dF, list(edges))
    bucket = np.digitize(dF, cut)
    bins = []
    for k in range(len(cut) + 1):
        sel = bucket == k
        if not sel.sum():
            continue
        bins.append({"bin": int(k), "n": int(sel.sum()),
                     "dF_lo": float(dF[sel].min()), "dF_hi": float(dF[sel].max()),
                     "q90_log_err": float(np.quantile(err[sel], 0.9)),
                     "median_log_err": float(np.median(err[sel]))})
    ratio = err / np.maximum(spread, 1e-12)
    return {
        "dF_edges": [float(v) for v in cut],
        "dF_bins": bins,
        "q90_err_over_spread": float(np.quantile(ratio, 0.9)),
        "median_err_over_spread": float(np.median(ratio)),
        "global_q90_log_err": float(np.quantile(err, 0.9)),
        "n": int(len(err)),
    }


ANALYSIS_JSON = os.path.join(E.OUT, "uncertainty_stageB_analysis.json")
MARGIN_Q_FALLBACK = 2.94        # q90(|log err| / spread) re-fit on the ROSS-verified


def margin_q(default=MARGIN_Q_FALLBACK):
    """The constant of the honest screen.

    The test split gives 2.04, the ROSS-verified submissions give 2.94: the
    proxy is systematically less accurate outside the data manifold, so the
    error bar has to widen by about 45 % when the candidate is generated
    rather than sampled.  Reading the value from the stage-B analysis keeps a
    single source of truth; the constant is only a fallback.
    """
    if os.path.exists(ANALYSIS_JSON):
        try:
            with open(ANALYSIS_JSON, encoding="utf-8") as handle:
                return float(json.load(handle)["q_used"])
        except (ValueError, KeyError):
            pass
    return float(default)


def screen_submission(frame, spec, bank, q=None):
    """Score one submitted batch under a point-band spec.

    Returns the point prediction, the honest interval [lo, hi] and the keep
    mask.  A candidate is kept only when the whole interval sits inside the
    band; that is the rule whose precision was measured at 0.90-1.00 on the
    ROSS-verified batches, against 0.40-0.60 for the point prediction's own
    "in band" claim.
    """
    if q is None:
        q = margin_q()
    x = np.asarray(frame[S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64))
    rows = S.canonical_to_model_rows(x, frame["material"].tolist(),
                                     frame["n_disks"].tolist(),
                                     frame["n_bearings"].tolist())
    sig = bank.score(rows, x)
    proxy = np.exp(sig["log_mean"])
    half = np.exp(q * sig["log_spread"])
    lo, hi = proxy / half, proxy * half
    return {"proxy": proxy, "lo": lo, "hi": hi, "keep": (lo >= spec["lower"]) & (hi <= spec["upper"]),
            "spread": sig["log_spread"], "dF": sig["dF"], "dD": sig["dD"], "box": sig["box"],
            "q": float(q)}


def err90_from_bins(calib, dF):
    """Interpolate the bin q90 onto arbitrary novelty values, held flat outside."""
    edges = calib["dF_edges"]
    bins = calib["dF_bins"]
    dF = np.atleast_1d(np.asarray(dF, dtype=np.float64))
    out = np.empty_like(dF)
    bucket = np.digitize(dF, edges)
    for k, row in enumerate(bins):
        sel = bucket == k
        out[sel] = row["q90_log_err"]
    if not np.isfinite(out).all():
        out = np.where(np.isfinite(out), out, calib["global_q90_log_err"])
    return out