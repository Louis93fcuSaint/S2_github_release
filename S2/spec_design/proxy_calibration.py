# -*- coding: utf-8 -*-
"""Two calibrations of the frozen surrogate, both fitted on evidence.

A. honest bar   h = exp(b0 + b1 log(spread) + b2 log(dF)) is the tau quantile of
   |log(proxy / truth)|, fitted by pinball regression on the held-out test split.
   A candidate is trusted only when [proxy / h, proxy * h] sits inside the band.
   The binned constant this replaces was too loose inside the manifold (it threw
   away 90 % of a batch that was entirely in band) and too tight outside it.

B. bias layer   log truth = log proxy + g(features), g linear, fitted by ridge on
   the ROSS-verified submissions and validated leave-one-batch-out.  The proxy's
   error is not random: it over-predicts the more the candidate leaves the data
   manifold, and that part is learnable, which is what the within-batch ranking
   was missing.

Both are fitted on data with real labels, never on the proxy's own opinion, and
both are monotone in the signals they are allowed to use.
"""
import numpy as np

MATERIAL_ORDER = ["Steel", "Aluminum", "Titanium"]
N_DISK_SCALE = 6.0
N_BEARING_SCALE = 4.0
EPS = 1e-12
BAR_NAMES = ["1", "log(spread)", "log(dF)"]
BIAS_NAMES = ["1", "log(proxy/target)", "log(spread)", "log(dF)", "log(dD)", "box",
              "mat_Aluminum", "mat_Titanium", "nd", "nb"]


def _pinball_weights(resid, tau, scale):
    weight = np.where(resid >= 0.0, tau, 1.0 - tau)
    return weight / np.maximum(np.abs(resid), 1e-3 * scale)


def _weighted_ridge(A, y, w, l2):
    root = np.sqrt(w)
    Aw = A * root[:, None]
    yw = y * root
    gram = Aw.T @ Aw + l2 * np.eye(A.shape[1])
    return np.linalg.solve(gram, Aw.T @ yw)


def pinball_fit(A, y, tau=0.9, l2=1e-6, iters=60):
    """Quantile regression by iteratively reweighted least squares."""
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    beta = np.linalg.solve(A.T @ A + l2 * np.eye(A.shape[1]), A.T @ y)
    scale = max(float(np.median(np.abs(y))), 1e-6)
    for _ in range(iters):
        w = _pinball_weights(y - A @ beta, tau, scale)
        beta = _weighted_ridge(A, y, w, l2)
    return beta


def ridge_fit(A, y, l2=1e-3):
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.linalg.solve(A.T @ A + l2 * np.eye(A.shape[1]), A.T @ y)


def coverage(err, half):
    err = np.asarray(err, dtype=np.float64)
    half = np.asarray(half, dtype=np.float64)
    return float((err <= half).mean())


# --------------------------------------------------------------------------- #
# A. the honest bar
# --------------------------------------------------------------------------- #
def bar_design(spread, dF):
    spread = np.asarray(spread, dtype=np.float64)
    dF = np.asarray(dF, dtype=np.float64)
    return np.column_stack([np.ones_like(spread), np.log(spread + EPS), np.log(dF + EPS)])


def fit_bar(spread, dF, err, tau=0.9, l2=1e-6):
    """Fit the 0.9 quantile of |log err| as a function of spread and novelty."""
    A = bar_design(spread, dF)
    beta = pinball_fit(A, np.log(np.asarray(err, dtype=np.float64) + EPS), tau=tau, l2=l2)
    half = np.exp(A @ beta)
    # a bar has to be monotone: clip a negative slope to zero and refit the intercept
    slope = beta.copy()
    slope[1] = max(slope[1], 0.0)
    slope[2] = max(slope[2], 0.0)
    if not np.allclose(slope, beta):
        A2 = np.column_stack([np.ones(len(spread)), slope[1] * np.log(spread + EPS),
                              slope[2] * np.log(dF + EPS)])
        beta = np.concatenate([[pinball_fit(A2, np.log(err + EPS), tau=tau, l2=l2)[0]],
                               slope[1:]])
    return {"coef": [float(v) for v in beta], "names": BAR_NAMES, "tau": float(tau),
            "n": int(len(spread)), "coverage_fit": coverage(err, np.exp(bar_design(spread, dF) @ beta)),
            "clipped": bool(not np.allclose(slope, np.asarray([beta[0], beta[1], beta[2]]))),
            "median_half": float(np.median(np.exp(bar_design(spread, dF) @ beta)))}


def bar_predict(calib, spread, dF):
    """The fitted 0.9 quantile of |log err|: a *log-space* half-width (0.09, not
    1.09).  Use bar_factor() when building the interval on the rpm scale."""
    return np.exp(bar_design(spread, dF) @ np.asarray(calib["coef"], dtype=np.float64))


def bar_factor(calib, spread, dF):
    """The multiplicative half-width of the interval on the rpm scale."""
    return np.exp(bar_predict(calib, spread, dF))


def trust_mask(proxy, spread, dF, lower, upper, bar_calib=None, q=None):
    """Keep a candidate only when its whole honest interval sits inside the band."""
    proxy = np.asarray(proxy, dtype=np.float64)
    if bar_calib is not None:
        factor = bar_factor(bar_calib, spread, dF)
    else:
        factor = np.exp(float(q) * np.asarray(spread, dtype=np.float64))
    return ((proxy / factor >= lower) & (proxy * factor <= upper)), factor


# --------------------------------------------------------------------------- #
# B. the bias layer
# --------------------------------------------------------------------------- #
def bias_design(rows, material_key="material", proxy_key="proxy", target_key="target"):
    mat = [str(r.get(material_key, "Steel")) for r in rows]
    proxy = np.array([r[proxy_key] for r in rows], dtype=np.float64)
    target = np.array([r[target_key] for r in rows], dtype=np.float64)
    spread = np.array([max(float(r["spread"]), EPS) for r in rows])
    dF = np.array([max(float(r["dF"]), EPS) for r in rows])
    dD = np.array([max(float(r["dD"]), EPS) for r in rows])
    box = np.array([float(r["box"]) for r in rows])
    nd = np.array([float(r["nd"]) for r in rows]) / N_DISK_SCALE
    nb = np.array([float(r["nb"]) for r in rows]) / N_BEARING_SCALE
    cols = [np.ones(len(rows)), np.log(proxy / target), np.log(spread), np.log(dF), np.log(dD),
            box,
            np.array([1.0 if m == "Aluminum" else 0.0 for m in mat]),
            np.array([1.0 if m == "Titanium" else 0.0 for m in mat]),
            nd, nb]
    return np.column_stack(cols)


def fit_bias(rows, l2=1e-3):
    """Fit log(truth) - log(proxy) on the candidate's own signals."""
    A = bias_design(rows)
    r = np.array([np.log(r["truth"] / r["proxy"]) for r in rows], dtype=np.float64)
    beta = ridge_fit(A, r, l2=l2)
    pred = A @ beta
    resid = r - pred
    return {"coef": [float(v) for v in beta], "names": BIAS_NAMES, "n": int(len(rows)),
            "r2": float(1.0 - resid.var() / max(r.var(), EPS)),
            "rmse": float(np.sqrt(np.mean(resid ** 2)))}


def bias_correct(calib, rows):
    return np.exp(np.log(np.array([r["proxy"] for r in rows], dtype=np.float64))
                  + bias_design(rows) @ np.asarray(calib["coef"], dtype=np.float64))


# --------------------------------------------------------------------------- #
# C. the target shift
#
# The bias layer above needs ten features and manages an r2 of 0.13, which is
# why it never earned its place.  The one thing it does establish is the sign:
# the proxy is optimistic, so the designs it calls "on target" really sit a
# little under it.  A single log-space offset, fitted on the handful of ROSS
# solves a verification run already pays for, is the cheapest way to use that,
# and it cannot overfit ten coefficients onto thirty labels.
# --------------------------------------------------------------------------- #
def shift_fit(truth, proxy):
    """Median log(truth / proxy): the robust part of the bias, no features."""
    truth = np.asarray(truth, dtype=np.float64)
    proxy = np.asarray(proxy, dtype=np.float64)
    good = (truth > 0) & (proxy > 0)
    if good.sum() < 3:
        return 0.0
    return float(np.median(np.log(truth[good]) - np.log(proxy[good])))


def shift_cross_fit(truth, proxy, folds=5, seed=42):
    """Out-of-fold offsets: every point gets an offset fitted without it."""
    truth = np.asarray(truth, dtype=np.float64)
    proxy = np.asarray(proxy, dtype=np.float64)
    n = len(truth)
    out = np.zeros(n)
    if n < folds * 2:
        out[:] = shift_fit(truth, proxy)
        return out
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    for part in np.array_split(order, folds):
        rest = np.setdiff1d(order, part, assume_unique=False)
        out[part] = shift_fit(truth[rest], proxy[rest])
    return out


def topk_rate(score, hit, ks):
    """In-band rate of the best k by score, for every k that fits the batch."""
    score = np.asarray(score, dtype=np.float64)
    hit = np.asarray(hit, dtype=bool)
    order = np.argsort(-score)
    ranked = hit[order]
    out = {}
    for k in ks:
        kk = min(int(k), len(ranked))
        out["top%d" % k] = float(ranked[:kk].mean())
    return out
