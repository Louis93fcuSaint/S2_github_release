# -*- coding: utf-8 -*-
"""Round-trip the whole pool through the latent box, then stress the decoder."""
import collections, io, os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import spec_common as S, spec_eval as E
import latent_design as LD

LOG = io.open("outputs/logs/latent_roundtrip.txt", "w", encoding="utf-8")
def say(*a):
    m = " ".join(str(v) for v in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()

t0 = time.time()
ctx = E.pool_ctx()
X, nd, nb, mat = ctx["designs"], ctx["nd"], ctx["nb"], ctx["material"]
say("pool %s  (%.0f s)" % (X.shape, time.time() - t0))

errs = np.zeros((len(X), S.DESIGN_DIM))
V = np.empty(len(X), dtype=bool)
BACK = np.empty_like(X)
for i in range(len(X)):
    fam = (int(nd[i]), int(nb[i]))
    xb = LD.decode(LD.encode(X[i], fam), fam)
    BACK[i] = xb
    errs[i] = np.abs(xb - X[i]) / np.maximum(np.abs(X[i]), 1e-12)
    V[i] = S.is_valid(xb, mat[i], fam[0], fam[1])[0]
say("round trip over %d rows (%.0f s)" % (len(X), time.time() - t0))
say("   decoded rows failing H1-H6 : %d" % int((~V).sum()))
say("   errs[i][|dx|<1e-15]=0 applied; per-column median / p99 / max:")
for c in range(S.DESIGN_DIM):
    if errs[:, c].max() > 1e-9:
        say("      %-8s %.2e  %.2e  %.2e" % (S.CANONICAL_COLUMNS[c],
            np.median(errs[:, c]), np.percentile(errs[:, c], 99), errs[:, c].max()))
say("   fraction of rows with any column error > 1e-3 : %.5f"
    % float((errs.max(axis=1) > 1e-3).mean()))

# the metric that actually matters: does the round trip move cs1?
say("-" * 70)
surrogate = S.load_surrogate("mlp_best", threads=8)
rng = np.random.default_rng(0)
idx = rng.choice(len(X), size=20000, replace=False)
def cs1(z, sel):
    out = np.full(len(z), np.nan)
    for m in S.MATERIAL_ORDER:
        k = mat[sel] == m
        if k.any():
            out[k] = surrogate.predict(S.canonical_to_model_rows(
                z[k], [m] * int(k.sum()), nd[sel][k], nb[sel][k]))[:, 0]
    return out
a = cs1(X[idx], idx); b = cs1(BACK[idx], idx)
rel = np.abs(b - a) / a
say("surrogate cs1 on 20000 pool rows, original vs round-tripped:")
say("   relative change median %.3e  p99 %.3e  max %.3e"
    % (np.median(rel), np.percentile(rel, 99), rel.max()))
say("   rows with > 0.1 %% cs1 change : %.5f" % float((rel > 1e-3).mean()))

# ------------------------------------------------------------------ stress
say("-" * 70)
rng = np.random.default_rng(0)
tot = bad = 0
c = collections.Counter()
for a_, b_ in [(x, y) for x in S.ND_CHOICES for y in S.NB_CHOICES]:
    mask = LD.used_mask((a_, b_))
    U = rng.random((400, LD.LAT_DIM)); U[:, ~mask] = 0.0
    for u in U:
        tot += 1
        ok, why = S.is_valid(LD.decode(u, (a_, b_)), "Steel", a_, b_)
        if not ok:
            bad += 1
            for w in why:
                c[w.split(":")[0].strip()] += 1
say("uniform-latent stress: %d samples, invalid %d (%.4f%%)  reasons %s"
    % (tot, bad, 100.0 * bad / tot, dict(c)))
tot2 = bad2 = 0
for a_, b_ in [(x, y) for x in S.ND_CHOICES for y in S.NB_CHOICES]:
    mask = LD.used_mask((a_, b_))
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        U = np.full((1, LD.LAT_DIM), value); U[:, ~mask] = 0.0
        tot2 += 1
        if not S.is_valid(LD.decode(U[0], (a_, b_)), "Steel", a_, b_)[0]:
            bad2 += 1
say("box-corner/grid stress: %d samples, invalid %d" % (tot2, bad2))
say("decoder guard fired %d / %d" % (LD.GUARD["fired"], LD.GUARD["total"]))
say("[done] %.0f s" % (time.time() - t0))
LOG.close()