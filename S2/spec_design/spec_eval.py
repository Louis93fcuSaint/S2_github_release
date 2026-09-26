# -*- coding: utf-8 -*-
"""Shared context + metrics for the spec-driven design track.

Frozen decisions (see S2/research/一阶临界转速设计需求调研.md):
  n_operating = 12000 rpm (max continuous), engineering margin = 20 %
  -> spec lower bound = 14400 rpm ; upper bound = the largest cs1 ever
  realised for that material, so no method can win by extrapolating the proxy.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_common as S

OUT = os.path.join(S.HERE, "outputs")
CACHE = os.path.join(OUT, "cache_features.npz")
SPEC_JSON = os.path.join(OUT, "specs.json")
CSV_COLUMNS = ["material", "n_disks", "n_bearings"] + S.CANONICAL_COLUMNS

TIERS = [("std", 14400.0), ("hard", 25000.0)]


ADHOC_JSON = os.path.join(OUT, "specs_adhoc.json")


def _tol_suffix(tol):
    """Tolerance in the auto name, so two bands on one target cannot collide.

    The protocol default (+-5 %) stays suffix-free because every arm recorded so
    far uses that bare name; +-0.5 % and +-1 % become `_t05` and `_t1`, which is
    the convention the existing t5000_t05 / t5000_t1 arms already follow.
    """
    tol = float(tol)
    if abs(tol - 0.05) < 1e-12:
        return ""
    return "_t%s" % ("%g" % (100.0 * tol)).replace(".", "")


def adhoc_spec(material, target, tol, upper=None, name=None, mode="band"):
    """A user-supplied point target, outside the frozen protocol.

    mode "band"  accept cs1 inside target * (1 +- tol), aim at the target.
    mode "lower" accept cs1 at or above target, and aim at target * (1 + tol),
                 i.e. tol becomes the aiming margin on top of the requirement.
                 A shortlist centred exactly on the requirement would spend half
                 its designs inside the surrogate's own error bar.  upper then
                 defaults to the material maximum, filled in by the caller.

    target / tol drive the generative sampling; lower / upper are the band that
    the metrics and the family prior read, exactly as in a frozen spec.
    """
    target = float(target)
    tol = float(tol)
    tag = (name or "%s_%g%s" % (material, target,
                                _tol_suffix(tol))).strip().replace(" ", "_")
    spec = {"name": tag, "material": material, "tier": "adhoc", "mode": mode,
            "target": target, "tol": tol, "aim": target}
    if mode == "lower":
        # Aim above the requirement, not at it.  The surrogate carries ~1 % of
        # its own error, so a shortlist centred exactly on the requirement loses
        # half its designs to that error; tol is the aiming margin here.
        spec["lower"] = target
        spec["aim"] = target * (1.0 + tol)
        spec["upper"] = float(upper) if upper else None
        return spec
    spec["lower"] = target * (1.0 - tol)
    spec["upper"] = float(upper) if upper else target * (1.0 + tol)
    return spec


def adhoc_specs():
    if not os.path.exists(ADHOC_JSON):
        return []
    with open(ADHOC_JSON, encoding="utf-8") as handle:
        return list(json.load(handle)["specs"])


def add_adhoc_spec(spec):
    """Persist an ad-hoc target so 05_ross_verify.py resolves the same spec."""
    rows = [sp for sp in adhoc_specs() if sp["name"] != spec["name"]]
    rows.append(spec)
    os.makedirs(OUT, exist_ok=True)
    with open(ADHOC_JSON, "w", encoding="utf-8") as handle:
        json.dump({"specs": rows}, handle, indent=2)
    return spec


def pool_ctx(pool=None):
    """Everything the metrics need: designs, engineered features, truth, scales."""
    if pool is None:
        pool = S.load_pool()
    blob = np.load(CACHE, allow_pickle=True)
    X = np.asarray(blob["X"], dtype=np.float64)
    Y = np.asarray(blob["Y"], dtype=np.float64)
    if X.shape[0] != len(pool):
        raise RuntimeError("feature cache is stale, rerun 01_proxy_family.py")
    designs = pool[S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
    scales = np.maximum(designs.std(axis=0), 1e-9)
    nd = pool["n_disks"].to_numpy(dtype=int)
    nb = pool["n_bearings"].to_numpy(dtype=int)
    material = pool["material"].to_numpy()
    ctx = {"pool": pool, "X": X, "Y": Y, "designs": designs, "scales": scales,
           "nd": nd, "nb": nb, "material": material}
    ctx["upper"] = {m: float(pool.loc[material == m, "cs_1_rpm"].max())
                    for m in S.MATERIAL_ORDER}
    rng = np.random.default_rng(7)
    idx = rng.choice(len(pool), size=2000, replace=False)
    ctx["novelty_ref"] = float(np.median(nearest_distance(
        designs[idx], designs, scales, query_ids=idx,
        ref_ids=np.arange(len(pool)))))
    return ctx


def nearest_distance(queries, ref, scales, block=512, ref_block=20000,
                    query_ids=None, ref_ids=None):
    """Distance from every query to its nearest reference design, measured in
    units of the per-dimension spread of the pool.

    When query_ids / ref_ids are supplied the self-match is excluded, which is
    what the novelty *reference* statistic needs; a submission of brand-new
    designs needs no exclusion, and a submission of copied pool designs then
    legitimately scores a novelty of zero.
    """
    queries = np.asarray(queries, dtype=np.float64) / scales
    out = np.empty(len(queries), dtype=np.float64)
    for start in range(0, len(queries), block):
        q = queries[start:start + block]
        best = np.full(len(q), np.inf)
        for lo in range(0, len(ref), ref_block):
            r = np.asarray(ref[lo:lo + ref_block], dtype=np.float64) / scales
            d = np.sqrt(np.maximum(
                (q * q).sum(axis=1)[:, None] + (r * r).sum(axis=1)[None, :]
                - 2.0 * (q @ r.T), 0.0))
            if query_ids is not None and ref_ids is not None:
                same = (query_ids[start:start + block][:, None]
                        == ref_ids[lo:lo + ref_block][None, :])
                d[same] = np.inf
            best = np.minimum(best, d.min(axis=1))
        out[start:start + block] = best
    return out


def specs(ctx, path=None, with_adhoc=False):
    """Frozen specs.  When a frozen spec file exists it wins, so that the same
    protocol definition is used by every script without recomputation.

    with_adhoc additionally appends the user-supplied point targets stored in
    specs_adhoc.json; the frozen protocol file itself is never touched.
    """
    path = path or SPEC_JSON
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            out = list(json.load(handle)["specs"])
    else:
        out = []
        for material in S.MATERIAL_ORDER:
            for tier, lower in TIERS:
                out.append({"name": "%s_%s" % (material, tier), "material": material,
                            "lower": lower, "upper": ctx["upper"][material],
                            "tier": tier})
    if with_adhoc and os.path.exists(ADHOC_JSON):
        known = {sp["name"] for sp in out}
        out += [sp for sp in adhoc_specs() if sp["name"] not in known]
    return out


def predict_cs1(frame, surrogate, ceil=None):
    x = frame[S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
    rows = S.canonical_to_model_rows(x, frame["material"].tolist(),
                                     frame["n_disks"].tolist(), frame["n_bearings"].tolist())
    out = np.asarray(surrogate.predict(rows), dtype=np.float64)[:, 0]
    # an unconstrained optimiser can drive the log-target into overflow; a
    # non-finite prediction must never be counted as a hit.
    return np.where(np.isfinite(out), out, np.inf)


def evaluate_submission(frame, spec, ctx, surrogate, truth=None):
    """Protocol metrics for one submitted batch against one spec."""
    frame = frame.reset_index(drop=True)
    pred = predict_cs1(frame, surrogate)
    x = frame[S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
    valid = np.array([S.is_valid(xi, m, int(nd), int(nb))[0]
                      for xi, m, nd, nb in zip(x, frame["material"],
                                               frame["n_disks"], frame["n_bearings"])])
    in_band = (pred >= spec["lower"]) & (pred <= spec["upper"])
    novelty = nearest_distance(x, ctx["designs"], ctx["scales"])
    keys, counts = np.unique(np.round(x, 6), axis=0, return_counts=True)
    mix = {}
    for nd, nb in zip(frame["n_disks"], frame["n_bearings"]):
        key = "%d_%d" % (int(nd), int(nb))
        mix[key] = mix.get(key, 0) + 1
    metrics = {
        "spec": spec["name"], "n": int(len(frame)),
        "valid_rate": float(valid.mean()),
        "spec_rate_proxy": float(in_band.mean()),
        "unique_rate": float(len(keys) / max(len(frame), 1)),
        "survivors": int((valid & in_band).sum()),
        "family_mix": mix, "n_families": len(mix),
        "novelty_min_median": float(np.median(novelty)),
        "novelty_min_p10": float(np.percentile(novelty, 10)),
        "novelty_ref_median": ctx["novelty_ref"],
        "proxy_cs1_median": float(np.median(pred)),
        "band": [spec["lower"], spec["upper"]],
        "finite_rate": float(np.isfinite(pred).mean()),
    }
    # A design that already exists in the dataset is not a design *result*, so
    # the headline metric has to be a joint one: inside the band AND genuinely
    # novel.  Thresholds are in units of the pool's own nearest-neighbour scale.
    for floor in (0.5, 1.0, 2.0):
        joint = valid & in_band & (novelty >= floor)
        metrics["spec_and_novel_%.1f" % floor] = float(joint.mean())
    if spec.get("target"):
        rel = np.abs(pred - float(spec["target"])) / float(spec["target"])
        rel = rel[np.isfinite(rel)]
        metrics["median_rel_err_proxy"] = float(np.median(rel)) if rel.size else None
        for tol in (0.01, 0.03, 0.05, 0.10):
            metrics["hit_proxy_%d" % int(round(tol * 100))] = \
                float((rel <= tol).mean()) if rel.size else 0.0
    if truth is not None:
        truth = np.asarray(truth, dtype=np.float64)
        keep = np.isfinite(truth) & (truth > 0)
        metrics["spec_rate_true"] = float(((truth >= spec["lower"])
                                           & (truth <= spec["upper"]))[keep].mean()) \
            if keep.any() else 0.0
        metrics["n_truth"] = int(keep.sum())
        if keep.sum() > 1:
            ape = np.abs(pred[keep] - truth[keep]) / truth[keep] * 100.0
            metrics["MAPE_cs1"] = float(ape.mean())
            metrics["median_APE_cs1"] = float(np.median(ape))
    return metrics
