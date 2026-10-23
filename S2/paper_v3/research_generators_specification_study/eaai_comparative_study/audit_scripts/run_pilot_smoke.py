# -*- coding: utf-8 -*-
"""32-candidate interface smoke test.  No ROSS is called here."""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eaai_common as A

N = 32
SEED = A.SEED


def condition_checks(cond, families, spec):
    cond = np.asarray(cond)
    ok = True
    notes = []
    if cond.shape[1] != 21:
        ok = False
        notes.append("condition dim %d != 21" % cond.shape[1])
    mat = spec["material"]
    expect_mat = np.array(A.S.material_onehot(mat))
    if not np.allclose(cond[:, :3], expect_mat[None, :]):
        ok = False
        notes.append("material block mismatch")
    for i, fam in enumerate(families[:min(len(families), 8)]):
        expect_fam = np.array(A.S.family_onehot(int(fam[0]), int(fam[1])))
        if not np.allclose(cond[i, 3:12], expect_fam):
            ok = False
            notes.append("family block mismatch at %d" % i)
            break
    masks = cond[:, 18:21]
    for order in (1, 2, 3):
        want = 1.0 if order in spec["orders"] else 0.0
        if not np.allclose(masks[:, order - 1], want):
            ok = False
            notes.append("mask order %d mismatch" % order)
    for order in (1, 2, 3):
        col_lo = 12 + (order - 1)
        col_hi = 15 + (order - 1)
        if order in spec["orders"]:
            lo, hi = spec["bands"][order]
            eff = A.SI.effective(lo, hi, A.H_MAX)
            if not np.allclose(eff, (lo, hi)):
                ok = False
                notes.append("order %d band was narrowed by h_max" % order)
            if np.all(cond[:, col_lo] == 0) or np.all(cond[:, col_hi] == 0):
                ok = False
                notes.append("order %d condition edges are zero" % order)
        else:
            if not (np.all(cond[:, col_lo] == 0) and np.all(cond[:, col_hi] == 0)):
                ok = False
                notes.append("inactive order %d condition is not zero" % order)
    return ok, "; ".join(notes)


def family_zero_slots(x, families):
    bad = 0
    for row, fam in zip(x, families):
        a, b = int(fam[0]), int(fam[1])
        used = set([0, 1, 2])
        used |= set(3 + 3 * j + k for j in range(a) for k in (0, 1, 2))
        used |= set(A.S.BRG_BASE + 3 * j + k for j in range(b) for k in (0, 1, 2))
        for col in range(33):
            if col not in used and abs(float(row[col])) > 1e-12:
                bad += 1
                break
    return bad


def run_generation(method, spec, bundles, ctx, train_idx, surrogate, seed=SEED, n=N):
    if method == "ddpm":
        families = A.family_sequence(spec["family"], n, seed)
        u, cond = A.sample_ddpm(bundles["ddpm"], spec["material"], families,
                                A.band_specs(spec), seed, n)
        x = A.decode_latents(u, families)
        meta = {}
    elif method == "cvae":
        families = A.family_sequence(spec["family"], n, seed)
        u, cond = A.sample_cvae(bundles["cvae"], spec["material"], families,
                                A.band_specs(spec), seed, n)
        x = A.decode_latents(u, families)
        meta = {}
    elif method == "retrieval":
        x, families, origins, changes, meta = A.sample_retrieval(
            ctx, train_idx, spec, surrogate, seed=seed, n_attempts=n,
            n_originals=min(8, max(1, n // 4)), frac=A.JITTER_FRAC)
        cond = A.conditions_for(spec["material"], families, A.band_specs(spec),
                                bundles["ddpm"]["scalers"])
        u = np.stack([A.LD.encode(row, fam) for row, fam in zip(x, families)])
    else:
        raise ValueError(method)
    return x, u, cond, families, meta


def main():
    started = time.time()
    ctx, train_idx, _val_idx, _ = A.build_context_split(save=False)
    specs = A.load_calibration_specs()
    bundles = {"ddpm": A.load_ddpm(), "cvae": A.load_cvae()}
    surrogate = A.S.load_surrogate("mlp_best", threads=16)
    before = A.old_output_snapshot()
    rows = []
    for slot in ("CAL02", "CAL05"):
        spec = specs[slot]
        for method in ("ddpm", "cvae", "retrieval"):
            cell = "%s_%s" % (slot, method)
            t0 = time.time()
            x1, u1, cond1, fam1, meta = run_generation(
                method, spec, bundles, ctx, train_idx, surrogate)
            x2, u2, cond2, fam2, _ = run_generation(
                method, spec, bundles, ctx, train_idx, surrogate)
            valid = A.validity(x1, spec["material"], fam1)
            pred = A.predict_six(x1, spec["material"], fam1, surrogate)
            cond_ok, cond_notes = condition_checks(cond1, fam1, spec)
            repro_u = A.sha256_bytes(np.asarray(u1, dtype="<f8").tobytes()) == \
                A.sha256_bytes(np.asarray(u2, dtype="<f8").tobytes())
            repro_x = A.sha256_bytes(np.asarray(x1, dtype="<f8").tobytes()) == \
                A.sha256_bytes(np.asarray(x2, dtype="<f8").tobytes())
            fam_bad = family_zero_slots(x1, fam1)
            # CSV round-trip at 17 significant digits.
            frame = A.O.make_frame(x1, spec["material"], fam1)
            for i in range(6):
                frame["cs%d_pred" % (i + 1)] = pred[:, i]
            path = os.path.join(A.SMOKE_DIR, "%s.csv" % cell)
            frame.to_csv(path, index=False, encoding="utf-8", float_format="%.17g")
            back = pd.read_csv(path)
            back_x = back[A.S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
            roundtrip = float(np.max(np.abs(back_x - x1) / np.maximum(np.abs(x1), 1.0)))
            np.savez_compressed(
                os.path.join(A.SMOKE_DIR, "%s.npz" % cell),
                x=np.asarray(x1, dtype=np.float64), u=np.asarray(u1, dtype=np.float64),
                pred=np.asarray(pred, dtype=np.float64),
                nd=np.array([int(f[0]) for f in fam1], dtype=np.int16),
                nb=np.array([int(f[1]) for f in fam1], dtype=np.int16))
            notes = []
            if cond_notes:
                notes.append(cond_notes)
            if fam_bad:
                notes.append("%d rows with nonzero unused slots" % fam_bad)
            status = "pass" if (cond_ok and repro_u and repro_x and roundtrip < 1e-12
                                and np.isfinite(pred).all() and fam_bad == 0) else "fail"
            rows.append({
                "cell": cell, "method": method, "spec": slot,
                "n": int(len(x1)), "condition_dim": int(cond1.shape[1]),
                "condition_ok": bool(cond_ok), "material_ok": True,
                "family_ok": bool(fam_bad == 0),
                "validity_rate": float(np.mean(valid)),
                "pred_finite": bool(np.isfinite(pred).all()),
                "pred_cs1_median": float(np.median(pred[:, 0])),
                "pred_cs2_median": float(np.median(pred[:, 1])),
                "pred_cs3_median": float(np.median(pred[:, 2])),
                "reproducible_u": bool(repro_u), "reproducible_x": bool(repro_x),
                "csv_roundtrip_max_abs": roundtrip,
                "n_families": int(len(set(fam1))),
                "family_list": "|".join("%dd-%db" % (f[0], f[1]) for f in sorted(set(fam1))),
                "retrieval_sources": int(meta.get("n_reference_feasible", -1)) if meta else -1,
                "elapsed_s": float(time.time() - t0),
                "status": status, "notes": "; ".join(notes),
            })
            print("[smoke] %-18s %s valid=%.3f cond=%s repro=%s %.1fs"
                  % (cell, status, np.mean(valid), cond_ok, repro_x, time.time() - t0),
                  flush=True)
    after = A.old_output_snapshot()
    unchanged = before["sha256"] == after["sha256"]
    frame = pd.DataFrame(rows)
    frame["old_dirs_unchanged"] = unchanged
    frame.to_csv(os.path.join(A.SMOKE_DIR, "smoke_cells.csv"), index=False)
    A.save_json(os.path.join(A.SMOKE_DIR, "old_output_before.json"), before)
    A.save_json(os.path.join(A.SMOKE_DIR, "old_output_after.json"), after)

    lines = [
        "# Smoke test (32 candidates per cell, no ROSS)",
        "",
        "- Created: %s" % A.utc_now(),
        "- Total wall: %.1f s" % (time.time() - started),
        "- Historical output directories unchanged: **%s**" % ("YES" if unchanged else "NO"),
        "",
        "| Cell | Status | Validity | Cond | Repro | Round-trip | Families |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| %s | %s | %.3f | %s | %s | %.1e | %d |" % (
            row["cell"], row["status"], row["validity_rate"],
            "OK" if row["condition_ok"] else "FAIL",
            "OK" if (row["reproducible_u"] and row["reproducible_x"]) else "FAIL",
            row["csv_roundtrip_max_abs"], row["n_families"]))
    lines += [
        "",
        "## Checks performed",
        "",
        "- 21-D condition shape, material block, family block, order masks;",
        "- CAL05 inactive order 2 condition edges are zero;",
        "- CAL05 bands are not narrowed by `h_max`;",
        "- decoded 33-D canonical vector; unused slots are zero;",
        "- `O.validity` on every candidate;",
        "- six surrogate outputs are finite;",
        "- CSV written with 17 significant digits and read back;",
        "- two independent runs with the same seed produce identical latent and design hashes;",
        "- historical output directories are fingerprinted before/after.",
        "",
        "No ROSS call is made in this smoke test.",
        "",
        "## Cell notes",
        "",
    ]
    for row in rows:
        lines.append("- **%s**: %s" % (row["cell"], row["notes"] or "no issue"))
    with open(os.path.join(A.SMOKE_DIR, "smoke_test.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("[smoke] wrote", os.path.join(A.SMOKE_DIR, "smoke_test.md"))
    if not unchanged:
        raise SystemExit("historical output directories changed during smoke test")
    if (frame["status"] != "pass").any():
        raise SystemExit("smoke test failed; see smoke_cells.csv")


if __name__ == "__main__":
    main()
