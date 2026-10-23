# -*- coding: utf-8 -*-
"""Analyse the frozen TST ROSS acceptance results (8 specs x 3 methods x 2 sampling seeds)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROSS = ROOT / "tst_runs" / "03_ross"
ALL = ROSS / "tst_ross_all_slots.csv"
META = ROSS / "tst_ross_run_meta.json"
OUT_POOL = ROSS / "tst_results_by_pool.csv"
OUT_MATRIX = ROSS / "tst_method_spec_matrix.csv"
OUT_MD = ROOT / "tst_results_analysis.md"

df = pd.read_csv(ALL)
meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


rows = []
for (pid, lt), g in df.groupby(["pool_id", "list_type"]):
    den = int(g["expected_denominator"].iloc[0])
    k = int(g["delivered"].sum())
    lo, hi = wilson(k, den)
    rows.append({
        "pool_id": pid, "spec": g["spec"].iloc[0], "method": g["method"].iloc[0],
        "list_type": lt, "n_slots": len(g), "denominator": den,
        "n_delivered": k, "delivery_rate": k / den,
        "wilson_lo": lo, "wilson_hi": hi,
        "n_proxy_in_band": int(g["proxy_in_band"].sum()),
        "n_shortlist_fallback": int((~g["proxy_in_band"]).sum()) if lt == "shortlist" else 0,
        "n_invalid_generation": int((~g["valid_generation"]).sum()),
        "n_solved_ok": int((g["status"] == "ok").sum()),
        "n_error": int((g["status"] == "error").sum()),
        "n_timeout": int((g["status"] == "timeout").sum()),
        "n_cache_hit": int(g["cache_hit"].sum()),
        "median_elapsed_s": float(g["elapsed_s"].median()) if len(g) else np.nan,
    })
pool = pd.DataFrame(rows)
pool.to_csv(OUT_POOL, index=False, encoding="utf-8-sig")

# method x spec matrix (mean over the two sampling seeds)
mat = (pool.groupby(["spec", "method", "list_type"])
       .agg(rate_mean=("delivery_rate", "mean"), rate_min=("delivery_rate", "min"),
            rate_max=("delivery_rate", "max"), n_delivered=("n_delivered", "sum"),
            denominator=("denominator", "sum"), n_proxy_in_band=("n_proxy_in_band", "mean"),
            n_fallback=("n_shortlist_fallback", "mean"))
       .reset_index())
mat.to_csv(OUT_MATRIX, index=False, encoding="utf-8-sig")

short = mat[mat.list_type == "shortlist"].pivot(index="spec", columns="method", values="rate_mean")
rand = mat[mat.list_type == "random"].pivot(index="spec", columns="method", values="rate_mean")
specs = list(short.index)

lines = []
lines.append("# TST frozen-specification ROSS results\n")
lines.append("Protocol: 8 frozen test specifications (TST08 excluded because the frozen reference "
             "library contains zero feasible designs), 3 methods, sampling seeds 42 and 7, 5,000 raw "
             "generation attempts per pool, 100 uniform pool-level random samples and a 200-design "
             "shortlist per pool. Denominators are fixed (100 and 200); failures, timeouts and missing "
             "slots count as non-delivery. ROSS 1.6.1, forward positive, n_base=15, num_modes=48.")
if meta:
    lines.append("\nRun meta: %d logical slots, %d unique designs, %d physical solves, %d cache hits, "
                 "batch wall %.0f s with %d workers.\n" % (
                     meta.get("logical_slots", -1), meta.get("unique_designs", -1),
                     meta.get("physical_solves", -1), meta.get("cache_hits", -1),
                     meta.get("ross_wall_s", float("nan")), meta.get("workers", -1)))
lines.append("## Shortlist delivery by specification (mean of two sampling seeds)\n")
lines.append("| Spec | DDPM | CVAE | Retrieval+jitter | DDPM in-band / 200 | CVAE in-band / 200 | Retrieval in-band / 200 |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for s in specs:
    row = short.loc[s]
    ib = mat[(mat.spec == s) & (mat.list_type == "shortlist")].set_index("method")["n_proxy_in_band"]
    lines.append("| %s | %.1f%% | %.1f%% | %.1f%% | %.0f | %.0f | %.0f |" % (
        s, 100 * row.get("ddpm", np.nan), 100 * row.get("cvae", np.nan),
        100 * row.get("retrieval", np.nan), ib.get("ddpm", np.nan), ib.get("cvae", np.nan),
        ib.get("retrieval", np.nan)))
lines.append("")
lines.append("## Random pool-level feasible rate (mean of two sampling seeds)\n")
lines.append("| Spec | DDPM | CVAE | Retrieval+jitter |")
lines.append("|---|---:|---:|---:|")
for s in specs:
    row = rand.loc[s]
    lines.append("| %s | %.1f%% | %.1f%% | %.1f%% |" % (
        s, 100 * row.get("ddpm", np.nan), 100 * row.get("cvae", np.nan), 100 * row.get("retrieval", np.nan)))
lines.append("")
lines.append("## Descriptive method differences\n")
for a, b in (("cvae", "ddpm"), ("retrieval", "ddpm"), ("cvae", "retrieval")):
    d = (short[a] - short[b]).dropna()
    if len(d):
        lines.append("- %s - %s: mean %+.1f pp over %d specs, %d/%d specs positive, range %+.1f to %+.1f pp." % (
            a, b, 100 * d.mean(), len(d), int((d > 0).sum()), len(d), 100 * d.min(), 100 * d.max()))
lines.append("")
lines.append("## Reading rules\n")
lines.append("- The shortlist is built from proxy in-band candidates; when fewer than 200 are in-band, "
             "the remainder is filled by out-of-band candidates ranked by specification violation. "
             "In-band counts must not be reported as delivery rates.")
lines.append("- Retrieval+jitter selects source designs using the stored labels of the frozen reference "
             "library. Generative models do not access those labels. This is an information-condition "
             "difference, not only an algorithmic one, and must be stated wherever retrieval is compared.")
lines.append("- The eight specifications are stratified by reference-library support; they are not a "
             "random sample of the specification space, and each specification appears at only two "
             "sampling seeds. Inference is therefore descriptive, with the specification as the unit.")
lines.append("- Sampling seeds are repeated sampling of the same trained checkpoint, not independent "
             "training runs.")
OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", OUT_POOL)
print("wrote", OUT_MATRIX)
print("wrote", OUT_MD)
print(short.round(3).to_string())
