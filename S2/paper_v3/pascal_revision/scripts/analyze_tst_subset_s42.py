# -*- coding: utf-8 -*-
"""Summarise the explicitly partial TST subset (5 specs x 3 methods x seed 42)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tst_runs" / "03_ross"
OUT = ROOT / "tst_runs" / "03_ross_subset_s42"
ALL = SRC / "tst_ross_all_slots.csv"
META = SRC / "tst_ross_run_meta.json"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ALL)
    expected_specs = {"TST01", "TST02", "TST04", "TST07", "TST09"}
    if set(frame["spec"].unique()) != expected_specs:
        raise SystemExit("unexpected specification set: %s" % sorted(frame["spec"].unique()))
    if set(frame["seed"].unique()) != {42}:
        raise SystemExit("unexpected seed set: %s" % sorted(frame["seed"].unique()))

    rows = []
    for (pid, list_type), group in frame.groupby(["pool_id", "list_type"]):
        denominator = int(group["expected_denominator"].iloc[0])
        delivered = int(group["delivered"].sum())
        lo, hi = wilson(delivered, denominator)
        rows.append(
            {
                "pool_id": pid,
                "spec": group["spec"].iloc[0],
                "method": group["method"].iloc[0],
                "seed": int(group["seed"].iloc[0]),
                "list_type": list_type,
                "n_slots": int(len(group)),
                "denominator": denominator,
                "n_delivered": delivered,
                "delivery_rate": delivered / denominator,
                "wilson_lo": lo,
                "wilson_hi": hi,
                "n_proxy_in_band": int(group["proxy_in_band"].sum()),
                "n_shortlist_fallback": int((~group["proxy_in_band"]).sum())
                if list_type == "shortlist"
                else 0,
                "n_invalid_generation": int((~group["valid_generation"]).sum()),
                "n_solved_ok": int((group["status"] == "ok").sum()),
                "n_error": int((group["status"] == "error").sum()),
                "n_timeout": int((group["status"] == "timeout").sum()),
                "n_cache_hit": int(group["cache_hit"].sum()),
                "median_elapsed_s": float(group["elapsed_s"].median()) if len(group) else np.nan,
            }
        )
    pool = pd.DataFrame(rows)
    pool.to_csv(OUT / "tst_results_by_pool.csv", index=False, encoding="utf-8-sig")

    matrix = (
        pool.groupby(["spec", "method", "list_type"])
        .agg(
            n_delivered=("n_delivered", "sum"),
            denominator=("denominator", "sum"),
            delivery_rate=("delivery_rate", "mean"),
        )
        .reset_index()
    )
    matrix.to_csv(OUT / "tst_method_spec_matrix.csv", index=False, encoding="utf-8-sig")
    short = matrix[matrix.list_type == "shortlist"].pivot(
        index="spec", columns="method", values="delivery_rate"
    )
    random = matrix[matrix.list_type == "random"].pivot(
        index="spec", columns="method", values="delivery_rate"
    )

    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    lines = [
        "# TST subset ROSS results (five specifications, seed 42)",
        "",
        "**Status: partial validation, not the complete eight-specification / two-seed TST protocol.**",
        "",
        "The subset was frozen before running: TST01, TST02, TST04, TST07 and TST09; DDPM, CVAE and retrieval+jitter; sampling seed 42; 100 uniformly sampled raw-pool candidates and a 200-design shortlist per pool. Denominators remain 100 and 200. Failures, timeouts, invalid generations and missing shortlist entries count as non-delivery. Sampling seed 42 is a sampling replicate of one trained checkpoint, not an independent training run.",
        "",
    ]
    if meta:
        lines += [
            "Run meta: %d logical slots, %d unique designs, %d physical solves, %d cache hits, "
            "batch ROSS wall %.0f s with %d workers."
            % (
                meta.get("logical_slots", -1),
                meta.get("unique_designs", -1),
                meta.get("physical_solves", -1),
                meta.get("cache_hits", -1),
                meta.get("ross_wall_s", float("nan")),
                meta.get("workers", -1),
            ),
            "",
        ]
    lines += [
        "## Shortlist delivery by specification",
        "",
        "| Spec | DDPM | CVAE | Retrieval+jitter | DDPM in-band/200 | CVAE in-band/200 | Retrieval in-band/200 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in sorted(short.index):
        row = short.loc[spec]
        sub = pool[(pool.spec == spec) & (pool.list_type == "shortlist")].set_index("method")
        lines.append(
            "| %s | %.1f%% | %.1f%% | %.1f%% | %d | %d | %d |"
            % (
                spec,
                100 * row.get("ddpm", np.nan),
                100 * row.get("cvae", np.nan),
                100 * row.get("retrieval", np.nan),
                int(sub.loc["ddpm", "n_proxy_in_band"]),
                int(sub.loc["cvae", "n_proxy_in_band"]),
                int(sub.loc["retrieval", "n_proxy_in_band"]),
            )
        )
    lines += [
        "",
        "## Raw-pool random-sample feasibility",
        "",
        "| Spec | DDPM | CVAE | Retrieval+jitter |",
        "|---|---:|---:|---:|",
    ]
    for spec in sorted(random.index):
        row = random.loc[spec]
        lines.append(
            "| %s | %.1f%% | %.1f%% | %.1f%% |"
            % (
                spec,
                100 * row.get("ddpm", np.nan),
                100 * row.get("cvae", np.nan),
                100 * row.get("retrieval", np.nan),
            )
        )
    lines += ["", "## Descriptive paired differences over the five specifications", ""]
    for left, right in (("cvae", "ddpm"), ("retrieval", "ddpm"), ("cvae", "retrieval")):
        differences = (short[left] - short[right]).dropna()
        lines.append(
            "- `%s - %s`: mean %+.1f pp; positive in %d/%d specifications; range %+.1f to %+.1f pp."
            % (
                left,
                right,
                100 * differences.mean(),
                int((differences > 0).sum()),
                len(differences),
                100 * differences.min(),
                100 * differences.max(),
            )
        )
    lines += [
        "",
        "No p-value is attached to these five-specification descriptive differences. The omitted "
        "specifications are TST03, TST05, TST06 and TST08; their omission was a compute-budget decision, "
        "not a selection based on observed delivery rates.",
        "",
        "## Reading rules",
        "",
        "- Retrieval+jitter can use stored labels of the frozen reference library to select source designs; the generative models cannot. This is an information-condition difference and must be stated wherever retrieval is compared.",
        "- Proxy in-band counts are not delivery rates. The final decision is the ROSS result against the original user specification.",
        "- Distances and diversity statistics, if reported, describe parameter dispersion only; they are not engineering quality.",
        "- This subset cannot support a complete method ranking, a significance claim, or generalisation to unseen specifications.",
    ]
    (OUT / "tst_results_analysis_subset_s42.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("wrote", OUT)
    print(short.round(3).to_string())


if __name__ == "__main__":
    main()
