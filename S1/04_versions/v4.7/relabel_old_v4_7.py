"""Relabel an existing v4.4 feature table with v4.7 forward-whirl labels.

The script reads the input feature CSV read-only, rebuilds the same physical
rotor dictionary, reruns ROSS 2.3.0 through the v4.7 model builder, and writes
primary cs1-cs3 labels plus a long-form table containing every raw modal result.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import time
from typing import Any, Dict, List, Optional, Tuple

import s1_run_pipeline_v4_7 as pipeline


DEFAULT_FEATURES = (
    r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S1"
    r"\02_datasets\output_v4.4\merged_100k\features_v4.4_100000.csv"
)
DEFAULT_DATASET = (
    r"D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S1"
    r"\02_datasets\output_v4.4\merged_100k\dataset_v4.4_100000.csv"
)

LABEL_FIELDS = [
    "source_row_id", "source_sample_id", "status", "qc_note",
    "forward_mode_count", "raw_mode_count",
    "cs_1_rpm", "cs_2_rpm", "cs_3_rpm", "cs_4_rpm", "cs_5_rpm", "cs_6_rpm",
    "L_m", "OD_m", "ID_m", "material", "n_disks", "n_bearings",
    "S1_volume_ratio", "S2_support_sag", "S3_bearing_span",
    "volume_ratio", "support_sag_m", "bearing_span_frac",
    "modal_direction_counts",
]

MODAL_FIELDS = [
    "source_row_id", "source_sample_id", "status", "mode_index", "mode_order",
    "wn_rpm", "wd_rpm", "whirl_direction", "log_dec", "damping_ratio",
]

AUDIT_FIELDS = [
    "source_row_id", "source_sample_id", "sample_id", "status", "qc_note",
    "error_msg", "forward_mode_count", "raw_mode_count",
    "cs_1_rpm", "cs_2_rpm", "cs_3_rpm", "cs_4_rpm", "cs_5_rpm", "cs_6_rpm",
    "L_m", "OD_m", "ID_m", "material", "bearing_type",
    "n_disks", "n_bearings",
    "S1_volume_ratio", "S2_support_sag", "S3_bearing_span",
    "volume_ratio", "support_sag_m", "bearing_span_frac",
    "modal_direction_counts",
]


def _float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def feature_row_to_sample(row: Dict[str, str]) -> Dict[str, Any]:
    """Map the 39-column v4.4 feature schema back to a sample dictionary."""
    nd = int(_float(row, "n_disks"))
    nb = int(_float(row, "n_bearings"))
    if nd < 1 or nd > 6 or nb < 2 or nb > 4:
        raise ValueError(f"invalid nd={nd}, nb={nb}")

    material = None
    for name in pipeline.MATERIALS.keys():
        if _float(row, f"mat_{name}") > 0.5:
            material = name
            break
    if material is None:
        material = "Steel"

    sample = {
        "sl": _float(row, "L"),
        "od": _float(row, "OD"),
        "id": min(_float(row, "ID"), 0.8 * _float(row, "OD")),
        "material": material,
        "bearing_type": row.get("brg_type") or "isotropic",
        "nd": nd,
        "disk_od_list": [_float(row, f"d{i}_OD") for i in range(nd)],
        "disk_w_list": [_float(row, f"d{i}_W") for i in range(nd)],
        "dp": [_float(row, f"d{i}_pos") for i in range(nd)],
        "nb": nb,
        "bk_list": [_float(row, f"b{i}_K") for i in range(nb)],
        "bc_list": [_float(row, f"b{i}_C") for i in range(nb)],
        "bp": [_float(row, f"b{i}_pos") for i in range(nb)],
    }
    sample["soft"] = pipeline._cf.check_soft_v4(sample)
    return sample


def _source_ids(dataset_csv: Optional[str], n_rows: int) -> List[Optional[str]]:
    if not dataset_csv:
        return [None] * n_rows
    ids = []
    with open(dataset_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(row.get("sample_id"))
    if len(ids) != n_rows:
        raise ValueError(
            f"dataset row count {len(ids)} != feature row count {n_rows}"
        )
    return ids


def _worker(args: Tuple[int, Optional[str], Dict[str, Any], float, float, int]):
    row_id, source_id, sample, cs_min, cs_max, num_modes = args
    row, ok, modal_rows = pipeline.evaluate_sample(
        row_id, sample, cs_min=cs_min, cs_max=cs_max, num_modes=num_modes
    )
    row["source_row_id"] = row_id
    row["source_sample_id"] = source_id
    for modal in modal_rows:
        modal["source_row_id"] = row_id
        modal["source_sample_id"] = source_id
    return row, ok, modal_rows


def relabel(features_csv: str, dataset_csv: Optional[str], out_dir: str,
            start: int = 0, limit: int = 0, workers: int = 4,
            num_modes: int = pipeline.NUM_MODES,  # 48 in v4.7
            cs_min: float = pipeline.CS_MIN_RPM,
            cs_max: float = pipeline.CS_MAX_RPM) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    with open(features_csv, newline="", encoding="utf-8") as f:
        feature_rows = list(csv.DictReader(f))
    n_available = len(feature_rows)
    source_ids = _source_ids(dataset_csv, n_available)
    if start < 0 or start >= n_available:
        raise ValueError(f"start={start} outside [0, {n_available - 1}]")
    stop = n_available if limit <= 0 else min(n_available, start + limit)
    selected = list(range(start, stop))
    n_selected = len(selected)
    stem = f"{start}_{n_selected}"

    label_path = os.path.join(out_dir, f"relabeled_v4.7_{stem}.csv")
    modal_path = os.path.join(out_dir, f"relabel_modes_v4.7_{stem}.csv")
    audit_path = os.path.join(out_dir, f"relabel_audit_v4.7_{stem}.csv")
    summary_path = os.path.join(out_dir, f"relabel_summary_v4.7_{stem}.json")

    counts = {"success": 0, "partial": 0, "rejected_cs1": 0, "error": 0}
    t0 = time.time()
    print("=" * 72)
    print("  v4.7 old-data relabel")
    print(f"  input: {features_csv}")
    print(f"  rows: start={start}, count={n_selected}, workers={workers}")
    print("  primary targets: first 3 positive damped forward-whirl speeds")
    print("=" * 72)

    tasks = []
    for row_id in selected:
        sample = feature_row_to_sample(feature_rows[row_id])
        tasks.append((
            row_id, source_ids[row_id], sample, cs_min, cs_max, num_modes
        ))

    with open(label_path, "w", newline="", encoding="utf-8") as fl, \
            open(modal_path, "w", newline="", encoding="utf-8") as fm, \
            open(audit_path, "w", newline="", encoding="utf-8") as fa:
        label_writer = csv.DictWriter(fl, fieldnames=LABEL_FIELDS)
        modal_writer = csv.DictWriter(fm, fieldnames=MODAL_FIELDS)
        label_writer.writeheader()
        modal_writer.writeheader()

        audit_writer = csv.DictWriter(fa, fieldnames=AUDIT_FIELDS)
        audit_writer.writeheader()
        with mp.Pool(processes=workers) as pool:
            for idx, (row, ok, modal_rows) in enumerate(
                pool.imap_unordered(_worker, tasks)
            ):
                status = row.get("status", "error")
                counts[status] = counts.get(status, 0) + 1
                label_writer.writerow(
                    {key: row.get(key) for key in LABEL_FIELDS}
                )
                for modal in modal_rows:
                    modal_writer.writerow(
                        {key: modal.get(key) for key in MODAL_FIELDS}
                    )

                audit_writer.writerow(
                    {key: row.get(key) for key in AUDIT_FIELDS}
                )
                if (idx + 1) % 50 == 0 or idx + 1 == n_selected:
                    print(f"    [{idx + 1}/{n_selected}] {counts}")

    summary = {
        "version": "v4.7",
        "input_features": features_csv,
        "input_dataset": dataset_csv,
        "start": start,
        "n_selected": n_selected,
        "n_success": counts.get("success", 0),
        "n_partial": counts.get("partial", 0),
        "n_rejected_cs1": counts.get("rejected_cs1", 0),
        "n_error": counts.get("error", 0),
        "workers": workers,
        "num_modes": num_modes,
        "cs_min_rpm": cs_min,
        "cs_max_rpm": cs_max,
        "primary_target": "positive_damped_forward_whirl",
        "all_modes_preserved": True,
        "fast_critical_speed": os.environ.get("ROSS_FAST_RELABEL", "0") == "1",
        "numba_jit": os.environ.get("ROSS_ENABLE_NUMBA_JIT", "0") == "1",
        "undamped_root_solved": os.environ.get("ROSS_FAST_RELABEL", "0") != "1",
        "wn_definition": (
            "absolute eigenvalue at the damped critical speed"
            if os.environ.get("ROSS_FAST_RELABEL", "0") == "1"
            else "separate s=wn synchronism root"
        ),
        "outputs": {
            "labels": label_path,
            "modes": modal_path,
            "audit": audit_path,
        },
        "python_version": platform.python_version(),
        "ross_version": getattr(pipeline.rs, "__version__", "unknown"),
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary: {summary_path}")
    print(f"  Time: {summary['elapsed_s']:.1f}s")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("-o", "--out-dir", default="relabeled_v4.7")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="0 means all remaining rows; use a small value for a pilot",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-modes", type=int, default=pipeline.NUM_MODES)
    parser.add_argument("--cs-min", type=float, default=pipeline.CS_MIN_RPM)
    parser.add_argument("--cs-max", type=float, default=pipeline.CS_MAX_RPM)
    args = parser.parse_args()
    relabel(
        args.features, args.dataset, args.out_dir, args.start, args.limit,
        args.workers, args.num_modes, args.cs_min, args.cs_max,
    )
