"""Combine checkpointed v4.7 relabel chunks into sorted final CSV files."""

import argparse
import csv
import glob
import json
import os
from pathlib import Path


def read_rows(pattern: str) -> tuple[list[str], list[dict]]:
    paths = sorted(
        path for path in glob.glob(pattern)
        if not path.endswith("_all.csv")
        and not path.endswith("_success.csv")
    )
    if not paths:
        raise FileNotFoundError(f"No files matched {pattern}")
    rows = []
    fieldnames = None
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"Schema mismatch in {path}")
            rows.extend(reader)
    return fieldnames or [], rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--total", type=int, required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    summaries = []
    for path in sorted(glob.glob(str(out_dir / "relabel_summary_v4.7_*.json"))):
        if path.endswith("_all.json"):
            continue
        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle)
        summaries.append(summary)
    summaries.sort(key=lambda item: item["start"])
    selected_total = sum(int(item["n_selected"]) for item in summaries)
    if selected_total != args.total:
        raise ValueError(
            f"Chunk summaries cover {selected_total} rows, expected {args.total}"
        )

    label_fields, labels = read_rows(str(out_dir / "relabeled_v4.7_*.csv"))
    modal_fields, modes = read_rows(str(out_dir / "relabel_modes_v4.7_*.csv"))
    audit_fields, audits = read_rows(str(out_dir / "relabel_audit_v4.7_*.csv"))

    labels.sort(key=lambda row: int(row["source_row_id"]))
    audits.sort(key=lambda row: int(row["source_row_id"]))
    modes.sort(
        key=lambda row: (int(row["source_row_id"]), int(row["mode_index"]))
    )

    label_out = out_dir / "relabeled_v4.7_all.csv"
    modal_out = out_dir / "relabel_modes_v4.7_all.csv"
    audit_out = out_dir / "relabel_audit_v4.7_all.csv"
    training_out = out_dir / "relabeled_v4.7_success.csv"
    write_rows(label_out, label_fields, labels)
    write_rows(modal_out, modal_fields, modes)
    write_rows(audit_out, audit_fields, audits)

    training_fields = [
        "source_row_id",
        "source_sample_id",
        "cs_1_rpm",
        "cs_2_rpm",
        "cs_3_rpm",
        "material",
        "n_disks",
        "n_bearings",
        "L_m",
        "OD_m",
        "ID_m",
        "S1_volume_ratio",
        "S2_support_sag",
        "S3_bearing_span",
        "volume_ratio",
        "support_sag_m",
        "bearing_span_frac",
    ]
    training_rows = [
        {key: row.get(key) for key in training_fields}
        for row in labels
        if row.get("status") == "success"
        and all(row.get(f"cs_{order}_rpm") not in (None, "") for order in (1, 2, 3))
    ]
    write_rows(training_out, training_fields, training_rows)

    counts = {"success": 0, "partial": 0, "rejected_cs1": 0, "error": 0}
    for row in labels:
        status = row.get("status", "error")
        counts[status] = counts.get(status, 0) + 1

    first = summaries[0]
    combined = {
        "version": "v4.7",
        "input_features": first.get("input_features"),
        "input_dataset": first.get("input_dataset"),
        "n_selected": len(labels),
        "n_success": counts.get("success", 0),
        "n_partial": counts.get("partial", 0),
        "n_rejected_cs1": counts.get("rejected_cs1", 0),
        "n_error": counts.get("error", 0),
        "num_modes": first.get("num_modes"),
        "cs_min_rpm": first.get("cs_min_rpm"),
        "cs_max_rpm": first.get("cs_max_rpm"),
        "primary_target": first.get("primary_target"),
        "all_modes_preserved": True,
        "fast_critical_speed": first.get("fast_critical_speed"),
        "numba_jit": first.get("numba_jit"),
        "undamped_root_solved": first.get("undamped_root_solved"),
        "wn_definition": first.get("wn_definition"),
        "ross_version": first.get("ross_version"),
        "python_version": first.get("python_version"),
        "chunks": [
            {
                "start": item.get("start"),
                "n_selected": item.get("n_selected"),
                "elapsed_s": item.get("elapsed_s"),
            }
            for item in summaries
        ],
        "outputs": {
            "labels": os.fspath(label_out),
            "modes": os.fspath(modal_out),
            "audit": os.fspath(audit_out),
            "training_success": os.fspath(training_out),
        },
    }
    combined["n_training_success"] = len(training_rows)
    summary_out = out_dir / "relabel_summary_v4.7_all.json"
    with summary_out.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, ensure_ascii=False)
    print(
        f"Combined {len(labels)} samples, {len(modes)} modal rows. "
        f"Summary: {summary_out}"
    )


if __name__ == "__main__":
    main()
