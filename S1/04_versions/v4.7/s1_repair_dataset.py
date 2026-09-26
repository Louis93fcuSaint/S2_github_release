"""Repair an existing rotor dataset with the S1 v4.7 pipeline.

Given ANY feature table produced by an older S1 version, this re-runs
ROSS 2.3.0 on exactly the same geometry with v4.7 semantics and rewrites
the labels.  The geometry is never resampled, so the repaired dataset is
directly comparable with the original one.

Why a repair is usually needed
------------------------------
v4.4 and earlier had two defects.  Neither is visible in the geometry:

1. Precession direction was not filtered.  ROSS returns the damped whirl
   frequencies alternating backward / forward, and taking the three lowest
   gives backward/forward/backward in most cases.  cs1 and cs2 then describe
   the SAME mode in two directions instead of two consecutive orders.
2. The mode budget was too small (num_modes=12), so only six whirl pairs came
   back.  cs4 and above were missing entirely.

v4.7 fixes both: num_modes=48 and the primary targets are the first six
positive, damped, forward-whirl critical speeds.  cs1-cs3 are numerically
identical to v4.6 for the same geometry, so this is a pure label upgrade.

Steps
-----
1. diagnose - schema / label sanity / mode-budget check (no ROSS needed)
2. relabel  - ROSS re-solve, chunked and resumable
3. combine  - merge chunks into one sorted label table
4. compare  - old labels vs new labels, order by order
5. bundle   - repaired dataset + repair_report.md

Usage
-----
    python s1_repair_dataset.py --features features.csv --dataset dataset.csv \
        -o repaired_v4.7 --workers 20 --chunk-size 2000

Add --diagnose-only to stop after step 1, or --skip-relabel to redo only the
combine / compare / bundle steps on chunks that already exist.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import s1_diagnose_dataset as diag  # noqa: E402

DEFAULT_NUM_MODES = 48


def count_rows(path):
    with open(path, "rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def prepare_env():
    os.environ.setdefault("ROSS_FAST_RELABEL", "1")
    os.environ.setdefault("ROSS_ENABLE_NUMBA_JIT", "0")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    prefix = Path(sys.executable).resolve().parent
    library_bin = prefix / "Library" / "bin"
    if library_bin.is_dir():
        parts = [str(library_bin), str(prefix / "Scripts"), str(prefix)]
        current = os.environ.get("PATH", "")
        if current:
            parts.append(current)
        os.environ["PATH"] = os.pathsep.join(parts)


def chunk_paths(out_dir, stem):
    return {
        "labels": out_dir / ("relabeled_v4.7_%s.csv" % stem),
        "modes": out_dir / ("relabel_modes_v4.7_%s.csv" % stem),
        "audit": out_dir / ("relabel_audit_v4.7_%s.csv" % stem),
        "summary": out_dir / ("relabel_summary_v4.7_%s.json" % stem),
    }


def chunk_is_complete(out_dir, stem, limit, num_modes):
    paths = chunk_paths(out_dir, stem)
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        with paths["summary"].open(encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError):
        return False
    return (int(saved.get("n_selected", -1)) == int(limit)
            and int(saved.get("num_modes", -1)) == int(num_modes))


def run_relabel(features, dataset, out_dir, total, chunk_size, workers,
                num_modes):
    import relabel_old_v4_7 as relabeler

    done = 0
    for start in range(0, total, chunk_size):
        limit = min(chunk_size, total - start)
        stem = "%d_%d" % (start, limit)
        if chunk_is_complete(out_dir, stem, limit, num_modes):
            print("[skip] chunk %s" % stem, flush=True)
            done += limit
            continue
        print("[run ] chunk %s" % stem, flush=True)
        relabeler.relabel(
            features_csv=str(features),
            dataset_csv=str(dataset) if dataset else None,
            out_dir=str(out_dir),
            start=start,
            limit=limit,
            workers=workers,
            num_modes=num_modes,
        )
        done += limit
        print("[ok  ] chunk %s (%d/%d rows)" % (stem, done, total), flush=True)
    return done


def run_combine(out_dir, total):
    script = HERE / "combine_relabel_chunks_v4_7.py"
    command = [sys.executable, str(script), "--out-dir", str(out_dir),
               "--total", str(total)]
    result = subprocess.run(command, cwd=str(HERE))
    if result.returncode != 0:
        raise RuntimeError("combine failed with exit code %d"
                           % result.returncode)


def read_combined_labels(path):
    column_to_order = {}
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for index, name in enumerate(fieldnames):
            if name in diag.REQUIRED_FEATURES:
                continue
        cs_columns = diag.pick_cs_columns(fieldnames) or []
        for name in cs_columns:
            column_to_order[name] = int(name.split("_")[1][0])
        for row in reader:
            try:
                row_id = int(row.get("source_row_id", ""))
            except ValueError:
                continue
            rows[row_id] = [row.get(name) for name in cs_columns]
    return cs_columns, rows


def compare(old_matrix, new_rows, cs_columns, out_csv):
    results = []
    new_matrix = np.full(old_matrix.shape, np.nan)
    for row_id, values in new_rows.items():
        if row_id >= new_matrix.shape[0]:
            continue
        for index, text in enumerate(values):
            try:
                new_matrix[row_id, index] = float(text)
            except (TypeError, ValueError):
                pass

    for index, name in enumerate(cs_columns):
        old = old_matrix[:, index]
        new = new_matrix[:, index]
        common = (~np.isnan(old)) & (~np.isnan(new))
        entry = {
            "order": index + 1,
            "column": name,
            "n_common": int(np.sum(common)),
            "median_ape_pct": None,
            "share_ape_gt_1pct": None,
            "median_abs_delta_rpm": None,
        }
        if int(np.sum(common)) >= 1:
            delta = np.abs(new[common] - old[common])
            denominator = np.where(old[common] == 0.0, np.nan, old[common])
            ape = 100.0 * delta / denominator
            ape = ape[~np.isnan(ape)]
            entry["median_abs_delta_rpm"] = float(np.median(delta))
            if ape.size:
                entry["median_ape_pct"] = float(np.median(ape))
                entry["share_ape_gt_1pct"] = float(np.mean(ape > 1.0))
        results.append(entry)

    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    return results


def bundle_repaired_dataset(dataset_csv, new_rows, cs_columns, out_csv):
    with open(dataset_csv, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        original_fields = list(reader.fieldnames or [])
        rows = list(reader)

    kept = [name for name in original_fields if name not in cs_columns]
    fields = kept + list(cs_columns) + ["label_source"]
    written = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_id, row in enumerate(rows):
            values = new_rows.get(row_id)
            if values is None:
                continue
            out = {name: row.get(name) for name in kept}
            for name, text in zip(cs_columns, values):
                out[name] = text
            out["label_source"] = "v4.7"
            writer.writerow(out)
            written += 1
    return written


def coverage_of(new_rows, cs_columns):
    stats = []
    for index, name in enumerate(cs_columns):
        present = 0
        for values in new_rows.values():
            text = values[index]
            if text not in (None, ""):
                try:
                    float(text)
                    present += 1
                except ValueError:
                    pass
        total = max(len(new_rows), 1)
        stats.append({"column": name, "n_present": present,
                      "coverage": present / float(total)})
    return stats


def _fmt(value, digits=4):
    if value is None:
        return "-"
    return ("%%.%df" % digits) % value


def write_report(path, payload):
    lines = []
    lines.append("# 数据集修复报告 (v4.7)")
    lines.append("")
    lines.append("- 特征表: %s" % payload["features"])
    lines.append("- 标签表: %s" % (payload["dataset"] or "未提供"))
    lines.append("- 输出目录: %s" % payload["out_dir"])
    lines.append("- 行数: %d" % payload["n_rows"])
    lines.append("- 模态预算: %d" % payload["num_modes"])
    lines.append("- 进程数: %d" % payload["workers"])
    lines.append("")
    lines.append("## 1. 修复前诊断")
    lines.append("")
    for note in payload["diagnose"]["verdict"]["notes"]:
        lines.append("- %s" % note)
    lines.append("")
    lines.append("## 2. 修复后各阶覆盖率")
    lines.append("")
    lines.append("| 阶次 | 列名 | 有效样本 | 覆盖率 |")
    lines.append("|---|---|---|---|")
    for item in payload["new_coverage"]:
        lines.append("| %s | %s | %d | %.1f%% |" % (
            item["column"].split("_")[1], item["column"], item["n_present"],
            100.0 * item["coverage"]))
    lines.append("")
    if payload["comparison"]:
        lines.append("## 3. 新旧标签对比")
        lines.append("")
        lines.append("| 阶次 | 共同样本 | 绝对差中位数 (RPM) | APE 中位数 | APE>1% 的比例 |")
        lines.append("|---|---|---|---|---|")
        for item in payload["comparison"]:
            lines.append("| cs%d | %d | %s | %s | %s |" % (
                item["order"], item["n_common"],
                _fmt(item["median_abs_delta_rpm"], 1),
                _fmt(item["median_ape_pct"], 2) + "%"
                if item["median_ape_pct"] is not None else "-",
                _fmt(100.0 * item["share_ape_gt_1pct"], 1) + "%"
                if item["share_ape_gt_1pct"] is not None else "-"))
        lines.append("")
        lines.append("cs1-cs3 的差异应当很小（同一套正进动定义），差异主要应出现在"
                     "原先被删失的高阶上。若 cs1 或 cs2 差异很大，说明原标签的"
                     "进动方向定义确实有问题。")
        lines.append("")
    lines.append("## 4. 产物")
    lines.append("")
    for name, value in payload["outputs"].items():
        lines.append("- %s: %s" % (name, value))
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Repair an existing rotor dataset with S1 v4.7 labels")
    parser.add_argument("--features", required=True,
                        help="feature CSV to repair (geometry is kept as is)")
    parser.add_argument("--dataset", default=None,
                        help="optional label CSV, used for the old/new compare")
    parser.add_argument("-o", "--out-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--num-modes", type=int, default=DEFAULT_NUM_MODES)
    parser.add_argument("--limit", type=int, default=0,
                        help="0 means all rows")
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--skip-relabel", action="store_true")
    args = parser.parse_args()

    features = Path(args.features)
    dataset = Path(args.dataset) if args.dataset else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows = count_rows(features)
    total = n_rows if args.limit <= 0 else min(n_rows, args.limit)

    print("=" * 72)
    print("  S1 v4.7 dataset repair")
    print("  features : %s (%d rows)" % (features, n_rows))
    print("  dataset  : %s" % (dataset or "(none)"))
    print("  out-dir  : %s" % out_dir)
    print("  modes=%d  workers=%d  chunk=%d  rows_to_process=%d"
          % (args.num_modes, args.workers, args.chunk_size, total))
    print("=" * 72)

    print("[1/5] diagnose ...", flush=True)
    diagnose = diag.diagnose(features, dataset, out_dir, args.workers,
                             args.num_modes)
    verdict = diagnose["verdict"]
    if verdict["label_definition_suspect"] or verdict["mode_budget_insufficient"]:
        print("      estimated repair time: %.1f h" %
              verdict["repair_hours_estimate"], flush=True)
    if args.diagnose_only:
        print("done (--diagnose-only)")
        return

    if args.skip_relabel:
        print("[2/5] relabel skipped", flush=True)
    else:
        print("[2/5] relabel ...", flush=True)
        prepare_env()
        run_relabel(features, dataset, out_dir, total, args.chunk_size,
                    args.workers, args.num_modes)

    print("[3/5] combine ...", flush=True)
    run_combine(out_dir, total)

    combined = out_dir / "relabeled_v4.7_all.csv"
    cs_columns, new_rows = read_combined_labels(combined)
    new_coverage = coverage_of(new_rows, cs_columns)

    comparison = []
    outputs = {
        "诊断报告": str(out_dir / "diagnose_report.md"),
        "修复标签表": str(combined),
        "模态长表": str(out_dir / "relabel_modes_v4.7_all.csv"),
        "审计表": str(out_dir / "relabel_audit_v4.7_all.csv"),
    }

    if dataset is not None:
        print("[4/5] compare old vs new ...", flush=True)
        old_matrix = diag.load_columns(dataset, cs_columns)
        comparison = compare(old_matrix, new_rows, cs_columns,
                             out_dir / "comparison_v4.7.csv")
        repaired = out_dir / ("repaired_dataset_v4.7_%d.csv" % len(new_rows))
        written = bundle_repaired_dataset(dataset, new_rows, cs_columns,
                                          repaired)
        outputs["修复后数据集"] = "%s (%d rows)" % (repaired, written)
        outputs["新旧对比表"] = str(out_dir / "comparison_v4.7.csv")
    else:
        print("[4/5] compare skipped (no --dataset)", flush=True)
        rebuilt = out_dir / ("repaired_labels_v4.7_%d.csv" % len(new_rows))
        with open(combined, newline="", encoding="utf-8-sig") as src, \
                open(rebuilt, "w", newline="", encoding="utf-8") as dst:
            dst.write(src.read())
        outputs["修复后标签表"] = str(rebuilt)

    print("[5/5] report ...", flush=True)
    payload = {
        "features": str(features),
        "dataset": str(dataset) if dataset else None,
        "out_dir": str(out_dir),
        "n_rows": n_rows,
        "num_modes": args.num_modes,
        "workers": args.workers,
        "diagnose": diagnose,
        "new_coverage": new_coverage,
        "comparison": comparison,
        "outputs": outputs,
    }
    with (out_dir / "repair_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    write_report(out_dir / "repair_report.md", payload)

    print("")
    print("repaired rows : %d" % len(new_rows))
    print("report        : %s" % (out_dir / "repair_report.md"))
    for name, value in outputs.items():
        print("%-14s: %s" % (name, value))


if __name__ == "__main__":
    main()