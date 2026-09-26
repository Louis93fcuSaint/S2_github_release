"""Diagnose any rotor feature / label table against the S1 v4.7 schema.

Uses only csv + numpy and never imports ROSS, so it runs in seconds on any
interpreter.  Run it BEFORE a repair so you know what is actually wrong.

What it looks for
-----------------
1. Schema  - the 39 required v4.7 feature columns.  Column ORDER does not
             matter, because feature rows are read by name; extra columns
             are ignored.
2. Labels  - per-order coverage, monotonicity, adjacent-order ratios and the
             near-duplicate rate.  The v4.4 pipeline took the N lowest whirl
             frequencies WITHOUT filtering precession direction, so cs1/cs2
             were very often the backward/forward pair of the SAME mode.
             That shows up here as a cs2/cs1 ratio hugging 1.0.
3. Verdict - critical-speed definition suspect and/or mode budget too small,
             plus a runtime estimate for the repair.

Usage
-----
    python s1_diagnose_dataset.py --features features.csv \
        [--dataset dataset.csv] [-o diagnose_out] [--workers 20]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MATERIALS = ["Steel", "Aluminum", "Titanium"]
REQUIRED_FEATURES = (
    ["L", "OD", "ID", "n_disks", "n_bearings"]
    + ["mat_%s" % m for m in MATERIALS]
    + ["d%d_%s" % (i, a) for i in range(6) for a in ("OD", "W", "pos")]
    + ["brg_type"]
    + ["b%d_%s" % (i, a) for i in range(4) for a in ("K", "C", "pos")]
)
CS_PATTERNS = [
    ["cs_%d_rpm" % k for k in range(1, 7)],
    ["cs%d_rpm" % k for k in range(1, 7)],
    ["cs_%d" % k for k in range(1, 7)],
    ["cs%d" % k for k in range(1, 7)],
]
NEAR_DUPLICATE_RATIO = 1.02
MIN_PAIR_SAMPLES = 30
SECONDS_PER_ROTOR_48_MODES = 9.8


def read_header(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return next(csv.reader(handle))


def pick_cs_columns(header):
    for pattern in CS_PATTERNS:
        if all(name in header for name in pattern):
            return pattern
    return None


def load_columns(path, columns):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values = []
            for name in columns:
                text = (row.get(name) or "").strip()
                try:
                    values.append(float(text))
                except ValueError:
                    values.append(np.nan)
            rows.append(values)
    if not rows:
        return np.zeros((0, len(columns)))
    return np.asarray(rows, dtype=float)


def _safe_stats(values):
    values = values[~np.isnan(values)]
    if values.size == 0:
        return None, None, None
    return (float(np.min(values)), float(np.median(values)),
            float(np.max(values)))


def analyse_orders(matrix, names):
    n_rows = matrix.shape[0]
    orders = []
    for index in range(matrix.shape[1]):
        column = matrix[:, index]
        low, mid, high = _safe_stats(column)
        orders.append({
            "order": index + 1,
            "column": names[index],
            "coverage": float(np.mean(~np.isnan(column))) if n_rows else 0.0,
            "n_present": int(np.sum(~np.isnan(column))),
            "min": low,
            "median": mid,
            "max": high,
        })
    return {"n_rows": n_rows, "orders": orders}


def analyse_pairs(matrix, names):
    pairs = []
    for index in range(matrix.shape[1] - 1):
        low = matrix[:, index]
        high = matrix[:, index + 1]
        common = (~np.isnan(low)) & (~np.isnan(high))
        entry = {
            "pair": "%s->%s" % (names[index], names[index + 1]),
            "n_common": int(np.sum(common)),
        }
        if int(np.sum(common)) >= MIN_PAIR_SAMPLES:
            denominator = np.where(low[common] == 0.0, np.nan, low[common])
            ratio = high[common] / denominator
            ratio = ratio[~np.isnan(ratio)]
            if ratio.size:
                entry["ratio_median"] = float(np.median(ratio))
                entry["ratio_p10"] = float(np.percentile(ratio, 10))
                entry["ratio_p90"] = float(np.percentile(ratio, 90))
                entry["near_duplicate_rate"] = float(
                    np.mean(ratio < NEAR_DUPLICATE_RATIO))
                entry["non_monotonic_rate"] = float(np.mean(ratio <= 1.0))
        pairs.append(entry)
    return pairs


def build_verdict(order_stats, pairs, workers, num_modes):
    coverage = [order["coverage"] for order in order_stats["orders"]]
    highest = 0
    for index, value in enumerate(coverage, start=1):
        if value > 0.001:
            highest = index
    top_coverage = coverage[highest - 1] if highest else 0.0

    def near_duplicate(index):
        if index < len(pairs):
            return pairs[index].get("near_duplicate_rate")
        return None

    first_rate = near_duplicate(0)
    first_median = pairs[0].get("ratio_median") if pairs else None

    even = [near_duplicate(i) for i in (0, 2, 4)]
    odd = [near_duplicate(i) for i in (1, 3)]
    even = [value for value in even if value is not None]
    odd = [value for value in odd if value is not None]
    even_mean = sum(even) / len(even) if even else None
    odd_mean = sum(odd) / len(odd) if odd else None

    definition_suspect = bool(
        (first_rate is not None and first_rate > 0.25)
        or (first_median is not None and first_median < 1.2)
    )

    alternating_pair_signature = bool(
        even_mean is not None and even_mean > 0.05
        and (odd_mean is None or even_mean > 1.5 * odd_mean)
    )

    censored_high_orders = highest >= 3 and top_coverage < 0.9
    budget_insufficient = bool(alternating_pair_signature
                               or censored_high_orders)

    n_rows = order_stats["n_rows"]
    hours = (n_rows * SECONDS_PER_ROTOR_48_MODES * (float(num_modes) / 48.0)
             / max(int(workers), 1) / 3600.0)

    notes = []
    if definition_suspect:
        notes.append(
            "前两阶比值异常: cs2/cs1 中位数 %s, 近重复率(比值<%.2f) %.1f%%. "
            "这是未过滤进动方向、直接取最低几个涡动频率的典型特征, "
            "cs1 与 cs2 很可能是同一阶模态的后向/前向一对, 而不是相邻两阶."
            % (_fmt(first_median), NEAR_DUPLICATE_RATIO,
               100.0 * (first_rate or 0.0)))
    if alternating_pair_signature:
        notes.append(
            "存在成对结构: 第1-2、3-4、5-6 阶的平均近重复率 %.1f%%, 明显高于"
            "第2-3、4-5 阶的 %s. 这说明求解时模态预算偏小 (例如 num_modes=12 "
            "只返回 6 个涡动频率), 偶数序号的阶其实是前一阶的反向模态."
            % (100.0 * even_mean,
               ("%.1f%%" % (100.0 * odd_mean)) if odd_mean is not None else "-"))
    elif censored_high_orders:
        notes.append(
            "最高有效阶 cs%d 的覆盖率只有 %.1f%%, 说明求解时的模态预算不足, "
            "高阶临界转速被删失." % (highest, 100.0 * top_coverage))
    if not notes:
        if first_rate is None and first_median is None:
            notes.append(
                "样本量不足 (每个阶对至少需要 %d 个共同样本) 或缺少数阶, "
                "无法判断标签定义; 修复流程本身不受影响." % MIN_PAIR_SAMPLES)
        else:
            notes.append("未发现方向混叠或高阶删失的迹象, 标签可用于训练.")

    return {
        "highest_order_present": highest,
        "top_order_coverage": top_coverage,
        "cs2_over_cs1_near_duplicate_rate": first_rate,
        "cs2_over_cs1_ratio_median": first_median,
        "even_pair_near_duplicate_mean": even_mean,
        "odd_pair_near_duplicate_mean": odd_mean,
        "alternating_pair_signature": alternating_pair_signature,
        "label_definition_suspect": definition_suspect,
        "mode_budget_insufficient": budget_insufficient,
        "repair_hours_estimate": float(hours),
        "workers_used": int(workers),
        "num_modes_planned": int(num_modes),
        "notes": notes,
    }


def _fmt(value):
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return "%.0f" % value
    return "%.4f" % value


def _pct(value):
    if value is None:
        return "-"
    return "%.1f%%" % (100.0 * value)


def write_report(path, feature_path, label_path, header, order_stats, pairs,
                 verdict):
    lines = []
    lines.append("# 数据集诊断报告")
    lines.append("")
    lines.append("- 特征表: %s" % feature_path)
    lines.append("- 标签表: %s" % (label_path or "未提供"))
    lines.append("- 行数: %d" % order_stats["n_rows"])
    lines.append("- 特征列数: %d" % len(header))
    lines.append("")
    lines.append("## 1. 结论")
    lines.append("")
    for note in verdict["notes"]:
        lines.append("- %s" % note)
    if verdict["label_definition_suspect"] or verdict["mode_budget_insufficient"]:
        lines.append(
            "- 建议修复: 用 s1_repair_dataset.py 在同样的几何上按 48 模态重跑 "
            "ROSS，预计 %.1f 小时（%d 个进程的估算值，仅供参考）。"
            % (verdict["repair_hours_estimate"], verdict["workers_used"]))
    else:
        lines.append("- 不需要修复，可直接用于训练。")
    lines.append("")
    lines.append("## 2. 各阶标签覆盖")
    lines.append("")
    lines.append("| 阶次 | 列名 | 有效样本 | 覆盖率 | 最小值 | 中位数 | 最大值 |")
    lines.append("|---|---|---|---|---|---|---|")
    for order in order_stats["orders"]:
        lines.append("| cs%d | %s | %d | %.1f%% | %s | %s | %s |" % (
            order["order"], order["column"], order["n_present"],
            100.0 * order["coverage"], _fmt(order["min"]),
            _fmt(order["median"]), _fmt(order["max"])))
    lines.append("")
    lines.append("## 3. 相邻阶比值 (判断标签定义是否可疑)")
    lines.append("")
    lines.append("| 阶对 | 共同样本 | 比值中位数 | P10 | P90 | 近重复率(<1.02) | 非单调率 |")
    lines.append("|---|---|---|---|---|---|---|")
    for pair in pairs:
        lines.append("| %s | %d | %s | %s | %s | %s | %s |" % (
            pair["pair"], pair["n_common"], _fmt(pair.get("ratio_median")),
            _fmt(pair.get("ratio_p10")), _fmt(pair.get("ratio_p90")),
            _pct(pair.get("near_duplicate_rate")),
            _pct(pair.get("non_monotonic_rate"))))
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnose(feature_path, label_path=None, out_dir=None, workers=20,
             num_modes=48):
    header = read_header(feature_path)
    label_header = read_header(label_path) if label_path else header
    cs_columns = pick_cs_columns(label_header) or pick_cs_columns(header)
    label_source = label_path if pick_cs_columns(label_header) else feature_path

    if cs_columns:
        matrix = load_columns(label_source, cs_columns)
        order_stats = analyse_orders(matrix, cs_columns)
        pairs = analyse_pairs(matrix, cs_columns)
        verdict = build_verdict(order_stats, pairs, workers, num_modes)
    else:
        order_stats = {"n_rows": 0, "orders": []}
        pairs = []
        verdict = {
            "highest_order_present": 0,
            "top_order_coverage": 0.0,
            "cs2_over_cs1_near_duplicate_rate": None,
            "label_definition_suspect": False,
            "mode_budget_insufficient": False,
            "repair_hours_estimate": 0.0,
            "workers_used": int(workers),
            "num_modes_planned": int(num_modes),
            "notes": ["未提供标签表或未找到 cs_1..cs_6 列，只做了模式检查。"],
        }

    missing = [name for name in REQUIRED_FEATURES if name not in header]
    extra = [name for name in header if name not in REQUIRED_FEATURES]

    summary = {
        "features": str(feature_path),
        "labels": str(label_path) if label_path else None,
        "n_feature_columns": len(header),
        "missing_required_columns": missing,
        "extra_columns": extra,
        "cs_columns": cs_columns,
        "n_rows": order_stats["n_rows"],
        "orders": order_stats["orders"],
        "pairs": pairs,
        "verdict": verdict,
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "diagnose_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        write_report(out / "diagnose_report.md", feature_path, label_path,
                     header, order_stats, pairs, verdict)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose a rotor feature/label table against S1 v4.7")
    parser.add_argument("--features", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("-o", "--out-dir", default="diagnose_out")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--num-modes", type=int, default=48)
    args = parser.parse_args()

    summary = diagnose(args.features, args.dataset, args.out_dir,
                       args.workers, args.num_modes)
    print("rows                : %d" % summary["n_rows"])
    if summary["missing_required_columns"]:
        print("MISSING columns     : %s"
              % ", ".join(summary["missing_required_columns"]))
    if summary["cs_columns"]:
        print("cs columns          : %s" % ", ".join(summary["cs_columns"]))
    if summary["extra_columns"]:
        print("extra columns       : %s"
              % ", ".join(summary["extra_columns"]))
    verdict = summary["verdict"]
    print("highest order       : cs%d (coverage %.1f%%)" % (
        verdict["highest_order_present"],
        100.0 * verdict["top_order_coverage"]))
    print("definition suspect  : %s" % verdict["label_definition_suspect"])
    print("budget insufficient : %s" % verdict["mode_budget_insufficient"])
    for note in verdict["notes"]:
        print("  - %s" % note)
    print("report              : %s"
          % (Path(args.out_dir) / "diagnose_report.md"))


if __name__ == "__main__":
    main()