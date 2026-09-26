"""
S1 Module: Data Quality Check
Validates dataset quality: no truncation/anomalies, ROSS failures,
anomalous critical speeds, missing labels.

Supports both the old schema (shaft_length_m / critical_speed_1_rpm)
and the v4.x schema (L_m / cs_1_rpm).
"""

import csv, json, sys
from typing import Any, Dict, List, Optional
import numpy as np
from pathlib import Path


def load_dataset(csv_path: str) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _row_value(row, keys, default=None):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def quality_check(
    rows: List[Dict[str, Any]],
    report_path: str = None,
    min_cs_rpm: float = 50.0,
    max_cs_rpm: float = 950000.0,
    modal_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run quality checks on the dataset.
    Returns a report dict with pass/fail for each check.
    """
    report = {
        "n_total": len(rows),
        "checks": {},
        "warnings": [],
        "errors": [],
    }

    if not rows:
        report["checks"]["has_data"] = {"status": "FAIL", "detail": "Empty dataset"}
        return report
    report["checks"]["has_data"] = {"status": "PASS", "detail": f"{len(rows)} rows"}

    # 1. Check for ROSS calculation failures
    n_error = sum(1 for r in rows if r.get("status") == "error")
    error_rate = n_error / len(rows)
    report["checks"]["ross_failures"] = {
        "status": "PASS" if n_error == 0 else ("FAIL" if error_rate >= 0.2 else "WARN"),
        "detail": f"{n_error}/{len(rows)} errors ({error_rate:.1%})",
    }
    if n_error > 0:
        message = (
            f"High ROSS failure rate: {error_rate:.1%}"
            if error_rate >= 0.2
            else f"ROSS errors detected: {n_error}/{len(rows)}"
        )
        report["warnings"].append(message)

    # 2. Check for missing labels
    success_rows = [
        r for r in rows
        if r.get("status") in ("success", "ok") or r.get("status") in (None, "")
    ]
    missing_cs = 0
    for r in success_rows:
        values = [
            _row_value(r, (f"cs_{j}_rpm", f"critical_speed_{j}_rpm"))
            for j in range(1, 4)
        ]
        if any(v in (None, "") for v in values):
            missing_cs += 1
    report["checks"]["missing_labels"] = {
        "status": "PASS" if missing_cs == 0 else "FAIL",
        "detail": f"{missing_cs} rows missing critical speed labels",
    }
    if missing_cs > 0:
        report["errors"].append(f"{missing_cs} rows missing critical speed labels")

    # 3. cs1 boundary check required by the S1 sampling document
    out_of_range = []
    for r in success_rows:
        cs1 = _row_value(r, ("cs_1_rpm", "critical_speed_1_rpm"))
        if cs1 is None:
            continue
        try:
            cs1 = float(cs1)
        except (TypeError, ValueError):
            continue
        if cs1 < min_cs_rpm or cs1 > max_cs_rpm:
            out_of_range.append(cs1)
    report["checks"]["cs1_boundary"] = {
        "status": "PASS" if not out_of_range else "FAIL",
        "detail": (
            f"{len(out_of_range)} rows outside "
            f"[{min_cs_rpm:.0f}, {max_cs_rpm:.0f}] RPM"
        ),
    }
    if out_of_range:
        report["errors"].append(
            f"{len(out_of_range)} rows outside "
            f"[{min_cs_rpm:.0f}, {max_cs_rpm:.0f}] RPM"
        )

    # 4. Check parameter ranges for anomalies
    if success_rows:
        shaft_lengths = []
        for r in success_rows:
            value = _row_value(r, ("L_m", "shaft_length_m"))
            if value is not None:
                try:
                    shaft_lengths.append(float(value))
                except (TypeError, ValueError):
                    pass
        cspeeds = []
        for r in success_rows:
            value = _row_value(r, ("cs_1_rpm", "critical_speed_1_rpm"))
            if value is not None:
                try:
                    cspeeds.append(float(value))
                except (TypeError, ValueError):
                    pass

        if shaft_lengths:
            zero_len = sum(1 for l in shaft_lengths if l < 0.01)
            report["checks"]["shaft_length_anomalies"] = {
                "status": "PASS" if zero_len == 0 else "FAIL",
                "detail": f"{zero_len} rows with near-zero shaft length",
            }

            at_boundary = sum(1 for l in shaft_lengths if abs(l - 0.3) < 0.001)
            report["checks"]["truncation"] = {
                "status": "WARN" if at_boundary > len(shaft_lengths) * 0.1 else "PASS",
                "detail": f"{at_boundary} rows at lower boundary (0.3m)",
            }

        if cspeeds:
            log_cspeeds = np.log10(np.maximum(np.asarray(cspeeds, dtype=float), 1e-12))
            q1, q3 = np.percentile(log_cspeeds, [25, 75])
            iqr = q3 - q1
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            n_anomalous = int(np.sum((log_cspeeds < lower) | (log_cspeeds > upper)))
            report["checks"]["anomalous_critical_speeds"] = {
                "status": "WARN" if n_anomalous > len(cspeeds) * 0.02 else "PASS",
                "detail": (
                    f"{n_anomalous} anomalous in log10 space "
                    f"(log-IQR={iqr:.3f}, bounds=[{lower:.3f}, {upper:.3f}])"
                ),
            }
            if n_anomalous > 0:
                report["warnings"].append(f"{n_anomalous} anomalous critical speeds detected")

    # 5. Raw-modal audit: directions are preserved, not silently discarded.
    if modal_rows:
        direction_counts = {}
        for modal in modal_rows:
            direction = modal.get("whirl_direction") or "None"
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        report["checks"]["modal_directions"] = {
            "status": "PASS",
            "detail": ", ".join(
                f"{key}={direction_counts[key]}" for key in sorted(direction_counts)
            ),
        }
        n_partial_modal = sum(
            1 for r in rows
            if r.get("status") == "partial"
            or (
                r.get("status") in ("success", "ok", None, "")
                and int(float(r.get("forward_mode_count") or 0)) < 3
            )
        )
        report["checks"]["incomplete_forward_modes"] = {
            "status": "WARN" if n_partial_modal > 0 else "PASS",
            "detail": f"{n_partial_modal} rows with fewer than 3 forward modes",
        }
        if n_partial_modal > 0:
            report["warnings"].append(
                f"{n_partial_modal} rows have fewer than 3 forward-modal labels"
            )

    # 6. Post-sim status accounting (partial/rejected_cs1 only in audit files)
    n_partial = sum(1 for r in rows if r.get("status") == "partial")
    n_rejected_cs1 = sum(1 for r in rows if r.get("status") == "rejected_cs1")
    if n_partial or n_rejected_cs1:
        report["checks"]["post_sim_status"] = {
            "status": "WARN",
            "detail": f"{n_partial} partial, {n_rejected_cs1} rejected_cs1 in audit",
        }
        report["warnings"].append(f"{n_partial} partial, {n_rejected_cs1} rejected_cs1")

    # Summary
    n_fail = sum(1 for c in report["checks"].values() if c["status"] == "FAIL")
    n_warn = sum(1 for c in report["checks"].values() if c["status"] == "WARN")
    report["summary"] = f"{n_fail} FAIL, {n_warn} WARN, {len(report['checks']) - n_fail - n_warn} PASS"
    report["is_clean"] = (n_fail == 0 and n_warn == 0)

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Quality report saved to {report_path}")

    return report


def print_report(report: dict):
    """Pretty-print the quality check report."""
    print(f"\n{'='*60}")
    print(f"  Data Quality Report ({report.get('n_total', 0)} rows)")
    print(f"{'='*60}")
    for check, data in report.get("checks", {}).items():
        icon = {"PASS": "+", "WARN": "!", "FAIL": "x"}
        status_icon = icon.get(data["status"], "?")
        print(f"  [{status_icon}] {check}: {data['detail']}")
    for w in report.get("warnings", []):
        print(f"  [!] Warning: {w}")
    for e in report.get("errors", []):
        print(f"  [x] Error: {e}")
    print(f"  {'-'*56}")
    print(f"  Summary: {report.get('summary', 'N/A')}")
    print(f"  Clean: {'YES' if report.get('is_clean') else 'NO - REVIEW REQUIRED'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python s1_quality_check_v4_7.py <dataset.csv> [--save report.json]")
        sys.exit(1)

    csv_path = sys.argv[1]
    report_path = None
    if len(sys.argv) > 2 and sys.argv[2] == "--save":
        report_path = sys.argv[3] if len(sys.argv) > 3 else "quality_report.json"

    rows = load_dataset(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")
    report = quality_check(rows, report_path)
    print_report(report)
