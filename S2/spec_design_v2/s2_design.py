# -*- coding: utf-8 -*-
"""s2_design.py -- 一条命令，输入规格，输出交付清单。

用法示例
--------
  python s2_design.py --spec "cs1=3000-4500" --material Steel
  python s2_design.py --spec "cs1=3000-4500, cs2=6000-9000" --material Steel
  python s2_design.py --cs1 3681 --material Steel --tol 5%
  python s2_design.py --spec "cs1>=3681" --material Steel --disks 3-3 --bearings 2-2
  python s2_design.py --spec "cs1=3500-3870" --material Steel --filter-disks 2-4
  python s2_design.py --spec "cs1=3000-4500" --material Steel --proxy-only   # 快速预览，不做 ROSS

规格写法（多项用逗号隔开，阶数 1..6）
  cs1=3681            点值
  cs1=3000-4500       区间
  cs1>=3681           单边（不低于）
  cs1=3681+           单边（同上）
  1=[3000,4500]       原生写法，也认

默认做 ROSS 逐台复验 —— 那是唯一诚实的交付口径；只想要快，加 --proxy-only。

产出
----
  outputs/delivered_<tag>/
      deliverable.csv           交付清单（已通过 ROSS 复验的设计 + 几何参数）
      shortlist_with_truth.csv  短名单全部候选的真值（含未通过的）
      report.json               完整报告（各阶误差、在带率、排序可信度）
      summary.md                人可读清单
      summary.json              同上，机器可读
"""
import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
V1_DIR = os.path.join(os.path.dirname(HERE), "spec_design")
for path in (V1_DIR, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import spec_common as S          # noqa: E402
import spec_interval as SI       # noqa: E402

PY = sys.executable
INF_CAP = SI.INF_CAP


# --------------------------------------------------------------------- 规格解析
def _token_to_native(token):
    """'cs1=3000-4500' / 'cs2>=6000' / '1=[3000,4500]' -> '1=[3000,4500]'."""
    m = re.match(r"^\s*(?:cs)?\s*([1-6])\s*(.*)$", token, re.I)
    if not m:
        raise SystemExit("[spec] 这段看不懂：%r" % token)
    order, rest = m.group(1), m.group(2).strip()

    if rest.startswith(">=") or rest.startswith(">") or rest.startswith("\u2265"):
        value = rest.lstrip(">=\u2265").strip()
        return "%s=[%s,+]" % (order, value)
    if rest.startswith("<=") or rest.startswith("<") or rest.startswith("\u2264"):
        raise SystemExit(
            "[spec] 暂不支持只给上界（cs%s %s）。请改写成区间，例如 cs%s=1000-5000"
            % (order, rest, order))

    rest = rest.lstrip("=").strip()
    if not rest:
        raise SystemExit("[spec] 缺少目标值：%r" % token)
    if rest[0] in "[(":
        return "%s=%s" % (order, rest)
    if rest.endswith("+"):
        return "%s=[%s,+]" % (order, rest[:-1].strip())
    if rest.endswith("-") or rest.endswith("~"):
        body = rest[:-1].strip()
        try:
            float(body)
        except ValueError:
            pass
        else:
            return "%s=[%s,+]" % (order, body)

    parts = [p for p in re.split(r"\s*(?:-|~|\.\.)\s*", rest) if p != ""]
    if len(parts) == 2:
        try:
            return "%s=[%g,%g]" % (order, float(parts[0]), float(parts[1]))
        except ValueError:
            pass
    try:
        float(rest)
    except ValueError:
        raise SystemExit("[spec] 目标值看不懂：%r" % rest)
    return "%s=%s" % (order, rest)


def normalize_spec(text):
    tokens = [_token_to_native(t) for t in SI.split_specs(text)]
    if not tokens:
        raise SystemExit("[spec] 规格是空的")
    return ",".join(tokens)


def parse_tol(text, default):
    if text is None:
        return default
    raw = str(text).strip()
    percent = raw.endswith("%")
    if percent:
        raw = raw[:-1]
    value = float(raw)
    if percent or value > 1.0:
        value = value / 100.0
    return value


def slug(text):
    out = re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_").lower()
    return out or "query"


# --------------------------------------------------------------------- 小工具
def read_json(path):
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_text(path, text):
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def fmt_value(value, digits=0):
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number >= INF_CAP:
        return "+inf"
    return ("%%.%df" % digits) % number


def band_label(lo, hi, digits=0):
    if hi >= INF_CAP:
        return ">= %s" % fmt_value(lo, digits)
    if hi <= lo:
        return "%s" % fmt_value(lo, digits)
    return "%s - %s" % (fmt_value(lo, digits), fmt_value(hi, digits))


def unique_tag(base):
    tag, index = base, 1
    while os.path.exists(os.path.join(OUT_DIR, "%s.csv" % tag)) or \
            os.path.isdir(os.path.join(OUT_DIR, "delivered_%s" % tag)):
        index += 1
        tag = "%s_%d" % (base, index)
    return tag


# --------------------------------------------------------------------- 主流程
def main():
    parser = argparse.ArgumentParser(
        description="一条命令：输入临界转速规格，输出经 ROSS 复验的交付清单。")
    parser.add_argument("--spec", default="",
                        help='例如 "cs1=3000-4500" 或 "cs1=3000-4500, cs2=6000-9000"')
    for order in (1, 2, 3, 4, 5, 6):
        parser.add_argument("--cs%d" % order, default="",
                            help="第 %d 阶的目标，写法同 --spec 里的单条" % order)
    parser.add_argument("--material", required=True, choices=list(S.MATERIAL_ORDER))
    parser.add_argument("--tol", default=None,
                        help="判定容差，写 5%% 或 0.05 都行；默认区间 0、点值 5%%")
    parser.add_argument("--disks", default="", help="生成前限定盘数，如 3-3 或 2-4")
    parser.add_argument("--bearings", default="", help="生成前限定轴承数，如 2-4")
    parser.add_argument("--filter-disks", default="", help="生成后筛选盘数，如 2-4")
    parser.add_argument("--filter-bearings", default="", help="生成后筛选轴承数")
    parser.add_argument("--n-generate", type=int, default=5000, help="采样多少台候选")
    parser.add_argument("--n-submit", type=int, default=200, help="送 ROSS 复验多少台")
    parser.add_argument("--tag", default="v2i", help="用哪个生成器：v2i（区间臂）或 v2m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--workers", type=int, default=18, help="ROSS 并行进程数")
    parser.add_argument("--verify-timeout", type=float, default=180.0)
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--guide-lambda", type=float, default=0.05)
    parser.add_argument("--proxy-only", action="store_true",
                        help="只出代理排序的预览清单，不做 ROSS 复验（快，但不保证真实在带）")
    parser.add_argument("--out-tag", default="")
    args = parser.parse_args()

    pieces = [args.spec] if args.spec else []
    for order in (1, 2, 3, 4, 5, 6):
        value = getattr(args, "cs%d" % order)
        if value:
            pieces.append("cs%d=%s" % (order, value))
    if not pieces:
        parser.error("至少要给一个目标：--spec 或 --cs1/--cs2/--cs3")

    targets = normalize_spec(",".join(pieces))
    specs = SI.parse_specs(targets)
    orders = sorted(specs)
    interval = any(hi > lo for lo, hi in specs.values())
    tol = parse_tol(args.tol, 0.0 if interval else 0.05)

    tag = args.out_tag or unique_tag("%s_%s" % (slug(targets), slug(args.material)))
    os.makedirs(OUT_DIR, exist_ok=True)

    family = ""
    if args.disks or args.bearings:
        family = "%sx%s" % (args.disks, args.bearings)
    family_filter = ""
    if args.filter_disks or args.filter_bearings:
        family_filter = "%sx%s" % (args.filter_disks, args.filter_bearings)

    print("=" * 72)
    print("[规格] " + "，".join("cs%d %s" % (o, band_label(*specs[o])) for o in orders)
          + "   材料 %s   容差 +-%g%%" % (args.material, 100.0 * tol))
    print("[家族] 生成前 %s | 生成后筛选 %s"
          % (family or "不限制", family_filter or "不限制"))
    print("[模式] " + ("只出代理预览（不做 ROSS 复验）" if args.proxy_only
                       else "ROSS 逐台复验（交付的都是实测在带）"))
    print("=" * 72)

    t0 = time.time()
    cmd = [PY, os.path.join(HERE, "latent_ddpm_v2.py"), "sample",
           "--tag", args.tag, "--out-tag", tag, "--targets", targets,
           "--material", args.material, "--tol", "%g" % tol, "--use-ema", "1",
           "--guidance", "%g" % args.guidance, "--guide-lambda", "%g" % args.guide_lambda,
           "--sample-steps", str(args.sample_steps),
           "--n-generate", str(args.n_generate), "--n-submit", str(args.n_submit),
           "--threads", str(args.threads), "--seed", str(args.seed)]
    if family:
        cmd += ["--fam-allowed", family]
    if family_filter:
        cmd += ["--fam-filter", family_filter]
    subprocess.run(cmd, cwd=HERE, check=True)
    batch_csv = os.path.join(OUT_DIR, "%s.csv" % tag)
    gen_seconds = time.time() - t0

    verify_json = None
    if not args.proxy_only:
        t1 = time.time()
        subprocess.run([PY, os.path.join(HERE, "v2_ross_verify.py"),
                        "--csv", batch_csv, "--targets", targets,
                        "--tols", "%g" % tol, "--top-n", str(args.n_submit),
                        "--select", "top", "--workers", str(args.workers),
                        "--timeout", str(args.verify_timeout),
                        "--name", tag, "--seed", str(args.seed)],
                       cwd=HERE, check=True)
        verify_json = os.path.join(OUT_DIR, "verify_%s.json" % tag)
        verify_seconds = time.time() - t1

    # ---------------- 收集成一份交付文件夹 ----------------
    folder = os.path.join(OUT_DIR, "delivered_%s" % tag)
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    entry = None
    if verify_json:
        report = read_json(verify_json)
        entry = report["runs"][0] if report.get("runs") else None
        deliverable_src = ((entry or {}).get("deliverable")
                           or os.path.join(OUT_DIR, "verify_%s__passed.csv" % tag))
        shortlist_src = (entry or {}).get("csv")
        if shortlist_src and os.path.exists(shortlist_src):
            shutil.copy2(shortlist_src, os.path.join(folder, "shortlist_with_truth.csv"))
    else:
        deliverable_src = batch_csv

    if os.path.exists(deliverable_src):
        shutil.copy2(deliverable_src, os.path.join(folder, "deliverable.csv"))
    if verify_json and os.path.exists(verify_json):
        shutil.copy2(verify_json, os.path.join(folder, "report.json"))

    summary = build_summary(tag, folder, specs, orders, args, tol, family,
                            family_filter, batch_csv, gen_seconds, entry,
                            os.path.exists(os.path.join(folder, "deliverable.csv")))
    write_text(os.path.join(folder, "summary.md"), summary["markdown"])
    write_text(os.path.join(folder, "summary.json"),
               json.dumps(summary["json"], ensure_ascii=False, indent=2))

    print("=" * 72)
    print(summary["console"])
    print("-" * 72)
    print("[产出] %s" % folder)
    print("       deliverable.csv           交付清单")
    print("       summary.md                人可读清单")
    if verify_json:
        print("       shortlist_with_truth.csv  短名单真值")
        print("       report.json               完整报告")
    print("=" * 72)


def build_summary(tag, folder, specs, orders, args, tol, family, family_filter,
                  batch_csv, gen_seconds, entry, has_deliverable):
    import pandas as pd

    lines = []
    info = {
        "tag": tag,
        "spec": {str(o): [specs[o][0], min(specs[o][1], INF_CAP)] for o in orders},
        "spec_text": "，".join("cs%d %s" % (o, band_label(*specs[o])) for o in orders),
        "material": args.material,
        "tolerance": tol,
        "family_generate": family or None,
        "family_filter": family_filter or None,
        "n_generate": args.n_generate,
        "n_submit": args.n_submit,
        "proxy_only": bool(args.proxy_only),
        "generate_seconds": round(gen_seconds, 1),
    }

    lines.append("# 交付清单 · %s" % tag)
    lines.append("")
    lines.append("- 规格：%s" % info["spec_text"])
    lines.append("- 材料：%s　判定容差：+-%g%%" % (args.material, 100.0 * tol))
    lines.append("- 家族：生成前 %s ｜ 生成后筛选 %s"
                 % (family or "不限制", family_filter or "不限制"))
    lines.append("- 采样 %d 台，用时 %.1f 秒" % (args.n_generate, gen_seconds))

    console = []
    if entry is None:
        n = 0
        if has_deliverable:
            frame = pd.read_csv(os.path.join(folder, "deliverable.csv"))
            n = len(frame)
        info["mode"] = "proxy_only"
        info["delivered"] = n
        lines.append("- **模式：只出代理预览，未做 ROSS 复验** —— 下面这些是代理认为最好的，"
                     "真实是否在带没有验证")
        lines.append("- 候选清单：%d 台" % n)
        console.append("[结果] 代理预览 %d 台（未做 ROSS 复验）" % n)
    else:
        truth = entry.get("truth", {})
        proxy = entry.get("proxy", {})
        gate = entry.get("gate_tol%.0f" % round(100.0 * tol)) or {}
        if not gate:
            gate = entry.get("gate_tol0") or {}
        n_solved = entry.get("n_solved")
        info["mode"] = "ross_verified"
        info["delivered"] = entry.get("deliverable_n")
        info["n_solved"] = n_solved
        info["true_in_band"] = truth.get("joint_in_band_tol0")
        info["proxy_in_band"] = proxy.get("joint_in_band_tol0")
        info["rho"] = entry.get("spearman_joint_proxy_vs_true")
        info["seconds_per_delivered"] = gate.get("seconds_per_passed")
        info["ross_seconds"] = round(entry.get("seconds"), 1)
        lines.append("- ROSS 复验 %s 台，用时 %.1f 秒" % (n_solved, entry.get("seconds", 0.0)))
        lines.append("- **交付 %s 台**（ROSS 真值落在规格内）" % entry.get("deliverable_n"))
        lines.append("- 代理口径：代理自报短名单 %.1f%% 在带 → 真值 %.1f%%"
                     % (100.0 * (proxy.get("joint_in_band_tol0") or 0.0),
                        100.0 * (truth.get("joint_in_band_tol0") or 0.0)))
        lines.append("- 代理排序可信度 rho = %s"
                     % (("%.3f" % entry["spearman_joint_proxy_vs_true"])
                        if entry.get("spearman_joint_proxy_vs_true") is not None else "-"))
        lines.append("- 每台交付的仿真成本：%.2f 秒" % (gate.get("seconds_per_passed") or 0.0))
        console.append("[结果] 交付 %s 台（ROSS 实测在带）｜ 代理自报 %.0f%% → 真值 %.0f%% ｜ rho %.2f ｜ %.2f 秒/台"
                       % (entry.get("deliverable_n"),
                          100.0 * (proxy.get("joint_in_band_tol0") or 0.0),
                          100.0 * (truth.get("joint_in_band_tol0") or 0.0),
                          entry.get("spearman_joint_proxy_vs_true") or 0.0,
                          gate.get("seconds_per_passed") or 0.0))
        lines.append("")
        lines.append("## 各阶误差（ROSS 真值，单位 %）")
        lines.append("")
        lines.append("| 阶 | 规格 | 中位 APE | 平均 APE | P90 APE | 在带率 | 代理在带率 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for order in orders:
            per_true = truth.get("order%d" % order, {})
            per_proxy = proxy.get("order%d" % order, {})
            lines.append("| cs%d | %s | %.2f | %.2f | %.2f | %.3f | %.3f |"
                         % (order, band_label(*specs[order]),
                            per_true.get("median_ape_pct", float("nan")),
                            per_true.get("mean_ape_pct", float("nan")),
                            per_true.get("p90_ape_pct", float("nan")),
                            per_true.get("inside_rate", float("nan")),
                            per_proxy.get("inside_rate", float("nan"))))

    path = os.path.join(folder, "deliverable.csv")
    if has_deliverable:
        frame = pd.read_csv(path)
        lines.append("")
        lines.append("## 交付清单（前 20 台）")
        lines.append("")
        columns = ["rank", "material", "n_disks", "n_bearings"]
        for order in orders:
            for suffix in ("_ross", "_pred"):
                name = "cs%d%s" % (order, suffix)
                if name in frame.columns:
                    columns.append(name)
                    break
            name = "cs%d_relerr_truth_pct" % order
            if name in frame.columns:
                columns.append(name)
        columns = [c for c in columns if c in frame.columns]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for _, row in frame.head(20).iterrows():
            cells = []
            for column in columns:
                value = row[column]
                if isinstance(value, float):
                    cells.append("%.2f" % value if column.endswith("_pct") else "%g" % value)
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("完整清单（%d 行，含全部几何参数）：`deliverable.csv`" % len(frame))
        if not entry:
            lines.append("")
            lines.append("> 注意：这是代理排序结果，**没有经过 ROSS 复验**。"
                         "要去掉 --proxy-only 重跑才会得到实测交付。")

    return {"markdown": "\n".join(lines) + "\n", "json": info,
            "console": "\n".join(console)}


if __name__ == "__main__":
    main()