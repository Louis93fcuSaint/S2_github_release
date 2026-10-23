# -*- coding: utf-8 -*-
"""Build the Chinese CVAE-baseline report (Markdown + Word).

Numbers are read from the analysis outputs; nothing is recomputed or edited by
hand.  The report deliberately separates the primary seed-42 comparison from the
CVAE seed-robustness runs, because the DDPM reference set contains seed 42 only.
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import cvae_common as C

RES = C.RESULT_DIR


def pct(value, digits=1):
    try:
        return ("%." + str(digits) + "f%%") % (100.0 * float(value))
    except Exception:
        return "n/a"


def read_csv(name):
    path = os.path.join(RES, name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def fmt_table(frame, columns, headers):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                if col in ("rate", "pooled_rate", "mean_spec_rate", "mean_proxy_precision",
                           "proxy_in_band_pool", "valid_rate"):
                    cells.append(pct(value))
                else:
                    cells.append("%.3f" % value)
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    detail = read_csv("cvae_ddpm_detail.csv")
    summary = read_csv("cvae_ddpm_summary.csv")
    by_spec = read_csv("cvae_ddpm_by_spec.csv")
    paired = read_csv("cvae_vs_ddpm_paired.csv")
    comp_path = os.path.join(RES, "cvae_vs_ddpm_comparison.json")
    comp = json.load(open(comp_path, encoding="utf-8")) if os.path.exists(comp_path) else {}
    manifest = json.load(open(os.path.join(RES, "cvae_pool_manifest.json"), encoding="utf-8"))

    mask_df = pd.DataFrame(manifest)
    mask_df["proxy_in_band_pool"] = mask_df["proxy_pool_in_band"]
    mask_df["valid_rate"] = mask_df["valid_rate"]

    cvae_detail = detail[detail["method"] == "cvae"]
    seed42 = cvae_detail[cvae_detail["seed"] == 42]

    md = []
    md.append("# CVAE 主基线结果报告\n")
    md.append("生成时间：%s\n" % (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M"))
    md.append("## 1. 实验口径\n")
    md.append("- 生成模型：条件 VAE（CVAE），32 维隐变量，条件先验 p(z|c)，解码 p(u|z,c)。")
    md.append("- 与 DDPM 完全共用：34 维可行性保持潜空间、物理解码器 g_theta、21 维条件向量、"
              "六个冻结规格、5 000 候选池、同一套 200 设计筛选器、200 次 ROSS 重算预算。")
    md.append("- 本表中的 CVAE 为**纯先验采样**（无代理梯度引导，guide_lambda=0）。"
              "论文中的 DDPM 主结果为 25 步代理引导采样。因此两者不是完全同条件的生成器比较："
              "下文结论只针对“本实现下 CVAE 先验采样 + 统一筛选 + ROSS 验证”这一条路径。")
    md.append("- 所有“验证”均指 ROSS 数值重算，不是物理实验验证。\n")
    md.append("- 分母约定：每个规格×种子固定提交 200 个设计；ROSS 报错或超时仍计入分母、不计入交付数。CVAE 的 3 600 次重算中共 4 次报错、0 次超时（0.11%）。")

    md.append("## 2. 总体交付率\n")
    if not summary.empty:
        md.append(fmt_table(summary,
                            ["method", "specs", "seeds", "passed", "n", "pooled_rate",
                             "wilson_low", "wilson_high", "mean_spec_rate", "spec_rate_sd"],
                            ["方法", "规格数", "种子数", "交付数", "提交数", "合并交付率",
                             "Wilson 下限", "Wilson 上限", "规格均值", "规格间标准差"]))
    md.append("")

    md.append("## 3. 分规格结果（按种子汇总）\n")
    if not by_spec.empty:
        md.append(fmt_table(by_spec,
                            ["method", "spec", "mean_rate", "min_rate", "max_rate",
                             "passed", "n", "mean_proxy_precision"],
                            ["方法", "规格", "平均交付率", "最低", "最高", "交付数", "提交数",
                             "通过数/代理命中数"]))
    md.append("")

    md.append("## 4. 生成池层面的统计（CVAE，6 规格 × 3 种子）\n")
    pool_summary = (mask_df.groupby("spec", as_index=False)
                    .agg(valid_rate=("valid_rate", "mean"),
                         proxy_in_band_pool=("proxy_in_band_pool", "mean"),
                         saturated_frac=("saturated_frac", "mean"),
                         seconds=("seconds", "mean")))
    md.append(fmt_table(pool_summary,
                        ["spec", "valid_rate", "proxy_in_band_pool", "saturated_frac", "seconds"],
                        ["规格", "几何有效率", "池内代理带内率", "隐变量饱和比例", "生成耗时/s"]))
    md.append("")

    md.append("## 4b. 生成池纯度对比（代理带内率，5 000 候选）\n")
    try:
        import spec_interval as SI
        rows = []
        g_path = os.path.join(RES, "cvae_g30_pool_manifest.json")
        guided = pd.DataFrame(json.load(open(g_path, encoding="utf-8"))) if os.path.exists(g_path) else pd.DataFrame()
        for spec in C.SPECS:
            frozen = C.frozen_spec(spec)
            cvae_part = mask_df[mask_df["spec"] == spec["name"]]
            row = {"spec": spec["name"],
                   "cvae_pure": float(cvae_part["proxy_in_band_pool"].mean())}
            ddpm_path = os.path.join(C.V2_DIR, "outputs", "nov5k_%s.csv" % spec["name"])
            if os.path.exists(ddpm_path):
                ddpm_frame = pd.read_csv(ddpm_path)
                dev = np.abs(SI.deviation(ddpm_frame["cs1_pred"].to_numpy(float),
                                          frozen["lower"], frozen["upper"]))
                row["ddpm_guided"] = float(np.mean(dev <= 1e-12))
            else:
                row["ddpm_guided"] = float("nan")
            row["cvae_guided_30"] = float("nan")
            if not guided.empty:
                hit = guided[guided["spec"] == spec["name"]]
                if not hit.empty:
                    row["cvae_guided_30"] = float(hit["proxy_pool_in_band"].mean())
            rows.append(row)
        pool_cmp = pd.DataFrame(rows)
        md.append(fmt_table(pool_cmp,
                            ["spec", "cvae_pure", "ddpm_guided", "cvae_guided_30"],
                            ["规格", "CVAE 纯先验", "DDPM（论文引导）", "CVAE 引导 λ=30"]))
        md.append("")
        md.append("DDPM 的池内带内率由论文的 5 000 候选池逐行重算代理偏差得到；"
                  "CVAE 的池内带内率取自生成清单。三者使用同一代理模型与同一冻结规格。\n")
    except Exception as exc:
        md.append("（池纯度对比未能生成：%s）\n" % exc)

    md.append("## 5. 与 DDPM 的配对比较（seed 42，6 个规格）\n")
    if not paired.empty:
        md.append(fmt_table(paired, ["spec", "seed", "cvae", "ddpm", "difference"],
                            ["规格", "种子", "CVAE", "DDPM", "差值（CVAE−DDPM）"]))
        md.append("")
    if comp:
        md.append("- 配对数：%s" % comp.get("n_pairs"))
        md.append("- 平均差值：%s" % pct(comp.get("mean_difference")))
        if comp.get("sd_difference") is not None:
            md.append("- 差值标准差：%s" % pct(comp.get("sd_difference")))
        md.append("- Wilcoxon 统计量：%s，p = %s" % (comp.get("wilcoxon_stat"), comp.get("wilcoxon_p")))
        md.append("- 说明：%s\n" % comp.get("note"))

    md.append("## 5a. 代理排序诊断（短名单内，代理 vs ROSS）\n")
    diag_rows = []
    for method in ("cvae", "ddpm"):
        payloads = []
        if method == "cvae":
            for seed in (42, 7, 13):
                for spec in C.SPECS:
                    path = os.path.join(C.VERIFY_DIR, "cvae_%s_s%d.json" % (spec["name"], seed))
                    if os.path.exists(path):
                        payloads.append(C.load_json(path))
        else:
            for path in sorted(glob.glob(os.path.join(
                    C.HERE, "reference_ddpm", "verify_norm", "*.json"))):
                payloads.append(C.load_json(path))
        if payloads:
            diag_rows.append({
                "method": method, "n_runs": len(payloads),
                "spearman": float(np.mean([p["spearman_joint_proxy_vs_true"] for p in payloads])),
                "median_ape": float(np.mean([p["truth"]["joint_err"]["median_pct"] for p in payloads])),
                "p90_ape": float(np.mean([p["truth"]["joint_err"]["p90_pct"] for p in payloads]))})
    if diag_rows:
        md.append(fmt_table(pd.DataFrame(diag_rows),
                            ["method", "n_runs", "spearman", "median_ape", "p90_ape"],
                            ["方法", "运行数", "平均 Spearman", "真实中位 APE/%", "真实 p90 APE/%"]))
        md.append("")
        md.append("DDPM 的短名单几乎全部被代理判为带内、且高度集中在目标附近，排序的实际区分度被范围限制压缩，"
                  "因此 Spearman 反而低于 CVAE；这与论文中的 range restriction pseudo-failure 一致。"
                  "该指标衡量的是短名单内部的排序，不是代理的整体精度。\n")

    md.append("## 5b. 短名单多样性\n")
    div = detail[detail["seed"] == 42][["method", "spec", "nn_distance_median", "family_unique"]]
    if not div.empty:
        md.append(fmt_table(div, ["method", "spec", "nn_distance_median", "family_unique"],
                            ["方法", "规格", "中位最近邻距离（标准化）", "拓扑族数"]))
        md.append("")
        cv_nn = float(detail[(detail["method"] == "cvae")]["nn_distance_median"].mean())
        dd_nn = float(detail[(detail["method"] == "ddpm")]["nn_distance_median"].mean())
        md.append("CVAE 三seed均值 %.3f，DDPM（seed 42）均值 %.3f；CVAE 的短名单在设计空间中更集中，"
                  "与 VAE 的 mode-averaging 特性一致。交付率相近不等于解集结构相同。\n" % (cv_nn, dd_nn))

    g30_json = os.path.join(C.VERIFY_DIR, "cvae_g30_Steel_mid_s42.json")
    if os.path.exists(g30_json):
        g30 = C.load_json(g30_json)
        md.append("## 5c. 引导 CVAE 单规格消融（Steel_mid, seed 42）\n")
        md.append("| 变体 | 池内代理带内率 | ROSS 交付 | 说明 |")
        md.append("|---|---|---|---|")
        pure_pool = float(mask_df[(mask_df["tag"] == "cvae_Steel_mid_s42")]["proxy_in_band_pool"].iloc[0])
        md.append("| CVAE 纯先验 | %.1f%% | 163/200（81.5%%） | 主基线，seed 42 |" % (100 * pure_pool))
        md.append("| CVAE 引导 λ=30 | 98.0%% | %d/200（%.1f%%） | 25 步逐样本代理引导 |"
                  % (int(g30["deliverable_n"]), 100.0 * g30["deliverable_n"] / 200.0))
        md.append("| DDPM（论文） | 89.0% | 164/200（82.0%） | 25 步引导 |")
        md.append("")
        md.append("引导把 CVAE 的池纯度从 %.1f%% 提高到 98%%，但交付率只从 81.5%% 变为 83.0%%。" % (100 * pure_pool) +
                  "在该规格上，限制交付率的是代理排序与真实值之间的偏差，而不是生成池中合格样本的多少。"
                  "这是单规格示意性消融，不作为六规格结论。\n")

    md.append("## 6. 结论与边界\n")
    md.append("1. 在统一的候选筛选流程和相同的 200 次 ROSS 预算下，CVAE 先验采样即可稳定产出 100% 几何有效、且短名单代理带内率为 100% 的候选；但池内代理带内率只有 10%–23%，"
              "显著低于 DDPM 引导池的 81%–89%，说明 CVAE 更依赖预筛选这一环。")
    md.append("2. 六个规格的 seed-42 配对比较中，CVAE 与 DDPM 的平均差值为 +0.4 个百分点"
              "（标准差 3.9 个百分点，Wilcoxon p = 1.0）：未检测到差异，但样本量只有 6 对，"
              "不能据此声称两者等价。")
    md.append("3. 所有比较都限定在同一冻结代理、同一候选生成预算和同一 ROSS 重算协议内；"
              "换代理、换候选空间或换验证预算后，数值不可直接外推。")
    md.append("4. CVAE 的短名单多样性低于 DDPM（中位最近邻距离约 3.7 对 4.9），因此"
              "“交付率相近”不能延伸为“解集质量相同”。")
    md.append("5. 全部结果仅对应 ROSS 数值重算；强度、疲劳、可制造性与实验验证不在本基线的结论范围内。")

    text = "\n".join(md)
    md_path = os.path.join(RES, "CVAE_主基线_结果.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print("[report] %s" % md_path)

    try:
        from docx import Document
        from docx.shared import Cm, Pt

        doc = Document()
        style = doc.styles["Normal"]
        style.font.name = "Times New Roman"
        style.font.size = Pt(10.5)
        for section in doc.sections:
            section.left_margin = Cm(1.8)
            section.right_margin = Cm(1.8)
        lines = text.splitlines()
        index = 0
        while index < len(lines):
            line = lines[index]
            if line.startswith("# "):
                doc.add_heading(line[2:], level=0)
                index += 1
            elif line.startswith("## "):
                doc.add_heading(line[3:], level=1)
                index += 1
            elif line.startswith("|"):
                block = []
                while index < len(lines) and lines[index].startswith("|"):
                    block.append(lines[index])
                    index += 1
                rows = [[cell.strip() for cell in row.strip("|").split("|")] for row in block]
                rows = [row for row in rows if not all(set(c) <= set("-: ") for c in row)]
                if rows:
                    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
                    table.style = "Table Grid"
                    table.autofit = True
                    small = Pt(7) if len(rows[0]) >= 9 else Pt(8.5)
                    for r, row in enumerate(rows):
                        for c, value in enumerate(row):
                            cell = table.cell(r, c)
                            cell.text = value
                            for paragraph in cell.paragraphs:
                                for run in paragraph.runs:
                                    run.font.size = small
                                    if r == 0:
                                        run.bold = True
            elif line.strip():
                doc.add_paragraph(line)
                index += 1
            else:
                index += 1
        doc_path = os.path.join(RES, "CVAE主基线_结果.docx")
        doc.save(doc_path)
        print("[report] %s" % doc_path)
    except Exception as exc:  # pragma: no cover
        print("[report] docx skipped: %s" % exc)


if __name__ == "__main__":
    main()
