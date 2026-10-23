# -*- coding: utf-8 -*-
"""Write the DDPM three-seed follow-up report (Markdown + Word)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

import cvae_common as C

RES = C.RESULT_DIR
OUT_MD = os.path.join(RES, "DDPM_三种子补充结果_20261019.md")
OUT_DOCX = os.path.join(RES, "DDPM_三种子补充结果_20261019.docx")


def pct(x, digits=1):
    return ("%." + str(digits) + "f%%") % (100.0 * float(x))


def main():
    summary = pd.read_csv(os.path.join(RES, "cvae_ddpm_summary.csv"))
    by_spec = pd.read_csv(os.path.join(RES, "cvae_ddpm_by_spec.csv"))
    paired = pd.read_csv(os.path.join(RES, "cvae_vs_ddpm_paired.csv"))

    cv = summary[summary["method"] == "cvae"].iloc[0]
    dd = summary[summary["method"] == "ddpm"].iloc[0]
    spec_mean = paired.groupby("spec")["difference"].mean().sort_index()
    w18 = wilcoxon(paired["difference"], alternative="two-sided")
    w6 = wilcoxon(spec_mean, alternative="two-sided")
    rng = np.random.default_rng(42)
    vals = spec_mean.to_numpy()
    boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                     for _ in range(20000)])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    per_seed = paired.groupby("seed").agg(cvae_rate=("cvae", "sum"),
                                          ddpm_rate=("ddpm", "sum"),
                                          diff=("difference", "sum")).reset_index()
    per_seed["cvae_n"] = (per_seed["cvae_rate"] * 200).round().astype(int)
    per_seed["ddpm_n"] = (per_seed["ddpm_rate"] * 200).round().astype(int)
    per_seed["cvae_rate"] = per_seed["cvae_n"] / 1200.0
    per_seed["ddpm_rate"] = per_seed["ddpm_n"] / 1200.0
    per_seed["diff"] = per_seed["cvae_rate"] - per_seed["ddpm_rate"]

    order = ["Steel_mid", "Steel_high", "Aluminum_mid", "Aluminum_high",
             "Titanium_mid", "Titanium_high"]
    wide = by_spec.pivot(index="spec", columns="method", values="mean_rate").reset_index()
    wide["spec"] = pd.Categorical(wide["spec"], categories=order, ordered=True)
    wide = wide.sort_values("spec").reset_index(drop=True)
    wide["diff"] = wide["cvae"] - wide["ddpm"]
    nn = by_spec.groupby("method")["mean_nn_distance"].mean()

    md = []
    md.append("# DDPM 三种子补充结果（CVAE 主基线对照）\n")
    md.append("日期：2026-10-19\n")
    md.append("## 1. 实验口径\n")
    md.append("- 六个冻结规格：Steel_mid 3600、Steel_high 16000、Aluminum_mid 5100、Aluminum_high 23000、Titanium_mid 4300、Titanium_high 19000 rpm。")
    md.append("- 每个规格 × 每个种子：生成 5 000 个候选，按同一 `select_common` 选择器取 200 台，逐台 ROSS 复算；分母固定为 200，ROSS 报错或超时仍计入分母。")
    md.append("- 种子：42、7、13；DDPM 与 CVAE 现在都是 6 规格 × 3 种子 × 200 台 = 3 600 次提交。")
    md.append("- CVAE 主基线是**纯先验采样**（无代理梯度引导）；DDPM 主结果是 25 步代理引导采样（guide_lambda = 0.05）。这一点在解释交付率差异时必须说明。")
    md.append("- 所有“验证”均为 ROSS 数值复算，不是物理实验。\n")
    md.append("## 2. 总体结果\n")
    md.append("| 方法 | 规格×种子 | 交付/提交 | 合并交付率 | Wilson 95% CI |")
    md.append("|---|---:|---:|---:|---:|")
    md.append("| CVAE | 6×3 | %d/%d | %s | %s–%s |" % (cv["passed"], cv["n"], pct(cv["pooled_rate"]), pct(cv["wilson_low"]), pct(cv["wilson_high"])))
    md.append("| DDPM | 6×3 | %d/%d | %s | %s–%s |" % (dd["passed"], dd["n"], pct(dd["pooled_rate"]), pct(dd["wilson_low"]), pct(dd["wilson_high"])))
    md.append("")
    md.append("合并差：CVAE − DDPM = **%+.1f 个百分点**（CVAE %s，DDPM %s）。" % (100 * (cv["pooled_rate"] - dd["pooled_rate"]), pct(cv["pooled_rate"]), pct(dd["pooled_rate"])))
    md.append("")
    md.append("## 3. 分规格（3 种子平均）\n")
    md.append("| 规格 | CVAE | DDPM | 差值（CVAE−DDPM） |")
    md.append("|---|---:|---:|---:|")
    for _, row in wide.iterrows():
        md.append("| %s | %s | %s | %+.1f pp |" % (row["spec"], pct(row["cvae"]), pct(row["ddpm"]), 100 * row["diff"]))
    md.append("")
    md.append("## 4. 分种子（6 规格合计）\n")
    md.append("| 种子 | CVAE | DDPM | 差值 |")
    md.append("|---:|---:|---:|---:|")
    for _, row in per_seed.iterrows():
        md.append("| %d | %d/1200（%s） | %d/1200（%s） | %+.1f pp |" % (row["seed"], row["cvae_n"], pct(row["cvae_rate"]), row["ddpm_n"], pct(row["ddpm_rate"]), 100 * row["diff"]))
    md.append("")
    md.append("## 5. 配对检验\n")
    md.append("- 18 个规格×种子配对：平均差 **%+.1f pp**，标准差 %.1f pp，Wilcoxon p = %.4f。" % (100 * paired["difference"].mean(), 100 * paired["difference"].std(ddof=1), w18.pvalue))
    md.append("- 先按 6 个规格各自对 3 个种子取平均，再做 6 对检验：6 个差值全部为正，Wilcoxon p = %.4f（6 对下双侧最小可达到值）。" % w6.pvalue)
    md.append("- 按规格聚类的 bootstrap 95%% CI：%+.1f 至 %+.1f pp。" % (100 * ci_lo, 100 * ci_hi))
    md.append("- 18 对检验把同规格的不同种子视为独立配对；6 对规格检验更保守。两者方向一致，但前者会忽略规格内相关性，正文建议以 6 对规格检验为主、18 对为补充。\n")
    md.append("## 6. 多样性\n")
    md.append("- 短名单中位最近邻距离（标准化）：CVAE %.2f，DDPM %.2f；两者拓扑族覆盖均为 18/18。" % (nn["cvae"], nn["ddpm"]))
    md.append("- 因此 CVAE 的交付率更高，但解集更集中；不能据此说 CVAE 整体更优。交付率与多样性应一起报告。\n")
    md.append("## 7. 结论边界\n")
    md.append("1. 在相同的每规格候选预算、同一筛选器和同一 200 次 ROSS 复算协议下，CVAE 三种子合并交付率高于 DDPM（+3.3 pp；6 对规格检验 p = 0.0312，18 对检验 p = 0.0105）。")
    md.append("2. 该差异不能直接归因于“CVAE 生成器更好”：CVAE 是纯先验采样，DDPM 使用 25 步代理引导；CVAE 的短名单多样性也更低。它说明的是：在本框架下，DDPM 不是唯一可行的生成器，生成器类别不是交付率的唯一决定因素。")
    md.append("3. 若要把两种生成器做成严格同条件比较，下一步应对全部六个规格运行**带代理引导的 CVAE**（目前只有 Steel_mid seed 42 的单规格消融：引导 CVAE 166/200，DDPM 164/200），或对 DDPM 关闭引导做对称消融。")
    md.append("4. 所有结论仅限 ROSS 数值验证与本文的固定代理、候选空间和复算协议，不构成物理实验验证或普遍优越性主张。")
    text = "\n".join(md) + "\n"
    with open(OUT_MD, "w", encoding="utf-8") as handle:
        handle.write(text)

    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.0); section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.2); section.right_margin = Cm(2.2)
    style = doc.styles["Normal"]
    style.font.name = "宋体"; style.font.size = Pt(10.5)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    title = doc.add_paragraph(); title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("DDPM 三种子补充结果（CVAE 主基线对照）")
    run.bold = True; run.font.size = Pt(16); run.font.name = "微软雅黑"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    doc.add_paragraph("日期：2026-10-19").alignment = WD_ALIGN_PARAGRAPH.CENTER

    def heading(t):
        p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(10)
        r = p.add_run(t); r.bold = True; r.font.size = Pt(12)
        r.font.name = "微软雅黑"; r._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")

    def para(t):
        p = doc.add_paragraph(t); p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15

    def table(rows):
        t = doc.add_table(rows=0, cols=len(rows[0])); t.style = "Table Grid"
        for i, row in enumerate(rows):
            cells = t.add_row().cells
            for j, val in enumerate(row):
                cells[j].text = str(val)
                for p in cells[j].paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(9)
                        if i == 0: r.bold = True

    heading("1. 实验口径")
    para("六个冻结规格；每个规格 × 每个种子生成 5 000 个候选，按同一选择器取 200 台，逐台 ROSS 复算；分母固定为 200。种子为 42、7、13，DDPM 与 CVAE 均为 6 规格 × 3 种子 × 200 台 = 3 600 次提交。")
    para("CVAE 主基线为纯先验采样（无代理梯度引导）；DDPM 主结果为 25 步代理引导采样（guide_lambda = 0.05）。所有验证均为 ROSS 数值复算，不是物理实验。")
    heading("2. 总体结果")
    table([["方法", "规格×种子", "交付/提交", "合并交付率", "Wilson 95% CI"],
           ["CVAE", "6×3", "%d/%d" % (cv["passed"], cv["n"]), pct(cv["pooled_rate"]), "%s–%s" % (pct(cv["wilson_low"]), pct(cv["wilson_high"]))],
           ["DDPM", "6×3", "%d/%d" % (dd["passed"], dd["n"]), pct(dd["pooled_rate"]), "%s–%s" % (pct(dd["wilson_low"]), pct(dd["wilson_high"]))]])
    para("合并差：CVAE − DDPM = %+.1f 个百分点。" % (100 * (cv["pooled_rate"] - dd["pooled_rate"])))
    heading("3. 分规格（3 种子平均）")
    rows = [["规格", "CVAE", "DDPM", "差值（CVAE−DDPM）"]]
    for _, row in wide.iterrows():
        rows.append([row["spec"], pct(row["cvae"]), pct(row["ddpm"]), "%+.1f pp" % (100 * row["diff"])])
    table(rows)
    heading("4. 分种子（6 规格合计）")
    rows = [["种子", "CVAE", "DDPM", "差值"]]
    for _, row in per_seed.iterrows():
        rows.append(["%d" % int(row["seed"]), "%d/1200（%s）" % (row["cvae_n"], pct(row["cvae_rate"])), "%d/1200（%s）" % (row["ddpm_n"], pct(row["ddpm_rate"])), "%+.1f pp" % (100 * row["diff"])])
    table(rows)
    heading("5. 配对检验")
    para("18 个规格×种子配对：平均差 %+.1f pp，标准差 %.1f pp，Wilcoxon p = %.4f。" % (100 * paired["difference"].mean(), 100 * paired["difference"].std(ddof=1), w18.pvalue))
    para("6 个规格各对 3 个种子取平均后检验：6 个差值全部为正，Wilcoxon p = %.4f（6 对下双侧最小可达到值）。按规格聚类的 bootstrap 95%% CI：%+.1f 至 %+.1f pp。" % (w6.pvalue, 100 * ci_lo, 100 * ci_hi))
    para("18 对检验忽略规格内相关性，6 对规格检验更保守；正文建议以 6 对规格检验为主、18 对作为补充。")
    heading("6. 多样性与结论边界")
    para("短名单中位最近邻距离：CVAE %.2f，DDPM %.2f；两者拓扑族覆盖均为 18/18。CVAE 交付率更高但解集更集中，不能据此说 CVAE 整体更优。" % (nn["cvae"], nn["ddpm"]))
    para("在相同候选预算、同一筛选器和同一 ROSS 协议下，CVAE 三种子合并交付率高于 DDPM（+3.3 pp；6 对规格检验 p = 0.0312）。但 CVAE 为纯先验采样、DDPM 为引导采样，且 CVAE 多样性更低，因此该结果只能说明本框架不依赖 DDPM 这一生成器类别，不能表述为 CVAE 普遍优于 DDPM。")
    para("若要严格同条件比较生成器，下一步应对全部六个规格运行带代理引导的 CVAE，或对 DDPM 做关闭引导的对称消融。所有结论仅限 ROSS 数值验证与本文固定代理、候选空间和复算协议。")
    doc.save(OUT_DOCX)
    print(OUT_MD)
    print(OUT_DOCX)


if __name__ == "__main__":
    main()
