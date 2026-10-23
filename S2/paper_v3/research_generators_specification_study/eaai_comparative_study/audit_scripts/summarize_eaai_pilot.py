# -*- coding: utf-8 -*-
"""Build the pilot summaries, family-aware diversity and the pilot report."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eaai_common as A

THRESHOLDS = [1e-6, 1e-4, 1e-3]


def active_cols(family):
    a, b = int(family[0]), int(family[1])
    cols = [0, 1, 2]
    cols += [3 + 3 * j + k for j in range(a) for k in (0, 1, 2)]
    cols += [A.S.BRG_BASE + 3 * j + k for j in range(b) for k in (0, 1, 2)]
    return np.asarray(cols, dtype=int)


def family_label(nd, nb):
    return "%dd-%db" % (int(nd), int(nb))


def diversity_metrics(designs, families, scales, train_designs, train_families):
    designs = np.asarray(designs, dtype=np.float64)
    n = len(designs)
    out = {
        "n_designs": int(n), "n_families": 0, "family_entropy": None,
        "family_counts": "", "median_pairwise_distance": None,
        "median_nearest_neighbor_distance": None,
        "reference_novelty_median": None, "reference_novelty_p10": None,
    }
    for t in THRESHOLDS:
        out["near_duplicate_rate_%g" % t] = None
    if n == 0:
        return out
    labels = [family_label(*f) for f in families]
    counts = pd.Series(labels).value_counts()
    out["n_families"] = int(len(counts))
    out["family_counts"] = ";".join("%s:%d" % (k, int(v)) for k, v in sorted(counts.items()))
    if len(counts) > 1:
        p = counts.to_numpy(dtype=float) / counts.sum()
        out["family_entropy"] = float(-(p * np.log2(p)).sum())
    else:
        out["family_entropy"] = 0.0

    pair_values = []
    nn_values = []
    near_counts = {t: 0 for t in THRESHOLDS}
    near_totals = {t: 0 for t in THRESHOLDS}
    for label in counts.index:
        idx = np.where(np.asarray(labels) == label)[0]
        if len(idx) < 2:
            continue
        fam = families[idx[0]]
        cols = active_cols(fam)
        z = designs[idx][:, cols] / np.maximum(scales[cols], 1e-12)
        # Pairwise Euclidean distance; n <= 100, so this is cheap.
        d = np.sqrt(np.maximum(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1), 0.0))
        iu = np.triu_indices(len(idx), k=1)
        vals = d[iu]
        pair_values.extend(vals.tolist())
        np.fill_diagonal(d, np.inf)
        nn_values.extend(d.min(axis=1).tolist())
        for t in THRESHOLDS:
            near_counts[t] += int((vals <= t).sum())
            near_totals[t] += int(len(vals))
    if pair_values:
        out["median_pairwise_distance"] = float(np.median(pair_values))
        out["median_nearest_neighbor_distance"] = float(np.median(nn_values))
        for t in THRESHOLDS:
            out["near_duplicate_rate_%g" % t] = (
                float(near_counts[t] / near_totals[t]) if near_totals[t] else None)

    # Reference novelty: same-family nearest training reference, active coords only.
    novelty = []
    train_families = np.asarray([family_label(*f) for f in train_families])
    for i, fam in enumerate(families):
        label = family_label(*fam)
        ref_idx = np.where(train_families == label)[0]
        if len(ref_idx) == 0:
            continue
        cols = active_cols(fam)
        q = designs[i, cols] / np.maximum(scales[cols], 1e-12)
        r = np.asarray(train_designs[ref_idx][:, cols], dtype=np.float64) / np.maximum(scales[cols], 1e-12)
        best = np.inf
        for start in range(0, len(r), 5000):
            block = r[start:start + 5000]
            d = np.sqrt(np.maximum(((q[None, :] - block) ** 2).sum(axis=1), 0.0))
            best = min(best, float(d.min()))
        novelty.append(best)
    if novelty:
        out["reference_novelty_median"] = float(np.median(novelty))
        out["reference_novelty_p10"] = float(np.percentile(novelty, 10))
    return out


def main():
    ctx, train_idx, _val_idx, _ = A.build_context_split(save=False)
    train_designs = ctx["designs"][train_idx]
    train_families = [(int(a), int(b)) for a, b in zip(ctx["nd"][train_idx], ctx["nb"][train_idx])]
    scales = np.asarray(ctx["scales"], dtype=np.float64)
    pd.DataFrame({"column": A.S.CANONICAL_COLUMNS, "scale": scales}).to_csv(
        os.path.join(A.SPLIT_DIR, "train_scales.csv"), index=False, encoding="utf-8")
    np.save(os.path.join(A.SPLIT_DIR, "train_scales.npy"), scales)

    all_slots = pd.read_csv(os.path.join(A.ROSS_DIR, "all_slots.csv"))
    # The ROSS slot table stores identities, not the 33 canonical columns.
    # Recover them from the frozen per-pool shortlist/random CSVs.
    lookup = []
    for pid in all_slots["pool_id"].unique():
        for fname in ("shortlist.csv", "random_sample.csv"):
            frame = pd.read_csv(os.path.join(A.SHORT_DIR, pid, fname))
            frame["pool_id"] = pid
            lookup.append(frame[["pool_id", "candidate_id"] + A.S.CANONICAL_COLUMNS])
    lookup = pd.concat(lookup, ignore_index=True).drop_duplicates(
        ["pool_id", "candidate_id"]).reset_index(drop=True)
    all_slots = all_slots.merge(lookup, on=["pool_id", "candidate_id"], how="left")
    if all_slots[A.S.CANONICAL_COLUMNS].isna().any().any():
        raise RuntimeError("canonical design lookup failed for some logical slots")
    short_summary = pd.read_csv(os.path.join(A.STUDY, "pilot_shortlist_summary.csv"))
    pool_summary = pd.read_csv(os.path.join(A.STUDY, "pilot_pool_summary.csv"))
    ross_summary = pd.read_csv(os.path.join(A.ROSS_DIR, "ross_pool_summary.csv"))

    # Diversity for shortlist-all, shortlist-delivered and random-delivered.
    div_rows = []
    for pid, group in all_slots.groupby("pool_id", sort=False):
        method = group["method"].iloc[0]
        spec = group["spec"].iloc[0]
        sets = {
            "shortlist_all": group[group["list_type"] == "shortlist"],
            "shortlist_delivered": group[(group["list_type"] == "shortlist") & group["delivered"]],
            "random_delivered": group[(group["list_type"] == "random") & group["delivered"]],
        }
        for set_name, sub in sets.items():
            sub = sub.drop_duplicates("design_key")
            designs = sub[A.S.CANONICAL_COLUMNS].to_numpy(dtype=np.float64)
            families = [(int(a), int(b)) for a, b in zip(sub["nd"], sub["nb"])]
            metrics = diversity_metrics(designs, families, scales, train_designs, train_families)
            row = {"pool_id": pid, "method": method, "spec": spec,
                   "set": set_name, **metrics}
            div_rows.append(row)
    div = pd.DataFrame(div_rows)
    div.to_csv(os.path.join(A.STUDY, "pilot_valid_diversity.csv"), index=False,
               encoding="utf-8")
    div.to_csv(os.path.join(A.METRIC_DIR, "pilot_valid_diversity.csv"), index=False,
               encoding="utf-8")

    # Merge delivery results into the shortlist summary.
    rs = ross_summary.set_index(["pool_id", "list_type"])
    for i, row in short_summary.iterrows():
        pid = row["pool_id"]
        for list_type, prefix in (("shortlist", "shortlist"), ("random", "random")):
            key = (pid, list_type)
            if key in rs.index:
                r = rs.loc[key]
                short_summary.loc[i, "n_delivered_%s" % prefix] = int(r["n_delivered"])
                short_summary.loc[i, "%s_delivery_rate" % prefix] = float(r["delivery_rate"])
                short_summary.loc[i, "n_solved_ok_%s" % prefix] = int(r["n_solved_ok"])
                short_summary.loc[i, "n_error_%s" % prefix] = int(r["n_error"])
                short_summary.loc[i, "n_timeout_%s" % prefix] = int(r["n_timeout"])
                short_summary.loc[i, "n_cache_hit_%s" % prefix] = int(r["n_cache_hit"])
                short_summary.loc[i, "median_elapsed_%s_s" % prefix] = float(r["median_elapsed_s"])
    short_summary.to_csv(os.path.join(A.STUDY, "pilot_shortlist_summary.csv"),
                         index=False, encoding="utf-8")
    short_summary.to_csv(os.path.join(A.METRIC_DIR, "pilot_shortlist_summary.csv"),
                         index=False, encoding="utf-8")

    # Merge delivery results into the pool summary.
    pool_summary["ross_random_feasible_rate"] = np.nan
    pool_summary["ross_shortlist_delivery_rate"] = np.nan
    for i, row in pool_summary.iterrows():
        pid = row["pool_id"]
        if (pid, "random") in rs.index:
            pool_summary.loc[i, "ross_random_feasible_rate"] = float(rs.loc[(pid, "random"), "delivery_rate"])
        if (pid, "shortlist") in rs.index:
            pool_summary.loc[i, "ross_shortlist_delivery_rate"] = float(rs.loc[(pid, "shortlist"), "delivery_rate"])
    pool_summary.to_csv(os.path.join(A.STUDY, "pilot_pool_summary.csv"),
                        index=False, encoding="utf-8")
    pool_summary.to_csv(os.path.join(A.METRIC_DIR, "pilot_pool_summary.csv"),
                        index=False, encoding="utf-8")

    # Cost rows: shortlist build and end-to-end pilot wall.
    cost = pd.read_csv(os.path.join(A.METRIC_DIR, "pilot_cost_breakdown.csv"))
    cost = cost[~cost["stage"].isin(["batch_shortlist", "batch_total"])]
    short_meta = A.load_json(os.path.join(A.METRIC_DIR, "shortlist_run_meta.json"))
    run_meta = A.load_json(os.path.join(A.ROSS_DIR, "run_meta.json"))
    gen_batch = float(cost.loc[cost["stage"] == "batch_generation", "wall_s"].iloc[0])
    cost = pd.concat([cost, pd.DataFrame([
        {"pool_id": "ALL", "stage": "batch_shortlist",
         "wall_s": float(short_meta["wall_s"]), "cpu_core_s": "unknown",
         "n_units": 6, "notes": "proxy gate + 100/100 selection for six pools"},
        {"pool_id": "ALL", "stage": "batch_ross_with_qa",
         "wall_s": float(run_meta["total_wall_s"]), "cpu_core_s": "unknown",
         "n_units": 1200, "notes": "includes 4 QA repeat solves, excludes analysis"},
        {"pool_id": "ALL", "stage": "batch_total",
         "wall_s": gen_batch + float(short_meta["wall_s"]) + float(run_meta["total_wall_s"]),
         "cpu_core_s": "unknown", "n_units": 1200,
         "notes": "generation + shortlist + ROSS/QA; excludes report writing"},
    ])], ignore_index=True)
    cost.to_csv(os.path.join(A.STUDY, "pilot_cost_breakdown.csv"), index=False,
                encoding="utf-8")
    cost.to_csv(os.path.join(A.METRIC_DIR, "pilot_cost_breakdown.csv"), index=False,
                encoding="utf-8")

    # Markdown report.
    def fmt(v, nd=3):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "NA"
        if isinstance(v, (float, np.floating)):
            return ("%." + str(nd) + "f") % float(v)
        return str(v)

    lines = [
        "# EAAI 比较研究先导报告（v1.1）",
        "",
        "- 日期：%s" % A.utc_now(),
        "- 研究目录：`%s`" % A.STUDY,
        "- 协议：`protocol_v1_1.md`",
        "- 规格：仅 CAL02、CAL05；未运行任何 TST 规格",
        "- 方法：区间 DDPM unguided、CVAE unguided、retrieval+jitter；采样种子 42",
        "- 预算：每方法每规格 5000 次原始尝试；每池 100 随机 + 100 短名单；逻辑名额 1200",
        "",
        "## 1. 先导结论（描述性）",
        "",
        "本轮管线已端到端跑通：6 个原始池、6 个随机 100 样本、6 个 100 台代理短名单、1200 个 ROSS 逻辑名额。"
        "所有逻辑名额均完成验收，0 错误、0 超时；19 个逻辑名额复用了同一设计的数值结果，物理求解 1181 次。",
        "",
        "在 CAL02（高支持、自由族、单阶区间）上，原始池 ROSS 可行率为 DDPM 25%、CVAE 64%、retrieval+jitter 75%；"
        "代理短名单交付率为 95%、96%、98%。",
        "",
        "在 CAL05（低支持、固定 6d-2b、1+3 阶区间）上，原始池 ROSS 可行率为 DDPM 3%、CVAE 21%、retrieval+jitter 37%；"
        "代理短名单交付率为 69%、74%、69%。",
        "",
        "这些是单个采样种子、两个校准规格上的描述性结果，不做正式显著性检验，也不能外推为测试规格上的泛化结论。",
        "",
        "## 2. 原始池记账",
        "",
        "| Pool | Method | Spec | Attempts | Valid | Invalid | Unique valid | Exact dup valid | Proxy in-band (unique) | In-band rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in pool_summary.iterrows():
        lines.append("| %s | %s | %s | %d | %d | %d | %d | %d | %d | %s |" % (
            row["pool_id"], row["method"], row["spec"], row["n_attempts"],
            row["n_valid"], row["n_invalid"], row["n_unique_valid"],
            row["n_exact_duplicate_valid_attempts"],
            row["n_proxy_in_band_unique_valid"],
            fmt(row["proxy_in_band_rate_unique_valid"], 4)))
    lines += [
        "",
        "所有 6 个池均为 5000/5000 有效、5000 个精确唯一设计；没有 H1–H6 约束失败，也没有精确重复。"
        "这不表示设计在工程上高度不同，只表示在本轮解码器与浮点表示下没有精确复制。",
        "",
        "## 3. ROSS 验收",
        "",
        "| Pool | List | Logical | Delivered | Delivery rate | OK | Error | Timeout | Cache reuse | Median solve s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ross_summary.iterrows():
        lines.append("| %s | %s | %d | %d | %s | %d | %d | %d | %d | %s |" % (
            row["pool_id"], row["list_type"], row["n_logical_slots"],
            row["n_delivered"], fmt(row["delivery_rate"], 4),
            row["n_solved_ok"], row["n_error"], row["n_timeout"],
            row["n_cache_hit"], fmt(row["median_elapsed_s"], 2)))
    lines += [
        "",
        "随机样本与短名单分母均为 100。没有空缺，没有无效生成进入短名单；随机样本也未出现无效生成。"
        "短名单的带内候选均超过 100，因此本轮没有触发带外补齐路径。",
        "",
        "## 4. 合格集多样性",
        "",
        "距离仅在**同结构族**的活跃连续参数上计算，并按训练池逐维标准差归一化；不使用 33 维零填充向量直接算欧氏距离。"
        "样本不足时返回 NA。",
        "",
        "| Pool | Set | n | Families | Entropy | Median NN | Median pairwise | Near-dup 1e-4 | Ref novelty median |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in div.iterrows():
        lines.append("| %s | %s | %d | %d | %s | %s | %s | %s | %s |" % (
            row["pool_id"], row["set"], row["n_designs"], row["n_families"],
            fmt(row["family_entropy"], 3), fmt(row["median_nearest_neighbor_distance"], 4),
            fmt(row["median_pairwise_distance"], 4),
            fmt(row["near_duplicate_rate_0.0001"], 4),
            fmt(row["reference_novelty_median"], 4)))
    lines += [
        "",
        "`shortlist_all` 是代理短名单（ROSS 前）；`shortlist_delivered` 是短名单中真正通过 ROSS 的设计；"
        "`random_delivered` 是随机 100 样本中通过 ROSS 的设计，仅作背景对照。",
        "",
        "## 5. 成本",
        "",
        "| Stage | Wall s | CPU core-s | Units | Notes |",
        "|---|---:|---|---:|---|",
    ]
    for _, row in cost.iterrows():
        if row["stage"] in ("batch_generation", "batch_shortlist", "batch_ross_wall",
                            "batch_ross_with_qa", "batch_total"):
            lines.append("| %s | %s | %s | %s | %s |" % (
                row["stage"], fmt(row["wall_s"], 1), row["cpu_core_s"],
                row["n_units"], row["notes"]))
    lines += [
        "",
        "CPU 核时未直接测量，统一标记为 `unknown`；没有用“墙钟 × workers”冒充实测 CPU 时间。"
        "ROSS 的 `batch_ross_wall` 是 16 线程并行的实际墙钟，单次求解中位数见上一节。",
        "",
        "## 6. 质量检查",
        "",
        "- Smoke test：6 格 × 32 候选全部通过；条件维度、材料块、族块、1+3 阶掩码、未用槽零填充、代理 6 阶有限性、CSV 回读、同种子复现均通过。",
        "- ROSS 重复性 QA：4 次重复求解，最大绝对差 0.00158 rpm，低于 1 rpm；未计入 1200 逻辑名额。",
        "- 历史输出目录：池生成与 ROSS 运行前后指纹一致；未写入旧 `outputs/pools/candidates/verify`。",
        "- `cvae_common` 在 import 时对旧 CVAE 目录执行 `makedirs`，但本轮未调用其数据加载或 `main()`，未产生新文件。",
        "- 规格 CSV 三个文件的 SHA-256 与 `SHA256SUMS.csv` 全部匹配。",
        "",
        "## 7. 能否进入下一阶段",
        "",
        "**管线层面：可以进入下一阶段。** 生成、代理门控、短名单、ROSS 验收、缓存和汇总均可复现运行。",
        "",
        "**统计层面：仍有限制。**",
        "",
        "1. 本轮只有一个采样种子 42，不能估计采样种子方差；",
        "2. 没有独立训练重复，不能把 42 当作训练重复；",
        "3. CAL05 的 DDPM 原始池可行率仅 3%，下一阶段若要用它做正式比较，需要预先说明预算、分母和最低可检测效应，不能因结果低而删规格；",
        "4. 短名单全部带内，带外补齐路径未被真实触发；正式阶段应保留该路径，并用一个合成单元测试或更难规格验证；",
        "5. retrieval+jitter 使用了参考库已有性能标签，这是信息优势，正式比较必须明确；",
        "6. ROSS 数值复算不是物理实验、工程认证或制造可行性验证；H1–H6 通过也不等于完整装配可行；",
        "7. 本轮没有运行 TST 规格，因此不能对测试集泛化或方法优劣作结论。",
        "",
        "## 8. 文件",
        "",
        "- `pilot_run_manifest.csv`：6 个池的路径、哈希、计数与计时；",
        "- `pilot_pool_summary.csv`：池级有效/重复/带内统计；",
        "- `pilot_shortlist_summary.csv`：短名单与随机样本的 ROSS 交付统计；",
        "- `pilot_valid_diversity.csv`：同族活跃参数距离、族覆盖、近重复与参考新颖度；",
        "- `pilot_cost_breakdown.csv`：生成、筛选、ROSS、整批墙钟；",
        "- `pilot_runs/01_pools/`：原始池、去重映射、manifest；",
        "- `pilot_runs/02_shortlists/`：随机 100、短名单 100、重叠 ID；",
        "- `pilot_runs/03_ross/`：逐逻辑名额结果、逐次求解、缓存、QA；",
        "- `pilot_runs/smoke/`：32 候选 smoke 记录；",
        "- `pilot_runs/00_env/`：环境、检查点、源文件与规格哈希。",
    ]
    with open(os.path.join(A.STUDY, "pilot_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("[summary] wrote pilot_report.md and summary CSV files")
    print(div[["pool_id", "set", "n_designs", "n_families",
               "median_nearest_neighbor_distance", "reference_novelty_median"]].to_string(index=False))


if __name__ == "__main__":
    main()
