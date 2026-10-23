# 复现说明

本目录保存第一批研究（阶段 0–5）的全部分析脚本。所有脚本只读取既有数据与检查点，不重新训练、不执行新的 ROSS 计算。

## 环境

- 解释器：`D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2\neural_surrogate_experiment\.venv_torch\Scripts\python.exe`
- Python 3.8.18
- numpy 1.24.4, pandas 2.0.3, scipy 1.10.1, scikit-learn 1.3.2, matplotlib 3.7.5

## 输入

| 输入 | 路径 |
|---|---|
| 原始数据集 | `S1/04_versions/v4.7/dataset_v4.7/dataset_v4.7_all.csv` |
| 区间 DDPM 检查点 | `S2/spec_design_v2/outputs/v2i/model.pt` |
| 点值 DDPM 检查点 | `S2/spec_design_v2/outputs/v2m/model.pt` |
| CVAE 检查点 | `S2/CVAE_baseline/outputs/cvae_main/model_best.pt` |
| 四流程验证记录 | `paper_v2/revision/revision_after_fairness_audit/` |
| retrieval+jitter | `paper_v2/completer_materials/new_experiments/retrieval_jitter/s42/` |

## 脚本与输出

| 脚本 | 输出目录 | 说明 | 典型耗时 |
|---|---|---|---|
| `build_inventory.py` | `00_inventory/` | 资产盘点、SHA-256、状态与标签统计 | 2–5 min（需哈希大文件） |
| `build_spec_difficulty.py` | `02_specification_difficulty/` | 35 个规格的 p_ref 与支持等级 | <1 min |
| `reaggregate_four_way.py` | `03_existing_candidate_analysis/` | 四流程独立重聚合与配对检验 | <1 min |
| `build_cost_ci.py` | `03_existing_candidate_analysis/` | Wilson 区间与 ROSS 墙钟成本 | <1 min |
| `analyze_diversity.py` | `05_valid_diversity/` | 多样性、新颖度、近重复 | 5–6 min |
| `analyze_screening.py` | `04_generator_screening_analysis/` | 代理漏斗与随机/代理对照 | 1–2 min |
| `analyze_difficulty.py` | `02_specification_difficulty/` | 难度趋势、多阶、容差、带宽 | 1–2 min |
| `analyze_retrieval.py` | `06_retrieval_comparison/` | 检索扰动对照与等样本多样性 | 3–5 min |
| `../08_figures/scripts/make_figures.py` | `08_figures/` | 图 A–G（PDF/SVG/PNG） | 1–2 min |

## 一键运行

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

首次运行或需要重新计算哈希时：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Full
```

## 随机种子

- 采样种子：7、13、42（同一检查点下的重复采样，不是独立训练）。
- 多样性等样本抽样的分析种子：20261021（`analyze_diversity.py`、`analyze_retrieval.py` 内固定）。
- 参考库子采样种子：11（图 E 的 PCA 参考样本）。
- Bootstrap 种子：见 `analyze_screening.py` 内的固定值。

## 已知限制

1. 四个主流程的 5,000 候选池没有完整 ROSS 标签；`analyze_screening.py` 只报告代理在带率，不把它当作真实可行率。
2. ROSS 墙钟时间受线程配置与机器负载影响；`build_cost_ci.py` 报告的 `seconds_wall` 是批量时间，不是单次求解 CPU 时间。
3. `analyze_retrieval.py` 的 centre-ranking 使用了额外的在带标签信息，不能与生成模型作信息对等比较。
4. 图 E 的 PCA 使用 9 个可解释汇总特征，未使用零填充槽位；PC1+PC2 仅解释约 25% 方差，该图用于展示分布差异而非降维重构。

## 输出校验

- `03_existing_candidate_analysis/reaggregation_consistency.csv`：历史汇总与独立重聚合的逐项差值，应为 0。
- `06_retrieval_comparison/retrieval_comparison_summary.json`：应为 959/1,200 与 1,116/1,200。
- `08_figures/figure_manifest.json`：记录每张图的 PDF/SVG/PNG 路径。
