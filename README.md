# S2：临界转速规格驱动的生成式设计

用户给出材料 + 目标临界转速规格，系统返回一批**合法、互不相同、经 ROSS 复验**的转子设计。
核心是「条件 DDPM 在约束内隐空间里生成，代理排序，ROSS 闸门交付」这条链路。

两个版本并存：

| 版本 | 目录 | 条件 |
| --- | --- | --- |
| v1 | `S2/spec_design/` | 材料 + 一阶（cs1）点值，可带容差与家族约束 |
| v2 | `S2/spec_design_v2/` | cs1/cs2/cs3 任意组合，点值或**区间**（含单边），家族可选 |

- v1 技术报告：`S2/spec_design/生成式设计_报告.md`（全部结论、口径与限度）
- v1 使用说明：`S2/spec_design/README_初版生成式设计_v1.md`（参数、产出、常见用法）
- v2 使用说明与结论：`S2/spec_design_v2/README_v2_多阶条件.md`
- 代理模型说明：`S2/surrogate_v47/代理模型结果_v4.7.md`
- 文献综述（S2 独立成文准备）：`S2/research/生成式转子设计文献综述_20261002.md`

## 0. v2（2026-10-02 新增）：多阶 / 区间 / 家族条件

v1 只做「材料 + 一阶点值」。v2 在同一套反向链、解码器、代理与引导之上把条件扩成：

- **任意阶组合**：cs1 / cs2 / cs3 的 7 种非空组合，±5 % 容差；
- **区间规格**：某一阶可以写成 `[10000,12000]`，也可以写成 `[14400,+]`（「至少」）；
- **家族可选**：盘数与轴承数可以生成前限定（`--fam-allowed 3-3x2-2`），也可以生成后筛选；
- **以上可任意叠加**。

一条命令（默认用区间臂 v2i；先做第 2 节的准备）：

```bash
python S2/spec_design_v2/design_by_target_v2.py --targets "1=[3000,4500],2=[6000,9000]"     --material Steel --tol 0.05 --n-generate 5000 --n-submit 200 --out-tag myrun --threads 8
```

实测（5 000 台候选，取代理前 200 台送 ROSS 复验，均为 ROSS 真值口径）：

| 规格 | 交付 | 秒/台交付 |
| --- | --- | --- |
| 点值 cs1 ±5 % | 192/200 | ~0.6 |
| 区间 ±5 %（窄带 3500–3870） | 154/200 | 0.83 |
| 区间 [2600, 5200] | 194/200 | 0.65 |
| 区间 [1800, 7500] | 198/200 | 0.63 |
| 单边 [3681, +) | 197/200 | 0.62 |
| cs1 + cs2 联立 ±5 % | 162/199 | 1.24 |
| cs1 + cs2 + cs3 联立（2 万台候选） | 131/200 | — |
| 家族生成前限定 3-3x2-2 | 181/200 | 0.62 |
| 家族生成后筛选 | 192/199 | 0.75 |

权重：`outputs/v2i/model.pt`（区间臂）、`outputs/v2m/model.pt`（点值臂）。汇总数字见
`outputs/interval_summary.json`、`outputs/joint_summary.json`、`outputs/family_sweep.json`。

## 1. 环境

Python 3.8（其他版本未测）。安装依赖：

```bash
pip install -r requirements.txt
```

注意两代环境：S2 侧全部在 **Python 3.8 + ROSS 1.6.1** 下冻结；S1 的重标注与修复脚本
是按 **Python 3.12 + ROSS 2.3.0** 写的（`S1/04_versions/v4.7/README_v4.7.md` 有说明），
其中 `combine_relabel_chunks_v4_7.py` 用了 3.9+ 的内置泛型标注，在 3.8 下无法导入。

ROSS（转子动力学求解器，只有跑真值复验的步骤才需要，但 `s1_run_pipeline_v4_7.py`
在导入时就会 `import ross`，所以完整流程建议装上）：

```bash
pip install ross-rotordynamics==1.6.1
```

`numba` 被 ROSS 的求解路径用到；`lightgbm` 只在那条与 MLP 的对照实验里用到，可以后装。

## 2. 第一次使用前：两步准备

仓库里只保留了压缩后的数据集（GitHub 单文件上限 100 MB，原始 CSV 有 106 MB）：

```bash
python -m gzip -d S1/04_versions/v4.7/dataset_v4.7/dataset_v4.7_all.csv.gz
```

解压后约 106 MB。然后建一次特征缓存（约 16 秒）：

```bash
python S2/spec_design/01_proxy_family.py
```

这一步会生成 `S2/spec_design/outputs/cache_features.npz`（约 49 MB，所以没有入库）。

## 3. 快速开始

一条命令出设计（采样 + 生成几何 CSV + 人可读清单）：

```bash
python S2/spec_design/design_by_target.py --target 5000 --material Steel --tol 0.05 \
    --n-generate 5000 --n-submit 200 --guide-lambda 0.05 --out-tag my_run
```

产出 `S2/spec_design/outputs/latent_ddpm/my_run__Steel_5000.csv` 与 `..._sheet.md`。
CSV 里可以直接按 `n_disks` / `n_bearings` 筛家族。

加 ROSS 闸门（逐台求解，只交付实测在带的设计；约 1.2 秒/台，18 进程）：

```bash
python S2/spec_design/design_by_target.py --target 5000 --material Steel --tol 0.01 \
    --n-generate 5000 --n-submit 200 --guide-lambda 0.05 --verify --verify-workers 18
```

跑固定规格表（`outputs/specs.json` 里那六个冻结规格）：

```bash
python S2/spec_design/05_latent_ddpm.py sample --tag armV_ema --out-tag smoke \
    --predict v --use-ema 1 --guidance 1.0 --sample-steps 25 \
    --n-generate 200 --n-submit 20 --seed 42 --threads 8 --specs Steel_high
```

预期在带率落在 0.15 到 0.25 之间。

## 4. 参数里最容易踩的三个

| 参数 | 说明 |
| --- | --- |
| `--sample-steps 25` | **必须显式给**。默认值 0 表示走完整 1000 步链，更慢而且批次更散 |
| `--guide-lambda 0.05` | 采样期用可微代理把设计往目标上推。批次真值 ±1% 在带率 0.027 到 0.300、中位目标距离 15.4% 到 1.7%；采样从 11 秒到 21 秒。0 = 关闭 |
| `--tol` | 容差是主旋钮：±5% 时代理就够，±1% 及以下必须付 ROSS 的钱（`--verify`） |

其余常用项：`--spec-mode lower`（只要求「不低于」）、`--disks 3-3 --bearings 2-4`
（家族区间）、`--fam-policy learned`（用池子先验抽家族）、`--dump-batch N`
（额外落盘 N 台无偏样本，用来量批次质量）。

## 5. 目录
| `S2/spec_design_v2/*.py` | v2 主链路。`design_by_target_v2.py` 单入口；`latent_ddpm_v2.py` 训练/采样（`--layout point|interval`）；`spec_interval.py` 区间解析与判定；`v2_ross_verify.py` 真值复验；`run_v2_eval.py` / `run_v2_interval.py` / `run_v2_joint.py` / `run_v2_family_sweep.py` 四张评估表；`outputs/v2i/`、`outputs/v2m/` 两个生成器权重 |

| 路径 | 内容 |
| --- | --- |
| `S2/spec_design/*.py` | 主链路。`design_by_target.py` 单入口；`05_latent_ddpm.py` 条件 DDPM 训练/采样；`05_ross_verify.py` 真值复验与闸门；`latent_design.py` 34 维隐空间编解码；`spec_opt.py` 合法性/目标函数/可微代理；`06_report.py` 汇总 |
| `S2/spec_design/tools/*.py` | 产出报告数字的分析脚本 |
| `S2/spec_design/outputs/` | 规格表、缓存占位、家族先验、各次运行的漏斗记录 `latent_ddpm/runs.json`、复验真值 |
| `S2/spec_design/outputs/latent_ddpm/armV_ema/model.pt` | 现用生成器权重（512x512x512，v 预测，EMA） |
| `S2/surrogate_v47/` | 代理模型：冻结的 5 seed 多输出 MLP（`outputs/mlp_v47_best/`）；训练 `run_mlp.py`；对照实验 `run_tabular.py`（LightGBM / HistGBDT / ExtraTrees / Ridge，结果在 `outputs/tabular_v47/`）；排序敏感性 `rank_check.py`（结果在 `outputs/rank_check/`）；出图 `plot_results.py` 与 `diagnose_mape.py`，图已生成在 `outputs/figures/` |
| `S2/DDPM_v47/ddpm_v47_common.py`、`02_ddpm.py` | 代理加载器与冻结的 `Denoiser` 网络定义（隐空间版逐字节复用） |
| `S2/neural_surrogate_experiment/run_six_order_mlp_relabeled.py` | 47 维特征工程（代理的输入配方） |
| `S1/04_versions/v4.7/` | v4.7 数据集；采样/体检/修复统一入口 `s1_toolkit.py`（`s1_run_pipeline_v4_7.py` + `s1_diagnose_dataset.py` + `s1_repair_dataset.py`）；约束过滤器 `s1_constraint_filter_v4_7.py`；质量检查 `s1_quality_check_v4_7.py`；重标注 `relabel_old_v4_7.py` 与合并/提升脚本 `combine_relabel_chunks_v4_7.py`、`promote_relabel_to_v4_7.py` |

## 6. 重新训练（可选）

训一个去噪器，200 epoch、8 线程 CPU 约 21 分钟：

```bash
python S2/spec_design/05_latent_ddpm.py train --tag myarm --predict v \
    --hidden 512 512 512 --epochs 200 --batch-size 1024 --ema --threads 8 --seed 42
```

编码臂与引导臂的完整训练/采样命令见报告第 11 节「复现」。

## 7. 仓库里没有、但都能再生成的东西

| 缺什么 | 怎么来 |
| --- | --- |
| `outputs/cache_features.npz`（49 MB） | `python S2/spec_design/01_proxy_family.py`，约 16 秒 |
| `latent_ddpm/*_pool__*.csv`（整批无偏样本） | 采样时加 `--dump-batch N` |
| 对照权重 `armF` / `armD0` / `armX0` / `armN256` | 报告第 11 节的训练命令，各约 21 分钟（报告第 5 节解决七、八的对照臂） |
| S1 的 200k 原始特征分块与重标注中间结果 | 本仓库只带 v4.7 的代码与最终数据集 |
| `outputs/mlp_v47/pred_*.npz`、`outputs/tabular_v47/pred_*.npz`（约 27 MB，逐行预测） | `plot_results.py` 与 `diagnose_mape.py` 需要；`run_mlp.py`、`run_tabular.py` 重跑即得，结果表已在 `summary_*.json` |
| `outputs/tune/*/pred_multi.npz`（13 个调参臂） | `rank_check.py` 复跑需要；本次只带了它的结论 `outputs/rank_check/rank_check.json` |
| 生成器以外的历史交付 CSV | 报告与 `runs.json` 里有全部数字 |

## 8. 已知边界

- ±5% 及以上靠代理排序就够（交付实测在带 0.93 到 0.99）；±1% 与 ±0.5% 必须走
  `--verify` 闸门，成本 1.22 秒/台与 1.95 秒/台。
- ±5% 的整批真值在带率：无引导 0.171，加 `--guide-lambda 0.05` 后 0.870。
- 代理的排序精度在 ±1% 附近约 0.5，这是紧带交付的瓶颈；详见报告第 5、9 节。
- 所有结论基于 v4.7 数据集与 ROSS `num_modes=48`、正阻尼正进动语义。

## 9. 本包实测记录

以下命令都直接在 `S2_github_release/` 根目录跑通（Python 3.8 + torch 2.4.1 CPU + ROSS 1.6.1），
都依赖第 2 节解压后的数据集：

| 命令 | 结果 |
| --- | --- |
| `python S2/spec_design/01_proxy_family.py` | 15 秒建好缓存；`outputs/proxy_family_audit.json` 逐家族 MAPE 1.17% 到 2.83%、R2(log) 0.9956 到 0.9993 |
| `python S2/spec_design/05_latent_ddpm.py sample --tag armV_ema --out-tag smoke --predict v --use-ema 1 --guidance 1.0 --sample-steps 25 --n-generate 200 --n-submit 20 --seed 42 --threads 8 --specs Steel_high` | 整批在带 0.23（第 3 节预期区间内），top20 全中，中位相对误差 1.2% |
| `python S2/spec_design/design_by_target.py --target 5000 --material Steel --tol 0.05 --n-generate 2000 --n-submit 100 --guide-lambda 0.05 --out-tag smoke_guided --threads 8` | 12 秒出 CSV 与清单，整批在带 0.915，中位相对误差 0.02% |
| `... design_by_target.py --target 5000 --material Steel --tol 0.05 --n-generate 3000 --n-submit 20 --guide-lambda 0.05 --verify --verify-workers 8 --out-tag smoke_verify --threads 8` | ROSS 闸门 20/20 求解成功、19 台实测在带，8 进程下 0.86 秒/台 |

`python S2/spec_design/06_report.py` 也能直接跑（约 3 秒），会刷新 `outputs/auto_report.md`
与 `outputs/figures/` 三张图，并把本次运行并入 `runs.json`（已有条目不会丢）。
