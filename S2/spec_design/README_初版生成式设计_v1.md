# 初版生成式设计 v1：怎么运行

日期 2026-09-25。代码 `S2/spec_design/`，数据 v4.7（199811 台 success）。
技术内容与全部实测数字见同目录 `生成式设计_报告.md`；本文件只说「哪些文件要，怎么跑」。

---

## 1. 一次查询是什么

输入：材料 + 目标一阶临界转速 + 容差，可选盘数与轴承数的约束。

    --fam-allowed 3x2                        只要 3 盘 2 轴承
    --fam-allowed 2x2,3x2,4x2,2x3,3x3,4x3    2 到 4 盘、2 到 4 轴承
    不写则家族由系统决定

输出：每个规格一份 200 台的候选 CSV，含代理预测的 cs1、家族，以及 ROSS 复验结果。

## 2. 跑起来

环境：`S2/neural_surrogate_experiment/.venv_torch/Scripts/python.exe`（Python 3.8.18，torch 2.4.1 CPU）。
下面的命令在仓库根目录执行。

```powershell
$PY = "S2\neural_surrogate_experiment\.venv_torch\Scripts\python.exe"

# 第一步 特征缓存（已经有 outputs/cache_features.npz 就跳过，重算约 20 分钟）
& $PY S2\spec_design\01_proxy_family.py

# 第二步 规格表（六个点目标正负 5%）
& $PY S2\spec_design\02b_point_protocol.py

# 第三步 生成（每规格 5000 台，代理排序取前 200）
& $PY S2\spec_design\05_latent_ddpm.py sample --tag armV_ema --out-tag v1 `
    --predict v --use-ema 1 --guidance 1.0 --sample-steps 25 `
    --n-generate 5000 --n-submit 200 --seed 42 --threads 8

# 第四步 ROSS 真值复验（整批随机抽 20 台，另加 12 台硬件校准）
& $PY S2\spec_design\05_ross_verify.py --methods v1 --top-n 20 --select rand `
    --workers 18 --timeout 180 --n-base 15 --calibrate 12 --threads 8

# 第五步 汇总表与图（写到 outputs/auto_report.md）
& $PY S2\spec_design\06_report.py
```

实测耗时：第一步约 20 分钟（只需一次），第二步与第六步几秒，第三步六规格 24 秒，第四步 101 秒。
重任务不要并行跑两份，ROSS 用 18 进程、采样用 8 线程。

### 2.1 任意目标转速

不写规格表也能直接指定目标一阶临界转速：`--target`（rpm）、`--material`、`--tol`（相对带宽，0.05 就是正负 5 个点）。临时规格写进 `outputs/specs_adhoc.json`，冻结协议 `outputs/specs.json` 不动。`--target-upper` 可覆盖带宽上界，`--spec-name` 可给这次目标起名。

```powershell
& $PY S2\spec_design\05_latent_ddpm.py sample --tag armV_ema --out-tag myrun `
    --use-ema 1 --guidance 1.0 --sample-steps 25 `
    --n-generate 5000 --n-submit 200 --seed 42 --threads 8 `
    --target 5000 --material Steel --tol 0.05

& $PY S2\spec_design\05_ross_verify.py --methods myrun --top-n 20 --select rand `
    --workers 18 --timeout 180 --n-base 15 --calibrate 12 --threads 8
```

实测 Steel 5000 rpm 正负 5%：采样 12 秒，前 200 在带 1.00；ROSS 真值抽 20 台在带 18 台（0.90），MAPE 1.66%，中位 APE 0.86%。目标要落在该材料有数据支撑的区间（Steel 中位 3642、最大 73143 rpm），越靠上尾越难；超出量程就是外推。单侧「大于等于」需求见 2.3；容差能收到多紧见 2.5，紧到代理排不出名次时怎么交付见 2.6。

### 2.2 限定盘数与轴承数

`--fam-allowed` 接受家族区间，盘数和轴承数两条轴都支持范围，用 `x` 或 `/` 分隔，空的一端表示不设限：

- `--fam-allowed 3x2`：盘 3、轴承 2，钉死唯一家族（等价于 `--fam-policy pinned --fam-allowed 3x2`）
- `--fam-allowed 3-3/2-4`：盘 3、轴承 2 到 4，三个家族
- `--fam-allowed 2-4x2-3`：盘 2 到 4、轴承 2 到 3，六个家族
- `--fam-allowed 3x`：盘 3、轴承不限；`--fam-allowed x2`：轴承 2、盘不限

家族是在采样时就限定，不是事后过滤，所以 5000 台的预算全部花在允许的家族上。`--fam-policy uniform` 在允许集合里均抽，`--fam-policy learned` 按池子证据加权。日志和 `funnel_*.json` 会多两个字段：`family_band`（实际用到的家族）与 `support_pool_band_family`（池子里同时落在该目标带和该家族区间的真实设计数，为 0 说明交付的是外推设计）。

实测 Steel 5000 rpm 正负 5%，同一批参数只换家族约束，每臂随机抽 150 台真值：

| 家族约束 | 池子真实支撑 | 代理在带（交付批） | 真值在带 | MAPE | 中位 APE |
| --- | --- | --- | --- | --- | --- |
| 不限，18 个家族 | 2558 | 1.000 | 0.960 | 1.24% | 0.81% |
| 盘 3、轴承 2 到 4 | 462 | 1.000 | 0.953 | 1.42% | 0.94% |
| 盘 3、轴承 2 | 170 | 1.000 | 0.987 | 0.86% | 0.63% |

两两 Fisher 精确检验 p = 0.17 到 1.00：**家族约束不改变在带率**。它是用户约束机制（把盘数和轴承数钉在你想要的范围内），不是精度杠杆。此前用 20 台抽样得到的「限定家族更好」是噪声，150 台下被否掉了；MAPE 上钉死 3x2 看起来更好，但那三条臂没有落盘逐台真值，做不了配对检验，只能算方向性观察。

想让 CSV 多给几台候选就调大 `--n-submit`，例如 1000 台。

### 2.3 单侧要求（一阶不低于某值）

工程上更常见的说法是「一阶临界转速不低于工作转速乘 1.2」。用 `--spec-mode lower`，此时 `--tol` 的含义从「带半宽」变成「瞄准余量」：

```powershell
# 一阶不低于 5000 rpm，瞄准 5100（余量 2%）
& $PY 05_latent_ddpm.py sample --tag armV_ema --out-tag lb5000_m2 `
    --use-ema 1 --guidance 1.0 --sample-steps 25 `
    --n-generate 5000 --n-submit 200 --seed 42 --threads 8 `
    --target 5000 --material Steel --spec-mode lower --tol 0.02
```

瞄准余量不是可选项：瞄准点压在要求线上时，代理自己 1% 的误差会把接近一半的设计判到线下方，真值在带只有 0.733；瞄准 5100 之后是 0.960。余量取代理中位 APE 的两倍左右比较稳。

### 2.4 单入口与设计清单

不想记那串调参值时用单入口，它同时给出 CSV 和人可读的清单：

```powershell
& $PY design_by_target.py --target 5000 --material Steel --tol 0.05
& $PY design_by_target.py --target 5000 --material Steel --disks 3-3 --bearings 2-4
& $PY design_by_target.py --target 5000 --material Steel --spec-mode lower --tol 0.02
```

清单写到 `outputs/latent_ddpm/<out-tag>__<规格>_sheet.md`：抬头是目标、家族约束、池子支撑与代理误差，表格每台一行，长度 mm、轴承刚度 MN/m、阻尼 N s/m，未用槽位显示 `-`，最后两列是代理预测的 cs1 与相对目标差。`--sheet-top 0` 可列全部 200 台。CSV 里也补了 `cs1_pred` 与 `rel_err_pct` 两列，不用再自己算。

紧带（容差小于代理误差）时加 `--verify`：短名单逐台送 ROSS，只交付实测在带的设计，清单和 CSV 都换成已复验版本。

```powershell
& $PY design_by_target.py --target 5000 --material Steel --tol 0.01 --verify
& $PY design_by_target.py --target 5000 --material Steel --tol 0.01 --verify --verify-top 100
```

产出 `outputs/latent_ddpm/<out-tag>__<规格>_verified.csv`（多一列 `cs1_ross` 实测值）与 `..._verified_sheet.md`，复验记录写进同一个 `funnel_<out-tag>.json` 的 `gate` 段。成本见 2.6。

### 2.5 目标可达性：这个容差能收到多紧

`02d_tol_support.py` 一次算出「池子里有多少真实解」和「代理自己的误差」，写到 `outputs/tol_support.json`。每格是「最紧可行容差（该容差下的真实解数）」，可行定义为仍有 200 台以上真实解：

| 材料 | p10 | p50 | p90 | p95 | p99 |
| --- | --- | --- | --- | --- | --- |
| Steel | ±0.75% (264) | ±0.50% (275) | ±1.00% (273) | ±1.00% (213) | ±5.00% (245) |
| Aluminum | ±0.75% (241) | ±0.50% (275) | ±0.75% (208) | ±2.00% (370) | ±5.00% (266) |
| Titanium | ±0.75% (259) | ±0.50% (288) | ±0.75% (227) | ±2.00% (373) | ±5.00% (245) |

代理在 cs1 上的中位 APE 是 1.04% 到 1.14%。中位档的数据能支撑 ±0.5%，比代理误差还紧，所以那里限制精度的是代理；p95 往后是数据先到极限。实测印证：±5% 真值在带 0.95 到 0.99，±1% 只有 0.52，±0.5% 只有 0.22。所以**别指望代理在 1% 以内排序**：紧带要么放宽，要么按 2.6 付 ROSS 复验的钱。

**限定家族**时多走一步先验拟合，再在采样命令上加参数：

```powershell
& $PY S2\spec_design\07_family_policy.py
& $PY S2\spec_design\05_latent_ddpm.py sample --tag armV_ema --out-tag v1_fam `
    --predict v --use-ema 1 --guidance 1.0 --sample-steps 25 `
    --n-generate 5000 --n-submit 200 --seed 42 --threads 8 `
    --fam-policy learned --fam-allowed 2x2,3x2,4x2,2x3,3x3,4x3
```

`--fam-policy` 三个取值：`uniform`（默认，18 个家族均匀抽）、`learned`（按池子证据加权）、
`pinned`（用户钉死或限定范围，此时 `--fam-allowed` 必填）。

**最小冒烟**（改代码后先跑这个，约 20 秒）：

```powershell
& $PY S2\spec_design\05_latent_ddpm.py sample --tag armV_ema --out-tag smoke `
    --predict v --use-ema 1 --guidance 1.0 --sample-steps 25 `
    --n-generate 200 --n-submit 20 --seed 42 --threads 8 --specs Steel_high
```

预期：`outputs/latent_ddpm/funnel_smoke.json` 生成，在带率落在 0.15 到 0.25 之间。

### 2.6 紧带交付：闸门

容差小于代理误差时，代理排不出名次（2.5），但交付不必依赖名次：把短名单逐台解 ROSS，只交付实测在带的那部分。`05_ross_verify.py --gate` 与单入口 `--verify` 就是这件事，代价按「交付一台要付几次求解」算。实测（Steel 5000 rpm，`armV_ema`，25000 台臂的短名单 200 台，18 进程）：

| 容差 | 代理自称在带 | 真值在带＝交付 | 每交付一台 |
| --- | --- | --- | --- |
| ±0.5% | 200 | 66 | 1.95 s |
| ±1% | 200 | 105 | 1.22 s |
| ±2% | 200 | 164 | 0.78 s |
| ±3% | 200 | 182 | 0.71 s |
| ±5% | 200 | 192 | 0.67 s |
| ±10% | 200 | 199 | 0.65 s |

同一张表解释了「交付批代理在带 1.000」为什么不能直接信：±5% 时自称全在带、实测 192 台，直接交付会夹带 7 台不合格品，过闸门后为零。加大 `--n-generate` 没用：25000 台比 5000 台只多交付 2 台（真值在带 0.525 对 0.520），采样时间却多一倍。

把整批拉出来看，才知道每一级各值多少（`--dump-batch` 落盘整批的无偏样本，再用 `--select rand` 解真值，Steel 5000 rpm ±1%）：

| 环节 | 在带率 |
| --- | --- |
| 池子：66377 台真实 Steel 设计，不筛选 | 0.0077 |
| 条件 DDPM 生成 25000 台（300 台无偏抽样） | 0.0334 |
| 代理排序后取前 200 台 | 0.525 |
| 过闸门交付 | 1.000，1.22 秒/台 |

条件值 4.3 倍、排序值 15.7 倍。批样本的中位目标距离是 15.5%，说明紧带绝对值低是「条件推得散」，不是排序或候选数的问题。按代理自称预筛只省 4% 到 8%，不值得加开关。模型容量也试过减半（`armN256`，256x5）：验证损失只差 2.4%，但批样本 ±10% 份额从 0.324 掉到 0.247，所以不缩宽度。条件侧也试过三版：多尺度目标编码（`armF`）、`condition_dropout` 从 0.1 降到 0（`armD0`）、去噪器改成 x0 预测（`armX0`），同 seed 同配置各训一遍，六个目标各 3000 台配对比较，±1% 份额分别 +0.13 / −0.03 / +0.03 个百分点，都没收窄散度（报告第 5 节解决八）。改得动它的是**采样期的可微代理引导**（`--guide-lambda 0.05`，报告第 5 节解决九）：不用重训，采样从 11 秒到 21 秒，ROSS 真值口径下整批 ±1% 在带率从 0.0268 抬到 0.3000、中位目标距离从 15.41% 压到 1.72%。闸门交付量在 200 台预算下两条路等价（103 对 94 台，噪声内），但把预算放到 1000 台就是 **163 对 494 台**，引导便宜约 3 倍：小预算的瓶颈在代理排序精度，大预算的瓶颈在批次里有多少在带设计，后者已被引导解决。

## 3. 哪些文件是真正要的

**运行必需（缺一不可）**

| 文件 | 作用 |
|---|---|
| `spec_common.py` | 数据/特征/代理加载、canonical 与模型特征的转换、ROSS 调用 |
| `spec_eval.py` | 池子上下文、规格表、协议指标 |
| `spec_opt.py` | 合法性与软约束、目标函数 |
| `latent_design.py` | 34 维隐空间的编码与解码（硬约束按构造满足） |
| `latent_torch.py` | 可微解码器，只被基线 `adam` 用到 |
| `01_proxy_family.py` | 生成 `outputs/cache_features.npz` |
| `02b_point_protocol.py` | 生成 `outputs/specs.json` |
| `05_latent_ddpm.py` | 训练与采样条件 DDPM（`--dump-batch` 落盘整批的无偏样本） |
| `05_ross_verify.py` | ROSS 真值复验（`--dump-truth` 落盘逐台真值，`--gate` 只交付实测在带的设计） |
| `_ross_one_canon.py` | 上面这个脚本的 worker，一台设计一个进程 |
| `06_report.py` | 汇总表与图，并把逐次运行并进 `runs.json` |
| `design_by_target.py` | 单入口：采样 + 人可读设计清单（`--verify` 加 ROSS 闸门） |
| `02d_tol_support.py` | 目标可达性面，写到 `outputs/tol_support.json` |
| `07_family_policy.py` + `family_policy.py` | 家族先验，只在用 `learned` / `pinned` 时需要 |
| `ross_guard.py` | 退化标签判据，被复验与普查用到 |

模型权重：`outputs/latent_ddpm/armV_ema/model.pt`（推荐配置），`armD6b/model.pt` 与 `armN256/model.pt`（窄而深对照）。`*_pool__*.csv` 是整批的无偏样本，不是交付臂。
特征缓存：`outputs/cache_features.npz`（50 MB，重建要 20 分钟，别删）。

**基线对照（要跑基线时才需要）**

`spec_baselines_latent.py` 加上 `uncertainty_signals.py`、`proxy_calibration.py`（代理不确定度与校准、目标偏移）。
`tools/proxy_target_shift.py` 是目标偏移的检验脚本，读 `--dump-truth` 落盘的真值。

**历史步骤（保留作记录，当前流程不用）**

| 文件 | 说明 |
|---|---|
| `00_validate_canonical.py` | 一次性校验 canonical 表结构 |
| `02_protocol.py` | 带宽版规格表，已被 `02b_point_protocol.py` 取代 |
| `02d_tol_support.py` | 每个规格在各容差下的真实解数，即报告第 1 节那张表 |
| `03_cond_diag.py` | 条件的可分性诊断 |
| `03b_repair_compare.py` | 随机修复与结构化修复的对比 |
| `04_ddpm.py` | 直接生成 33 维原始设计的 DDPM，合法率只有 6.9%，被隐空间版取代 |
| `05b_test_latent.py` | 隐空间往返与解码器压力测试 |
| `spec_baselines.py` | 第一版基线（canonical 空间），已被 `spec_baselines_latent.py` 取代 |

**分析脚本（产出报告里的数字，不参与运行）**

`tools/` 下十个：`diag_family_reach.py`、`diag_family_ab.py`、`diag_family_ab2.py`、
`uncertainty_stage_a.py`、`uncertainty_stage_b.py`、`uncertainty_calibration.py`、
`uncertainty_analysis.py`、`uncertainty_extra_truth.py`、`uncertainty_noise.py`、
`uncertainty_significance.py`、`census_low_cs1.py`。

**产物**

| 路径 | 内容 |
|---|---|
| `outputs/specs.json`、`outputs/spec_table.md` | 六个规格 |
| `outputs/latent_ddpm/armV_ema/model.pt` | 模型权重 |
| `outputs/latent_ddpm/<out-tag>__<规格>.csv` | 提交清单（200 台） |
| `outputs/latent_ddpm/funnel_<out-tag>.json`、`meta_<out-tag>.json` | 该次运行的漏斗与配置 |
| `outputs/latent_ddpm/runs.json` | 全部历史运行的漏斗与配置（合并成一个文件） |
| `outputs/ross_verify.json`、`ross_verify_rand.json` | 按「规格\|方法」索引的 ROSS 复验结果（带 `--dump-truth` 的条目另存逐台真值） |
| `outputs/latent_ddpm/<out-tag>__<规格>_sheet.md` | 人可读设计清单（换算单位后的表格） |
| `outputs/tol_support.json` | 逐规格与逐分位的池子支撑、可达容差 |
| `outputs/latent_baselines/baseline_metrics.json` | 七个基线在同一隐空间的指标 |
| `outputs/family_policy.json`、`family_reachability.json` | 家族先验与池子普查 |
| `outputs/uncertainty_*.json`、`proxy_calibration_report.json` | 代理不确定度与校准的证据 |
| `outputs/auto_report.md`、`outputs/figures/` | `06_report.py` 自动生成的报告与图 |

## 4. 当前水平与已知边界

ROSS 真值：六规格平均真值在带 0.933（每规格随机抽 20 台）；本轮同一目标（Steel 5000 rpm）的新臂每臂抽 150 台，真值在带 0.953 到 0.987（±5%）、0.960（单侧加瞄准余量）、0.520（±1%）、0.220（±0.5%）。
同口径的对照：检索 1.000（但新颖度 0，是复制已有设计）、GA 0.750、Adam 0.575、DE 0.348、BO 0.125。

1. **容差是主旋钮**：±5% 能交付，±1% 与 ±0.5% 不能（真值在带 0.52 与 0.22），而事后校准已被证否（单参数目标偏移在 4 条臂上 0 条受益）。别许下 1% 以内的容差，除非把 ROSS 放进循环。
2. 交付清单的 0.933 到 0.987 与整批 5000 的代理在带率（约 0.18）是两个口径，不能混说；要整批交付就得把复验扩到整批。
3. 代理在分布外偏乐观（本版 +0.058，DE +0.652）。分析层的误差棒、门控与偏置层都没接进采样器的选择步骤，而且第 5 节证明接进去也救不了紧带。
4. 用户钉死池子对该带位没有支撑的家族时交付的是外推设计。`funnel_*.json` 里的 `support_pool_band_family` 与 2.5 节的表能量出这一点，但采样器还不会因此拒绝或告警。