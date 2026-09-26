# S1 v4.7 使用说明

## 为什么有 v4.7

v4.6 的临界转速标签定义（正阻尼、`wd > 0`、正进动的临界转速，按转速升序）
本身没有问题，问题出在**模态预算**：`NUM_MODES = 24` 只让 ROSS 返回 12 对
涡动模态，而一个转子的正进动模态平均只有 4.0 个，于是高阶临界转速被"删失"
（censoring）了。

实测（同一批 200 个转子，只改 `num_modes`）：

| 阶次 | 24 模态 | 48 模态 | 96 模态 |
|---|---|---|---|
| cs1–cs3 | 200 / 200 / 199 | 200 / 200 / 200 | 200 / 200 / 200 |
| cs4 | 152 | **200** | 200 |
| cs5 | 52 | **200** | 200 |
| cs6 | 1 | **198** | 200 |
| cs7 | 0 | 159 | 200 |
| 平均正进动模态数 | 4.02 | 7.30 | 13.02 |

关键校验：**cs1–cs3 在 24/48/96 模态下数值完全一致**（差 < 0.1 rpm，只是
四舍五入），说明提高模态预算只在高频端追加模态，不改变既有标签的定义。

全量数据上的后果：v4.5 的 cs4/cs5/cs6 覆盖率是 69.7% / 23.7% / **0.23%**，
合并后 cs6 只有约 520 行有效标签（0.26%）。48 模态后预计约 99%，
即约 **380 倍**的有效标签。

## v4.7 相对 v4.6 的改动

1. `NUM_MODES` 24 → **48**（24 对涡动模态）。
2. 标签列从 `cs_1..cs_3` 扩展到 **`cs_1..cs_6`**（`N_PRIMARY_CS = 6`）。
3. 新增 `MIN_FORWARD_MODES = 3`：`success` 的判定门槛保持 v4.6 的
   "至少 3 个正进动模态"，所以 v4.7 的数据集**不会比 v4.6 更小**，
   只是标签更完整（不足 6 阶的行仍然保留，缺的阶为 NaN）。
4. relabel / combine 的默认模态预算同步改为 48。

## 文件

| 文件 | 说明 |
|---|---|
| `s1_run_pipeline_v4_7.py` | 主流程（采样 → 过滤 → ROSS → 后处理） |
| `s1_constraint_filter_v4_7.py` | 序贯条件 LHS 采样与 H1–H6 硬约束 |
| `s1_quality_check_v4_7.py` | 数据质量检查 |
| `relabel_old_v4_7.py` | 对已有特征表重标注（默认 48 模态） |
| `combine_relabel_chunks_v4_7.py` | 合并分块结果 |
| `promote_relabel_to_v4_7.py` | 把在跑的 v4.6 命名分块提升为 v4.7 命名 |
| `run_relabel_v4_7.ps1` | 全量重标注启动脚本（可断点续跑） |

## 生成全新数据

```powershell
& "C:\Users\fcu15\ross230_py312\python.exe" s1_run_pipeline_v4_7.py `
  --n 100000 --workers 20 --seed 42 -o output_v4.7
```

## 重标注已有数据

```powershell
.\run_relabel_v4_7.ps1 -ChunkSize 2000 -Workers 20 -NumModes 48
```

输出：`relabeled_full_v4.7`（v4.4 特征表）与 `relabeled_full_v4.7_v45`
（v4.5 特征表）。每个 chunk 完成即落盘，中断后重跑会自动跳过已完成分块。

## 把正在运行的 v4.6 命名分块提升为 v4.7 命名

2026-09-20 启动的 48 模态重标注是从 v4.6 脚本发起的，落在
`S1/03_relabel/relabeled_full_v4.6_m48` 与 `S1/03_relabel/relabeled_full_v4.5_m48`，文件名是
`*_v4.6_*`，标签表只有 `cs_1..cs_3`。用下面的命令零成本转换（不重跑 FEM）：

```powershell
& "C:\Users\fcu15\ross230_py312\python.exe" promote_relabel_to_v4_7.py `
  --src-dir  "..\..\03_relabel\relabeled_full_v4.6_m48" `
  --out-dir  "relabeled_full_v4.7"
```

该脚本从长表 `relabel_modes_*_all.csv` 重新推导每个转子的正进动频率列表，
补齐 `cs_4..cs_6`，其余内容原样搬运。

## 环境

ROSS 2.3.0 / Python 3.12.14：`C:\Users\fcu15\ross230_py312\python.exe`
（原临时环境 `%TEMP%\ross230_conda_py312` 的标准库被系统清理删除过，
这是修复后复制出来的稳定副本）。

重标注时需要设置：

```powershell
$env:ROSS_FAST_RELABEL = "1"
$env:ROSS_ENABLE_NUMBA_JIT = "0"
```

启动脚本已包含这两项。

## 采样与修复：统一入口 s1_toolkit.py

v4.7 把"造新数据"和"修旧数据"收敛到一个入口：

```powershell
# 造新数据（序贯条件 LHS + H1-H6 硬约束 + ROSS）
python s1_toolkit.py sample --n 50000 --workers 20 -o output_v4.7

# 体检任意一份数据集（纯 numpy，不需要 ROSS）
python s1_toolkit.py diagnose --features features.csv --dataset dataset.csv

# 修复任意一份数据集（几何丝毫不变，只重算标签）
python s1_toolkit.py repair --features features.csv --dataset dataset.csv `
    -o repaired_v4.7 --workers 20 --chunk-size 2000
```

### 什么时候需要修复

旧版本（v4.4 及更早）有两个缺陷，它们在几何里完全看不出来：

1. 没有过滤进动方向。 ROSS 返回的涡动频率是"后向/前向"交替的，直接取最低三个，
   在多数情况下得到的是 后向/前向/后向 —— cs1 和 cs2 其实描述同一阶模态的两个方向，
   而不是相邻两阶临界转速。
2. 模态预算太小。 num_modes=12 只返回 6 个涡动频率，cs4 及以上根本不存在，
   cs3 也处于边缘。

实测对比（s1_diagnose_dataset.py 的输出）：

| 数据集 | cs2/cs1 比值中位数 | cs2/cs1 近重复率 | 判定 |
|---|---|---|---|
| v4.4 标签（缺陷版） | 1.045 | 32.8% | 方向混叠 + 成对结构 |
| v4.5 标签（修正版） | 1.870 | 0.1% | 定义正常，但高阶删失 |

一个 3 行样本的实例（同一转子，旧 vs 新）：

```text
旧 v4.4 : 1995.4, 1998.5, 2149.8,  3689.9,  12388.8,  24976.5
新 v4.7 : 1998.5, 3689.9, 32549.9, 78147.3, 168477.9, 215640.3
```

旧标签的第 2、4 个值正好是新标签的第 1、2 阶 —— 旧 cs2 就是新 cs1 本身。

### repair 的产物

- repaired_dataset_v4.7_<n>.csv：修复后的数据集（cs_1..cs_6 全部替换，多一列 label_source）
- comparison_v4.7.csv：每一阶的旧/新标签差异（绝对差、APE 中位数、APE>1% 占比）
- relabel_modes_v4.7_*.csv：每个转子的完整模态长表（含后向模态、方向、阻尼比）
- diagnose_report.md / repair_report.md / repair_summary.json

分块 + 断点续跑，中断后重跑自动跳过已完成分块。

### 成本估算

48 模态下单核约 9.8 s/转子：

| 行数 | 20 进程 | 30 进程 |
|---|---|---|
| 50,000 | 约 6.8 h | 约 4.5 h |
| 100,000 | 约 13.6 h | 约 9.1 h |

diagnose 会直接把这个估算写进报告，先体检再决定要不要修。
