# CVAE 主基线实验

本目录实现 Pascal 老师要求的“另一种生成模型”主基线：条件变分自编码器（CVAE）。
目的不是证明 CVAE 更好，而是在**同一候选预算、同一筛选器、同一 ROSS 重算协议**下，
检验论文的生成—筛选—验证流程是否依赖 DDPM 这一种生成器。

## 1. 公平性设计

- 训练数据与 DDPM 完全相同：199 808 条可用样本，189 818 / 9 990 训练/验证划分，21 维区间条件向量。
- 使用同一个 34 维可行性保持潜空间和同一个物理解码器 `g_theta`，H1–H6 由同一解码路径保证。
- 离散拓扑量 `n_disks`、`n_bearings` 与 DDPM 一样作为族条件输入，不单独生成。
- 六个冻结规格：Steel 3600/16000，Aluminum 5100/23000，Titanium 4300/19000 rpm。
- 每个规格、每个种子生成 5 000 个候选；用与论文基线实验完全相同的 `select_common`
  （先取代理带内、再在标准化设计空间做最远点多样性）选出 200 个；再逐个做 ROSS 重算。
- 交付率分母固定为提交数 200：ROSS 报错或超时的设计仍计入分母，不计入交付数。

## 2. 模型与训练

| 项目 | 配置 |
|---|---|
| 潜变量 | 32 维 |
| 编码器 / 解码器 | 3 × 512 MLP，sigmoid 输出 + 族掩码 |
| 条件先验 p(z\|c) | 2 × 256 MLP |
| 参数量 | 约 1.252 M |
| beta | 0.001（beta=0.005 出现明显后验坍塌，弃用） |
| KL warm-up | 10 epoch，warm-up 结束后才做模型选择 |
| 训练 | 200 epoch，CPU，约 1 675 s |
| 最佳验证损失 | 0.020261（最终 KL ≈ 11.5，重构 MSE ≈ 0.0088） |

检查点：`outputs/cvae_main/model_best.pt`。

## 3. 主结果（纯 CVAE 先验采样，无代理引导）

| 方法 | 规格 × 种子 | 交付/提交 | 合并交付率 | Wilson 95% CI |
|---|---|---:|---:|---|
| CVAE | 6 × 3 | 2 754 / 3 600 | 76.5% | 75.1%–77.9% |
| DDPM（论文，seed 42） | 6 × 1 | 897 / 1 200 | 74.8% | 72.2%–77.1% |

seed-42 六规格配对：平均差值 +0.4 个百分点，标准差 3.9 个百分点，Wilcoxon p = 1.0。
**未检测到差异，不等于等价；样本量只有 6 对。**

分规格（CVAE 三种子均值 / DDPM seed 42）：

| 规格 | CVAE | DDPM |
|---|---:|---:|
| Steel_mid | 82.3% | 82.0% |
| Steel_high | 76.0% | 75.5% |
| Aluminum_mid | 82.2% | 79.5% |
| Aluminum_high | 64.5% | 65.5% |
| Titanium_mid | 81.7% | 80.0% |
| Titanium_high | 72.3% | 66.0% |

两个必须一起报告的结构性差异：

1. **池纯度**：CVAE 纯先验的 5 000 池代理带内率只有 10%–23%，DDPM 引导池为 81%–89%。
   CVAE 更依赖预筛选环节，但短名单仍可被筛到 100% 代理带内。
2. **多样性**：CVAE 短名单的中位最近邻距离约 3.6–3.8，DDPM 约 4.9–5.1（同一标准化尺度）。
   交付率相近不代表解集结构相同，VAE 的 mode-averaging 在这里可见。

## 4. 引导 CVAE 消融（单规格）

`guided_prior` 在 CVAE 的 z 空间做 25 步逐样本代理引导，对应 DDPM 的 25 步反向链。
Steel_mid / seed 42：

| 变体 | 池内代理带内率 | ROSS 交付 |
|---|---:|---:|
| CVAE 纯先验 | 23.3% | 163/200（81.5%） |
| CVAE 引导 λ=30 | 98.0% | 166/200（83.0%） |
| DDPM（论文） | 89.0% | 164/200（82.0%） |

引导把池纯度从 23% 提高到 98%，交付率只提高 1.5 个百分点。
该规格上限制交付率的是代理排序与真实值之间的偏差，而不是生成池中合格样本的数量。
这只是单规格示意性消融，不作为六规格结论。

> 实现说明：`guided_prior` 的引导损失必须对族内样本取 **sum** 而不是 mean。
> 取 mean 会让梯度按 1/n_family 衰减，5 000 样本时引导强度约弱 25 倍；
> 早期一次 λ=30 的 5 000 池只得到 23.8% 带内率，就是该批大小依赖导致的，已修正。

## 5. 文件

- `cvae_common.py`：数据、条件、CVAE 模型、采样、引导、统一筛选与公共工具。
- `train_cvae.py`：训练 CVAE；模型选择只在 KL 升温结束后进行。
- `run_cvae_pools.py`：生成候选池与 200 设计短名单；`--guide-lambda 0` 为纯先验。
- `run_ddpm_pools.py`：用相同筛选器准备 DDPM 对照池。
- `run_verify.py`：调用论文现有 ROSS 验证器逐个重算短名单。
- `analyze_cvae.py`：汇总交付率、Wilson 区间、代理精度、多样性与配对差异。
- `prepare_ddpm_reference.py`：把论文 DDPM 的 `runs[0]` 结果规范化为本目录的分析格式（只重排键，不改数值）。
- `make_report.py`：生成 `results/CVAE_主基线_结果.md` 与 `results/CVAE主基线_结果.docx`。
- `results/cvae_ddpm_detail.csv`、`cvae_ddpm_summary.csv`、`cvae_ddpm_by_spec.csv`、
  `cvae_vs_ddpm_paired.csv`、`cvae_vs_ddpm_comparison.json`：机器可读结果。

## 6. 复现

```powershell
$py = "D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2\neural_surrogate_experiment\.venv_torch\Scripts\python.exe"

# 训练（已完成，可跳过）
& $py train_cvae.py --out-tag cvae_main --epochs 200 --warmup-epochs 10 --threads 16

# 候选池与短名单
& $py run_cvae_pools.py --seeds 42,7,13 --threads 8 --guide-lambda 0

# ROSS 重算（18 组，约 90 min / 16 workers）
& $py run_verify.py --methods cvae --seeds 42,7,13 --workers 16

# 汇总
& $py prepare_ddpm_reference.py
& $py analyze_cvae.py --methods cvae,ddpm --seeds 42,7,13
& $py make_report.py
```

## 7. 结论边界

- 所有“验证”均为 ROSS 数值重算，不是物理实验；结论只覆盖 ROSS 建模协议下的临界转速符合性。
- 比较限定在同一冻结代理、同一候选生成预算、同一 200 次 ROSS 重算协议内。
- 未检测到显著差异不等于等价；CVAE 与 DDPM 的多样性差异必须与交付率一起报告。
- 若 CVAE 数值更好，也不能表述为“生成模型普遍更优”或“DDPM 没有必要”。
