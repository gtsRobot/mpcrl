# MPC + LeWorldModel + Planning-Residual RL（论文实验级工程）

> **2026-09-30 重要更新**：这一轮修复改变了若干实验数值的语义，`docs/RESULT_DRIVEN_TUNING_20260831.md`
> 和 `docs/RESULTS_ANALYSIS_20260917.md` 里的旧数字**不能**和新运行的数字放在同一张表里。
> 动手前请先读 **[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)** —— 它也解释了为什么旧的
> 4-方法 suite 从一开始就测不出它声称的 5% 效应，以及应该改用哪个协议。

本工程实现三个可区分的研究组件：

1. **Ensemble World Model + CEM-MPC**：用于 MuJoCo 状态空间控制的可训练基线。
2. **Planning-level Residual SAC**：RL 不从零生成动作，而对 MPC 的短时 action chunk 做残差修正；可选世界模型不确定性自适应门控与残差幅度正则。
3. **LeWorldModel 复现路径**：ViT encoder + 自回归因果 transformer predictor（AdaLN-Zero 动作条件）+ SIGReg，以及 latent CEM-MPC 适配代码；另附官方仓库对齐说明。

> 说明：ZPRL 原论文建立在预训练生成式机器人策略（flow-matching policy）上。本工程的 `zprl_style` 是用于 MuJoCo 公平对比的**瓶颈潜空间扰动代理基线**，不是对 ZPRL 机器人实验的等价复现。这样可避免把“世界模型潜空间”和“冻结基础策略瓶颈潜空间”混为一谈。

## 目录

```text
MPC_RL_ZPRL_RTX5090/
├─ configs/                   # 环境/训练/消融配置
│  ├─ local/                  # 单卡轻量 profile（5060 Ti 级别，分钟级迭代）
│  └─ rtx5090/                # 论文配置
├─ mpcrl/
│  ├─ world_model.py          # 向量化 ensemble dynamics + reward model
│  ├─ cem.py                  # CEM-MPC（NaN 安全，execute 模式可选）
│  ├─ sac.py                  # Residual SAC（优势参数化 critic + 终端价值头）
│  ├─ gate.py                 # 动态归一化不确定性门控 + Residual Ramp
│  ├─ plan_cache.py           # next-state MPC 规划缓存
│  ├─ planning_residual.py    # 残差扩展 + 多模式 safeguard
│  ├─ residual_context.py     # SAC context 构造
│  ├─ paired_stats.py         # 配对统计（t 分布 CI、精确 Wilcoxon、MDE）
│  ├─ zprl_style.py           # 瓶颈潜空间扰动代理基线
│  ├─ lewm.py                 # ViT encoder + AR predictor + SIGReg
│  ├─ lewm_planner.py         # latent CEM planner
│  ├─ lewm_rl.py              # LeWM 侧残差工具
│  ├─ replay.py               # replay buffers
│  ├─ envs.py / config.py / utils.py
│  ├─ metrics.py              # 论文指标
│  └─ plotting.py             # 论文绘图
├─ experiments/
│  ├─ train.py                # mpc_only / action_residual / planning_residual
│  ├─ train_paired.py         # 配对协议（共享冻结 WM + CRN）<- 主实验
│  ├─ summarize_paired.py     # 配对统计汇总
│  ├─ train_zprl_style.py     # ZPRL-style proxy
│  ├─ train_lewm.py           # LeWM 目标层面复现
│  ├─ train_lewm_mpc_rl.py    # LeWM-MPC vs LeWM-MPC+RL 配对
│  ├─ run_suite.py            # 多任务多 seed 批量实验（失败不中断）
│  ├─ run_lewm_mpc_rl_suite.py
│  ├─ run_zprl_ablation.py
│  ├─ quick_diagnostic.py     # 10k 三方法快速诊断
│  ├─ ablation.py             # 消融实验矩阵
│  ├─ evaluate.py             # 对保存的 run 做确定性评估
│  ├─ aggregate.py            # CSV 聚合
│  ├─ summarize_eval.py / summarize_lewm_mpc_rl.py
│  └─ plot_results.py / plot_ablation.py / plot_lewm_mpc_rl.py / paper_figures.py
├─ scripts/
│  ├─ _common.sh              # nohup/setsid + 日志 + cd 到仓库根
│  ├─ smoke_test.sh
│  ├─ run_paired_suite.sh     # <- 主实验入口
│  ├─ run_diagnostic_10k.sh
│  ├─ run_main_suite.sh
│  └─ run_ablation.sh
├─ tests/
└─ docs/
```

## RTX 5090 环境配置

本分支提供 `configs/rtx5090/` 专用配置。RTX 5090 属于 NVIDIA Blackwell（SM 12.0），建议使用 **PyTorch CUDA 12.8+** 官方构建。专用配置默认启用 BF16 自动混合精度、TF32、cuDNN benchmark，并针对 32 GB 显存增大 world model、SAC batch 和 CEM 并行候选数。

### 1. 创建 Conda 环境

```bash
conda create -n mpc_rl_zprl_5090 python=3.11 -y
conda activate mpc_rl_zprl_5090
```

也可以使用工程中的 `environment.yml` 创建基础环境：

```bash
conda env create -f environment.yml
conda activate mpc_rl_zprl_5090
```

### 2. 安装 RTX 5090 可用的 PyTorch

优先安装 CUDA 12.8 官方 wheel：

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

随后安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

> 不建议给 RTX 5090 安装旧的 `cu121/cu124/cu126` PyTorch wheel。工程要求 `torch>=2.7`，并以 CUDA 12.8+ 构建为目标。

### 3. 检查 RTX 5090 是否正确工作

```bash
nvidia-smi
python scripts/check_rtx5090.py
```

正常情况下应看到类似：

```text
GPU: NVIDIA GeForce RTX 5090
Compute capability: 12.0
VRAM GiB: 31.x
BF16 supported: True
RTX 5090 environment check: PASS
```

### 4. 一键安装（Ubuntu / bash）

若本机已经安装 Conda，可直接执行：

```bash
bash scripts/install_rtx5090.sh
conda activate mpc_rl_zprl_5090
```

该脚本依次创建 Conda 环境、安装 CUDA 12.8 PyTorch、安装工程依赖并执行 GPU 自检。

## 最小训练

先在本地用轻量 profile 验证（5060 Ti 级别，30k 步约 10 分钟）：

```bash
python -m experiments.train --config configs/local/halfcheetah.yaml --method planning_residual --steps 30000
```

论文规模：

```bash
python -m experiments.train --config configs/rtx5090/halfcheetah.yaml --method mpc_only --steps 100000
python -m experiments.train --config configs/rtx5090/halfcheetah.yaml --method action_residual --steps 100000
python -m experiments.train --config configs/rtx5090/halfcheetah.yaml --method planning_residual --steps 100000
# 对保存的 run 做确定性评估：
python -m experiments.evaluate --run-dir runs/<run_name> --episodes 10
# 被 kill 之后续跑（需要 --run-name 才能定位目录）：
python -m experiments.train --config ... --run-name <run_name> --resume
```

`experiments.train` 的 `--method` 只接受三个值：`mpc_only` / `action_residual` / `planning_residual`。
**`zprl_style` 不是它的一个 method**，它有独立入口 `python -m experiments.train_zprl_style`
（`experiments.run_suite` 会自动分派）。

## 主实验：配对协议

**这是能真正测出残差效应的实验**，普通的 4-方法 suite 不行 —— 原因见
[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) §0。

`experiments/train_paired.py` 在**一次 run 内**训练并冻结一个世界模型，然后用
**同一个冻结 WM + 同一组 eval seed + 共同随机数（CRN）** 分别评估 MPC 和 MPC+residual：

```text
warmup → 训 WM（跑 MPC）→ 冻结（state_dict hash 断言）→ 评 Arm A
       → 只训残差 → 评 Arm B（同一 WM、同一 seed、同一 CEM 噪声）
```

```bash
# 单个 seed
python -m experiments.train_paired --config configs/rtx5090/halfcheetah.yaml --seed 0 --steps 100000

# 多 seed + 统计汇总（n>=6：n=5 时 Wilcoxon 两侧最小 p = 0.0625，数学上到不了 0.05）
CONFIG=configs/rtx5090/halfcheetah.yaml SEEDS="0 1 2 3 4 5 6 7" bash scripts/run_paired_suite.sh
python -m experiments.summarize_paired --root runs --out results/paired_summary.csv
```

汇总会给出 per-seed 配对差、**t 分布** 95% CI（不是正态 1.96）、正 delta 的 seed 数、
精确 Wilcoxon p 及其**理论下限**、以及当前 n 下的**最小可检测效应（MDE）**。
如果 MDE 大于你期望的效应，这个 n 就不值得跑。

预算摊分：`warmup + wm_steps + rl_steps = --steps`。Arm A 因此用的是较少数据训练的 WM，
比满预算的 `mpc_only` 弱，所以 `pair_summary.json` 会把三个数都记下来。

## RTX 5090 推荐运行顺序

先进行最小 GPU 验证：

```bash
conda activate mpc_rl_zprl_5090
python scripts/check_rtx5090.py
python -m unittest discover -s tests -v
bash scripts/smoke_test.sh      # 三个方法 + 配对协议 + LeWM，全部跑一遍
```

确认无报错后，运行单个正式实验：

```bash
python -m experiments.train \
  --config configs/rtx5090/halfcheetah.yaml \
  --method planning_residual \
  --steps 100000 \
  --seed 0
```

四种对比方法（注意入口不同）：

```text
mpc_only            python -m experiments.train --method mpc_only
action_residual     python -m experiments.train --method action_residual
planning_residual   python -m experiments.train --method planning_residual
zprl_style          python -m experiments.train_zprl_style     <- 独立入口
```

RTX 5090 主配置的关键参数（**以 yaml 为准**，下表按实际文件核对过）：

| 参数 | HalfCheetah | Hopper | Walker2d | Reacher |
|---|---|---|---|---|
| ensemble / hidden | 7 / 512 | 7 / 512 | 7 / 512 | **5 / 256** |
| SAC hidden / batch | 512 / 1024 | 512 / 1024 | 512 / 1024 | **256 / 512** |
| CEM candidates | 1024 | 1024 | 1024 | 1024 |
| CEM horizon | 20 | 15 | 15 | 20 |
| residual_scale | 0.10 | 0.10 | **0.06** | **0.05** |
| gate_min / gate_max | 0.08 / 0.90 | 0.15 / 0.90 | **0.00 / 0.65** | **0.00 / 0.70** |
| gate_power | 1.0 | 1.0 | **1.5** | **1.25** |
| ramp_steps | 4000 | 5000 | 5000 | 5000 |

`precision: bf16`、TF32 与 cuDNN benchmark 默认开启。

> `candidates` 从 4096 降到 1024 是实测结果：在 5060 Ti 上 162 → 47 ms/plan（3.5×），
> 而 CEM 质量在远低于 4096 处就饱和了。若显存或时间更紧，先降 `mpc.candidates`，
> 再降 `train.batch_size`。

### 门控与残差

- `gate_mode: running_zscore`：ensemble uncertainty 先按运行中的 EMA 均值/方差标准化，再进入 Sigmoid，不再依赖固定绝对阈值。
- `gate_initial_std: -1`：从信号本身推导初始尺度。uncertainty 现在是**回报单位**（跨成员回报标准差），量级随任务差几个数量级，固定的 0.05 会让 gate 前几百步饱和。
- `ramp_steps`：RL 开始更新后，Residual 修正强度由 0 线性增加至完整强度，避免随机初始策略立即破坏 MPC。
- `residual_scale: 0.10`：RTX 5090 主配置先采用保守残差幅度，后续根据 10k 诊断结果再调。
- `PlanCache`：Residual SAC 为 next-state context 计算出的 MPC plan 会在下一环境步复用，避免对同一状态重复规划。

训练日志新增：

- `adaptive_gate`：仅由标准化 uncertainty 得到的门控值；
- `residual_ramp`：当前 Residual Ramp 系数；
- `gate`：实际生效的 `adaptive_gate × residual_ramp`；
- `gate_z`：当前 uncertainty 的标准化 z-score；
- `effective_residual_norm`：真正加到 MPC action chunk 上的有效残差范数；
- `mpc_cache_hit_rate`：MPC next-plan 缓存命中率；
- `safeguard_scale` / `safeguard_gain`：shield 接受的残差比例与预测增益（`safeguard_mode: off` 时为 NaN）；
- `adv_q_std_ratio`：`std(A) / std(Q)`。**这是判断残差能不能学会的第一个诊断** ——
  低于约 `1e-3` 说明 critic 根本没有可用的残差信号，此时调 gate 是没有意义的；
- `adv_mean`、`q_loss`、`actor_loss`、`alpha`、`v_loss`：SAC 内部量。

## 10k 三方法快速诊断

在正式 100k 实验前，建议先运行 10k 快速诊断：

```bash
conda activate mpc_rl_zprl_5090
python -m experiments.quick_diagnostic \
  --config configs/rtx5090/halfcheetah.yaml \
  --steps 10000 \
  --seeds 0
```

也可以直接执行：

```bash
bash scripts/run_diagnostic_10k.sh
```

该命令依次运行：

```text
mpc_only
action_residual
planning_residual
```

结果保存在：

```text
results/diagnostic_10k/<时间戳>/
├─ episodes_all.csv
├─ eval_all.csv
├─ summary_per_seed.csv
├─ summary_mean_std.csv
├─ comparison.csv
└─ figures/
   ├─ episode_return.png
   ├─ mpc_ms.png
   ├─ prediction_mse.png
   ├─ action_d1.png
   ├─ effective_residual_norm.png
   ├─ gate.png
   └─ final_eval_return.png
```

正式 100k 实验前重点检查：`adv_q_std_ratio` 是否明显高于 `1e-3`；`gate` 是否脱离饱和；
`effective_residual_norm` 是否随 Ramp 平稳增加；`mpc_cache_hit_rate` 在非终止步应接近 1；
动作一阶/二阶平滑度不应明显恶化。

> **不要**用这个诊断判断"Planning Residual 是否超过 MPC" —— 它是非配对的、单 seed 的，
> 完全落在噪声里。要回答那个问题请用配对协议（见上文「主实验」）。

## 4-方法工程 benchmark

```bash
# 注意：configs/rtx5090/paper_suite.yaml 里已经写了 run_lewm: true，
# 所以 --include-lewm 对它是 no-op —— LeWM 的 20 个 job 本来就会跑。
bash scripts/run_main_suite.sh

# 或手动：
python -m experiments.run_suite --suite configs/rtx5090/paper_suite.yaml
python -m experiments.aggregate --root runs --out results/summary.csv
python -m experiments.plot_results --root runs --outdir results/figures
```

规模提醒：`paper_suite.yaml` = 4 env × 4 method × 5 seed = **80 个 job**，
外加 4 个 LeWM 配置 × 5 seed = **20 个 job**。按修复后的吞吐（HalfCheetah 100k ≈ 1.3 h）
仍然是**上百 GPU 小时**。一个 job 失败不会再带走其余的（`run_suite.py` 默认 `--keep-going`，
失败列表写到 `results/suite_failures.csv`），所有脚本也都 `nohup`/`setsid`，SSH 断开不会杀进程。

> 这个 suite 是**非配对**的（每个 method 独立训练自己的世界模型），所以它能给出各方法的
> 绝对水平，但**不适合**用来主张"残差比 MPC 好 5%" —— 那个差值被世界模型质量的 seed 方差
> 淹没。主张改进请用配对协议。

原始保守配置仍保留在 `configs/*.yaml`，用于硬件无关的对照和消融复现；
`configs/local/*.yaml` 是单卡分钟级迭代用的 profile。

`configs/rtx5090/refine/*.yaml` **不要用于论文结果** —— 它们是拿已完成的 5-seed 评估结果
反过来调参、且只重跑输掉的环境，属于 test-set tuning（文件头有说明）。

## LeWorldModel

`mpcrl/lewm.py` 现在是对官方 [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)
（arXiv:2511.19312）的忠实实现：

- **ViT encoder**，输出 patch token 序列（无 global pooling、无 BatchNorm）；
- **自回归因果 transformer predictor**，AdaLN-Zero 动作条件，对帧历史做 block-causal 注意；
- 目标 `L_pred + λ·SIGReg`，**无 EMA teacher、无 stop-gradient**（这是 LeJEPA 的核心主张）；
- reward probe 仍是 MuJoCo 适配层，不属于原论文的两项目标。

旧的 3 层 CNN + 2 层 MLP 保留为显式 ablation：

```yaml
lewm:
  encoder: cnn          # vit | cnn
  predictor: mlp        # ar_transformer | mlp
```

```bash
# 本地（分钟级）
python -m experiments.train_lewm_mpc_rl --config configs/local/lewm_halfcheetah.yaml --rl-steps 2000

# 论文规模
python -m experiments.train_lewm_mpc_rl --config configs/rtx5090/lewm_halfcheetah.yaml --seed 0
```

诊断三件套：`pred`（预测损失）、`sigreg`（**真 N(0,I) 的锚点是 ~1.0**）、
`token_effective_rank`（应为 `latent_dim` 的相当一部分）。
`λ` 必须按数据量和训练长度重新扫描 —— 见 [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) §2 的实测表。

> **上服务器前务必重新实测规划耗时**：token 级 predictor 放进 latent CEM 比旧的 MLP 贵得多
> （旧预算下 16.6 s/plan vs 48 ms/plan），所以 LeWM 配置的 CEM 预算已大幅下调。
