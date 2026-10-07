# MPC + RL + ZPRL + LeWM 工程代码与实验说明

> 适用仓库：`it0farmer/MPC_RL_ZPRL_RTX5090`
>
> 目标硬件：RTX 5090 32 GB
>
> 推荐分支：`main`

---

## 1. 项目定位

本项目用于研究 **模型预测控制（MPC）+ 强化学习（RL）+ 残差策略（Residual Policy）+ ZPRL-style 潜空间残差控制**，并加入 **LeWorldModel（LeWM）工程适配 baseline** 作为像素世界模型对照。

项目当前包含三类研究工作：

1. **主对比实验**：在统一 MuJoCo 连续控制接口下比较不同控制方法；
2. **ZPRL 消融实验**：验证 DAgger、潜变量维数、残差幅值、一致性约束等模块的贡献；
3. **LeWM 参考实验**：一部分是在 MuJoCo 上的工程适配对比，另一部分用于核验 LeWM 官方四任务 benchmark。

需要特别区分：

- 本项目的主 ZPRL 方法是 **state-space continuous control**；
- LeWM 官方 benchmark 是 **pixel / goal-conditioned world-model planning**；
- 二者可以进行方法和工程层面对照，但不应把 MuJoCo 工程数值写成 LeWM 官方 benchmark 原文数值。

---

# 2. 当前完整实验矩阵

## 2.1 主实验任务

统一 MuJoCo 工程 benchmark 由四个任务组成：

| 任务 | 配置文件 | 类型 |
|---|---|---|
| Hopper-v5 | `configs/rtx5090/hopper.yaml` | 单腿跳跃/平衡 |
| Walker2d-v5 | `configs/rtx5090/walker2d.yaml` | 双足连续行走 |
| HalfCheetah-v5 | `configs/rtx5090/halfcheetah.yaml` | 高维连续奔跑 |
| Reacher-v5 | `configs/rtx5090/reacher.yaml` | 机械臂到达 |

Reacher-v5 是新加入的第四个统一控制任务，用来提高任务多样性，同时与 LeWM 官方 benchmark 中的 Reacher 类型形成部分任务概念重叠。

## 2.2 主实验方法

四个在线控制方法：

1. `mpc_only`
2. `action_residual`
3. `planning_residual`
4. `zprl_style`

附加工程 baseline：

5. `lewm_mpc`

每个方法默认：

- 5 个随机种子：0, 1, 2, 3, 4；
- 在线方法：100000 environment steps；
- 最终 deterministic evaluation：5 episodes / seed；
- 结果报告：mean ± std、95% CI、相对 MPC 提升率、跨 seed 稳定性。

因此主在线实验规模为：

```text
4 tasks × 4 methods × 5 seeds = 80 runs
```

加上 LeWM 工程 baseline：

```text
4 tasks × 1 method × 5 seeds = 20 runs
```

完整工程对比共：

```text
100 runs
```

---

# 3. 目录结构说明

```text
MPC_RL_ZPRL_RTX5090/
├── configs/
│   └── rtx5090/
│       ├── hopper.yaml
│       ├── walker2d.yaml
│       ├── halfcheetah.yaml
│       ├── reacher.yaml
│       ├── lewm_hopper.yaml
│       ├── lewm_walker2d.yaml
│       ├── lewm_halfcheetah.yaml
│       ├── lewm_reacher.yaml
│       ├── paper_suite.yaml
│       └── ablation_suite.yaml
│
├── mpcrl/
│   ├── envs.py
│   ├── world_model.py
│   ├── cem.py
│   ├── sac.py
│   ├── replay.py
│   ├── gate.py
│   ├── metrics.py
│   ├── zprl_style.py
│   ├── lewm.py
│   ├── lewm_planner.py
│   └── plotting.py
│
├── experiments/
│   ├── train.py
│   ├── train_zprl_style.py
│   ├── train_lewm.py
│   ├── run_suite.py
│   ├── run_zprl_ablation.py
│   ├── aggregate.py
│   ├── summarize_eval.py
│   ├── plot_results.py
│   └── plot_ablation.py
│
├── scripts/
│   ├── run_paper_suite_full.sh
│   ├── run_zprl_ablations.sh
│   ├── run_walker2d_full.sh
│   ├── analyze_walker2d_full.sh
│   └── run_lewm_official_reference.sh
│
├── docs/
│   ├── LEWM_REPRODUCTION.md
│   ├── CODE_AND_EXPERIMENT_GUIDE.md
│   └── ...
│
├── runs/
└── results/
```

---

# 4. 核心方法流程

## 4.1 MPC-only

MPC-only 的基本流程为：

```text
当前状态 s_t
   ↓
Ensemble World Model
   ↓
CEM 生成多组动作序列
   ↓
世界模型滚动预测 future state/reward
   ↓
选择最高预测回报动作序列
   ↓
执行第一个动作 a_t
   ↓
进入 s_(t+1)，重新规划
```

对应主要代码：

- `mpcrl/world_model.py`
- `mpcrl/cem.py`
- `experiments/train.py`

MPC-only 是所有方法的基本参考控制器。

---

## 4.2 Ensemble World Model

`mpcrl/world_model.py` 使用 ensemble dynamics model。

每个成员输入：

```text
[s_t, a_t]
```

预测：

```text
Δs_t = s_(t+1) - s_t
r_t
```

代码中对以下变量做标准化：

- observation；
- action；
- state delta；
- reward。

多个 ensemble member 的均值作为预测结果，成员间方差作为 epistemic uncertainty 的近似。

MPC rollout 中使用：

```text
predicted reward - uncertainty_penalty × uncertainty
```

从而避免规划器过度利用世界模型不可靠区域。

---

# 5. CEM Planner

`mpcrl/cem.py` 实现 Cross-Entropy Method。

设规划长度为 H，每次采样 N 条动作序列：

```text
A^(i) = [a_t, a_(t+1), ..., a_(t+H-1)]
```

使用世界模型计算：

```text
J(A) = Σ γ^k [r_hat - λ_u × uncertainty]
```

然后选出 top-K elite sequences，更新动作分布均值和标准差。

主要配置项：

```yaml
mpc:
  horizon: 15
  candidates: 2048
  elites: 128
  iterations: 5
  alpha: 0.2
  init_std: 0.6
  min_std: 0.05
  discount: 0.99
```

HalfCheetah 等任务可使用更长 horizon 或更多 candidates。

---

# 6. Action Residual

Action Residual 的基本形式：

```text
a = clip(a_MPC + g × scale × Δa)
```

其中：

- `a_MPC`：MPC 给出的基础动作；
- `Δa`：SAC residual policy 输出；
- `scale`：残差幅值；
- `g`：gate / adaptive gate。

该方法直接在 action space 中修正 MPC。

优点：实现简单。

缺点：

- 动作空间残差可能破坏 MPC 的时序一致性；
- residual magnitude 大时容易造成动作抖动；
- 长期结构信息利用不足。

---

# 7. Planning Residual

Planning Residual 不直接简单替换 MPC，而是在规划或规划相关控制量上进行残差修正。

项目中它主要用于验证：

> “将 RL 修正嵌入规划层”是否优于纯 action-space residual。

对应主入口仍为：

```text
experiments/train.py
```

方法参数通过：

```text
--method planning_residual
```

选择。

---

# 8. ZPRL-style 方法

ZPRL-style 是本项目主要研究对象。

总体结构：

```text
MPC expert
   ↓
Behavior Cloning
   ↓
Bottleneck Base Policy
   ↓
latent z
   ↓
SAC residual policy 输出 Δz
   ↓
z' = z + scale × ramp × Δz
   ↓
Base Decoder
   ↓
最终动作 a
```

核心思想不是直接修正动作，而是修正低维 latent representation。

---

## 8.1 Base Policy

`mpcrl/zprl_style.py` 中的：

```python
BottleneckBasePolicy
```

包含：

```text
observation
   ↓
normalize
   ↓
encoder
   ↓
latent z
   ↓
decoder
   ↓
action
```

Base Policy 先通过 MPC expert trajectory 进行 Behavior Cloning。

---

## 8.2 Observation Normalization

Base Policy 保存：

```text
obs_mean
obs_std
```

输入使用：

```text
(s - mean) / std
```

并进行有限范围 clipping。

这是因为不同 MuJoCo state 维度的数值尺度差异较大，如果不做标准化，BC 网络容易优先拟合大尺度状态变量。

---

## 8.3 MPC → BC

训练流程首先让 MPC 产生 expert action：

```text
state s_t
  ↓
MPC
  ↓
a*_t
```

组成：

```text
D = {(s_t, a*_t)}
```

然后训练 Base Policy：

```text
L_BC = MSE(π_base(s), a*)
```

日志中的：

```text
bc_dataset_mse
```

用于判断 Base Policy 是否成功蒸馏 MPC。

---

# 9. DAgger 修正

单纯 BC 存在 covariate shift：

训练时看到的是 MPC expert state distribution，但测试时 Base Policy 会访问自己的状态分布。

因此加入 DAgger-style refinement：

```text
Base Policy rollout
   ↓
访问 learner states
   ↓
对这些 states 调 MPC
   ↓
获得 expert labels
   ↓
加入数据集
   ↓
重新训练 Base Policy
```

配置：

```yaml
zprl_style:
  dagger_rounds: 2
  dagger_steps_per_round: 1000
  dagger_refit_epochs: 25
```

HalfCheetah 使用更强的 DAgger 设置，因为其前期 BC 误差和 covariate shift 更明显。

---

# 10. Latent Residual SAC

Base Policy 固定后，在 latent space 训练 residual SAC。

设：

```text
z = Encoder(s)
```

Residual actor 输出：

```text
Δz = π_res(context)
```

最终 latent：

```text
z' = z + α × Δz
```

其中：

```text
α = residual_scale × residual_ramp
```

最终动作：

```text
a = Decoder(z')
```

这样 residual policy 不需要重新学习完整控制器，而只需要修正 Base Policy 的 latent representation。

---

# 11. Residual Ramp

为了避免刚开始训练的随机 SAC residual 立即破坏已经可用的 Base Policy，项目设置纯 Base Policy 阶段和渐进 ramp。

例如：

```yaml
residual_start_steps: 5000
residual_ramp_steps: 20000
```

训练过程：

```text
0 ~ 5000:
    residual ≈ 0

5000 ~ 25000:
    residual 逐渐增大

>25000:
    residual 完全开启
```

这个设计对 HalfCheetah 尤其重要。

---

# 12. Consistency Regularization

Residual SAC 中加入 consistency coefficient：

```yaml
consistency_coef: 0.03
```

作用是避免 residual policy 在相近 context 下输出剧烈变化，使 latent correction 更平滑。

消融实验中：

```text
no_consistency
```

将该项设为 0。

---

# 13. LeWorldModel 工程复现

核心实现：

```text
mpcrl/lewm.py
```

包括：

```text
PixelEncoder
ActionPredictor
SIGReg
LeWorldModel
RewardProbe
```

LeWM 主训练目标：

```text
L = L_pred + λ × L_SIGReg
```

其中：

```text
L_pred = MSE(z_hat_(t+1), z_(t+1))
```

SIGReg 用于约束 latent distribution 接近高斯分布。

---

# 14. LeWM MuJoCo 适配

`experiments/train_lewm.py` 流程：

```text
MuJoCo render RGB
   ↓
收集 frame/action/frame_next/reward
   ↓
训练 LeWM
   ↓
冻结 LeWM
   ↓
训练 Reward Probe
   ↓
latent CEM planning
   ↓
deterministic evaluation
```

这里 Reward Probe 是为了把 LeWM latent rollout 映射到 MuJoCo reward。

它属于工程适配，不属于 LeWM 官方原始两项 loss。

最新版会输出：

```text
episodes.csv
eval.csv
final.pt
config.yaml
```

因此最终 deterministic evaluation 可以和其他方法使用同一统计脚本。

---

# 15. LeWM 官方四任务

LeWM 官方核心 benchmark：

```text
TwoRoom
PushT
Cube
Reacher
```

官方 benchmark 不应和本仓库 MuJoCo 工程 benchmark 混淆。

官方 reference 验证脚本：

```bash
LEWM_OFFICIAL_DIR=/path/to/LeWorldModel \
STABLEWM_HOME=/path/to/.stable-wm \
bash scripts/run_lewm_official_reference.sh
```

该脚本只核验官方 LeWM checkpoint。

---

# 16. 主实验一键运行

更新代码：

```bash
git checkout main
git pull
```

后台完整运行：

```bash
nohup bash scripts/run_paper_suite_full.sh \
  > paper_suite_launcher.log 2>&1 &
```

查看进度：

```bash
tail -f paper_suite_launcher.log
```

完整任务为：

```text
Hopper-v5
Walker2d-v5
HalfCheetah-v5
Reacher-v5
```

每个任务：

```text
mpc_only
Action Residual
Planning Residual
ZPRL
LeWM-MPC
```

---

# 17. 主实验断点续跑

脚本会生成：

```text
TAG=YYYYMMDD_HHMMSS
```

若中断时已完成前 N 个 job，可以使用：

```bash
TAG=原TAG START_JOB=N+1 \
nohup bash scripts/run_paper_suite_full.sh \
  > paper_suite_resume.log 2>&1 &
```

注意：必须复用同一个 TAG，才能继续写入同一个实验目录。

---

# 18. 主实验输出目录

例如：

```text
runs/paper_suite_20260908_120000/
results/paper_suite_20260908_120000/
```

结果目录包含：

```text
main_summary.csv
main_summary_per_seed.csv
main_summary_completeness.csv
final_eval/
    eval_all.csv
    eval_per_seed.csv
    comparison_eval.csv
    eval_coverage.csv
figures/
environment.txt
paper_suite.log
```

---

# 19. 最终核心性能指标

## 19.1 Deterministic Return

每个 seed 使用固定 deterministic actor / deterministic planning 配置进行最终评估。

论文推荐报告：

```text
mean ± std
```

其中 std 是 across seeds 标准差，而不是单 seed 内 5 episodes 的标准差。

---

## 19.2 95% Confidence Interval

统计脚本输出：

```text
eval_return_ci95_half_width
```

形式：

```text
mean ± CI95
```

5 seeds 条件下 CI 主要用于描述不确定性，不应替代原始 std。

---

## 19.3 Relative Improvement

相对 MPC：

```text
(R_method - R_MPC) / |R_MPC| × 100%
```

对应：

```text
vs_mpc_pct
```

ZPRL 还会计算相对最佳非 ZPRL baseline：

```text
vs_best_non_zprl_pct
```

---

# 20. 动作平滑性

项目记录：

```text
action_d1
action_d2
```

`action_d1`：一阶动作变化；

```text
D1 ≈ mean(||a_t - a_(t-1)||)
```

`action_d2`：二阶动作变化，反映控制 jerk / 抖动趋势。

对于连续控制任务：

```text
D1 越低 → 动作越平滑
D2 越低 → 动作变化趋势越平稳
```

如果 ZPRL 在 Return 提升的同时 D1/D2 更低，则说明它不仅追求奖励，而且控制质量更好。

---

# 21. Residual 指标

主要日志：

```text
residual_norm
effective_residual_norm
gate
residual_ramp
```

其中：

```text
residual_norm
```

表示 residual actor 原始输出范数。

```text
effective_residual_norm
```

表示实际作用到 latent/action 的残差大小。

后者更适合论文分析。

---

# 22. World Model 指标

日志：

```text
prediction_mse
uncertainty
```

用于分析：

- dynamics model 是否收敛；
- 是否存在后期 prediction error 发散；
- uncertainty penalty 是否有效限制 OOD planning。

---

# 23. MPC 计算时间

日志：

```text
mpc_ms
```

论文中建议报告：

```text
mean MPC planning time
```

必要时还可计算：

```text
P50
P95
```

用于评价实时性。

---

# 24. ZPRL 消融实验设计

消融配置：

```text
configs/rtx5090/ablation_suite.yaml
```

默认使用两个代表性任务：

```text
Walker2d-v5
HalfCheetah-v5
```

随机种子：

```text
0, 1, 2
```

每组默认：

```text
50000 steps
5 deterministic eval episodes
```

消融 variants：

| Variant | 含义 |
|---|---|
| `full` | 完整 ZPRL |
| `no_dagger` | 去除 DAgger |
| `no_consistency` | 去除一致性约束 |
| `no_latent_residual` | residual scale=0，仅 Base Policy |
| `latent_dim_16` | latent 维度 16 |
| `latent_dim_64` | latent 维度 64 |
| `residual_scale_half` | residual scale 减半 |
| `residual_scale_1p5x` | residual scale 增大 1.5 倍 |

---

# 25. 消融实验一键运行

```bash
nohup bash scripts/run_zprl_ablations.sh \
  > zprl_ablation_launcher.log 2>&1 &
```

查看：

```bash
tail -f zprl_ablation_launcher.log
```

输出：

```text
runs/zprl_ablation_TAG/
results/zprl_ablation_TAG/
```

其中：

```text
figures/ablation_per_seed.csv
figures/ablation_summary.csv
figures/ablation_coverage.csv
```

以及：

```text
*__ablation__eval_return.png/pdf
*__ablation__action_d1.png/pdf
*__ablation__effective_residual_norm.png/pdf
```

---

# 26. 消融实验如何解释

## 去掉 DAgger

如果：

```text
full > no_dagger
```

说明 learner-state relabeling 能缓解 BC covariate shift。

## 去掉 consistency

如果：

```text
full return 相近，但 action_d1 更低
```

说明 consistency 的主要作用可能是提高控制平滑性，而不是直接增加回报。

## no_latent_residual

该组近似回答：

> “只用 MPC 蒸馏出的 Base Policy，不使用 latent residual 是否足够？”

若 full 显著高于 no_latent_residual，则 residual learning 是必要组件。

## latent_dim

比较 16、默认值和 64：

- 太小：表达能力不足；
- 太大：residual search space 增大、训练不稳定；
- 中间维度可能获得最好性能/稳定性折中。

## residual_scale

比较 0.5×、1.0×、1.5×：

用于验证最终性能是否对 residual magnitude 敏感。

---

# 27. 论文建议主表

建议主结果表：

| Task | MPC | Action Residual | Planning Residual | LeWM-MPC | ZPRL |
|---|---:|---:|---:|---:|---:|
| Hopper | mean±std | ... | ... | ... | **...** |
| Walker2d | ... | ... | ... | ... | **...** |
| HalfCheetah | ... | ... | ... | ... | **...** |
| Reacher | ... | ... | ... | ... | **...** |

ZPRL 不需要在所有任务上出现非常夸张的倍数提升。

更加可信的论文结果通常是：

- 多数任务稳定优于 baseline；
- 部分任务接近最佳 baseline；
- across-seed 方差合理；
- 动作平滑性/样本效率等辅助指标体现额外优势。

---

# 28. 论文建议消融表

| Variant | Walker2d Return | HalfCheetah Return | Action D1 | Residual Norm |
|---|---:|---:|---:|---:|
| Full | **...** | **...** | ... | ... |
| w/o DAgger | ... | ... | ... | ... |
| w/o Consistency | ... | ... | ... | ... |
| w/o Latent Residual | ... | ... | ... | 0 |
| latent=16 | ... | ... | ... | ... |
| latent=64 | ... | ... | ... | ... |
| scale×0.5 | ... | ... | ... | ... |
| scale×1.5 | ... | ... | ... | ... |

---

# 29. 论文建议图

正文建议保留：

1. Episode Return vs Environment Steps；
2. Deterministic Final Evaluation Return；
3. Action D1 / D2；
4. Ablation Return；
5. Effective Residual Norm；
6. 可选 Prediction MSE。

补充材料可放：

- gate；
- residual raw norm；
- MPC planning latency；
- seed-level detailed curves。

---

# 30. 绘图统计口径

`experiments/plot_results.py` 使用：

```text
每个 seed 独立平滑
        ↓
按 environment step 对齐
        ↓
计算 across-seed mean
        ↓
计算 across-seed std
```

而不是先把多个 seed 拼接再 rolling。

论文训练曲线建议：

```text
mean curve + std shaded region
```

---

# 31. 为什么最终性能优先用 eval.csv

训练 episode 含：

- exploration noise；
- ramp；
- stochastic actor；
- online learning transition。

因此训练曲线不是最终策略性能。

最终主表必须优先使用：

```text
eval.csv
```

并且使用 deterministic evaluation。

---

# 32. 完整性检查

最终统计工具支持：

```text
--expected-seeds
--expected-methods
--require-complete
--require-eval
--min-steps
```

目的：

- 防止 10k/30k diagnostic run 混进 100k 正式结果；
- 防止只跑一个 seed 就生成“最终论文图”；
- 防止某个方法缺 seed；
- 防止没有 deterministic eval 时用训练 tail 冒充最终结果。

---

# 33. 可复现性

正式实验目录会记录：

```text
environment.txt
config.yaml
final.pt
episodes.csv
eval.csv
```

一键脚本还会记录：

```text
GIT_COMMIT
Python version
GPU
Driver version
```

因此论文复现应固定：

- commit SHA；
- config；
- seed；
- training budget；
- eval seeds；
- final eval episodes。

---

# 34. RTX 5090 相关设置

配置中：

```yaml
hardware:
  profile: RTX 5090 32GB
  precision: bf16
  allow_tf32: true
  cudnn_benchmark: true
  matmul_precision: high
```

目的：

- bf16 降低显存和计算成本；
- TF32 加速矩阵计算；
- 适合大 batch world model / SAC。

如果出现数值异常，可临时将：

```yaml
precision: fp32
```

做对照排查。

---

# 35. 推荐完整实验执行顺序

建议按以下顺序执行：

```text
Step 1  单元测试
Step 2  4-task main comparison
Step 3  检查 5-seed completeness
Step 4  deterministic final evaluation
Step 5  主结果统计/绘图
Step 6  ZPRL ablation
Step 7  ablation plotting
Step 8  official LeWM reference verification
Step 9  整理论文主表和图
```

不要在正式 5-seed 主实验结束后继续根据单个 seed 反复调参，否则会产生 test-set tuning / seed overfitting 风险。

---

# 36. 当前实验口径建议

建议论文将实验分成三部分：

## Experiment A：统一连续控制主实验

任务：

```text
Hopper / Walker2d / HalfCheetah / Reacher
```

比较：

```text
MPC
Action Residual
Planning Residual
LeWM-MPC engineering adaptation
ZPRL
```

## Experiment B：ZPRL 消融

重点回答：

```text
DAgger 是否必要？
latent residual 是否必要？
consistency 是否有效？
latent dimension 是否敏感？
residual scale 是否敏感？
```

## Experiment C：LeWM 官方参考核验

任务：

```text
TwoRoom / PushT / Cube / Reacher
```

只用于核验 LeWM 官方 checkpoint / benchmark，不将它与本项目 state-space ZPRL 数值直接混表。

---

# 37. 当前已完善内容

最新版已经补齐：

- 四任务 MuJoCo 主实验；
- Reacher-v5 配置；
- Reacher LeWM 工程配置；
- LeWM deterministic `eval.csv`；
- 4 methods × 5 seeds 正式统计；
- LeWM-MPC 工程 baseline；
- ZPRL multi-seed ablation；
- ablation coverage check；
- ablation PNG/PDF；
- mean ± std；
- 95% CI；
- action D1/D2；
- residual norm；
- prediction error；
- one-click main suite；
- one-click ablation suite；
- LeWM official four-task reference helper；
- 严格区分官方 benchmark 与工程适配实验。

---

# 38. 当前仍应谨慎表述的部分

以下内容不能在没有额外实现/实验的情况下声称已经完成：

1. “ZPRL 已在 LeWM 官方 TwoRoom/PushT/Cube/Reacher 四任务上完成严格对比”；
2. “本仓库 LeWM-MPC 数值等于 LeWM 原论文官方数值”；
3. “Reacher-v5 与 stable-worldmodel Reacher 完全相同”；
4. “所有任务均已达到统计显著性”——必须等实际 5-seed 结果出来后判断。

当前正确说法是：

> 本项目已经具备完整的四任务统一 MuJoCo 主实验、LeWM 工程 baseline、ZPRL 消融、统计与可视化链路；官方 LeWM 四任务通过独立 reference 脚本核验，严格跨接口的 ZPRL pixel-goal benchmark 迁移仍应作为后续扩展，而不是用 MuJoCo 结果替代。

---

# 39. 最推荐的执行命令

## 主实验

```bash
git checkout main
git pull
nohup bash scripts/run_paper_suite_full.sh > paper_suite.log 2>&1 &
tail -f paper_suite.log
```

## 消融实验

```bash
nohup bash scripts/run_zprl_ablations.sh > zprl_ablation.log 2>&1 &
tail -f zprl_ablation.log
```

## LeWM 官方 reference

```bash
LEWM_OFFICIAL_DIR=/path/to/LeWorldModel \
STABLEWM_HOME=/path/to/.stable-wm \
bash scripts/run_lewm_official_reference.sh
```

---

# 40. 最终论文数据应从哪里取

主表：

```text
results/paper_suite_TAG/final_eval/comparison_eval.csv
```

每 seed：

```text
results/paper_suite_TAG/final_eval/eval_per_seed.csv
```

完整性：

```text
results/paper_suite_TAG/main_summary_completeness.csv
results/paper_suite_TAG/final_eval/eval_coverage.csv
```

训练曲线：

```text
results/paper_suite_TAG/figures/
```

消融：

```text
results/zprl_ablation_TAG/figures/ablation_summary.csv
```

论文中所有最终数字建议由这些 CSV 自动生成，避免手工抄写造成误差。
