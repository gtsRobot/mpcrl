# LeWM-MPC + RL 论文主线与实验协议

> 仓库：`it0farmer/MPC_RL_ZPRL_RTX5090`
>
> 当前推荐分支：`main`
>
> 核心研究问题：**在不修改 LeWM 表征学习和世界模型的前提下，在 MPC 执行动作阶段加入一个小幅、受约束的 RL 残差策略，能否稳定提高规划控制性能？**

---

## 1. 论文主线重新定义

本项目后续论文不再把“ZPRL 与若干 MuJoCo 控制器比较”作为唯一主线，而把更容易解释、也更贴近 LeWM 原工作的问题设为：

**LeWM-MPC baseline**

\[
a_t^{\mathrm{MPC}} = \operatorname{CEM}(z_t; f_{\theta}, r_{\psi})
\]

与

**LeWM-MPC+RL（本文方法）**

\[
a_t = \operatorname{clip}\left(
 a_t^{\mathrm{MPC}} + \rho_t\,\alpha\,\frac{a_{\max}-a_{\min}}{2}\odot
 \delta_{\phi}(c_t),
 a_{\min}, a_{\max}
\right)
\]

进行直接比较。

其中：

- `LeWM`、reward probe、CEM planner 在同一 seed 的两种方法之间完全共享；
- LeWM 在 RL 训练阶段冻结；
- RL 不重新学习世界模型；
- RL 只学习一个 bounded residual；
- `alpha` 控制 RL 最大干预强度；
- `rho_t` 是 residual ramp，避免训练初期突然破坏 MPC；
- `c_t=[z_t, normalize(a_mpc)]`，即 RL 同时看到 LeWM latent 和当前 MPC 动作。

因此实验的因果问题非常清楚：

> **在相同 LeWM、相同 MPC、相同任务、相同 seed 和相同 final-eval protocol 下，仅加入 RL residual 后性能是否改善？**

---

## 2. 为什么这样比原来的工程更适合论文

原工程包含 `mpc_only / action_residual / planning_residual / zprl_style / lewm_mpc` 等多种方法，它适合做宽泛 benchmark，但不够直接回答“在 LeWM 的 MPC 阶段加入 RL 是否有效”。

新的 paired protocol 每个 `(task, seed)` 只训练 **一个共享 LeWM**，然后：

1. 先直接测试 `LeWM-MPC`；
2. 冻结 LeWM 与 reward probe；
3. 在 MPC 第一动作上训练 SAC residual；
4. 用与 baseline 完全相同的 evaluation seeds 测试 `LeWM-MPC+RL`；
5. 做同 seed paired difference。

这比两个模型分别训练后再比较更严格，因为世界模型质量差异被控制掉了。

---

## 3. 当前工程中的两层 benchmark

### 3.1 直接机制验证：统一 MuJoCo LeWM 工程适配

当前可一键运行：

- Hopper-v5
- Walker2d-v5
- HalfCheetah-v5
- Reacher-v5

每个任务：

- 5 train seeds；
- 共享 LeWM 数据集与模型；
- LeWM-MPC baseline；
- LeWM-MPC+RL proposed method；
- 每个方法 10 个 deterministic final-eval episodes；
- RL 默认训练 50k environment steps。

这一层用于验证**“RL 加入 MPC action stage”这一机制本身**。

### 3.2 LeWM 官方 benchmark

LeWM 官方实验使用的核心任务是：

- Two-Room
- PushT
- OGB-Cube
- Reacher

官方代码基于 `stable-worldmodel`，以 goal-conditioned pixel planning 和 success rate 为主要评估口径。

**必须注意：MuJoCo Reacher-v5 不等同于 LeWM 官方 Reacher。**

因此论文若要写“在 LeWM 官方 benchmark 上超过 LeWM”，还需要把这里的 residual-RL controller 接到官方 `stable-worldmodel.WorldModelPolicy` 上，并按官方 start/goal dataset protocol 计算 success rate。当前本仓库已经保留官方 LeWM reference 入口，但不能把 MuJoCo 工程结果冒充官方 benchmark 数值。

---

## 4. 新增代码结构

### 4.1 `mpcrl/lewm_rl.py`

提供 MPC+RL 控制所需的公共函数：

- action normalization；
- bounded residual 合成；
- residual ramp；
- LeWM latent effective-rank 计算。

有效秩定义采用 entropy effective rank：

\[
p_i=\frac{\sigma_i}{\sum_j \sigma_j}
\]

\[
r_{\mathrm{eff}}=\exp\left(-\sum_i p_i\log p_i\right)
\]

其中 `sigma_i` 为中心化 latent matrix 的奇异值。

由于本方法发生在 MPC/control stage，LeWM 是冻结的，所以同一个 paired run 中 `LeWM-MPC` 与 `LeWM-MPC+RL` 的 effective rank 理论上应一致。**这里的 effective rank 是“保持 LeWM 表征不变”的控制变量/诊断量，而不是本文方法必须提高的主指标。**

### 4.2 `experiments/train_lewm_mpc_rl.py`

这是当前论文主实验最关键的训练入口。

单个 seed 完整执行五个阶段：

1. 收集一份共享 pixel/action/reward dataset；
2. 训练一份 LeWM + reward probe；
3. 测试冻结的 `LeWM-MPC`；
4. 在相同 frozen LeWM + CEM 上训练 residual SAC；
5. 使用同一组 eval seeds 测试 `LeWM-MPC+RL`。

输出两个 run directory：

```text
<env>__lewm_mpc__seedX__pairTAG/
<env>__lewm_mpc_rl__seedX__pairTAG/
```

二者共享同一个 LeWM 权重来源。

### 4.3 `experiments/run_lewm_mpc_rl_suite.py`

顺序执行：

```text
4 tasks x 5 seeds = 20 paired jobs
```

每个 paired job 同时生成 baseline 和 proposed result，所以最终得到 40 个方法级 run directory。

### 4.4 `experiments/summarize_lewm_mpc_rl.py`

输出：

- `direct_per_seed.csv`
- `direct_summary.csv`
- `paired_improvements.csv`
- `paired_summary.csv`
- `direct_coverage.csv`

其中最重要的是 paired comparison：

\[
\Delta R_s = R^{\mathrm{MPC+RL}}_s-R^{\mathrm{MPC}}_s
\]

因为两者来自同 seed、同一 LeWM，所以这个差值比普通跨模型平均值更有解释力。

### 4.5 `experiments/plot_lewm_mpc_rl.py`

自动生成：

- final return bar plot：LeWM-MPC vs LeWM-MPC+RL；
- success-rate plot（当任务定义了 success 时）；
- paired return improvement per seed；
- RL learning curve；
- shared LeWM effective-rank diagnostic；
- 300 dpi PNG + PDF。

所有论文图使用 ASCII/English labels，避免 Ubuntu 缺少中文字体造成乱码。

---

## 5. RL residual 的具体结构

### 5.1 Context

默认：

\[
c_t=[z_t, \tilde a_t^{MPC}]
\]

其中：

- `z_t`：冻结 LeWM encoder 对当前 frame 的 latent；
- `a_mpc`：CEM 当前规划得到的第一动作；
- `normalize(a_mpc)`：映射到约 `[-1,1]`。

这样 RL 学习的是：

> “当前世界状态 + MPC 已经准备执行的动作下，还应该修正多少？”

而不是重新从头学习一个控制器。

### 5.2 SAC residual

Actor 输出：

\[
\delta_t\in[-1,1]^{d_a}
\]

实际修正量：

\[
\Delta a_t=\rho_t\alpha\frac{a_{max}-a_{min}}{2}\odot\delta_t
\]

其中 `alpha` 默认仅为 0.08~0.12，避免 RL 完全覆盖 MPC。

### 5.3 Consistency regularization

SAC actor loss 额外包含 residual magnitude penalty：

\[
L_{actor}=L_{SAC}+\lambda_c\mathbb E[\|\delta_t\|_2^2]
\]

目的不是限制 RL 不学习，而是鼓励：

> MPC 已经正确时少改；MPC 存在系统误差时再进行必要补偿。

### 5.4 Ramp

训练前期：

\[
\rho_t\approx0
\]

随后逐渐增加到：

\[
\rho_t=1
\]

避免未训练的随机 residual 在一开始破坏 MPC trajectory distribution。

---

## 6. 四任务默认参数

| Task | residual scale | RL steps | ramp | LeWM latent |
|---|---:|---:|---:|---:|
| Hopper-v5 | 0.08 | 50k | 10k | 256 |
| Walker2d-v5 | 0.08 | 50k | 10k | 256 |
| HalfCheetah-v5 | 0.10 | 50k | 12k | 256 |
| Reacher-v5 | 0.12 | 50k | 10k | 256 |

公共 SAC 参数：

```yaml
batch_size: 512
hidden_dim: 512
lr: 3e-4
gamma: 0.99
tau: 0.005
init_alpha: 0.1
target_entropy_scale: 0.7
consistency_coef: 0.05
updates_per_step: 1
```

---

## 7. 一键完整实验

更新代码：

```bash
git checkout main
git pull
```

前台：

```bash
bash scripts/run_lewm_mpc_rl_full.sh
```

推荐后台：

```bash
nohup bash scripts/run_lewm_mpc_rl_full.sh > lewm_mpc_rl_full.log 2>&1 &
tail -f lewm_mpc_rl_full.log
```

中途断开后，先从日志找到原 TAG 和已完成 job，再恢复：

```bash
TAG=<原TAG> START_JOB=<下一任务编号> \
nohup bash scripts/run_lewm_mpc_rl_full.sh > lewm_mpc_rl_resume.log 2>&1 &
```

---

## 8. 最终结果目录

```text
runs/lewm_mpc_rl_<TAG>/
results/lewm_mpc_rl_<TAG>/
├── environment.txt
├── full_run.log
├── summary/
│   ├── direct_per_seed.csv
│   ├── direct_summary.csv
│   ├── paired_improvements.csv
│   ├── paired_summary.csv
│   └── direct_coverage.csv
└── figures/
    ├── Hopper-v5__return.png/pdf
    ├── Walker2d-v5__return.png/pdf
    ├── HalfCheetah-v5__return.png/pdf
    ├── Reacher-v5__return.png/pdf
    ├── *__paired_delta.png/pdf
    ├── *__rl_learning_curve.png/pdf
    └── shared_lewm_effective_rank.png/pdf
```

---

## 9. 论文应重点报告哪些指标

### 9.1 第一主指标：Success Rate

在 LeWM 官方 goal-conditioned benchmark 上应以 success rate 为核心：

\[
SuccessRate=\frac{N_{success}}{N_{eval}}\times100\%
\]

这与 LeWM 原论文展示方式最直接对应。

对于当前 MuJoCo engineering benchmark，如果环境没有原生 success 标志，代码不会人为制造 success rate，而是保留 NaN；此时使用 return / episode length 做主要控制指标。若论文需要 success rate，应优先在官方 LeWM 四任务上报告，而不是随意为 MuJoCo 人工设阈值。

### 9.2 第二主指标：Return / Goal Cost

统一 MuJoCo 验证使用 deterministic final return：

- seed 内先对 eval episodes 求均值；
- 再对 5 seeds 求 `mean ± std`；
- 同时计算 95% CI。

### 9.3 Paired improvement

这是本实验设计最重要的统计量之一：

\[
\Delta R_s=R_{RL,s}-R_{MPC,s}
\]

以及：

\[
Improvement_s=\frac{R_{RL,s}-R_{MPC,s}}{|R_{MPC,s}|}\times100\%
\]

建议论文主表同时报告普通 `mean±std` 和 paired delta。

### 9.4 Action smoothness

\[
D_1=\frac{1}{T-1}\sum_{t=2}^{T}\|a_t-a_{t-1}\|_2
\]

\[
D_2=\frac{1}{T-2}\sum_{t=3}^{T}\|a_t-2a_{t-1}+a_{t-2}\|_2
\]

用于证明 RL 提升成功率/return 时没有通过剧烈动作抖动取得优势。

### 9.5 Planning latency

`mpc_ms` 用于确认加入 residual actor 后主计算瓶颈仍然是 CEM，而 RL 网络前向不会显著增加 planning latency。

### 9.6 Effective rank

只作为 shared LeWM representation sanity check。

因为世界模型冻结，所以如果论文写“RL 使 effective rank 提升”反而不符合本方法定义。正确表述应是：

> 在保持 LeWM latent representation 不变的条件下，仅修改 MPC action execution policy 即可提升 control success/performance。

这个因果表述比硬追求 rank 上升更适合当前创新点。

---

## 10. 公平性要求

论文最终实验必须满足：

1. baseline 和 proposed 方法共享同一 LeWM；
2. 共享同一 reward probe；
3. 共享 CEM horizon/candidates/elites/iterations；
4. 同 seed pair 使用相同 final-eval seeds；
5. LeWM 在 RL 阶段严格冻结；
6. 不根据最终 test seed 单独调 residual scale；
7. 所有正式结果至少 5 train seeds；
8. 不以单个最好 seed 作为论文结果；
9. MuJoCo 与官方 LeWM benchmark 的数字分开报告。

---

## 11. 论文推荐实验章节结构

### 4.1 Experimental Setup

说明 LeWM、CEM、RL residual、任务、训练预算、随机种子与硬件。

### 4.2 Direct Comparison with LeWM-MPC

核心表：

| Environment | LeWM-MPC | LeWM-MPC+RL | Paired Delta | Improvement |
|---|---:|---:|---:|---:|
| Two-Room / engineering task | ... | ... | ... | ... |
| PushT / engineering task | ... | ... | ... | ... |
| Cube / engineering task | ... | ... | ... | ... |
| Reacher / engineering task | ... | ... | ... | ... |

官方 benchmark 应填 success rate；工程 MuJoCo 可填 return。

### 4.3 Learning Curves

展示 residual RL 训练过程中 performance 从 LeWM baseline 附近向上提升。

### 4.4 Ablation Study

后续建议围绕本文创新本身做：

- no residual RL = LeWM baseline；
- residual scale；
- no ramp；
- no residual magnitude regularization；
- latent-only context vs latent+MPC-action context；
- RL training budget。

### 4.5 Representation and Control Diagnostics

报告：

- shared LeWM effective rank；
- action D1/D2；
- MPC latency；
- residual magnitude。

强调**世界模型未改变，改善来自控制阶段**。

---

## 12. 当前结论

当前最新版代码已经把论文最核心的实验问题改成了：

```text
same LeWM
same data
same reward probe
same CEM
same seed
same final evaluation
        |
        +-- LeWM-MPC
        |
        +-- LeWM-MPC + bounded SAC residual
```

因此只要最终 5-seed paired results 显示：

- success rate / return 稳定提高；
- 多数 seed 的 paired delta 为正；
- 方差没有显著恶化；
- action smoothness 没有明显崩坏；
- effective rank 保持一致；

就可以非常直观地支撑论文核心结论：

> **在不重新训练 LeWM 表征模型的情况下，在 MPC action stage 引入轻量强化学习残差，可以补偿模型预测与有限时域规划误差，提高规划控制性能。**
