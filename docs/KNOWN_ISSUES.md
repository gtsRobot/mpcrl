# 2026-09-30 修复说明：哪些旧结果不再可比

本文件记录 2026-09-30 这一轮修复改变了什么，以及为什么 `docs/RESULT_DRIVEN_TUNING_20260831.md`
和 `docs/RESULTS_ANALYSIS_20260917.md` 里的数字不能和新运行的数字放在同一张表里。

---

## 0. 先说最重要的一件事：旧结果本来就测不出它声称的效应

两份文档对同一组实验**符号都对不上**：

| Env | 0831（训练尾部 20 episode） | 0917（确定性 eval） |
|---|---|---|
| HalfCheetah | planning **+13.14%** (2594 vs 2293) | planning **−4.76%** (2347 vs 2465) |
| Walker2d | planning **−5.68%** (208 vs 220) | planning **+5.08%** (225 vs 214) |
| Hopper | −7.82% | −15.33%（唯一符号稳定） |

唯一记了 std 的表里，HalfCheetah `mpc_only` 是 **2293.05 ± 1054.42**（n=5，SEM≈472）。
0917 那个 −117 的"差距"只有 **0.11σ**。同期 `action_residual` 在两张表之间漂了 **−632.67**，
是被声称效应的 **5.4 倍**。0917 表**完全没有记 std**。

而且 **n=5 时 Wilcoxon signed-rank 两侧最小可达 p = 2/2⁵ = 0.0625** —— 5 个 seed 的研究
**在数学上不可能达到 p<0.05**，无论效应多干净。n=6 → 0.031，n=8 → 0.0078。

所以：**不是"修复让旧结果失效"，而是旧结果从一开始就没有统计功效。**
新增的 `experiments/train_paired.py` + `experiments/summarize_paired.py` 才是能测出效应的设计。

---

## 1. 会改变数值的修复

### 1.1 `model_safeguard` → `safeguard_mode`（**默认从"开"变成"关"**）

旧代码在 `experiments/train.py:296` 用 `rc.setdefault('model_safeguard', True)` 默认开启，
**任何 yaml 里都没有这个键**，README 和 protocol 文档也没写。

它用**同一个世界模型、同一个 H 步 rollout 目标函数**给残差重新打分 —— 而 CEM 已经在这个目标上
做过近似 argmax。所以残差只有在能补上 CEM 自己没优化到位的部分时才会被接受，
**拿不到模型偏差修正，也拿不到 γ^H 尾巴**。它把方法的上限锁死在 ≈MPC。

新的 `residual.safeguard_mode` 有四档：

| 模式 | 判据 | 说明 |
|---|---|---|
| `off`（默认） | 不否决 | |
| `critic` | 学到的优势 `A(c, s·r)` | 基于真实 reward，**增加**信息 |
| `model_value` | WM rollout + `γ^H·V(s_H)` | 打分 CEM 没优化的东西 |
| `model` | WM rollout（旧行为） | 只保留作显式 ablation |

旧配置里如果还写着 `model_safeguard`，程序会**报错退出**并提示改名，不会静默按新语义跑。

γ^H 的量级：γ=0.99 时 H=15 → 86.0%、H=18 → 83.5%、H=20 → 81.8% 的折扣回报在 CEM 视野之外。

### 1.2 truncation 不再当成 termination

`done = term or trunc` 曾经直接当 SAC 的 bootstrap mask。HalfCheetah 和 Reacher **永远不会
terminate**，所以它们 100% 的 `done=True` 都是 TimeLimit 截断；`train.py:466` 还在截断时写了
全零 next_context。现在 bootstrap 只用 `terminated`，截断时照常计算真实 next context。
LeWM 臂（`train_lewm_mpc_rl.py`）同样修了 —— Reacher 50 步上限意味着 2% 的 transition 受影响。

### 1.3 replay 存实际执行的残差

旧代码存 actor 的原始输出，但实际执行的是 safeguard 之后的值（经常正好是 0）。
critic 在用"这个动作没发生"的 reward 回归"这个动作"。现在存 `residual × applied_scale`。

### 1.4 CEM 执行 elite 均值而非 argmax 采样

`cem.py` 旧版返回 `seq[idx[0]]` —— **单个打分最高的采样**，最大化模型利用。
新增 `mpc.execute: best_sample | elite_mean | cem_mean`，所有 shipped 配置设为 `elite_mean`
（PETS / TD-MPC 惯例）。**代码默认仍是 `best_sample`，不会静默改变行为** —— 由配置显式指定。

这条很可能会**抬高 MPC 基线**（现在 Hopper 371、Walker2d 214 比正常 CEM planner 低 4–10×），
从而**抹掉旧表里 Walker2d +5.08% 和 Reacher +3.13% 这两个"胜利"**。

### 1.5 `uncertainty_penalty` 语义变了

`rollout_return` 现在**每个集成成员各自传播自己的状态**，返回的 uncertainty 是整段 horizon 回报的
跨成员标准差。旧实现传播的是集成**均值**状态，成员轨迹永不发散，uncertainty 实质是单步分歧的
horizon 平均 ≈ 常数 —— 全篇论文依赖的 gate 实际上是个常数，而文档反复诊断的"gate collapse"
是在治标。

实测（5 成员，horizon 1→24）：新实现的 spread 单调上升 0.099 → 2.07；旧实现按构造恒定。

打分现在是 MOPO 式下界 `mean_members(R) − uncertainty_penalty · std_members(R)`，
所以 `uncertainty_penalty: 0.2` 现在的意思是"惩罚 0.2 个回报标准差"，和旧的绝对尺度不可比。
配套地 `gate_initial_std: -1` 让 gate 从信号本身推导初始尺度。

### 1.6 集成 bootstrap + 归一化统计量改 EMA + 梯度裁剪

- 每个成员用独立的索引抽样 + Bernoulli(0.8) mask，分歧才是 epistemic 的。
- `update_stats` 从**覆盖式重算**改成 EMA（momentum 0.99）。旧行为每 500 步用新采的 10k 样本
  覆盖归一化统计量，输入尺度在已训练权重下不连续跳变。
- `fit_batch` 加了 `clip_grad_norm_(10.0)` 和 finite 检查（同仓库 `train_lewm.py:106` 本来就有）。

### 1.7 SAC：优势参数化 critic、终端价值头、TD3 卫生、尺度感知信任域

- `critic: dueling` → `Q(c,a) = V(c) + A(c,a)`，且 `A(c,0)` 被减掉。残差只扰动动作几个百分点，
  而 HalfCheetah 上 `|Q| ≈ 250`，单头回归整体 Q 会把残差的信号埋进价值基线的逼近误差里。
- `terminal_value: true` → 多一个 `V(obs)` 头，供 `safeguard_mode: model_value` 当终端价值用。
- `actor_delay: 2` + `target_policy_noise: 0.1`：阻止 actor 在 ±4% 窄支撑上吃 critic 外推误差。
- `trust_region_relative: true`：`consistency_coef` 的惩罚按 `|Q|` 归一化。旧版固定 0.05 相对
  `|Q|≈250` 小了约 **5000 倍**，文档叫它"MPC 一致性正则"，实际什么也约束不了 ——
  这正好解释了 `no_consistency` ablation 从来没测出差别。
- `target_entropy_scale: 1.0 → 0.3`：残差是修正量，不是主策略。

### 1.8 残差保持 k 步（`residual_hold_steps`）

旧代码只执行 `corrected[0]` 就重新规划，所以 `chunk_len` / `temporal_decay` /
`expand_temporal_residual` **只影响 safeguard 的打分，从不影响执行的行为** ——
README 说的"对 MPC 的短时 action chunk 做残差修正"在执行层面没有发生。

现在 actor 每 k 步出一个残差，base plan 仍每步重规划（保留闭环），replay 存 γ 折扣的 k 步回报。

### 1.9 `zprl_style` 的步数预算不公平（**未修，已标注**）

`train_zprl_style.py` 额外消耗 `bc_collect_steps`（8000–20000）+ DAgger 步数（2000–10000），
在同样的 `--steps` 下比其他方法多拿 **15–35% 的真实环境交互**。这条需要改协议而不只是改代码，
留给你决定：要么扣减 `--steps`，要么在论文里明确报告总交互步数。

---

## 2. LeWM：架构从"简化基线"改成忠实复现

官方 `github.com/lucas-maes/le-wm`（arXiv:2511.19312）：ViT-Tiny/patch-14 encoder（192-d）
+ 自回归因果 transformer predictor（AdaLN-Zero 动作条件，history=3；TwoRoom 用 1），
目标 `L_pred + λ·SIGReg`，**无 EMA、无 stop-grad**（这一点旧代码是对的）。

旧实现的四个问题：

1. **3 层 CNN + `AdaptiveAvgPool2d(1)`** —— 全局平均池化把空间信息全抹了。
2. **末尾 `BatchNorm1d(affine=False)` 让 SIGReg 失效**。逐维标准化后所有边缘分布都是标准的，
   而标准化高维数据的随机 1-D 投影按 CLT 近似高斯，**与协方差无关**。实测（256×128）：
   各向异性+BN 得 0.579、均匀+BN 得 0.577，**都低于**真 N(0,I) 的 0.987 ——
   测试把坏 embedding 排在理想的前面。去掉 BN 后同一统计量分得很开：
   各向异性 28.7（29×）、rank-1 塌缩 43.0（44×）。
3. **`Linear(128, latent_dim)` 的 128 维瓶颈** —— conv 栈硬编码 128 通道，
   `latent_dim: 256` 的配置实际最多只有 128 个独立方向。
4. **单帧看不到速度** —— MuJoCo 直接从 qpos 光栅化，没有运动模糊可供推断。

修复后实测（2000 transitions / 8 epochs / latent_dim=192）：
`pred` 从 **0.93**（= 输出恒零的平凡 predictor 水平 0.987）降到 **0.003**，
token effective rank **98/192**。

**SIGReg 的 N 因子**：Epps–Pulley 的样本量归一化用的是**独立样本数（帧数）**，
不是 patch token 数 —— 同一张图的 token 强相关。按 token 数缩放会把统计量放大 36×
（96px/patch16），λ 就失去意义。λ 由实测扫描选定，不是猜的：

| λ | pred | sigreg | token_rank | probe_mse |
|---|---|---|---|---|
| 0 | 0.0011 | 264 | 103 | 0.4436 |
| **1e-3** | 0.0029 | 259 | **98** | 0.4434 |
| 1e-2 | 0.0286 | 251 | 52 | 0.4412 |
| 1e-1 | 0.292 | **41** | 13 | 0.4219 |

（真 N(0,I) 的锚点是 **~1.0**。）大 λ 确实把 embedding 推向各向同性，但在这个预算内会先让表征
塌缩且无法恢复。**全量跑之前请重跑这个扫描** —— 合适的 λ 取决于数据量和训练长度。
注意 `probe_mse` 在所有 λ 下基本不变（0.42–0.44），即在这个尺度上 SIGReg 没有改善
任务相关指标；这一点要如实报告。

规划开销：token 级 predictor 在旧预算（candidates=3072, H=18, iterations=5）下是
**16.6 s/plan**，旧 MLP 是 **48 ms/plan**（344×）。所以 LeWM 配置的 CEM 预算大幅下调，
**上服务器前必须重新实测**。

旧的 CNN+MLP 路径保留为显式 ablation：`lewm.encoder: cnn` / `lewm.predictor: mlp`。

---

## 3. 不改变数值、但会改变"能不能跑完"的修复

- **`cem.py` 不再可能返回 `actions=None`**。`torch.topk` 把 NaN 排在所有有限值**之前**，
  所以**一个** NaN candidate 就让 `float(vals[0]) > best_score` 恒假，跳过整轮赋值；
  若所有迭代都如此，`plan.actions` 为 None，`plan_chunk` 抛
  `TypeError: 'NoneType' object is not subscriptable`。已复现。`lewm_planner.py` 同样修了。
- **`elites > candidates` 不再崩**（旧版 `RuntimeError: selected index k out of range`，
  本地调小 candidates 做轻量测试必踩）。
- **`CSVLogger` 建文件时立刻写表头**。旧版留 0 字节空文件，`quick_diagnostic.py` 的
  `pd.read_csv` 直接抛 `EmptyDataError`，短 run 必崩。重名 run 现在轮转而不是静默 append。
- **`plot_results.py` 接受 `--summary`**。README 和 `scripts/run_main_suite.sh` 一直在传这个
  参数，实际不存在 → **exit 2**，跑完整个 suite 才炸在最后一步。已复现。
- **断点续跑**：`--resume`。checkpoint 以前每 10k 写一次但**全仓库没有一处读过**。
- **一个 job 失败不再带走整个 suite**（`run_suite.py` 默认 `--keep-going`，失败写
  `results/suite_failures.csv`）。
- **`scripts/` 全部 `nohup`/`setsid` + 日志落盘 + `cd` 到仓库根 + `$PYTHON`**，SSH 断开不再杀进程。
- **无头 Linux 自动设 `MUJOCO_GL=egl`**（Windows/macOS 不动，那里 glfw 才是对的）。
- **CUDA 不可用时可以大声失败**（`hardware.require_cuda`），不再静默退到 CPU 慢 100 倍。
- **`±` 和中文 print 改 ASCII**，非 UTF-8 locale 下不会在最后一行把跑完的 run 弄崩。

---

## 4. 测试

从 23 个（1.4 秒、全是 smoke test、算法写错也不会挂）扩到 **64 个**，新增的都针对具体失败模式：
单个/全部 NaN 打分、`elites > candidates`、truncation 不置零 bootstrap、replay 存执行值、
SIGReg 必须惩罚各向异性和塌缩、per-member uncertainty 随 horizon 单调增、
AdaLN-Zero 初始为恒等、因果 mask 方向、配对协议里 WM fingerprint 不变。

---

## 5. 我改配置时弄丢的东西

用脚本批量 patch 配置时，`yaml.safe_dump` 把**所有 shipped 配置里的注释都删掉了**。
我把关键的说明性注释以文件头的形式补了回来（含 `refine/*.yaml` 的 test-set-tuning 警告），
但**逐键的行内注释没有恢复**。如果你本地有 git 历史，`git diff` 能看到原文。
