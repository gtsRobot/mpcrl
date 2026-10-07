# LeWorldModel 复现与对比口径

本工程同时保留 **LeWorldModel（LeWM）方法核心复现**、**统一 MuJoCo 工程适配对比** 和 **官方四任务参考核验** 三条实验线。三者用途不同，论文中必须分开表述。

## A. LeWM 方法核心复现（本仓库可直接运行）

`mpcrl/lewm.py` 实现了与 LeWM 核心思想对应的轻量像素世界模型：

- raw RGB frame -> latent encoder；
- action-conditioned next-latent predictor；
- next-embedding MSE；
- SIGReg（随机投影 + Epps-Pulley/经验特征函数式高斯匹配）；
- 无 EMA teacher；
- 无 stop-gradient target；
- latent CEM planner。

训练目标保持两项结构：

```text
L = L_pred + lambda * SIGReg
```

MuJoCo locomotion 没有官方 LeWM goal-image benchmark 的同构奖励定义，因此 `experiments/train_lewm.py` 在冻结 LeWM 后单独训练 reward probe，再使用 CEM 最大化预测回报。**reward probe 是 MuJoCo 适配层，不属于 LeWM 原论文的两项训练目标。**

最新版 `train_lewm.py` 会同时输出 `episodes.csv` 与 deterministic `eval.csv`，从而能够和 MPC-only、Action Residual、Planning Residual、ZPRL 使用同一最终统计脚本比较。

## B. 本仓库统一 MuJoCo 工程 benchmark

主论文工程对比使用四个 Gymnasium/MuJoCo 连续控制任务：

1. Hopper-v5
2. Walker2d-v5
3. HalfCheetah-v5
4. Reacher-v5

对应配置：

```text
configs/rtx5090/hopper.yaml
configs/rtx5090/walker2d.yaml
configs/rtx5090/halfcheetah.yaml
configs/rtx5090/reacher.yaml
```

LeWM 工程适配 baseline：

```text
configs/rtx5090/lewm_hopper.yaml
configs/rtx5090/lewm_walker2d.yaml
configs/rtx5090/lewm_halfcheetah.yaml
configs/rtx5090/lewm_reacher.yaml
```

这里的 `Reacher-v5` 是为了增加第四个统一连续控制任务，并和 LeWM 官方 benchmark 中的 Reacher 概念形成部分任务类型重叠；**它不等同于官方 stable-worldmodel 的 Reacher 环境，也不能替代官方论文数值。**

## C. LeWM 官方四任务 benchmark

LeWM 官方代码/数据使用的核心四任务为：

1. TwoRoom
2. PushT
3. Cube
4. Reacher

若论文需要声称“严格复现 LeWM 官方 benchmark 数值”，应使用官方仓库、官方数据、官方 checkpoint 与 stable-worldmodel 评估接口，而不是拿本仓库的 MuJoCo locomotion 数值代替。

本仓库提供辅助脚本：

```bash
LEWM_OFFICIAL_DIR=/path/to/LeWorldModel \
STABLEWM_HOME=/path/to/.stable-wm \
bash scripts/run_lewm_official_reference.sh
```

该脚本只做 **官方 LeWM checkpoint 的四任务参考核验**。它不会把本仓库 state-space ZPRL 的结果伪装成官方 pixel-goal benchmark 结果。

## D. 论文中建议的表述

推荐表述为：

> 本文首先在统一的 MuJoCo 连续控制接口下比较 MPC、Action Residual、Planning Residual、ZPRL 与 LeWM 工程适配版本，以保证训练预算、随机种子和评估口径一致；同时使用 LeWM 官方仓库在 TwoRoom、PushT、Cube 和 Reacher 上进行参考数值核验，用于验证 LeWM 基线实现和原论文 benchmark 的一致性。由于官方 LeWM benchmark 属于像素目标条件规划任务，而本文主 ZPRL 实现采用状态空间连续控制接口，两类结果分别报告，不直接混为同一张数值表。

这样可以避免把“方法核心复现”“工程适配”和“严格官方 benchmark 复现”混为一谈。
