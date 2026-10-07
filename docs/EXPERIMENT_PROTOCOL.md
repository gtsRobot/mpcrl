# 论文实验协议

## 主实验

每个环境、每种方法运行 5 个随机种子：0,1,2,3,4。

对比方法：

1. Ensemble WM + CEM-MPC
2. MPC + Action Residual SAC
3. ZPRL-style bottleneck latent residual proxy
4. Proposed Planning-level Residual SAC
5. LeWorldModel-MPC（独立实验）

## 统计

- 主要指标：episodic return；若任务有明确成功判据，则报告 success rate。
- 次要指标：sample efficiency、MPC latency、action D1/D2 smoothness、world-model one-step MSE、uncertainty、residual norm。
- 推荐报告 mean ± std，并保留每个 seed 原始 CSV。
- 不应仅报告最优 seed。

## 消融

- Adaptive gate on/off
- Uncertainty penalty on/off
- MPC consistency penalty on/off
- planning chunk length = 1/3/5
- residual scale = 0.1/0.2/0.3

## 成功判据

Gymnasium locomotion 默认并不提供统一的 binary success。若使用 Hopper/Walker2d/HalfCheetah，应以 return 为主要指标；只有在提前定义并固定 return threshold 后，才将阈值成功率作为补充指标。
