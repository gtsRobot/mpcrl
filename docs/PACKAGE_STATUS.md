# 整理范围、实验状态与已知限制

日期：2026-10-07。来源：`/home/ubuntu22/Private/321/MPC_RL_ZPRL_RTX5090-main`。

## 本次做了什么

- 原始目录保留。将完整代码、所有轻量实验记录复制到新目录；没有只挑选涨点的 seed。
- 原算法源码、训练入口、配置和历史数据保持字节一致，映射见 `packaging/copied_files.csv`。
- 归档记录分为 benchmark、独立 LeWM、正式 LeWM 配对、短测试，降低错误混表的风险。
- 新增 checkpoint 评估入口和小型流程配置，补充从安装到汇总的导航。
- 排除模型权重、大日志和缓存；排除项逐文件列在 `packaging/excluded_files.csv`。
- 只保留 8 张常用 PNG；其他诊断图、PDF 和绘图中间数据可用保留的代码与原始 CSV 重新生成。
- 原 README 保存为 `docs/README_ORIGINAL.md`。历史文档保留原文，但其中的“推荐主线”、模型参数和已完成工作可能过时，应以当前源码、配置及本状态页为准。

## 实际完成情况

| 实验 | 已有记录 |
|---|---|
| 常规四方法 × 四环境 × 五 seed | 80 组；每组 5 回合最终评估 |
| 独立 LeWM-MPC 四环境基线 | 19/20 组；Reacher seed 1 失败，退出码 -9 |
| 正式 LeWM-MPC / LeWM-MPC+RL 配对 | HalfCheetah seed 0，RL 50000 步，双方各 10 回合 |
| 短测试配对 | `_t_lewm2`，RL 300 步，单独归档 |
| LeWM+RL 成套消融 | 没有完成记录 |
| 官方 LeWM 四任务实验 | 本来源目录没有正式结果 |

正式单 seed 配对的原始回报是 -86.06395 → -75.68993，差值 +10.37402。这是历史单 seed 结果，不是多 seed 稳定有效的证据。所有当前 success 均缺少有效判定，NaN 不代表成功率为零。

## 尚未修复的研究协议问题

1. 原 LeWM 汇总程序按 env/method/seed 各取最新 run，没有严格验证 pair ID、权重指纹、步数或评估数量。整理后的分类目录避免把本次独立旧基线混入示例配对，但不等于算法层面的修复。
2. 确定性 actor 仍调用随机采样，CEM 也使用全局 Torch 随机流，两条控制分支的采样没有严格对齐。
3. LeWM replay 记录原始残差，而实际动作还受到 ramp/可选 safeguard 影响；context 没有完整描述这一变化。
4. 默认探索阶段 ramp=1，随后在 start_steps 降到 0 再增加，协议与“始终渐进开启”并不相同。
5. 独立 `train_lewm.py` 强制 history=1，配对入口使用配置 history，当前为 2。新加载器按实际入口恢复这一差异。
6. 配对入口没有单任务中途恢复、定期确定性评估或冻结世界模型的独立缓存加载入口，模型主要在最终保存。批处理 `--start-job` 不等于训练断点恢复。
7. 原 paired summary 的 CI 仍使用正态近似；单个 seed 无法估计训练种子间方差。图中训练回报与确定性 final eval 的口径不同。
8. `mpc_ms` 仅测 planner 部分，不能直接当作完整 RL 控制延迟。表征有效秩只是冻结模型的辅助指标。
9. reward probe 的记录误差来自训练集；尚缺独立验证轨迹上的质量检查。

## 新加载器的边界

它恢复本来源项目的 checkpoint，并复用原有评估 helper，不会修改权重、重训模型或修复上述算法问题。支持 paired checkpoint 的 baseline/RL 两条路径，以及 standalone baseline。没有权重时不能仅靠 CSV 推理。

现有独立基线保存的 YAML history 与真实入口不一致，因此加载器对无 pair_summary 的来源 checkpoint 使用 history=1。若未来更改 checkpoint 格式，请同步更新加载器，不能把它当作任意 LeWM 实现的通用加载器。

原文档中 `arXiv:2511.19312` 的引用应核对修正：LeWM 官方仓库链接到 [2603.19312](https://arxiv.org/abs/2603.19312)。这里保留历史原文以便追溯，没有声称已完成官方架构逐项复核。
