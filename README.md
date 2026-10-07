# LeWM-MPC + Residual RL：代码与实验记录整理版

这是一份适合放入 GitHub 的项目快照，整理于 **2026-10-07**。包含安装配置、模型源码、训练入口、推理评估、统计绘图、测试，以及已有实验的轻量记录。

本项目研究在冻结的 LeWM 工程适配世界模型上，使用 RL 残差修正 MPC 动作。任务为 MuJoCo 的 Hopper、Walker2d、HalfCheetah、Reacher；不是 LeWM 官方四任务的严格复现。

**先读 [完整流程](docs/WORKFLOW.md)，再看 [当前实验状态与限制](docs/PACKAGE_STATUS.md)。**

## 目录导航

| 路径 | 用途 |
|---|---|
| `environment.yml`、`requirements.txt` | 安装环境与依赖 |
| `mpcrl/` | LeWM、CEM、SAC、回放与诊断等模型源码 |
| `configs/` | 任务、硬件、实验矩阵与消融配置 |
| `experiments/` | 训练、加载权重评估、汇总与绘图入口 |
| `scripts/` | 原有批处理脚本 |
| `tests/` | 源工程测试及新增权重恢复测试 |
| `results/archived_runs/benchmark/` | 常规四方法的 80 组原始轻量记录 |
| `results/archived_runs/lewm_baselines/` | 独立 LeWM 基线的 19 组记录 |
| `results/archived_runs/lewm_paired/` | HalfCheetah seed 0 的一组正式配对记录 |
| `results/archived_runs/smoke/` | 历史短测试，不能当作正式结果 |
| `results/reference/` | 原汇总表、失败记录和 8 张关键图，均为历史快照 |
| `results/overview.csv`、`results/experiment_index.csv` | 分类别结果总览，以及 103 条评估记录的文件导航 |
| `docs/` | 安装到结果的操作指南、协议及历史说明 |
| `packaging/` | 原文件映射、排除清单、703 个权重文件索引及打包校验 |

## 快速开始

从项目根目录执行，先按 [环境安装](docs/WORKFLOW.md#1-环境安装) 配好 PyTorch 与 MuJoCo：

```bash
python -m unittest discover -s tests -v

# 很小的集成测试；只用于确认流程可运行，不能写入论文结果。
python -m experiments.train_lewm_mpc_rl \
  --config configs/packaging/lewm_smoke.yaml \
  --seed 0 --rl-steps 64 --eval-episodes 1 \
  --run-root runs/package_smoke
```

正式训练：

```bash
python -m experiments.train_lewm_mpc_rl \
  --config configs/rtx5090/lewm_halfcheetah.yaml \
  --seed 0 --rl-steps 50000 --eval-episodes 10 \
  --run-root runs/lewm_paired_new
```

加载已有配对权重进行基线与 RL 评估，无需重训：

```bash
python -m experiments.evaluate_lewm_checkpoint \
  --checkpoint checkpoints/lewm_halfcheetah_seed0.pt \
  --method both --episodes 10 \
  --outdir results/generated/reevaluation
```

**压缩包不含权重。** 请先训练，或按 [权重获取说明](docs/WORKFLOW.md#3-取得已有权重) 从原目录复制对应 `final.pt`。权重位置与大小列在 [checkpoints.csv](packaging/checkpoints.csv)。

## 整理范围

- 原有 `mpcrl/`、训练源码、配置与历史指标按原内容复制；没有借整理之机修改算法或筛掉不涨点的 seed。
- 新增 LeWM checkpoint 评估入口、流程说明、轻量集成配置与打包校验工具。
- 去掉权重、运行大日志、缓存、重复 PDF 和可重新生成的绘图中间数据；原目录文件仍完整保留。
- 历史结果与新生成输出分开。新训练写入 `runs/`，新分析建议写入 `results/generated/`；两者不默认提交 Git。
- 本整理版不是修复完全部研究协议问题的“论文最终版”，详见 [状态说明](docs/PACKAGE_STATUS.md)。

上传时把本目录内容作为仓库根目录。可以先运行 `python scripts/verify_package.py` 检查已登记文件完整性。模型文件以后单独分发；不要把训练输出重新加入这个轻量源码仓库。
