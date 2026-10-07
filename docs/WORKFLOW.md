# 从安装到训练、推理评估、结果汇总

所有命令在项目根目录执行。本机整理目录是 `/home/ubuntu22/111`，换机器后使用自己的克隆目录。下文代码路径保持源工程结构，以免破坏 Python 导入和配置引用。

## 1. 环境安装

```bash
cd /home/ubuntu22/111
conda env create -f environment.yml
conda activate mpc_rl_zprl_5090

# 沿用源项目 RTX 5090 安装路线；按实际机器驱动确认 CUDA 可用。
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/check_rtx5090.py
python -m unittest discover -s tests -v
```

已有 `mpc_rl_zprl_5090` 环境时直接激活，不重复创建。无显示器 Linux 可设置 `export MUJOCO_GL=egl`。原配置中的 `hardware.require_cuda: true` 会在 CUDA 不可用时报错。

仅验证文件、读取已有 CSV 不需要 GPU。像素规划训练及推理需要 MuJoCo 和可用的渲染后端；CPU 推理可能很慢。

## 2. 训练

### 2.1 先跑一次轻量集成流程

```bash
python -m experiments.train_lewm_mpc_rl \
  --config configs/packaging/lewm_smoke.yaml \
  --seed 0 --rl-steps 64 --eval-episodes 1 \
  --run-root runs/package_smoke
```

它仍执行收集像素数据、训练模型与 reward probe、评估 baseline、训练 RL、评估 RL 五个阶段，但预算很小。此配置不是正式实验配置，不能与历史正式结果混合。

### 2.2 单任务正式 LeWM 配对实验

```bash
python -m experiments.train_lewm_mpc_rl \
  --config configs/rtx5090/lewm_halfcheetah.yaml \
  --seed 0 --rl-steps 50000 --eval-episodes 10 \
  --run-root runs/lewm_paired_new
```

任务配置可换为 `lewm_hopper.yaml`、`lewm_walker2d.yaml`、`lewm_reacher.yaml`。一个配对任务生成两个目录：

```text
runs/lewm_paired_new/
  <env>__lewm_mpc__seed0__pair<TAG>/
  <env>__lewm_mpc_rl__seed0__pair<TAG>/
```

两者包含 `config.yaml`、`eval.csv`、`representation.csv` 与 `final.pt`；RL 目录还包含 `episodes.csv`、`pair_summary.json`。必须保留相同 `pair<TAG>` 的两个目录。

### 2.3 四任务批量 LeWM 配对

```bash
python -m experiments.run_lewm_mpc_rl_suite \
  --suite configs/rtx5090/lewm_mpc_rl_suite.yaml \
  --run-root runs/lewm_paired_suite_new \
  --results-dir results/generated/lewm_paired_suite_new
```

默认是四任务、五个 seed、每组 50k RL 步和 10 个评估回合。此处只说明入口，并不表示历史上已完成这一整套实验。每次正式批次使用新的输出根目录，避免不同配置或重复 seed 混入同一汇总。

`--start-job` 只跳过前面的任务，不是中途恢复单个 RL 训练。此批处理入口遇到失败会停止；参见状态说明。

### 2.4 其他保留的入口

| 目的 | 入口 | 说明 |
|---|---|---|
| 状态空间四方法 benchmark | `bash scripts/run_main_suite.sh` | 不是 LeWM+RL 配对主实验；脚本默认后台运行 |
| 独立 LeWM 工程基线 | `python -m experiments.train_lewm --config configs/rtx5090/lewm_reacher.yaml --seed 0 --episodes 5` | 原入口实际使用 history=1 |
| 状态空间 MPC 配对 | `bash scripts/run_paired_suite.sh` | 与 LeWM 像素主线不同 |
| 状态空间残差消融 | `bash scripts/run_ablation.sh` | 不是 LeWM residual 的完整消融 |
| ZPRL-style 消融 | `bash scripts/run_zprl_ablations.sh` | 相关入口保留，不代表已完成 |

## 3. 取得已有权重

为了缩小 GitHub 包，703 个 `.pt` 文件全部留在原始实验目录。不是删除，也不是训练结果丢失。位置和大小见 `packaging/checkpoints.csv`。

在原机器上，可取回当前唯一正式 LeWM+RL 配对的 RL checkpoint；它同时包含共享 LeWM、reward probe 与 residual SAC：

```bash
mkdir -p checkpoints
cp /home/ubuntu22/Private/321/MPC_RL_ZPRL_RTX5090-main/runs/HalfCheetah-v5__lewm_mpc_rl__seed0__pair1790931152/final.pt \
  checkpoints/lewm_halfcheetah_seed0.pt
```

换机器时需要自己传输这个文件，或今后单独发布模型资产；**当前没有自动下载地址**。不需要为重评估复制所有中途 checkpoint，也不需要同时复制配对 baseline 权重。

## 4. 推理 / 重新评估

新增 `experiments.evaluate_lewm_checkpoint` 支持本目录实际使用的两种 LeWM checkpoint：独立 baseline 和 paired checkpoint。使用只读权重加载；输出写入新的目录，拒绝覆盖已有非空结果。

```bash
# 先检查权重与模型结构是否匹配，不执行控制回合。
python -m experiments.evaluate_lewm_checkpoint \
  --checkpoint checkpoints/lewm_halfcheetah_seed0.pt --inspect-only --device cpu

# 同一份 RL checkpoint 评估 baseline 与 RL 两种控制器。
python -m experiments.evaluate_lewm_checkpoint \
  --checkpoint checkpoints/lewm_halfcheetah_seed0.pt \
  --method both --episodes 10 \
  --outdir results/generated/reevaluation
```

`--method baseline` 只评估 MPC；`--method rl` 只评估 RL；`auto` 对有 RL 权重的文件评估两种方法，对独立文件只评估 baseline。独立 baseline checkpoint 不能凭空恢复 RL。

输出包括：

```text
results/generated/reevaluation/
  checkpoint_config.yaml
  evaluation.json                 # 权重 SHA256、设备、实际 history、均值等
  lewm_mpc/eval.csv
  lewm_mpc_rl/eval.csv             # 只有评估 RL 时生成
```

此处“推理”指加载策略，与 MuJoCo 环境交互并计算指标；不包含视频录制界面。评估沿用源工程 helper，种子规则为 `700000 + train_seed*1000 + episode`；原独立 LeWM 使用另一套种子，不能承诺逐数复现独立旧 CSV。

**此入口没有修复旧研究协议问题**，如确定性 actor 消耗随机数、CEM 随机流对齐，见 `PACKAGE_STATUS.md`。新增入口用于加载和操作完整性，不应被描述为已完成公平性修复。

原有 `python -m experiments.evaluate --run-dir <目录> --episodes 5` 只支持状态空间 `mpc_only/action_residual/planning_residual` checkpoint，且会写入该 run 的 `eval.csv`。不要用它读取 LeWM 或 ZPRL checkpoint。

## 5. 从已有记录生成结果

不必重新训练即可汇总打包的全部轻量记录。务必选择正确分类根目录，不能把 `results/archived_runs` 整体交给配对汇总。

### 5.1 常规四方法

```bash
python -m experiments.aggregate \
  --root results/archived_runs/benchmark \
  --out results/generated/benchmark/summary.csv \
  --min-steps 99000 --require-eval --require-complete

python -m experiments.plot_results \
  --root results/archived_runs/benchmark \
  --outdir results/generated/benchmark/figures
```

训练日志按完整 episode 写入，少数最后一条步数稍低于 100000；这里的 99000 是历史记录筛选阈值，不是把训练预算改成 99000。

### 5.2 当前唯一正式 LeWM 配对

```bash
python -m experiments.summarize_lewm_mpc_rl \
  --root results/archived_runs/lewm_paired \
  --outdir results/generated/lewm_pair/summary \
  --envs HalfCheetah-v5 --seeds 0 --require-complete

python -m experiments.plot_lewm_mpc_rl \
  --root results/archived_runs/lewm_paired \
  --summary-dir results/generated/lewm_pair/summary \
  --outdir results/generated/lewm_pair/figures
```

这里限制一个任务、一个 seed 是因为档案实际只有这一组，并不代表四任务全完成。旧绘图的 learning curve 是带探索的训练回报，不是周期性确定性评估曲线。

### 5.3 历史结果目录

`results/reference/` 保存源工程已生成的表格和关键 PNG，仅供对照。CSV 中的 `runs/...` 路径是原实验路径；对应整理后路径见 `packaging/copied_files.csv`。不改写原数据，新的汇总会产生正确的新路径。

`results/archived_runs/lewm_baselines/` 是独立模型，不得替代配对 baseline。`smoke/` 是短测试，仅用于追溯。

## 6. 上传 GitHub 与持续整理

本目录自身就是仓库根目录，`mpcrl/`、`experiments/`、`configs/` 应同级。新的 `.gitignore` 保留归档指标与参考结果，忽略权重、缓存、训练输出和新生成图表。

```bash
python scripts/verify_package.py
git init -b main
git add .
git status --short
git commit -m "Add organized LeWM MPC RL project"
# 将下方地址换成实际目标仓库：
# git remote add origin https://github.com/<账号>/<仓库>.git
# git push -u origin main
```

新实验跑完后，审阅需要保留的 `eval.csv`、`episodes.csv`、`config.yaml`、`pair_summary.json`，再另行归档；不要直接取消整个 `runs/` 的忽略规则。后续修改代码会导致快照 SHA256 校验不一致，这是正常的版本变更，需要重新生成清单。
