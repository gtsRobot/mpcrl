# 结果导航

- **[overview.csv](overview.csv)**：按实验类别、环境和方法汇总种子均值；单 seed 的跨 seed 标准差留空，不能解读为零方差。
- **[experiment_index.csv](experiment_index.csv)**：全部 103 条评估记录，包括 80 条常规 benchmark、19 条独立 LeWM、2 条正式配对控制器记录、2 条短测试记录。2 条配对记录代表同一个模型/seed 的两种方法，不是两个独立 seed。
- **[reference/](reference/)**：原始汇总快照、失败清单与关键图表；表内旧路径原样保留。
- **[archived_runs/](archived_runs/)**：全部轻量逐回合原始数据，按实验类别隔离，未包含权重。

新的输出放到 `generated/`（Git 忽略）。不要把四种类别混在一起求平均或配对，不要把 smoke 当成论文正式结果。

总览由现有 eval.csv 计算，不是新增训练。重建导航表：

```bash
python scripts/build_results_index.py
```

重建学习曲线和其他统计图的方法见 [完整流程](../docs/WORKFLOW.md#5-从已有记录生成结果)。
