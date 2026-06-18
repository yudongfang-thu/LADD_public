# LADD Capacity-aware KD 2026-06-11 证据包

本文档对应实验结果分析：

```text
docs/experiments/LADD_CAPACITY_KD_DIAG_RESULTS_20260611_CN.md
```

## 内容

```text
alpha0p5_b400/
alpha0p25_b400/
bdetonly_b400_r2/
m_a2_probe/
log_extracts/
```

- `*_results.csv`：从远端 run 目录同步的原始结果 CSV 或指定时间快照。
- `*_args.yaml`：Ultralytics/LADD 训练参数。
- `*_manifest.txt`：阶段启动时记录的 `git_commit`、run name、phase、关键开关。
- `log_extracts/`：从大 outer log 中抽取的 resume、phase diagnostic、grad clip、完成状态等关键行。

## 排除项

未同步以下内容：

- `weights/`
- `*.pt`
- `*.pth`
- 大体积 tqdm outer log 全量文本

原因：本目录只作为可审阅的轻量证据包，不作为 checkpoint 存储。
