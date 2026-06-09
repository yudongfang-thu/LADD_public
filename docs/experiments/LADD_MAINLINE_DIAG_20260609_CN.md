# LADD H1 主线诊断计划

最后更新：2026-06-09

## 1. 当前问题

- YOLO11n 三个 seed 已有稳定正向证据。
- YOLO11s 存在 B 阶段 best 正向但 last 后期退化。
- YOLO11m 在 A2 或 B 入口存在明显退化。
- 因此当前 H0 不能直接冻结为跨 seed、跨容量主线。

## 2. H1 修改定义

```text
H1 = H0 + BN-freeze 不改 requires_grad + ladd_diag_log_grad 不再隐式裁剪梯度
```

H1 不改变 LADD 方法本身，只消除训练和诊断实现污染：

- `FREEZE_BN_STATS=1` 只冻结 BatchNorm running statistics。
- BN-freeze 不再改变任何参数的 `requires_grad`。
- `ladd_diag_log_grad=1` 只记录梯度范数。
- 梯度裁剪只能通过显式 `LADD_GRAD_CLIP_NORM>0` 启用；H1 主线使用 `0.0`。
- B phase 可通过 `LADD_ASSERT_PHASE_FREEZE=1` 检查 `teacher_decomposition`、`student_reachability`、`teacher_task_heads` 是否保持 frozen。

## 3. 合格判据

单 run：

- 无 NaN/Inf。
- 无 B stage collapse。
- best mAP50-95 > SAR baseline。
- last mAP50-95 >= SAR baseline - 0.002。

n/s：

- 至少 seed0/42/123 三 seed 全部不负。
- mean gain > 0。

m：

- seed0 的 A2 不能低于 baseline。
- 否则不进入 B full。

只有 n/s 三 seed 和 m seed0 通过，才能考虑把 H1 作为正式主线。如果 H1 仍不能稳住 s/m，再进入 H2：capacity-adaptive KD，例如 s/m 使用 `alpha_kd=0.5` 或 `0.25`。

## 4. 实验队列

P1：

- `diag_h1_n_seed0_b100_smoke`
- `diag_h1_s_seed0_b400`
- `diag_h1_m_seed0_a2only`

P2_s：

- `diag_h1_s_seed0_alpha_kd_0p5_b400`
- `diag_h1_s_seed0_alpha_kd_0p25_b400`
- `diag_h1_s_seed0_detonly_b400`

P2_m：

- `diag_h1_m_seed0_a2_lr3e4`
- `diag_h1_m_seed0_a2_short25`
- `diag_h1_m_seed0_a2_lambda05`

## 5. 执行入口

默认只 dry-run：

```bash
DRY_RUN=1 bash scripts/launch_ladd_mainline_diag_20260609.sh
RUN_SET=p2_s DRY_RUN=1 bash scripts/launch_ladd_mainline_diag_20260609.sh
RUN_SET=p2_m DRY_RUN=1 bash scripts/launch_ladd_mainline_diag_20260609.sh
```

只在代码自检通过、提交并 push 后启动 P1：

```bash
SERVER_TAG=<server_name> GPU_ID=<gpu_id> DRY_RUN=0 RUN_SET=p1 \
  bash scripts/launch_ladd_mainline_diag_20260609.sh
```

不要根据 P1 自动启动 P2。P2 只能在 P1 结果人工确认后再启动。

## 6. 结果汇总

P1 结果出来后运行：

```bash
python tools/summarize_ladd_diag_runs.py \
  --runs <run_dir_1> <run_dir_2> <run_dir_3> \
  --out-csv docs/experiments/ladd_mainline_diag_20260609_summary.csv \
  --out-md docs/experiments/LADD_MAINLINE_DIAG_RESULTS_20260609_CN.md
```

汇总脚本会读取 `results.csv`、`args.yaml`，并优先从 `docs/experiments/BASELINE_LADD_STATUS_CN.md` 解析 SAR baseline。

## 7. 注意事项

- 不混用 4090D 的 invalid 或疑似协议错误结果作为主线证据。
- 不改 dataset yaml、imgsz、augmentation、YOLO 版本、teacher/student 配对协议。
- 不提交 raw logs、checkpoints、runs、wandb、tensorboard event。
- 所有结果必须记录 git commit SHA 和 server/GPU 信息。
- H1 不能直接宣传为最终主线，只能作为下一轮诊断候选。
