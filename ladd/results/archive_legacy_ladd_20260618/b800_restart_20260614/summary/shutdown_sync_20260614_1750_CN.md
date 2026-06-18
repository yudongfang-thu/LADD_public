# 2026-06-14 17:50 关机前轻量结果同步记录

本记录用于确认关闭远端服务器前，当前 LADD B800 诊断批次已经同步到本机的轻量证据。

## 同步范围

已同步到本机：

- `results.csv`
- `ladd_diagnostics.csv`
- `args.yaml`
- `initial_val_metrics.json`
- `b_split_load_manifest.json`（存在时）
- launch `manifest.txt`
- phase `master.log`（轻量日志）

未同步到仓库内：

- `weights/*.pt`
- `.pth`
- TensorBoard event
- wandb
- 完整 run 目录

已额外备份到仓库外：

- `/Users/yudongfang/Desktop/光sar/LADD_public_checkpoints_shutdown_20260614/b800_restart`
- 14 个 `.pt` checkpoint，约 357 MB。

本轮轻量同步后，本机目录：

- `ladd/results/b800_restart_20260614/evidence_raw/`
- `ladd/results/b800_restart_20260614/summary/`
- `docs/experiments/LADD_B800_RESTART_CURVE_ANALYSIS_20260614_CN.md`
- `docs/experiments/LADD_B800_RESTART_CURVE_ANALYSIS_20260614_LOCAL_EMBEDDED_CN.html`
- `docs/experiments/ladd_b800_restart_curve_summary_20260614.csv`

## 最新结果快照

| run | server | latest epoch | best AP50-95 | best epoch | latest AP50-95 | status |
|---|---|---:|---:|---:|---:|---|
| N0 YOLO-init det-only | AutoDL | 332 | 0.45155 | 332 | 0.45155 | running |
| N1 SAR-best det-only | AutoDL | 332 | 0.57521 | 324 | 0.57494 | running |
| N1 SAR-last det-only | AutoDL | 338 | 0.57687 | 337 | 0.57661 | running |
| N2 A2-best full LADD | 4090 | 229 | 0.55681 | 214 | 0.54271 | stopped, NaN at 229 |
| N2 A2-last full LADD | 4090 | 319 | 0.56073 | 271 | 0.46290 | stopped, NaN at 319 |
| N3 YOLO-init + A2 decomp | 4090 | 525 | 0.48742 | 525 | 0.48742 | running |
| N4 YOLO-init + A2 decomp KD-warmup | 4090 | 512 | 0.46180 | 512 | 0.46180 | running |

## 权重大小参考

已将 checkpoint 单独复制到本机仓库外：

- 4090 N3/N4 `best.pt` + `last.pt`: about 101 MB total.
- AutoDL N0/N1-basebest/N1-baselast `best.pt` + `last.pt`: about 150 MB total.
- 另含 4090 N2 stopped runs 的 `best.pt` + `last.pt`。
- Current listed LADD checkpoints total: about 357 MB.

这些权重不应提交 GitHub。
