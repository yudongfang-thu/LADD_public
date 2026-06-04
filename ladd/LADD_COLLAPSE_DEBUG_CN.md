# LADD 崩溃与修复证据链

最后更新：2026-06-04 08:55 CST

本文档专门给外部老师排查 LADD 主方法的问题。baseline 本身已经稳定，当前困难集中在 LADD 的 A2/B 阶段训练稳定性，尤其是 B 阶段后期塌缩。

## 1. 当前主线协议

```text
OGSOD-1.0 HBB
YOLO11n/s/m/l 容量轴
imgsz=256
formal no-mosaic
A1=10 -> A2=50 -> B=800
cap2 reach-rank
teacher = same-capacity same-seed RGB detector
student/inference = SAR-only detector
```

当前代码快照在 `ladd/code/` 和 `ladd/code_versions/current_hbb/`，两者已于 2026-06-04 同步。核心文件：

| 文件 | 作用 |
|---|---|
| `tools/train_ladd_hbb.py` | HBB LADD/对比方法训练入口 |
| `src/teacher_student_decomposition_kd_hbb/loss.py` | LADD loss、cap2、FGD/LD/HalluciDet-style 等 profile |
| `src/teacher_student_decomposition_kd_hbb/trainer.py` | A/B 阶段训练控制、BN-freeze 逻辑 |
| `scripts/ogsod_public/run_ladd_phase.sh` | 单阶段启动脚本 |
| `scripts/ogsod_public/run_hbb_ladd_converged_chain.sh` | A1/A2/B 链式启动脚本 |

## 2. 版本时间线

| 版本 | 配置/代码变化 | 目的 | 证据目录 |
|---|---|---|---|
| v0 cap2 初版 | cap2 reach-rank，A2/B 使用较激进默认优化器设置 | 原始主线，验证 cap2 相对 original rank loss 是否有效 | `ladd/results/90_formal_nomosaic_20260528/...cap2...` |
| v1 A2 稳定修复 | A2 改为 `MuSGD lr0=0.001 no warmup` | 修复 A2 早期 detection loss 失控 | `ladd/diagnostics/a2_stability/` |
| v2 B 稳定修复 | B 改为 `MuSGD lr0=0.001 no warmup` | 避免 B 入口 KD/检测冲击 | 90 的 `a2mu1e3_b...` runs |
| v3 B stable1e3 | seed123 使用更温和 B 设置完整跑 800 epoch | 验证 seed123 是否可救 | `...s123_a2mu1e3_bstable1e3...` |
| v4 BN-freeze | 新增 `--freeze-bn-stats` / `FREEZE_BN_STATS=1`，冻结 BN running mean/var，保留 affine 梯度 | 针对 4090D/90 坏 run 中 BN running stats 污染 | 90 BN-freeze 诊断 runs |

## 3. 关键结果

| Run | 服务器 | epoch | current AP50-95 | best AP50-95 | 状态 |
|---|---|---:|---:|---:|---|
| YOLO11n seed0 cap2 `a2mu1e3` | 90 | 800 | 0.57504 | 0.57662@725 | 完成，正向 |
| YOLO11n seed42 cap2 `a2mu1e3` | 90 | 800 | 0.57293 | 0.57420@735 | 完成，正向 |
| YOLO11n seed123 cap2 old B | 90 | 483 | 0.00000 | 0.52182@1 | 后期塌缩 |
| YOLO11n seed123 `bstable1e3` | 90 | 800 | 0.52875 | 0.56161@165 | 完整跑完但后期退化 |
| YOLO11n seed0 BN-freeze | 90 | 37 | 0.53266 | 0.53487@34 | 诊断 run，早期 |
| YOLO11n seed123 BN-freeze | 90 | 38 | 0.54597 | 0.55198@37 | 诊断 run，早期 |
| YOLO11n seed0 4090D r2 | 4090D | 346 | 0.00000 | 0.54925@227 | 已停，BN 污染塌缩 |
| YOLO11n seed123 4090D r2 | 4090D | 88 | 0.00006 | 0.54864@2 | 已停，BN 污染塌缩 |
| YOLO11n seed42 4090D r2 | 4090D | 659 | 约 0.565 | 约 0.570 | 运行中，未塌 |

## 4. 已确认现象

1. A2 早期崩溃主要是检测分支更新过猛。证据见 `ladd/diagnostics/a2_stability/cap2_s0_a2_old_default_results.csv` 与 `cap2_s0_a2_mu1e3_results.csv`。
2. B 阶段 seed0/seed123 的坏 run 权重没有明显 NaN/Inf，但 BN `running_mean/running_var` 被污染，导致 `last.pt` 几乎不出框或出大面积错误框。
3. seed42 在 90 和 4090D 上都更稳定，说明这不是所有 seed 必现的问题。
4. `bstable1e3` 能让 seed123 完整跑完，但 best 出现在较早 epoch，后期仍退化，因此还不是最终可接受修复。
5. BN-freeze 是当前正在验证的最新修复，目的是切断 BN running stats 污染路径。
6. 4090D 上 YOLO11s 三个 seed 当前明显低于 90 的 YOLO11s seed0，需要独立排查协议、数据增强、环境或 checkpoint 选择差异。

## 5. 已有证据文件

| 文件 | 内容 |
|---|---|
| `diagnostics/a2_stability/cap2_s0_a2_old_default_results.csv` | A2 旧默认设置早期崩溃记录 |
| `diagnostics/a2_stability/cap2_s0_a2_mu1e3_results.csv` | A2 MuSGD lr0=0.001 稳定修复记录 |
| `diagnostics/a2_stability/ladd_a2_stability_fix_20260601.png/pdf` | A2 修复可视化 |
| `diagnostics/b_collapse/ladd800r2_six_runs_loss_diagnostics_20260528.png` | B 阶段 6 run loss 诊断图 |
| `diagnostics/b_collapse/ladd_b_entry_kd_shock.png` | B 阶段入口 KD shock 图 |
| `diagnostics/b_collapse/LADD_CRASH_EVIDENCE_20260604_CN.md` | 本次补充的结构化崩溃证据与待查问题 |

## 6. 希望外部老师重点看

1. 冻结 BN running stats 是否是合理修复，还是应该只冻结 teacher/student 某一侧、只冻结检测 head 的 BN、或改为 BN recalibration / smaller batch / EMA 策略。
2. B 阶段是否应从 A2 `best.pt` 启动，还是应选择 A2 最后一轮、固定 epoch、或不用 EMA 权重。
3. KD loss 与 detection loss 的相对尺度是否在长训练中诱导 detector 逐步偏离 SAR baseline。
4. 4090D 上 YOLO11s 明显低于 90 上 seed0 的现象，是否提示代码/协议/数据增强存在未对齐。
