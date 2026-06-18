# LADD 主线诊断线 forensic map（2026-06-14）

## 1. 这条线在回答什么

LADD 主线诊断线的核心问题是：

> 为什么历史上健康的 LADD 主线，在近期 no-mosaic / B-stage 诊断中出现崩溃、后期退化或低位平台？

当前不要继续铺新实验。先把历史健康主线、近期 A2/B 诊断、B800 restart 结果放到同一张因果图中复盘。

## 2. Canonical 入口

| 类型 | 路径 |
|---|---|
| Forensic review 初版 | [LADD_FORENSIC_REVIEW_20260614_CN.md](LADD_FORENSIC_REVIEW_20260614_CN.md) |
| LADD 线 registry summary | [ladd_line_registry_summary_20260614.csv](ladd_line_registry_summary_20260614.csv) |
| LADD 主线旧结果台账 | [../../../ladd/results/LADD_RESULTS_CN.md](../../../ladd/results/LADD_RESULTS_CN.md) |
| 历史健康主线对比 | [../LADD_CONVERGED_MAINLINE_COMPARISON_20260613_CN.md](../LADD_CONVERGED_MAINLINE_COMPARISON_20260613_CN.md) |
| Mosaic vs no-mosaic 协议重合对比 | [LADD_PROTOCOL_MOSAIC_VS_NOMOSAIC_OVERLAP_20260614_CN.md](LADD_PROTOCOL_MOSAIC_VS_NOMOSAIC_OVERLAP_20260614_CN.md) |
| 当前 baseline overlay | [../LADD_CURRENT_BASELINE_OVERLAY_ANALYSIS_20260614_CN.md](../LADD_CURRENT_BASELINE_OVERLAY_ANALYSIS_20260614_CN.md) |
| B800 restart 曲线分析 | [../LADD_B800_RESTART_CURVE_ANALYSIS_20260614_CN.md](../LADD_B800_RESTART_CURVE_ANALYSIS_20260614_CN.md) |
| B entrance 压缩 schedule 归档 | [../LADD_B_ENTRANCE_COMPRESSED_SCHEDULE_ARCHIVE_20260614_CN.md](../LADD_B_ENTRANCE_COMPRESSED_SCHEDULE_ARCHIVE_20260614_CN.md) |
| 90 formal baseline 证据 | [../../../ladd/results/ladd90_formal_baselines_20260612/summary/ladd90_formal_baseline_summary_20260612.csv](../../../ladd/results/ladd90_formal_baselines_20260612/summary/ladd90_formal_baseline_summary_20260612.csv) |
| 双卡关机同步证据 | [../../../ladd/results/ladd4090_shutdown_sync_20260614/](../../../ladd/results/ladd4090_shutdown_sync_20260614/) |

## 3. 当前事实分层

| 事实 | 证据 | 读法 |
|---|---|---|
| N1 baseline continuation 很稳 | B800 restart 中 N1 best/last 约 `0.575-0.577` | B800 schedule 和 detection-only continuation 本身不是主要问题 |
| N2 A2-best/A2-last full LADD 出现 NaN/退化 | B800 restart 中 N2 在 B 200-300 epoch 区间异常 | 问题更像 A2 checkpoint + B LADD loss/数值稳定性组合 |
| N3/N4 yolo-init + A2 decomp 低位平台 | shutdown sync 中 N3 约 `0.494`，N4 约 `0.476` | 只加载 decomposition 不能代表原主线，也不能救弱 detector |
| 历史 90 主线曾健康 | converged mainline 对比与 `LADD_RESULTS_CN.md` | 必须做配置 diff，确认近期实验是否偏离真实主方法 |
| mosaic100 与 no-mosaic 的重合证据主要在 n | protocol overlap 对比 | no-mosaic 绝对 AP 更高，但相对同协议 baseline 的 LADD gain 更小，且 seed123 稳定性更窄 |
| B100/B120 早期实验不能解释 B800 后劲 | compressed schedule archive | 这些 run 只能作为入口/smoke 诊断 |

## 4. Forensic review 要检查的轴

| 轴 | 问题 |
|---|---|
| phase lineage | detector 来自 YOLO init / SAR baseline / A2 best / A2 last？decomp 来自哪里？ |
| loss definition | A1/A2/B 各阶段到底开了 detection、rec、reach、KD、sep、residual aux 中哪些？ |
| optimizer/schedule | A2/B 的 LR、warmup、cos schedule、训练长度是否和历史健康主线一致？ |
| BN/stability | BN freeze、grad clip、NaN recovery、diagnostic logging 是否改变训练行为？ |
| server/protocol | 90、4090、AutoDL 的数据路径、代码 commit、batch、mosaic、Albumentations 是否一致？ |
| metric window | best、last、late-regression、NaN 前后窗口是否分开解释？ |

## 5. 当前建议

1. 暂停新 LADD 训练。
2. 先生成 `LADD_FORENSIC_REVIEW_20260614_CN.md`，把所有近期实验标成 `mainline / diagnostic / invalid / compressed-schedule`。
3. 画同一协议下的四组曲线：历史健康主线、N1 baseline continuation、N2 A2-best/last、N3/N4 split-load。
4. 对 N2 做 NaN 前后 20 epoch 事件窗口分析。
5. 只有 forensic review 完成后，才考虑一个严格 replay：完全复现历史健康主线的配置和 loss 组合。
