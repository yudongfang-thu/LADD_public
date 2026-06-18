# LADD B-Entrance 曲线核查（2026-06-14）

本目录用于澄清当前 N/S B-entrance 实验的真实设定，并画出 AP/loss 曲线判断训练长度是否足够。

## 图

![AP curves](/Users/yudongfang/Desktop/光sar/LADD_public/docs/experiments/figures/ladd_b_entrance_trends_20260614/fig1_b_entrance_ap_curves.png)

![Detector loss curves](/Users/yudongfang/Desktop/光sar/LADD_public/docs/experiments/figures/ladd_b_entrance_trends_20260614/fig2_b_entrance_detector_loss_curves.png)

## 表

- 曲线摘要：`b_entrance_curve_summary_20260614.csv`
- 设定澄清：`b_entrance_setting_clarification_20260614.csv`

## 关键澄清

当前 N3/N4、S3/S4 的 detector 初始化不是 yolo 初始权重，而是 SAR converged baseline best。它们加载 A2 last 的 decomposition/reachability 模块，因此诊断的是“干净 SAR detector + A2 decomposition 是否有收益”，不是“从 yolo-init 开始的主线 LADD”。

缺失实验包括：N2-last/S2-last，以及如果需要复现你说的 yolo-init 语义，则需要新增 N3y/S3y、N4y/S4y。
