# 当前证据摘要（2026-06-25 11:15 CST 快照）

本页总结 `current_evidence_20260625_1115` 快照。该快照只整理证据，不新增实验，不把 stopped combo/frankenstein 当主线正结果。

## 快照内容

- 原始轻量数据：`evidence/current_evidence_20260625_1115/`
  - 每条 run 包含 `results.csv`、`ladd_diagnostics.csv`、`args.yaml`。
- 启动与运行日志：`runtime_logs/current_evidence_20260625_1115/`
  - 包含 `.cmd.sh`、外层 `.log`、smoke 日志、停止记录和 runtime snapshot。
- 对比表：`tables/current_evidence_20260625_1115/`
  - `current_evidence_focus_table.csv`
  - `current_evidence_3090_vs_det_and_plain.csv`
  - `current_evidence_4090_vs_det_and_plain.csv`
  - `run_copy_manifest.csv`
  - `file_manifest.csv`
- 代码快照：`code_refs/current_20260625_1115/`
- 压缩包：`current_evidence_20260625_1115.tar.gz`

## 当前核心观察

3090 同机 matched evidence 显示：

- `dynamic_plain_yoloinit` 是当前最重要的最小 anchor：rows 364，latest AP50-95 `0.486110`，late20 delta vs det-only `+0.017197`。
- `dynamic_singleproj_yoloinit`：rows 463，late20 delta vs det-only `+0.014497`，但 late20 delta vs plain `-0.000660`。
- `dynamic_wo_s_rec_yoloinit`：rows 492，late20 delta vs det-only `+0.015645`，但 late20 delta vs plain `-0.003238`。
- `dynamic_reach_rawinput_yoloinit`：rows 340，late20 delta vs det-only `+0.015467`，late20 delta vs plain `+0.000039`，基本与 plain 持平。
- `dynamic_capR2_yoloinit` / `dynamic_capR4_yoloinit`：约 135/138 rows，当前只弱正于 det-only，尚不能说明 capR 带来稳定增益。
- `dynamic_plain_detach_reach_probe_yoloinit`：10 rows，过早，不能下结论。

4090 当前证据显示：

- `dynamic_plain_anchor_4090_yoloinit` 只有 14 rows，太早，不适合作为强 anchor。
- `combo_*_STOPPED` 与 `frank_*_STOPPED` 是已停止诊断线，不作为主线证据。
- `dynamic_resume_fixed` 属于 resume/reload-like context，不是 YOLO-init 主线正结果。

## 审阅时最容易误读的点

- “相对 det-only 正”不代表“相对 plain 正”。当前 `singleproj` 和 `wo_s_rec` 在 3090 上都没有超过 plain。
- 已停止的 combo/frankenstein 线可以用于判断盲目缝合的风险，但不能被包装为候选主线。
- 4090 plain anchor 太早，4090 上相对 plain 的比较只适合作为早期诊断。
- reload/resume 线只解释混淆，不作为当前主线证据。

## 推荐高级模型入口

请使用：

- `REVIEW_PROMPT_CURRENT_20260625_CN.md`
- `tables/current_evidence_20260625_1115/current_evidence_focus_table.csv`
- `tables/current_evidence_20260625_1115/current_evidence_3090_vs_det_and_plain.csv`
- `tables/current_evidence_20260625_1115/current_evidence_4090_vs_det_and_plain.csv`

然后按需打开对应原始 `results.csv`、`ladd_diagnostics.csv` 和 `.cmd.sh`。
