# 给高级模型的当前证据审阅提示（2026-06-25）

请从 GitHub 仓库审阅：

- 仓库：`https://github.com/yudongfang-thu/LADD_public`
- 审阅包路径：`docs/review_packages/mainline_method_search_20260624/`
- 当前证据快照：`docs/review_packages/mainline_method_search_20260624/evidence/current_evidence_20260625_1115/`
- 当前对比表：`docs/review_packages/mainline_method_search_20260624/tables/current_evidence_20260625_1115/`
- 当前运行日志：`docs/review_packages/mainline_method_search_20260624/runtime_logs/current_evidence_20260625_1115/`
- 当前代码快照：`docs/review_packages/mainline_method_search_20260624/code_refs/current_20260625_1115/`
- 真实实现位置：`ladd/code/src/teacher_student_decomposition_kd_hbb/`
- 训练入口：`ladd/code/train_ladd_hbb.py`

## 审阅问题

我们现在不需要继续盲目缝合实验，而是需要判断已有证据到底说明了什么：

1. LADD/dynamic 当前的 YOLO-init 正增益，是否主要来自 `plain` 结构，而不是 `singleproj`、`wo_s_rec`、`capR` 或更复杂缝合？
2. `probe/reach` 结构是否有独立必要性？它是否和 student 单投影头作用重复？
3. 可达损失是否应该向 student backbone 反传？当前 `detach_reach_probe` 证据还太早，应如何设计最小对照？
4. 为什么“相比 det-only 正”在当前证据中没有转化成“相比 plain 正”？
5. 下一步应该先做机制审阅、最小消融，还是继续大规模训练？

## 必看文件

先读：

- `README_CN.md`
- `tables/current_evidence_20260625_1115/current_evidence_focus_table.csv`
- `tables/current_evidence_20260625_1115/current_evidence_3090_vs_det_and_plain.csv`
- `tables/current_evidence_20260625_1115/current_evidence_4090_vs_det_and_plain.csv`
- `tables/current_evidence_20260625_1115/run_copy_manifest.csv`

再看原始数据：

- `evidence/current_evidence_20260625_1115/remote_3090/*/results.csv`
- `evidence/current_evidence_20260625_1115/remote_3090/*/ladd_diagnostics.csv`
- `evidence/current_evidence_20260625_1115/remote_3090/*/args.yaml`
- `evidence/current_evidence_20260625_1115/remote_4090/*/results.csv`
- `evidence/current_evidence_20260625_1115/remote_4090/*/ladd_diagnostics.csv`
- `evidence/current_evidence_20260625_1115/remote_4090/*/args.yaml`

必要时追启动与异常：

- `runtime_logs/current_evidence_20260625_1115/remote_3090/`
- `runtime_logs/current_evidence_20260625_1115/remote_4090/`

代码审阅：

- `code_refs/current_20260625_1115/model.py`
- `code_refs/current_20260625_1115/loss.py`
- `code_refs/current_20260625_1115/trainer.py`
- `code_refs/current_20260625_1115/base_hbb.py`
- `code_refs/current_20260625_1115/train_ladd_hbb.py`

## 当前关键数字

3090 同机 matched evidence：

- `dynamic_plain_yoloinit`：rows 364，latest AP50-95 `0.486110`，late20 delta vs det-only `+0.017197`。
- `dynamic_singleproj_yoloinit`：rows 463，late20 delta vs det-only `+0.014497`，但 late20 delta vs plain `-0.000660`。
- `dynamic_wo_s_rec_yoloinit`：rows 492，late20 delta vs det-only `+0.015645`，但 late20 delta vs plain `-0.003238`。
- `dynamic_reach_rawinput_yoloinit`：rows 340，late20 delta vs det-only `+0.015467`，late20 delta vs plain `+0.000039`，基本与 plain 持平。
- `capR2/capR4`：约 135/138 rows，仍偏早，当前只弱正于 det-only，不能说明 capR 有稳定增益。
- `detach_probe`：10 rows，非常早，不能下结论。

4090 evidence：

- `dynamic_plain_anchor_4090_yoloinit`：14 rows，太早。
- combo/frankenstein 多条已停止，仅作诊断；大多相对 4090 plain anchor 的早期 matched delta 不好。
- `dynamic_resume_fixed` 不是 YOLO-init 主线证据，只能作为 reload/resume context。

## 审阅边界

- 主线证据只看 YOLO-init；不要把 reload/resume 当正结果。
- 3090 候选只和 3090 same-pipeline control 比；4090 候选只和 4090 same-pipeline control 比。
- 100 epoch 只能早筛，最终 claim 需要 e800、late-window、final/best 和 seed。
- 已停止 combo/frankenstein run 不能作为主线正结果。
- 不要只看 AP；请结合 `args.yaml`、`ladd_diagnostics.csv` 和启动命令判断是否存在协议混淆。

## 期望输出

请输出：

1. 当前证据是否支持继续把 `plain` 当作最小主线 anchor。
2. `singleproj`、`wo_s_rec`、`reach_rawinput`、`capR2/capR4` 相对 plain 的真实价值判断。
3. 对 probe/reach 结构的机制解释：它是否必要、是否重复、是否应阻断梯度。
4. 一组最小且可判别的下一步实验，不超过 4 条，每条说明必要对照和早筛判据。
5. 明确哪些实验应暂停，哪些应继续跑满。
