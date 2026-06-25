# 高级模型审阅 Prompt：OGSOD 3090/4090 LADD 主线探索

请审阅这个 GitHub 仓库中的审阅包：

```text
docs/review_packages/ogsod_3090_4090_mainline_review_20260625/
```

当前主线代码位置：

```text
ladd/code/src/teacher_student_decomposition_kd_hbb/
ladd/code/train_ladd_hbb.py
```

请先读：

```text
docs/review_packages/ogsod_3090_4090_mainline_review_20260625/README_CN.md
docs/review_packages/ogsod_3090_4090_mainline_review_20260625/tables/experiment_catalog_20260625.csv
docs/review_packages/ogsod_3090_4090_mainline_review_20260625/runtime_snapshots/live_status_20260625_1414.csv
```

然后根据需要检查每条 run 的原始数据：

```text
docs/review_packages/ogsod_3090_4090_mainline_review_20260625/evidence/current_evidence_20260625_1115/remote_3090/*/results.csv
docs/review_packages/ogsod_3090_4090_mainline_review_20260625/evidence/current_evidence_20260625_1115/remote_3090/*/args.yaml
docs/review_packages/ogsod_3090_4090_mainline_review_20260625/evidence/current_evidence_20260625_1115/remote_3090/*/ladd_diagnostics.csv
docs/review_packages/ogsod_3090_4090_mainline_review_20260625/evidence/current_evidence_20260625_1115/remote_4090/*/results.csv
docs/review_packages/ogsod_3090_4090_mainline_review_20260625/evidence/current_evidence_20260625_1115/remote_4090/*/args.yaml
docs/review_packages/ogsod_3090_4090_mainline_review_20260625/evidence/current_evidence_20260625_1115/remote_4090/*/ladd_diagnostics.csv
```

以及 4090 旧 restart/old-commit 证据：

```text
docs/review_packages/ogsod_3090_4090_mainline_review_20260625/evidence/ogsod_4090_restart_20260625/
```

## 审阅目标

我们需要找到一个可靠的 YOLO-init 主线方法，最好能稳定超过同机同协议 det-only control。请不要把 reload/resume 结果作为主线正结果。

请重点回答：

1. 当前最有希望的主线候选是哪几条？请分别说明证据强度、风险和下一步需要补什么。
2. `dynamic_wo_s_rec` 的正趋势是否说明 student reconstruction 应该被移除或大幅降权？
3. `dynamic_singleproj` 的正趋势是否说明 student split/probe 结构冗余？
4. `dynamic_plain`、`dynamic_singleproj`、`dynamic_wo_s_rec` 三者之间是否能形成一个更简洁的新主线？
5. `dynamic_reach_rawinput` 与 projected/raw teacher 相关实验是否说明 reach input/source 设计有问题？
6. `capR2` 与 `capR4` 的结果和 `ladd_diagnostics.csv` 是否支持 capR 机制？如果不支持，问题在哪里？
7. `no_reach`、`detach_reach_backbone`、`freeze_probeB`、`plain_detach_reach_probe` 这些线应该如何解释 reach 梯度路径？
8. 4090 `oldcommit_ProbeA` 早期正结果为什么可能不能复现？最可能的差异来源是什么？
9. stopped combo/frankenstein 线为什么大多不好？是否说明多个 knob 之间存在冲突？
10. 基于现有证据，请给出一个优先级排序的实验矩阵，尽量少而关键。

## 硬性判断规则

请遵守：

```text
3090 候选只和 3090 detonly_control 比。
4090 候选只和 4090 same-pipeline det-only/control 比。
跨机结果只能作为趋势参考。
reload/resume 不能作为 YOLO-init 主线证据。
100 epoch 只能 early screen，不能当最终 claim。
e800 + late50/final/best + seed 才能作为最终主线 claim。
```

请明确区分：

```text
mainline candidate
diagnostic evidence
negative control
stopped/invalid/frankenstein evidence
reload/resume context
```

## 期望输出

请输出：

1. 对当前证据的总体判断。
2. 一个按优先级排序的候选方法列表。
3. 每个候选方法的机制解释与反证风险。
4. 需要停止、继续、补跑的实验列表。
5. 对当前 LADD HBB 代码设计的具体修改建议，尤其是 student_rec、student split/probe、reach gradient path、capR/KD gating。
6. 哪些结论目前证据不足，不能写进论文或主线 claim。
