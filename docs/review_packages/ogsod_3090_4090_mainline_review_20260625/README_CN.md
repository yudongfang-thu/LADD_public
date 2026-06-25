# OGSOD 3090/4090 主线探索审阅包

生成时间：2026-06-25  
当前分支：`codex/ladd-mainline-review-evidence-20260625`  
范围：`ladd3090-zw1` 与 `ladd4090-zw1` 上围绕 OGSOD YOLO-init LADD/Dynamic/ProbeA/capR/reach 机制探索的轻量证据。  
不包含：checkpoint 权重、TensorBoard event、W&B 原始目录、数据集原图。

## 1. 审阅目的

这个包用于回答当前主线探索的核心问题：

```text
我们是否能在 YOLO-init、same-machine、same-pipeline det-only control 下，
找到一条稳定正增益的 LADD-like 主线？
```

并进一步拆开机制问题：

```text
1. dynamic/plain 结构本身是否已经有稳定正增益？
2. student_rec 是否拖累性能？
3. student split/probe 是否必要，还是和 single projection / student-z 作用重复？
4. reach loss 是否真的提供正贡献，还是只是普通辅助正则？
5. reach 梯度是否应该传到 student backbone / probe？
6. capR=2 与 capR=4 是否能解释 current dynamic 的小正增益？
7. 4090 上旧 ProbeA/old-commit 的早期大增益为什么不能稳定复现？
```

## 2. 包结构

```text
docs/review_packages/ogsod_3090_4090_mainline_review_20260625/
├── README_CN.md
├── REVIEW_PROMPT_CN.md
├── evidence/
│   ├── current_evidence_20260625_1115/
│   │   ├── remote_3090/*/{results.csv,args.yaml,ladd_diagnostics.csv}
│   │   └── remote_4090/*/{results.csv,args.yaml,ladd_diagnostics.csv}
│   ├── ladd_current_matrix_20260625_131506/
│   │   ├── data/{3090,4090}/*.csv
│   │   └── plots_*/*.png, comparison_table.*
│   ├── current_active_20260625_1001/
│   │   ├── data/*.csv
│   │   └── 3090/4090 active plot tables
│   └── ogsod_4090_restart_20260625/
│       ├── data/*.csv
│       └── comparison_table.*, curves/delta plots
├── runtime_snapshots/
│   ├── live_status_20260625_1414.csv
│   └── live_status_20260625_1414.md
└── tables/
    ├── experiment_catalog_20260625.csv
    ├── experiment_catalog_20260625.md
    ├── source_current_evidence_20260625_1115/*.csv
    ├── capr_existing_run_audit_3090_20260625.csv
    ├── capr_existing_run_audit_4090_20260625.csv
    └── learnability_*_20260625.csv
```

最重要的入口：

- `tables/experiment_catalog_20260625.csv`：统一实验目录，66 条记录，含配置字段、性能、状态、验证目的。
- `runtime_snapshots/live_status_20260625_1414.csv`：2026-06-25 14:14 CST 四卡实时进度。
- `evidence/current_evidence_20260625_1115/`：当时从远端拉回的 `results.csv`、`args.yaml`、`ladd_diagnostics.csv`。
- `evidence/ogsod_4090_restart_20260625/`：4090 早期 restart/old ProbeA/dynamic 一批证据。

## 3. 证据使用规则

### 3.1 主线正结果只看 YOLO-init

`dynamic_resume_fixed` 属于 resume/reload-like context，只能用于理解 reload confound，不能作为主线正结果。

### 3.2 必须同机同协议比较

```text
3090 候选只和 3090 detonly_control 比。
4090 候选只和 4090 same-pipeline det-only/control 比。
跨机结果只能看趋势，不能直接算 claim delta。
```

### 3.3 100 epoch 只是早筛

当前采用的早筛标准：

```text
PROMISING_EARLY:
  matched >= 100
  late20_delta >= +0.010 AP50-95
  latest_delta > 0

WATCH:
  late20_delta > 0 但 < +0.010

LOW_PRIORITY:
  matched >= 120 且 late20_delta <= 0
```

最终主线仍需要：

```text
e800 + late50/final/best + seed 补充。
```

### 3.4 stopped/frankenstein 线不能作为主线

`combo_*_STOPPED`、`frank_*_STOPPED` 是诊断线，用来判断多个 knob 合体是否明显失败或互相冲突。它们不能作为主线正结果。

## 4. 3090 当前主要实验

实时快照：`runtime_snapshots/live_status_20260625_1414.csv`

| 方法 | 14:14 rows | latest AP50-95 | 当前状态 | 验证内容 |
|---|---:|---:|---|---|
| `detonly_control` | 664/800 | 0.54340 | control，预计 06-25 18:44 完成 | 3090 same-pipeline YOLO-init det-only 对照 |
| `dynamic_plain` | 433/800 | 0.50395 | running，早期正 | dynamic/plain 主线 anchor |
| `dynamic_wo_s_rec` | 562/800 | 0.53528 | running，接近完成 | 去掉 student_rec 是否更稳定/更高 |
| `dynamic_singleproj` | 529/800 | 0.52863 | running | student split 是否可被 single projection 替代 |
| `dynamic_reach_rawinput` | 412/800 | 0.49504 | running | reach 输入从 adapter 改为 raw/input 的影响 |
| `dynamic_capR2` | 193/800 | 0.40494 | running/WATCH | capR=2 是否真实带来机制收益 |
| `dynamic_capR4` | 197/800 | 0.40622 | running/WATCH | capR=4/no-cap-ish 对照 |
| `plain_detach_reach_probe` | 68/800 | 0.32244 | pre100 | reach 梯度是否应在 probe 前截断 |
| `studentz_split_detach` | 20/800 | 0.17495 | pre100 | 直接用 student-z split 做 reach/KD target 的诊断 |
| `studentz_singleproj_detach` | 20/800 | 0.16276 | pre100 | student-z singleproj + detach 诊断 |

当前 3090 上最值得继续等 e800 的线：

```text
dynamic_wo_s_rec
dynamic_singleproj
dynamic_plain
dynamic_reach_rawinput
```

注意：这些 early positive 是相对同机 det-only control 的同 epoch 对照，不代表已经高于 e800 final control。

## 5. 4090 当前主要实验

实时快照：`runtime_snapshots/live_status_20260625_1414.csv`

| 方法 | 14:14 rows | latest AP50-95 | 当前状态 | 验证内容 |
|---|---:|---:|---|---|
| `detonly_control_caprgroup` | 326/800 | 0.46078 | control，预计 06-26 03:43 完成 | 4090 same-pipeline det-only 对照 |
| `plain_anchor` | 92/800 | 0.34782 | pre100 接近 100 | 4090 current mechanism matrix 的 plain anchor |
| `no_reach` | 47/800 | 0.28543 | pre100 | 去掉 reach，看正增益是否仍存在 |
| `detach_reach_backbone` | 47/800 | 0.28945 | pre100 | reach 梯度不进 student backbone |
| `freeze_probeB` | 44/800 | 0.27657 | pre100 | B 阶段冻结 reach probe |
| `wo_srec` | 32/800 | 0.22995 | pre100 | 去 student_rec |
| `detach_reach_wo_srec` | 33/800 | 0.22852 | pre100 | detach reach + 去 student_rec |
| `studentz_split` | 26/800 | 0.19818 | pre100 | student-z split 直接路径 |
| `studentz_singleproj` | 26/800 | 0.19072 | pre100 | student-z single projection 直接路径 |
| `dynamic_resume_fixed` | 341/800 | 0.55466 | context only | resume/reload-like，不作为 YOLO-init 主线证据 |

4090 当前批次整体太早，主要用于验证机制走向，不宜直接下正负结论。

## 6. 4090 旧 restart / old-commit 证据

来源：`evidence/ogsod_4090_restart_20260625/`

| 方法 | rows/progress | best AP50-95 | latest AP50-95 | 解读 |
|---|---:|---:|---:|---|
| `det-only` | 503 rows | 0.50665 | 0.50665 | 旧 4090 对照，未完成 800 |
| `ProbeA` | 403 rows | 0.48823 @384 | 0.00886 | 后期异常掉崩，不能直接当正常 ProbeA |
| `dynamic` | 345 rows | 0.47031 | 0.47031 | 早期小正趋势，但未完成 |
| `dynamic_kd0p5` | 78 rows | 0.31847 @61 | 0.03215 | 后期掉崩 |
| `dynamic_reach0p5` | 79 rows | 0.31928 | 0.31928 | pre100，小正趋势 |
| `dynamic_srec0p05` | 74 rows | 0.31065 @58 | 0.02954 | 后期掉崩 |
| `dynamic_teacher_projectedraw` | 86 rows | 0.32369 | 0.32369 | pre100，小正趋势 |
| `oldcommit_ProbeA` | 215/700 | 0.42046 | 0.42046 | old-code/old-condition 早期相对 det-only 约 +0.01，不能直接视为当前主线 |

这一批最重要的作用不是证明 ProbeA 成立，而是说明：

```text
旧实现/旧条件下确实出现过较明显 early positive；
但 current-code/当前重启批次没有稳定复现，且部分曲线后期异常掉崩。
```

## 7. stopped combo/frankenstein 诊断线

来源：

```text
tables/source_current_evidence_20260625_1115/current_evidence_4090_vs_det_and_plain.csv
tables/source_current_evidence_20260625_1115/current_evidence_3090_vs_det_and_plain.csv
```

典型方法：

```text
combo_singleproj_raw_STOPPED
combo_raw_wo_srec_STOPPED
frank_singleproj_adapter_wo_srec_capR2_STOPPED
frank_singleproj_adapter_lowsrec_capR2_STOPPED
frank_singleproj_raw_lowsrec_capR2_STOPPED
frank_singleproj_adapter_wo_srec_capR4_STOPPED
frank_singleproj_raw_wo_srec_capR4_STOPPED
```

这些线用来验证“把几个看似有希望的 knob 直接缝合”是否有效。当前证据显示多数早期相对 plain 并不好，因此只能作为负面/交互证据。

## 8. 配置字段索引

`tables/experiment_catalog_20260625.csv` 中保留了以下关键配置字段：

```text
rank_d_neg_cap
alpha_s_rec
alpha_kd
lambda_reach
student_branch_mode
reach_input_mode
```

更完整的参数请看每条 run 对应的：

```text
evidence/current_evidence_20260625_1115/remote_*/<run_key>/args.yaml
```

训练期 capR/reach 诊断请看：

```text
evidence/current_evidence_20260625_1115/remote_*/<run_key>/ladd_diagnostics.csv
tables/capr_existing_run_audit_3090_20260625.csv
tables/capr_existing_run_audit_4090_20260625.csv
```

learnability 相关诊断请看：

```text
tables/learnability_*_20260625.csv
```

## 9. 建议审阅问题

请重点审阅：

1. `dynamic_wo_s_rec`、`dynamic_singleproj`、`dynamic_plain`、`dynamic_reach_rawinput` 的 early gain 是否只是同 epoch schedule 偏差，还是有持续机制信号。
2. `dynamic_wo_s_rec` 是否说明 student reconstruction 是负项；如果是，是否应从主线默认项中移除或降权。
3. `dynamic_singleproj` 是否说明 student split/probe 设计过度复杂，与 single projection 功能重复。
4. `capR2` 与 `capR4` 差异是否足以支持 capR 机制；当前看二者差异不大，需要结合 `ladd_diagnostics.csv`。
5. `no_reach`、`detach_reach_backbone`、`freeze_probeB`、`plain_detach_reach_probe` 能否回答 reach 梯度应该流向哪里。
6. `oldcommit_ProbeA` 的正信号是否来自旧代码、旧 A1 cache、旧数据/增强、还是 ProbeA 本身。
7. stopped combo/frankenstein 失败是否说明 knob 之间互相冲突，或者只是启动太早/控制组太弱。

## 10. 当前局限

- 4090 SSH 在 2026-06-25 下午多次遇到 provider/fake-DNS 断连；本包优先使用已经成功拉回的轻量证据和 14:14 进度快照。
- 部分 active run 仍未到 100 epoch，特别是 4090 current mechanism matrix，不能做强结论。
- `experiment_catalog_20260625.csv` 同时保留多个时间点的快照，因此同一方法可能出现多行，这是有意保留 provenance。
- 后续若 3090/4090 active run 达到 e800，应追加新的 `results.csv`、`ladd_diagnostics.csv`、`args.yaml`，并重新生成 catalog。
