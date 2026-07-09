# LADD project onboarding guide

日期: 2026-07-09 CST

本文给新成员一个从 GitHub 接手 `LADD_public` 的阅读顺序。目标不是证明 LADD 已经成功, 而是让读者能理解:

- LADD 想解决什么问题。
- 当前代码和实验目录怎么组织。
- 已经做了哪些实验。
- 哪些结果可作为事实, 哪些仍是 progress / blocked / diagnostic。
- 为什么当前 direct-400 阶段还不能升级 claim。

## 1. 项目一句话

LADD, Learnability-Aware Decomposition Distillation, 研究 RGB-guided SAR object detection 中 teacher-student decomposition KD。核心设想是把 RGB teacher features 拆成可迁移 / 不可迁移部分, 再让 SAR student 学可迁移部分, 同时保护 SAR 自有结构。

当前仓库是 debug / evidence package, 不是 polished release。它保留 runnable code、协议记录、compact result summaries 和大量审计文档, 但不包含 checkpoint 权重、数据集、私有连接信息或完整远端日志。

## 2. 推荐阅读顺序

### 第一层: 当前状态

1. `README.md`
2. `docs/PROJECT_STATUS_20260709_CN.md`
3. `docs/PROJECT_ONBOARDING_20260709_CN.md`
4. `docs/experiments/DIRECT400_EXPERIMENT_LEDGER_20260709_CN.md`
5. `docs/CODE_AND_DOC_CHANGE_MAP_20260709_CN.md`

读完这一层, 应该能知道当前 direct-400 阶段的结论: LADD rescue rows 还低于 strong comparison floor, 当前主路线是 final fact audit / failure localization, 不是 claim writing。

### 第二层: 方法定义和代码入口

1. `docs/ladd_method_definition.md`
2. `docs/method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md`
3. `docs/method/METHOD_OVERVIEW_CN.md`
4. `ladd/code/train_ladd_hbb.py`
5. `ladd/code/src/teacher_student_decomposition_kd_hbb/model.py`
6. `ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py`
7. `ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py`

这一层回答: LADD loss surface 有哪些分量, student/teacher branch 怎么走, phase A/B/C 如何组织。

### 第三层: 当前实验记录

1. `docs/experiments/EXPERIMENT_INDEX_CN.md`
2. `docs/experiments/DIRECT400_EXPERIMENT_LEDGER_20260709_CN.md`
3. `docs/CODE_AND_DOC_CHANGE_MAP_20260709_CN.md`

这一层用于追溯公开实验矩阵和当前发布边界。更深的 PM / Coordinator 记录仍保留在本地工作区, 但其中部分包含远端路径和操作细节, 需要清洗后再进入 public GitHub。

### 第四层: 运行入口

Baseline:

```bash
python baseline/code/train_ogsod_baseline.py \
  --task hbb --model yolo11n.pt \
  --data shared/configs/datasets_public/ogsod1_sar_detect.yaml \
  --imgsz 256 --epochs 400 --cos-lr --mosaic 0.0 --close-mosaic 0
```

LADD HBB:

```bash
python ladd/code/train_ladd_hbb.py \
  --phase b --model yolo11n.pt \
  --data shared/configs/datasets_public/ogsod1_sar_detect.yaml \
  --teacher-data shared/configs/datasets_public/ogsod1_rgb_detect.yaml \
  --teacher-weights <rgb_teacher.pt> \
  --imgsz 256 --epochs 400
```

Comparison:

```bash
bash comparison/code/launch_formal_from_yolo_kd_job.sh fgd n 0 <gpu_id>
```

不要直接把这些命令当当前 claim protocol。当前 direct-400 rescue 阶段有额外 source / matched-control / final-fact audit gate。

## 3. 当前证据规则

| 规则 | 含义 |
|---|---|
| pure direct-400 only | 当前正式 evidence 只看 400 epoch 从头训练, 不混 row400-from-800 / 800 / 1600 / reload |
| running != final | progress row 不能进最终表 |
| no cross-server gain | 3090 / 90 / 4090 不跨机器算 matched gain |
| comparison is comparison | CMDistill / FGD / LD 只作为 comparison lines |
| LADD gain requires matched controls | LADD plain / singleproj / fusion 只能按 same-machine / same-seed / same-protocol controls 判断 |
| claim upgrade requires audit | final fact audit、comparison stability、matched control、provenance / health scan 都必须齐 |

## 4. 当前结论

截至 2026-07-09:

- 当前 active goal 仍是 `NOT_COMPLETE`。
- 当前 claim status 是 `claim_ready=no`。
- 3090 五条 LADD seed0 direct-400 rescue rows 已完成审计候选, 但全部低于 FGD seed0 floor `0.55147` AP50-95。
- 当前 best LADD AP50-95 是 `0.50350`, gap to FGD floor 是 `-0.04797`。
- 90 CMDistill / LD seed42 / seed123 有强 monitor 数字, 但最新 final fact audit 因 90 SSH/TUN 访问阻断, 还不能转为 accepted final facts。
- 下一步主线应是 `DISPATCH_FAILURE_LOCALIZATION`, 而不是 seed expansion 或 claim writing。

## 5. 新成员接手 checklist

1. 先确认 GitHub 分支和本地 dirty tree, 不要 `git add -A`。
2. 阅读 `docs/PROJECT_STATUS_20260709_CN.md` 和 direct-400 ledger。
3. 确认任何要用的结果是否是 `FINAL_FACT_READY`, 而不是 `running`, `progress-only`, `final-pending`, 或 `blocked`。
4. 如果要继续实验, 先解决 launcher / CLI parity blocker, 再做 no-launch remote preflight。
5. 如果要写论文, 只使用 claim ledger / final evidence table 已确认的事实, 不要把失败定位假设写成结论。
6. 如果要更新 GitHub, 优先提交 compact sanitized docs; 含远端绝对路径、权重路径、raw logs 的 PM 原始记录需要先清洗。
