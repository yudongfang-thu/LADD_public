# CoLD 复现结果

论文：CoLD (Category-Oriented Localization Distillation), TGRS 2023
原文报告：YOLOv5x on OGSOD-1.0, AP=0.567 (baseline=0.463, +0.104)

## 复现历程

### Attempt 1: matched 模式 (4090D, 400ep)

- 模式：`COLD_LOSS_MODE=matched`，仅 GT positive anchors 蒸馏
- 结果：AP=0.470 (+0.025 over baseline)
- 问题：matched 只覆盖少量正样本，NCLD 无法大规模做 nontarget 蒸馏
- 结论：方向对但增益远低于原文

实现文件见 `attempt_1_matched_4090d/`

### Attempt 2: candidate 模式 (5880 Ada, 50ep)

- 模式：`COLD_LOSS_MODE=candidate, CANDIDATE_TOPK=1000`
- 在线 NCLD 50ep：mAP@.5=0.520, mAP=0.260 (baseline @50ep: 0.429/0.200, **+30%**)
- 在线 TCLD 50ep：mAP@.5=0.604, mAP=0.323 (**+61%**)
- 问题：TCLD > NCLD，与论文趋势相反
- 速度：~2 s/it（candidate 模式 Python 逐图循环）

实现文件见 `attempt_2_candidate_5880ada/`

### Attempt 3: frozen teacher (5880 Ada, 50ep)

- 模式：`TEACHER_DET_WEIGHT=0`，预训练 RGB teacher 冻结
- NCLD 50ep：mAP 低于 baseline
- 结论：OKD（在线教师训练）是 CoLD 的关键组件，不可省略

实现文件见 `attempt_3_offline_frozen/`

## 核心问题

1. **TCLD > NCLD**：与论文 Table III (NCLD=0.563 >> TCLD=0.502) 相反
2. **速度**：candidate 模式 ~2 s/it，是正常训练的 4-5 倍，论文报告仅 3.6% 开销
3. **绝对性能差距**：50ep NCLD mAP=0.260，按论文 400ep 趋势推算最终约 0.40-0.45，距原文 0.563 仍有差距

## 可能的根因

1. bbox distribution 实现：YOLOv5 v5.0 无 DFL bins，用候选框 softmax KL 近似
2. 单 seed 噪声：NCLD 和 TCLD 串行跑，随机状态不同导致 teacher 收敛差异
3. candidate 循环未向量化
