# CoLD 复现

CoLD (Category-Oriented Localization Distillation), TGRS 2023.

原文报告 YOLOv5x on OGSOD-1.0: AP=0.567, baseline=0.463 (+0.104).

## 三次尝试

### v1: matched 模式 (4090D, 400ep)

仅 GT positive anchors 蒸馏。结果 AP=0.470 (+0.025)，远低于原文。

代码和诊断见 `v1_matched_4090d/`

### v2: candidate CPM 模式 (5880 Ada, 50ep)

topk=1000 候选框 + 逐图逐类 KL 蒸馏。
- NCLD 相对 baseline +30%，方向对
- 但 TCLD > NCLD，与论文趋势相反
- ~2 s/it，比正常训练慢 4-5x

代码见 `v2_candidate_5880ada/code/`

### v3: frozen teacher (5880 Ada, 50ep)

预训练 RGB teacher 冻结，做纯离线蒸馏。NCLD 低于 baseline，完全失效。证明 OKD 是 CoLD 的必要组件。

## 核心问题

1. TCLD > NCLD 与论文 Table III 相反
2. candidate 模式 Python 循环速度瓶颈
3. bbox distribution 实现与原文 DFL-bin 不严格等价

详见 `COLD_REPRO_FINAL_CN.md` 和 `results_summary.md`
