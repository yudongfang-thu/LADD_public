# CoLD 原文方法与消融提取说明

原文：`paper/CoLD__2023_TGRS__Category_Oriented_Localization_Distillation_OGSOD.pdf`

全文文本：`paper/CoLD__2023_TGRS__full_text_pdftotext.txt`

关键词上下文：`method_extraction/CoLD_method_and_ablation_keyword_context.txt`

## 原文核心机制

CoLD 的关键组件包括：

| 组件 | 作用 |
| --- | --- |
| CPM | Category Prior Matching，将候选定位蒸馏按类别先验分成 target 和 non-target 两条线 |
| TCLD | Target Category Localization Distillation，只处理与当前类别一致的候选组 |
| NCLD | Non-target Category Localization Distillation，处理非当前类别候选组 |
| IWM | IoU Weighting Module，用 IoU 相关权重调整定位蒸馏贡献 |
| OKD | Online KD，teacher 和 student 在线同步训练，而不是固定离线 teacher |

## 原文消融要点

当前复现主要参考两类消融：

1. TCLD / NCLD 机制线：比较 TCLD-only、NCLD-only、TCLD+NCLD。
2. 模块线：比较 CPM、IWM、OKD 的组合贡献。

我们当前阶段先做 CPM 机制线，因此 90 服务器三条 no-IWM 实验为：

| 实验 | 目的 |
| --- | --- |
| `NCLD no-IWM` | 单独验证 NCLD 是否主导 |
| `TCLD no-IWM` | 单独验证 TCLD 贡献 |
| `BOTH no-IWM` | 验证 TCLD+NCLD 是否接近/超过单项 |

117 服务器当前增加：

| 实验 | 目的 |
| --- | --- |
| `BOTH + IWM` | 验证当前 IWM 近似实现是否稳定、是否改善 BOTH |

## 当前实现相对原文的关键偏差

| 偏差 | 说明 |
| --- | --- |
| YOLOv5-v5.0 无 DFL bins | 当前用候选框集合上的 distribution KL 近似定位蒸馏 |
| candidate 循环未完全向量化 | 存在逐 batch / 逐类 Python 循环，速度显著慢于原文报告 |
| IWM 是近似版 | 当前实现为 candidate-to-GT IoU 的 group mean weighting，不保证完全等价原文 |
| OKD 已修 teacher KD 梯度污染 | teacher 接收 detection loss 梯度，但 CoLD loss 对 teacher prediction 已 detach |

## 建议老师重点看

- `code_versions/v5p0_hbb_current_local/train_cold_v5p0_hbb.py`
- `experiment_records/90_current_online_noiwm_20260604/extracted/code/train_cold_v5p0_hbb.py`
- `experiment_records/117_current_iwm_and_history_20260604/extracted/code/train_cold_v5p0_hbb.py`
- `analysis/NCLD_LOW_DATA_DIAGNOSIS_CN.md`

