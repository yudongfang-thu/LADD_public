# 已排除方法与无效结果

最后更新：2026-06-04

## 归档边界

| 路径 | 内容 | 排除原因 |
|---|---|---|
| `cold/` | CoLD 全部复现代码、日志、论文摘录、服务器记录和旧顶层文档 | 协议、容量与当前 YOLO11 controlled table 不可直接比较；降级为历史复现证据 |
| `crosskd/` | CrossKD-style 说明与历史结果 | 当前 YOLO port 没有真正 cross-head routing |
| `legacy_results/fgd_pre_20260604/` | FGD 修复前结果 | 旧实现缺少 teacher feature attention |
| `legacy_results/ld_softlogit_pre_20260604/` | 旧 LD profile 结果 | 旧实现实际是分类 soft-logit KD，不是 Localization Distillation |
| `literature/COMPARISON_METHOD_SURVEY_20260528_CN.md` | 旧五方法筛选方案 | 仍把 CoLD/CrossKD 列为必跑，已被 2026-06-04 四方法方案取代 |

## 使用规则

- 不从本目录启动正式实验。
- 不把本目录结果计入修正版主表或多 seed 统计。
- 可以引用这些材料解释方法淘汰、复现失败和实现修正过程。
- CoLD 原文数值如需展示，只能进入 external reported table，并明确协议不可比。
