# LADD Public 精简仓库说明

最后更新：2026-06-06 16:25 CST

本文档说明当前公开仓库的范围。此前用于排查问题的大体量证据包已经移出当前
public 分支；当前 public 分支只保留论文主线代码、协议文档、关键结果摘要和必要论文资料。

## 1. 已放入的核心材料

| 模块 | 已放入内容 | 位置 |
|---|---|---|
| Baseline 说明 | 最新 baseline 数字、gap、缺口 | `baseline/results/BASELINE_RESULTS_CN.md`, `docs/experiments/BASELINE_LADD_STATUS_CN.md` |
| LADD 当前代码 | 当前 HBB LADD trainer/loss/model/train script/run scripts | `ladd/code_versions/current_hbb/` |
| LADD 结果摘要 | 当前主线结果和主线规范 | `ladd/results/LADD_RESULTS_CN.md`, `docs/experiments/LADD_MAINLINE_STANDARD_CN.md` |
| 受控对比 | FGD/LD 修正版、HalluciDet-style 与 CCLKD online-trainer gap 复核 | `comparison/IMPLEMENTATION_REVIEW_CN.md`, `comparison/{fgd,ld,cclkd,hallucidet}/` |
| CCLKD 原文复现 | CCLKD 论文 PDF、400ep/数据增强/online trainer、YOLO11n 消融计划、旧失败运行的小型诊断快照 | `cclkd_reproduction/` |
| 协议审计 | 双卡 4090 `nc=5` 事故、无效结果归档、CCLKD loss 修正与 online 缺口 | `docs/experiments/PROTOCOL_AND_CCLKD_AUDIT_20260605_CN.md` |

## 2. 未放入或刻意排除

| 内容 | 原因 |
|---|---|
| SSH 密码、密钥、完整连接命令 | 不能公开 |
| 原始 OGSOD 数据集 | 数据量大且涉及数据授权 |
| `weights/*.pt` / `*.pth` checkpoint | GitHub 不适合承载大量 checkpoint；调试时优先看 CSV、args、代码和日志 |
| 私有服务器详细 IP/端口 | 不能公开；文档只保留 90/117/4090D 这种代号 |
| 原始 run 目录、训练曲线图片、压缩日志、旧方法归档 | 已从 public 分支移除，避免仓库继续作为事故取证包膨胀；CCLKD 只保留小型 CSV / YAML / code 快照 |

## 3. 公开排查重点

1. LADD 的 B 阶段为什么在部分 seed/机器上出现后期塌缩，尤其是 BN running stats 是否被污染。
2. A2 修复已经避免检测 loss 早期失控，但 B 阶段仍可能有长期退化，需要判断是否与 BN、EMA、batch 统计、teacher/student 输入分布或学习率日程有关。
3. 4090D 上 YOLO11s LADD 当前明显低于 90 上 seed0 结果，需要复核协议/代码/数据增强差异。
4. 双卡 4090 旧 smoke/formal partial runs 使用错误 `nc=5` yaml，已全部作废；请优先复核 `PROTOCOL_AND_CCLKD_AUDIT_20260605_CN.md`。
5. FGD/LD 旧实验分别缺少 teacher attention、误用了分类 logits；请重点复核 2026-06-04 修正版。
6. CCLKD 已在 2026-06-06 修正 LLD/FLD/RLD/CCL loss 语义，其中 CCL 现在使用 neck feature 而不是 DFL regression logits；请同时复核 `comparison/cclkd/README.md`、`cclkd_reproduction/` 和 loss 实现。

## 4. 安全检查状态

已删除公开目录中的服务器连接文档，已把 public dataset YAML 的绝对路径改成 `/path/to/...` 占位符，已移除 checkpoint 权重、原始日志、压缩日志包和大体量 run 目录。
