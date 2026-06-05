# LADD Public 证据包说明

最后更新：2026-06-05 16:45 CST

本文档说明这个公开包给外部老师排查时应该怎么看。它的目标不是隐藏问题，而是把当前遇到的困难、代码版本、配置、曲线和服务器记录尽量放齐。

## 1. 已放入的核心材料

| 模块 | 已放入内容 | 位置 |
|---|---|---|
| Baseline | 90 服务器 formal no-mosaic baseline 的 `results.csv`/`args.yaml`，n/s 三 seed 和 m/l/x seed0 | `baseline/results/90_formal_nomosaic_20260528/` |
| Baseline 说明 | 最新 baseline 数字、gap、缺口 | `baseline/results/BASELINE_RESULTS_CN.md`, `docs/experiments/BASELINE_LADD_STATUS_CN.md` |
| LADD 当前代码 | 当前 HBB LADD trainer/loss/model/train script/run scripts | `ladd/code_versions/current_hbb/` |
| LADD 结果 | 90 与 4090D 的 LADD 结果 CSV、args、训练图片、压缩训练日志 | `ladd/results/90_formal_nomosaic_20260528/`, `ladd/results/4090d_formal_nomosaic_20260528/`, `server_logs/` |
| LADD 诊断图 | A2 稳定性修复、B 入口 KD 冲击、loss 诊断图 | `ladd/diagnostics/` |
| 受控对比 | FGD/LD 修正版、CCLKD paper-structured reimplementation、HalluciDet-style 与实现复核 | `comparison/IMPLEMENTATION_REVIEW_CN.md`, `comparison/{fgd,ld,cclkd,hallucidet}/` |
| 协议审计 | 双卡 4090 `nc=5` 事故、无效结果归档、CCLKD 重写边界 | `docs/experiments/PROTOCOL_AND_CCLKD_AUDIT_20260605_CN.md` |
| 降级归档 | CoLD、CrossKD、修复前 FGD/LD 结果 | `comparison/archive/excluded_methods/` |

## 2. 未放入或刻意排除

| 内容 | 原因 |
|---|---|
| SSH 密码、密钥、完整连接命令 | 不能公开 |
| 原始 OGSOD 数据集 | 数据量大且涉及数据授权 |
| `weights/*.pt` / `*.pth` checkpoint | GitHub 不适合承载大量 checkpoint；调试时优先看 CSV、args、代码和日志 |
| 私有服务器详细 IP/端口 | 不能公开；文档只保留 90/117/4090D 这种代号 |

## 3. 公开排查重点

1. LADD 的 B 阶段为什么在部分 seed/机器上出现后期塌缩，尤其是 BN running stats 是否被污染。
2. A2 修复已经避免检测 loss 早期失控，但 B 阶段仍可能有长期退化，需要判断是否与 BN、EMA、batch 统计、teacher/student 输入分布或学习率日程有关。
3. 4090D 上 YOLO11s LADD 当前明显低于 90 上 seed0 结果，需要复核协议/代码/数据增强差异。
4. 双卡 4090 旧 smoke/formal partial runs 使用错误 `nc=5` yaml，已全部作废；请优先复核 `PROTOCOL_AND_CCLKD_AUDIT_20260605_CN.md`。
5. FGD/LD 旧实验分别缺少 teacher attention、误用了分类 logits；请重点复核 2026-06-04 修正版。
6. CCLKD 已在 2026-06-05 改为 COP/ATKD/CCL/RLD 结构，但仍不是官方严格复现；请重点复核 `comparison/cclkd/README.md` 和 loss 实现。

CoLD/CrossKD 与无效旧结果已统一归档，仅供追溯，不再列为当前公开排查重点。

## 4. 安全检查状态

已删除公开目录中的服务器连接文档，已把 public dataset YAML 的绝对路径改成 `/path/to/...` 占位符，已移除 checkpoint 权重。当前仍保留一些 `args.yaml` 和压缩日志中的训练机本地路径，用于说明实验 provenance；这些路径不是登录信息。
