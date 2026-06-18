# Legacy Comparison Archive

最后更新：2026-06-18

本目录记录已从 active comparison surface 移出的早期对比方法入口。当前论文/主表只允许使用
`docs/method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md` 和
`comparison/FINAL_LOCKED_METHODS_CN.md` 中列出的 locked implementation。

## 本次清理

| 方法 | 旧内容 | 当前处理 |
|---|---|---|
| FGD | `comparison/fgd/compare_normalizations.sh` normalization sweep | 删除 active 可执行脚本；旧 sweep 仅作为历史调试，不再可启动 |
| FGD | `comparison/fgd/verify_fix.sh` 修复期 smoke launcher | 删除 active 可执行脚本；当前 smoke 改为 `comparison/code/smoke_check_comparison_losses.py` |
| HalluciDet | custom U-Net standalone runtime/plateau analysis | 移到 `comparison/hallucidet/archive_legacy_20260618/analysis/` |

CMDistill 相关内容本轮不清理，因为实验仍在继续。
