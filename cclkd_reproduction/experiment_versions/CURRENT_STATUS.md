# CCLKD experiment versions current status

更新时间：2026-06-10 09:26 CST。

当前主归档版本：`v3_paper_pair_boxdist_20260609`。

## YOLO11n full ablation final status

| rank | variant | last epoch | last AP50 | last AP | delta AP vs SAR baseline | best epoch | best AP |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | lld_fld | 400 | 0.79701 | 0.53073 | 0.01527 | 400 | 0.53073 |
| 2 | lld_fld_rld | 400 | 0.79541 | 0.52793 | 0.01247 | 400 | 0.52793 |
| 3 | lld | 400 | 0.79417 | 0.52762 | 0.01216 | 399 | 0.52774 |
| 4 | full | 400 | 0.78764 | 0.52531 | 0.00985 | 400 | 0.52531 |
| 5 | atkd | 400 | 0.78892 | 0.52487 | 0.00941 | 399 | 0.52500 |
| 6 | ccl_only | 400 | 0.78560 | 0.52055 | 0.00509 | 400 | 0.52055 |

结论：六组均完成 400 epoch，全部相对 SAR YOLO11n baseline 正增益；`lld_fld` 最强，`RLD` 和 `Full` 的额外贡献不干净，说明当前 LADD 强训练协议下 CCLKD gain 小且组件互补性弱。
