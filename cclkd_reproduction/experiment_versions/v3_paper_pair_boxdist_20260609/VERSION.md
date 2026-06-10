# v3_paper_pair_boxdist_20260609

最终快照：2026-06-10 09:26 CST。

本版本对应 YOLO11n CCLKD paper-aligned 消融：`formulation=paper`，`ccl_mode=paper_pair`，`ccl_source=box_distribution`，`rld_mode=paper_instance`。

## Final status

六条 run 均已完整跑到 400 epoch：`lld`、`lld_fld`、`lld_fld_rld`、`ccl_only`、`atkd`、`full`。

## Final ranking

| rank | variant | last epoch | last AP50 | last AP | delta AP vs SAR baseline | best epoch | best AP |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | lld_fld | 400 | 0.79701 | 0.53073 | 0.01527 | 400 | 0.53073 |
| 2 | lld_fld_rld | 400 | 0.79541 | 0.52793 | 0.01247 | 400 | 0.52793 |
| 3 | lld | 400 | 0.79417 | 0.52762 | 0.01216 | 399 | 0.52774 |
| 4 | full | 400 | 0.78764 | 0.52531 | 0.00985 | 400 | 0.52531 |
| 5 | atkd | 400 | 0.78892 | 0.52487 | 0.00941 | 399 | 0.52500 |
| 6 | ccl_only | 400 | 0.78560 | 0.52055 | 0.00509 | 400 | 0.52055 |

主要分析见 `FINAL_ANALYSIS_20260610_CN.md`。
