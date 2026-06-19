# 旧 LADD 实验归档说明

归档日期：2026-06-18

本目录保存 LADD 主方法重新定义之前的历史实验记录。归档不是删除证据，而是把旧口径从当前主线入口移开，避免 A1-A2-B、旧 mosaic 或调试型 loss 结果污染当前 `LADD` 主表。

## 当前有效口径

当前 LADD 主线只认：

- 方法：`LADD`
- 阶段：`A -> B`
- 内部实现 profile / tag：`dynamic_probe` / `clean_a1b_dynprobe`
- 当前证据入口：`docs/experiments/ladd_mosaic100_mainline_curves_20260618/`
- 方法定义：`docs/ladd_method_definition.md`
- 训练规范：`docs/experiments/LADD_MAINLINE_STANDARD_CN.md`

`clean_a1b` static 和 `clean_a1b_dyn` dynamic 仍保留为消融实验；它们不是主线方法，但可以用于解释当前 LADD 中动态 teacher core 与冻结 student probe 的贡献。`clean_a1b_dynprobe` 只作为内部实验 tag，不作为论文方法名。

## 归档范围

| 目录 | 内容 | 当前用途 |
|---|---|---|
| `a1a2b_and_bstage/` | 旧 A1-A2-B 链路、skip-A2、B-stage collapse、loss audit、source/init 对比等诊断记录 | 只作历史诊断或附录背景 |
| `legacy_mosaic_protocols/` | 旧 mosaic90、mosaic 对比、早期 mosaic progress 证据 | 不进入当前 mosaic100 clean 主表 |
| `legacy_progress_snapshots/` | 旧 active run 快照、旧动态/静态 loss 临时分析、no-mosaic LADD 快照 | 只作运行过程追溯 |
| `historical_figures/` | 旧曲线图与图生成脚本 | 只作历史可视化证据 |
| `historical_reports/` | 顶层旧 LADD 报告和 CSV 汇总 | 只作历史报告，不作为当前入口 |

对应的 `ladd/results/` 旧 evidence 包已迁到：

```text
ladd/results/archive_legacy_ladd_20260618/
```

## 使用规则

1. A1-A2-B 结果不能改称 clean A1B。
2. 旧 mosaic90、旧 no-mosaic、旧 formal 结果不能和当前 mosaic100 clean 主表直接比较。
3. 任何含 A2、sep/aux/debug loss、BN-freeze repair、短 B schedule、manual retry 的结果默认是 diagnostic。
4. 旧结果如需引用，必须明确写成“历史诊断/失败分析/附录证据”，不能作为主方法有效数字。
5. 当前主表候选的论文方法名是 `LADD`，内部 tag 必须是 `clean_a1b_dynprobe`，并满足同容量、同 seed、同协议 baseline 对齐。

## 仍在当前入口保留的 clean 结果

`docs/experiments/ladd_mosaic100_mainline_curves_20260618/` 不是旧 mosaic 归档包。它保存当前 LADD 口径下的 mosaic100 主线曲线，其中：

- `clean_a1b_dynprobe`：LADD 主线的内部实现 tag；
- `clean_a1b` static：消融；
- `clean_a1b_dyn` dynamic：消融。

这些 clean 结果应和同协议 baseline/comparison results 一起管理，不与本归档目录中的旧实验混用。
