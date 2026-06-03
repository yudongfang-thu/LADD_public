# 4090D formal baseline / KD snapshot

拉取时间：2026-06-02 CST

## 文件范围

- YOLO11n/s seed0 formal no-mosaic SAR/RGB baseline: `results.csv`, `best.pt`, `last.pt`, logs/env metadata。
- YOLO11n FGD / CrossKD-style from-yolo-pretrain: `results.csv`, `best.pt`, `last.pt`, logs/manifest。
- 未拉取数据集、cache、图片、optimizer 中间 checkpoint。

## 结果表

| run | type | epochs | best epoch | best AP50 | best AP | last AP50 | last AP | delta AP vs SAR-n |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAR-n baseline | baseline | 800/800 | 716 | 0.82770 | 0.55916 | 0.82619 | 0.55602 | +0.00000 |
| RGB-n baseline | baseline | 800/800 | 746 | 0.92230 | 0.62502 | 0.92182 | 0.62274 | +0.00000 |
| SAR-s baseline | baseline | 800/800 | 716 | 0.90787 | 0.62382 | 0.90680 | 0.61710 | +0.00000 |
| RGB-s baseline | baseline | 800/800 | 716 | 0.94078 | 0.65463 | 0.93813 | 0.64551 | +0.00000 |
| FGD-n from-yolo | comparison | 800/800 | 749 | 0.82532 | 0.55867 | 0.82693 | 0.55514 | -0.00049 |
| CrossKD-style-n from-yolo | comparison | 800/800 | 737 | 0.82658 | 0.55764 | 0.82878 | 0.55670 | -0.00152 |

## 结论

- FGD-n 完整跑满 800 epoch，best AP `0.55867`，相对 4090D SAR-n baseline `0.55916` 为 `-0.00049`，基本打平但无正向提升。
- CrossKD-style-n 完整跑满 800 epoch，best AP `0.55764`，相对 baseline 为 `-0.00152`，无正向提升。
- 两个 KD run 均没有 early stopping，日志为 `Phase b finished`。
- 4090D 当前这批实验可作为通用 KD 对照诊断，不建议继续在 4090D 上扩展 s/multi-seed。

## 本地路径

根目录：`remote_snapshots/4090d_formal_kd_20260602/root/autodl-tmp/LADD`

关键日志：

- `logs/formal_nomosaic_20260528/comparisons/from_yolo_pretrain/formal_nomosaic_yolo11n_fgd_from_yolo_s0_r3_gpu0.outer.log`
- `logs/formal_nomosaic_20260528/comparisons/from_yolo_pretrain/formal_nomosaic_yolo11n_crosskd_from_yolo_s0_r1_gpu0.outer.log`

