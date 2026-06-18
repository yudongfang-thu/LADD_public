# 90 服务器 Mosaic/Close@100 LADD 主线证据

日期：2026-05-28 快照，2026-06-10 补充归档

用途：记录 90 服务器在 `mosaic=1.0, close_mosaic=700`，即前 100 epoch 使用 mosaic、后 700 epoch 关闭 mosaic 的收敛主线下，LADD 没有出现 B 阶段崩溃，并且三 seed 上稳定高于同协议 SAR baseline。

原始轻量证据目录：

```text
docs/experiments/ladd_mosaic90_20260528_artifacts/
```

该目录来自旧本地快照：

```text
/Users/yudongfang/Desktop/光sar/LADD/docs/remote_snapshots/ogsod_20260528_0110/
```

快照内容只包含 `results.csv`、`args.yaml` 和诊断图，不含权重。旧轻量快照没有保存对应训练 `.log` 文件，因此本次不能补交完整 raw log；可审计依据是每条 run 的原始 CSV 曲线与 args。

## 1. 协议

共同设置：

| 项 | 值 |
|---|---|
| 服务器 | 90 |
| 任务 | OGSOD HBB YOLO11n |
| B 阶段 | 800 epoch target |
| batch | 64 |
| optimizer | auto |
| lr | `lr0=0.01`, `lrf=0.01`, `cos_lr=true` |
| mosaic | `1.0` |
| close_mosaic | `700` |
| patience | 80 |

`close_mosaic=700` 在 800 epoch 训练中等价于前 100 epoch 保持 mosaic，后 700 epoch 关闭 mosaic。

同协议 baseline：

| Baseline | best AP50-95 | best epoch | last AP50-95 |
|---|---:|---:|---:|
| SAR YOLO11n `cos+close@100` | 0.54091 | 746 | 0.53836 |
| RGB YOLO11n `cos+close@100` | 0.61610 | 758 | 0.61345 |

## 2. LADD B 阶段结果

| Run | seed | last epoch | last AP50-95 | best AP50-95 | best epoch | vs SAR baseline | NaN/Inf |
|---|---:|---:|---:|---:|---:|---:|---|
| legacy | 0 | 755 | 0.56638 | 0.56678 | 746 | +0.02587 | no |
| legacy | 42 | 800 | 0.55920 | 0.56688 | 763 | +0.02597 | no |
| legacy | 123 | 800 | 0.56220 | 0.56526 | 770 | +0.02435 | no |
| cap2 | 0 | 800 | 0.56792 | 0.56841 | 798 | +0.02750 | no |
| cap2 | 42 | 800 | 0.56044 | 0.56799 | 750 | +0.02708 | no |
| cap2 | 123 | 800 | 0.56163 | 0.56163 | 800 | +0.02072 | no |

均值：

| 方法 | mean best AP50-95 | std | 相对 SAR baseline |
|---|---:|---:|---:|
| legacy | 0.56631 | 0.00091 | +0.02540 |
| cap2 | 0.56601 | 0.00380 | +0.02510 |

## 3. 结论

这批 90 服务器结果说明：在旧的 `cos_lr + mosaic open then close@100` 收敛主线下，YOLO11n LADD 的 B 阶段可以稳定训练到后期，未出现当前 no-mosaic/H1 诊断里关注的 NaN 或 collapse。

因此它是非常重要的反证信息：

- LADD 方法本身并非必然在 B 阶段崩溃；
- 后来 no-mosaic 主线和跨机器结果中的退化，更可能与训练协议、BN running stats、学习率/调度、或实现细节交互有关；
- 不能把 no-mosaic H1 的 s 后期退化直接解释成 LADD 机制不可行。

同时，这批结果不应直接混入当前 formal no-mosaic 主表，因为 baseline、增强协议和 teacher/student 起点均属于旧 `mosaic + close@100` 收敛主线。

## 4. Artifact 清单

```text
ladd_mosaic90_20260528_artifacts/
  baseline_cos_closeAt100/
    sar_yolo11n_hbb_800ep_cos_closeAt100_pat80_s0_gpu4/{args.yaml,results.csv}
    rgb_yolo11n_hbb_800ep_cos_closeAt100_pat80_s0_gpu5/{args.yaml,results.csv}
  ladd_b_runs/
    six legacy/cap2 B-stage runs, each with {args.yaml,results.csv}
  figures/
    ladd800r2_six_runs_b_curves_20260528.png
    ladd800r2_six_runs_loss_diagnostics_20260528.png
```

未归档：

- checkpoint 权重；
- TensorBoard event；
- 完整训练 `.log` 文件，因为旧 2026-05-28 轻量快照没有保存它们。
