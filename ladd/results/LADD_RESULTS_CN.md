# LADD 主线结果

最后更新：2026-06-04 23:20 CST

配置基线：`A1=10 -> A2=50 -> B=800, cap2 reach-rank, A2/B MuSGD lr0=0.001 no warmup`。最新稳定候选在 B 阶段额外启用 `FREEZE_BN_STATS=1`，冻结 BN running mean/var，保留 BN affine 参数梯度。

表中 `best AP50-95` 均标注 best epoch。`vs SAR baseline` 使用同容量同 seed 的 formal no-mosaic SAR baseline best。

## YOLO11n cap2 候选

| seed/run | 服务器 | B epoch | last AP50-95 | best AP50-95 | vs SAR baseline | 状态 |
|---|---|---:|---:|---:|---:|---|
| seed0 `a2mu1e3` | 90 | 800 | 0.57504 | 0.57662@725 | +0.02008 | 完成，健康强证据 |
| seed42 `a2mu1e3` | 90 | 800 | 0.57293 | 0.57420@735 | +0.01626 | 完成，健康强证据 |
| seed123 old B | 90 | 483 | 0.00000 | 0.52182@1 | -0.03946 | 检测 loss NaN，作废 |
| seed123 `bstable1e3` | 90 | 800 | 0.52875 | 0.56161@165 | +0.00033 | 无 NaN 但后期退化，不作主线 |
| seed0 BN-freeze | 90 | 800 | 0.57254 | 0.57276@793 | +0.01622 | 完成，稳定候选 |
| seed123 BN-freeze | 90 | 800 | 0.57219 | 0.57269@779 | +0.01141 | 完成，修复 seed123 退化 |
| seed0/42/123 high-LR BN-freeze diag | 双卡 4090 | running | — | — | — | `optimizer=auto -> MuSGD(lr=0.01)` + 默认 warmup + `FREEZE_BN_STATS=1`，诊断 old B 是否可由 BN-freeze 单独救回 |
| seed0 r2 | 4090D | 346 | 0.00000 | 0.54925@227 | -0.00729 | 已停，BN 污染塌缩 |
| seed123 r2 | 4090D | 88 | 0.00006 | 0.54864@2 | -0.01264 | 已停，BN 污染塌缩 |
| seed42 r2 | 4090D | 800 | 0.55222 | 0.57191@420 | +0.01397 | 跑满但后期退化，跨机器 sanity |

### YOLO11n 当前读法

- `seed0/42 a2mu1e3` 证明 cap2 主线在健康 seed 上有稳定正收益。
- `seed123 bstable1e3` 证明单纯降低 B 学习率不够，能防 NaN 但不能防后期退化。
- `seed0/123 BN-freeze` 证明 BN running stats 污染是关键路径之一；seed123 从几乎无收益/后期退化恢复到 `0.57269@779`，这是当前最重要的新证据。
- 若追求严格同协议三 seed 主表，下一步应补 `seed42 BN-freeze`。在补齐前，最保守口径是 “BN-freeze 已在 seed0/123 成立，seed42 原协议健康成立”。

## YOLO11n original (no cap2)

| seed/run | 服务器 | B epoch | last AP50-95 | best AP50-95 | vs SAR baseline | 备注 |
|---|---|---:|---:|---:|---:|---|
| seed0 original `a2mu1e3` | 90 | 800 | 0.57517 | 0.57821@730 | +0.02167 | no-cap2 消融；AP 略高于 cap2 seed0 |

original 结果说明 no-cap2 在 seed0 上可取得更高 AP，但 cap2 的价值主要是约束 reach-rank 几何、避免反平行坍缩。它应作为消融/诊断结果，而不是当前主线替代。

## YOLO11s cap2

| seed/run | 服务器 | B epoch | last AP50-95 | best AP50-95 | vs SAR baseline | 状态 |
|---|---|---:|---:|---:|---:|---|
| seed0 `a2mu1e3` | 90 | 608 | 0.63527 | 0.63551@605 | +0.00654 | 未满 800，但已有正收益 |
| seed0 r2 | 4090D | 800 | 0.58962 | 0.61787@570 | -0.01110 | 跑满但低于 baseline |
| seed42 r2 | 4090D | 800 | 0.58895 | 0.60838@638 | -0.02041 | 跑满但低于 baseline |
| seed123 r2 | 4090D | 800 | 0.58143 | 0.60849@513 | -0.01408 | 跑满但低于 baseline |

YOLO11s 的 90 seed0 是唯一正向证据，但未满 800；4090D 三 seed 全部低于 baseline，说明该机器/版本上的 s 链路不能直接作为主线证据。s 容量需要在 public 最终代码上重新排。

## YOLO11m cap2

| seed/run | 服务器 | B epoch | last AP50-95 | best AP50-95 | vs SAR baseline | 状态 |
|---|---|---:|---:|---:|---:|---|
| seed0 `a2mu1e3` | 90 | 121 | 0.52361 | 0.59796@1 | -0.05784 | 异常，暂不纳入主表 |

YOLO11m 的 A2 best 为 `0.64330@33`，已经低于 SAR baseline `0.65580@704`；B 入口进一步恶化。m/l 不应在当前主线未稳定前继续扩展。

## 当前主线建议

当前最合理的 LADD 主线备选是：

```text
cap2 + A2 MuSGD lr0=0.001 no warmup
     + B MuSGD lr0=0.001 no warmup
     + B FREEZE_BN_STATS=1
```

证据强度：

| 证据 | 结论 |
|---|---|
| YOLO11n seed0/42 `a2mu1e3` 完成且正向 | LADD 在 n 容量确有收益 |
| YOLO11n seed123 `bstable1e3` 后期退化 | 只降 LR 不够 |
| YOLO11n seed0/123 BN-freeze 完成且正向 | BN-freeze 是当前最可信的稳定修复 |
| YOLO11s 90 seed0 正向但未满 | s 可作为后续容量验证，需重跑/补齐 |
| 4090D s/n 多 run 退化 | 不能作为主线结果，只作崩溃证据 |

阶段性结论：LADD 主方法目前可以以 YOLO11n 为主表核心推进，但正式三 seed 统一协议还缺 `seed42 BN-freeze`。完成该 run 后，若三 seed 均保持正收益，就可以把 BN-freeze 版本定为主线。
