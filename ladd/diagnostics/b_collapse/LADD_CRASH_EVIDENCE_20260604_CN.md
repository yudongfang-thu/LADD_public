# LADD B 阶段崩溃证据汇总

最后更新：2026-06-04 08:55 CST

本文档把目前已知的 LADD 崩溃/退化现象拆成“事实、证据、当前解释、还缺什么”。目的是方便外部老师只看 public 包也能判断下一步该改哪里。

## 1. 三类不同问题

| 类型 | 发生位置 | 典型 run | 表现 | 当前判断 |
|---|---|---|---|---|
| A2 早期失控 | A2 前几轮 | cap2 seed0 old default | detection loss 早期 NaN/发散 | 优化器/学习率过激 |
| B 后期塌缩 | B 长训练后期 | 90 seed123 old B | AP 后期归零或严重退化 | B 阶段仍存在稳定性问题 |
| BN running stats 污染 | 4090D r2 | n seed0/123 | 权重无明显 NaN/Inf，但推理几乎不可用 | BN running mean/var 被污染 |

这三类问题不应混为同一个 bug。A2 修复已经比较明确；B 阶段和 BN 污染仍是主问题。

## 2. A2 早期失控

证据文件：

```text
ladd/diagnostics/a2_stability/cap2_s0_a2_old_default_results.csv
ladd/diagnostics/a2_stability/cap2_s0_a2_mu1e3_results.csv
ladd/diagnostics/a2_stability/ladd_a2_stability_fix_20260601.png
```

事实：

- old default A2 在早期出现 detection loss 失控，不是 reach loss 本身先坏。
- A2 改为 `MuSGD lr0=0.001 no warmup` 后可以稳定完成。
- 因此当前主线不再使用 old default A2。

当前解释：

- A2 同时引入学生骨干与 detection loss，默认优化器/学习率对这个阶段过猛。
- 温和学习率修复能降低入口冲击，但并不自动解决 B 阶段长训练退化。

## 3. 90 服务器 seed123 B 阶段塌缩/退化

关键结果：

| Run | epoch | current AP50-95 | best AP50-95 | 说明 |
|---|---:|---:|---:|---|
| YOLO11n seed123 old B | 483 | 0.00000 | 0.52182@1 | 后期塌缩 |
| YOLO11n seed123 `bstable1e3` | 800 | 0.52875 | 0.56161@165 | 可跑满，但后期退化 |

事实：

- `bstable1e3` 说明更温和的 B 设置可以避免直接 NaN/归零。
- 但 best 出现在 epoch 165，后续到 epoch 800 明显退化，不能作为最终可接受修复。
- seed0/seed42 在 90 上完整完成并正向，seed123 是更敏感的坏 seed。

当前解释：

- B 阶段不是单纯“能不能跑满”，而是长训练中学生 detector 是否逐步偏离。
- 可能原因包括 KD/detection loss 尺度长期不平衡、A2 checkpoint 选择过于敏感、EMA/BN 统计与阶段切换不匹配。

## 4. 4090D r2 的 BN running stats 污染

关键结果：

| Run | epoch | current AP50-95 | best AP50-95 | 状态 |
|---|---:|---:|---:|---|
| YOLO11n seed0 4090D r2 | 346 | 0.00000 | 0.54925@227 | 已停 |
| YOLO11n seed123 4090D r2 | 88 | 0.00006 | 0.54864@2 | 已停 |
| YOLO11n seed42 4090D r2 | 659 | 约 0.565 | 约 0.570 | 运行中，未塌 |

已有诊断：

- 坏 run 的 `last.pt` 权重没有明显 NaN/Inf。
- 但 BN `running_mean/running_var` 异常放大。
- 之前诊断到的最大 running variance 量级：
  - seed0 坏 run：约 1726
  - seed123 坏 run：约 1333
  - 健康 seed42：约 47.7
- 这解释了“训练过程不是立刻 NaN，但推理/验证突然不可用”的现象。

可复查脚本：

```bash
python3 ladd/diagnostics/b_collapse/analyze_bn_stats.py /path/to/last.pt --repo-root /path/to/LADD_public --csv bn_stats.csv
```

该脚本会按层输出 `running_var` 和 `running_mean` 的最大值、均值和 finite 状态，便于复查坏 run 与健康 run 的 BN 统计差异。

当前修复尝试：

```text
--freeze-bn-stats
FREEZE_BN_STATS=1
```

该修复冻结 BN running mean/var，同时保留 affine 参数梯度。它是诊断性修复，不是已经证明的最终方案。

## 5. 4090D YOLO11s 偏低问题

90 服务器 YOLO11s seed0 LADD：

| Run | epoch | current AP50-95 | best AP50-95 | vs SAR baseline |
|---|---:|---:|---:|---:|
| YOLO11s seed0 90 `a2mu1e3` | 608 | 0.63527 | 0.63551@605 | +0.00654 |

4090D YOLO11s r2 当前记录：

| Run | epoch | current/best AP50-95 | 状态 |
|---|---:|---:|---|
| YOLO11s seed0 4090D r2 | 509 | 约 0.614 | 运行中，偏低 |
| YOLO11s seed42 4090D r2 | 493 | 约 0.594 | 运行中，偏低 |
| YOLO11s seed123 4090D r2 | 492 | 约 0.606 | 运行中，偏低 |

这个现象不能直接归因于 LADD 方法失败，因为 90 上同容量 seed0 已有正向收益。需要重点复核：

- 4090D 与 90 的代码是否完全一致。
- 数据 YAML、no-mosaic、Albumentations、batch、workers、缓存、路径是否一致。
- A2/B 启动 checkpoint 是 `best.pt`、`last.pt` 还是固定 epoch。
- 4090D 上是否存在不同 torch/cuda/ultralytics 行为导致的训练差异。

## 6. public 包当前还缺的证据

这些证据尚未充分整理进 public 包，建议后续补齐：

| 缺口 | 为什么重要 |
|---|---|
| BN stats 数值表 | 已补可重复提取脚本，但还缺坏 run/健康 run 的逐层 CSV 表 |
| BN-freeze 完整曲线 | 现在只有早期结果，不能判断最终是否损失收益 |
| 90 vs 4090D 同 seed 同 epoch 曲线 | 判断 BN-freeze 或机器差异是否影响最终收益 |
| seed123 `bstable1e3` loss/mAP 曲线 | 需要看退化是平滑发生还是某个 epoch 后突变 |
| A2 checkpoint 选择对 B 的影响 | 需要比较 A2 best/last/fixed epoch 启动 B |

## 7. 建议外部老师优先判断的问题

1. BN-freeze 是否应该作为正式修复，还是只作为定位工具。
2. B 阶段是否需要在切换阶段时重置或重新校准 BN running stats。
3. 是否应该取消从 A2 `best.pt` 进入 B，改为固定 epoch 或 `last.pt`。
4. 是否需要对 KD loss 加入随 epoch 衰减、梯度裁剪、或 detection loss 下限保护。
5. 4090D/4090 与 90 服务器之间是否存在协议或环境差异，尤其是 YOLO11s。
