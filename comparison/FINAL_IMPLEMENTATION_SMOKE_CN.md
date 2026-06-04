# 最终对比方法实现 Smoke 记录

最后更新：2026-06-04

本文记录老师确认后的最终受控对比实现 smoke。测试目的仅是验证代码路径、损失数据流、
反向传播和 checkpoint 保存，不用于比较方法性能。

## 1. 测试环境与协议

- 服务器：117，NVIDIA RTX 5880 Ada Generation
- 工作路径：`/home/xmu/djd/ladd`
- 模型：YOLO11n，seed 0
- 初始化：同 seed SAR formal baseline `best.pt`
- 教师：同 seed RGB formal baseline `best.pt`
- 数据与增强：formal no-mosaic transferred-KD 协议
- smoke 范围：`fraction=0.01`，`epochs=1`，约 3 个训练 batch
- `PROFILE_KD_REPLACE_BASE=1`：日志中的 `train/kd_loss` 仅来自被测 profile

四个方法均使用同一 student、teacher、数据子集和训练配置。

## 2. 最终结果

| 方法 | profile | `train/kd_loss` | 有限非零 | 无方法级异常 | `best.pt` / `last.pt` | 结论 |
|---|---|---:|---|---|---|---|
| FGD-style | `fgd` | 1.38700 | 是 | 是 | 均生成 | 通过 |
| LD | `ld` | 1.05935 | 是 | 是 | 均生成 | 通过 |
| CCLKD-style | `cclkd` | 5.49214 | 是 | 是 | 均生成 | 通过 |
| HalluciDet-style | `hallucidet` | 0.82760 | 是 | 是 | 均生成 | 通过 |

统一验收条件：

1. 完成真实训练 batch，而非只完成 import 或 forward；
2. `train/kd_loss` 为有限非零值；
3. 日志中无 `Traceback`、`RuntimeError` 或 CUDA OOM；
4. 训练结果 CSV、`best.pt` 和 `last.pt` 均成功生成；
5. smoke 进程正常退出。

四个方法全部满足以上条件。

## 3. 关键专项结论

### 3.1 LD raw DFL 数据流有效

LD 完整完成训练且 `train/kd_loss=1.05935`，没有触发 raw DFL fail-fast。这证明当前
Ultralytics eval teacher forward、`_unwrap_teacher_preds()` 和 LD DFL distribution
loss 的数据流在真实训练中有效。无需、也不应将 teacher 切换到 train 模式。

### 3.2 CCLKD 新 profile 有效

CCLKD 完整完成训练，未发生 NaN 或 OOM。启动命令确认独立 feature/logit/contrastive
权重、温度参数和 `cclkd_max_tokens=512` 已传入最终实现。

### 3.3 FGD 与 HalluciDet-style 有效

FGD teacher-attention weighted 实现与 HalluciDet-style 三项组合均完成反向传播和
checkpoint 保存。当前 smoke 只能证明实现可运行，不能证明方法能够提升最终 AP。

## 4. 环境与执行备注

- 第一次 FGD smoke 已完成训练并得到非零 loss，但 117 的 `deco` 环境缺少 `polars`，
  在保存 checkpoint 时失败。补齐 `polars=1.41.2` 后，四个最终 smoke 均完整通过。
- 117 文件 I/O 较慢。四方法并发冷启动会长时间争抢模型读取，因此最终验收采用顺序执行。
- 正式实验启动前仍需在目标服务器复核代码 hash、环境依赖和短 smoke。

## 5. 已验证的最终代码 Hash

117 与 public 最终实现一致：

```text
e4fdaad2b0c1f53d88ebb84e3b88c56e32225bea73a355ac598c2f8405bd6b8d  loss.py
9aa638ae7d264381f2856b847f83829278ff87aea465f11e848bfdbbd65e731d  trainer.py
5c7c2003cecc94cec62c1215215fcac84b2c88091351b00e366f8b0dd9c00520  train_ladd_hbb.py
144492b719d8483d8658170c338358bb4b5abf97a123c673a0d03eb29bd1deca  run_ladd_phase.sh
```

结论：当前 FGD-style、LD、CCLKD-style 和 HalluciDet-style 代码可作为后续正式实验的
最终实现基线。
