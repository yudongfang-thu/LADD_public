# CCLKD YOLOv5x Custom Trainer Loss Scaling 对齐结果

日期：2026-06-12

## 背景

之前 YOLOv5x b32/80epoch 结果显示，custom `det_only_same_trainer`
明显低于 standard YOLOv5 `train.py`：

| 实验 | AP50 | AP50:95 |
|---|---:|---:|
| standard YOLOv5 `train.py` SAR baseline @80 | 0.57056 | 0.30964 |
| custom `det_only_same_trainer` @80（修复前） | 0.33064 | 0.13490 |

排查发现 custom trainer 缺少 YOLOv5 `train.py` 在 `ComputeLoss`
初始化前执行的 `hyp["box"]`、`hyp["cls"]`、`hyp["obj"]` loss-gain
scaling。该问题会显著改变 detection loss 的相对权重。

## 修复

在 `cclkd_reproduction/yolov5_sanity/code/train_yolov5_cclkd_full.py`
中，在读取 `dataset.labels` 后、初始化 `ComputeLoss(model)` 和
`ComputeLoss(teacher)` 前加入与 YOLOv5 `train.py` 等价的 scaling：

```python
nl = de_parallel(model).model[-1].nl
hyp["box"] *= 3 / nl
hyp["cls"] *= nc / 80 * 3 / nl
hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl
hyp["label_smoothing"] = opt.label_smoothing
```

没有修改 ATKD/CCL 公式、candidate source、projection、decoded box
representation，也没有修改 `train_yolov5_cclkd_from_trainpy.py`。

## 80 Epoch 对齐结果

| 实验 | 模式 | GPU | AP50 | AP50:95 | AP50 差值 vs train.py | AP 差值 vs train.py |
|---|---|---:|---:|---:|---:|---:|
| standard train.py det-only | `det_only_same_trainpy` | 0 | 0.57056 | 0.30964 | 0.00000 | 0.00000 |
| custom det-only scaled | `det_only_same_trainer` | 1 | 0.56807 | 0.30862 | -0.00249 | -0.00102 |
| two-branch no KD scaled | `two_branch_no_kd` | 3 | 0.56609 | 0.30616 | -0.00447 | -0.00348 |

## 结论

1. 修复 loss-gain scaling 后，custom `det_only_same_trainer` 与 standard
   YOLOv5 `train.py` 已基本对齐：AP 差距为 0.00102，低于 0.02 验收阈值。
2. `two_branch_no_kd` 也接近 standard baseline：AP 差距为 0.00348。
3. 因此，之前 custom det-only 的大幅低 AP 主要来自 missing loss-gain
   scaling，而不是 two-branch 结构本身。
4. 现在可以在该对齐基础上继续解释 ATKD-only、CCL-only、paper_full 的
   80epoch 结果；在此之前不应再使用修复前的 custom trainer 结果做方法判断。

## 证据路径

完整轻量证据包：

`cclkd_reproduction/yolov5_sanity/results/diagnostics_20260612_trainpy_scaling_fix_b32_e80/`

包含：

- 三条 run 的 `results.csv`
- custom run 的 `cclkd_yolov5_diagnostics.csv`
- `run_meta.txt`、`command.sh`、`hyp.yaml`、`opt.yaml`
- 压缩完整日志 `nohup.log.gz`
- 可读日志尾部 `nohup_tail.txt`
- standard train.py run 的 YOLOv5 曲线图

未包含 checkpoint weights、TensorBoard event 文件或其它大文件。
