# Raw Feature KD

日期：2026-06-23

## 作用

预留给 raw feature KD 对照。它用于判断 DSN shared latent 是否优于直接蒸馏 teacher feature。

## 计划目录

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/raw_feature_kd/ir_to_rgb/
logs/dronevehicle_method_search/sub2k_seed0_fullval/raw_feature_kd/ir_to_rgb/
```

## 已启动 run：raw feature KD, alpha 0.25

时间：2026-06-24 00:03 CST

远端：`ladd4090-zw1`，GPU1

目的：在不使用 CMDistill 的 logit/relation KD、不使用 LADD reach/rec/taskL 的情况下，测试最直接的 raw teacher feature KD 是否比 det-only/reload control 更稳。

协议：

```text
student modality: RGB
teacher modality: IR
model: YOLO11n
imgsz: 512
batch: 64
epochs: 200
optimizer: SGD
lr0/lrf: 0.01 / 0.01
mosaic/close_mosaic/mixup: 0.0 / 0 / 0.1
seed: 0
student init: RGB baseline best
teacher weights: IR baseline best
```

关键方法参数：

```text
COMPARISON_KD_PROFILE=none
PROFILE_KD_WEIGHT=0.0
STUDENT_BRANCH_MODE=raw
TEACHER_FEATURE_MODE=raw
KD_CALIBRATION_MODE=affine
ALPHA_KD=0.25
LAMBDA_REACH=0.0
LAMBDA_REC=0.0
LAMBDA_TASKL=0.0
ALPHA_S_REC=0.0
```

远端 run：

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/raw_feature_kd/ir_to_rgb/rawfeatkd_ir2rgb_yolo11n_a0p25_affine_e200_b64_img512_mosaic0p0_mixup0p1_s0_20260624_000336_b
logs/dronevehicle_method_search/sub2k_seed0_fullval/raw_feature_kd/ir_to_rgb/rawfeatkd_ir2rgb_yolo11n_a0p25_affine_e200_b64_img512_mosaic0p0_mixup0p1_s0_20260624_000336_gpu1
```

启动后检查：

- 进程 pid：`16963`
- 参数已确认：`imgsz=512`，`batch=64`，`alpha_kd=0.25`，`kd_calibration_mode=affine`，`comparison_kd_profile=none`。
- 在 GPU1 与 CMDistill、OGSOD RGB baseline 并发运行时，显存约 `23.6GB / 24.6GB`，很满但已完成 epoch 1。
- epoch 1 快照：`mAP50=0.56589`，`mAP50-95=0.35732`。该值接近 det-only/reload control 当前 best `0.56705/0.35876`，明显高于 CMDistill 当前 best `0.55672/0.34968`，但尚未超过 control，暂不算正向方案。
- epoch 6 快照：latest `mAP50-95=0.25078`，late-window 明显下滑。由于同协议 det-only 也在下滑，当前更像 high-LR reload protocol 不稳定，而不是 raw KD 单独失败。已排队 low-LR/no-warmup det-only 与 raw KD control。
