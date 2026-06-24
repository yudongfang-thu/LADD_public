# DroneVehicle Baselines

日期：2026-06-23

## 作用

记录 DroneVehicle sub2k seed0 小风洞的 S0 输入：RGB student baseline 与 IR teacher baseline。所有后续方法都必须明确引用这里的 checkpoint 和指标。

## 当前结果

| role | best epoch | best AP50 | best AP50-95 | final AP50 | final AP50-95 |
|---|---:|---:|---:|---:|---:|
| RGB student | 141 | `0.56886` | `0.36087` | `0.55255` | `0.35385` |
| IR teacher | 142 | `0.63800` | `0.43299` | `0.62123` | `0.42480` |

## 远端路径

```text
runs_public/cross_dataset/cclkd_yolo11n/dronevehicle_sub2k_seed0/baselines/student_rgb/dronevehicle_sub2k_student_rgb_yolo11n_cclkdproto_e200_b64_img512_mosaic0p0_close0_mixup0p1_s0_20260623_221620
runs_public/cross_dataset/cclkd_yolo11n/dronevehicle_sub2k_seed0/baselines/teacher_ir/dronevehicle_sub2k_teacher_ir_yolo11n_cclkdproto_e200_b64_img512_mosaic0p0_close0_mixup0p1_s0_gpu0_20260623_221936
```
