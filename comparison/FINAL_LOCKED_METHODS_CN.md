# 对比方法最终实现锁定

最后更新：2026-06-18

本文档锁定当前主表可用的对比方法实现。除 CMDistill 外，FGD、LD、HalluciDet 不再保留
早期版本或可调实现分支；旧结果只能作为 archived diagnostic。

## Locked Methods

| 方法 | locked implementation | 有效入口 | 不再允许的旧版本 |
|---|---|---|---|
| FGD-style / FGD-YOLO adaptation | `locked_fgd_yolo_gtbox_attention_20260618`：GT-box fg/bg mask；spatial/channel softmax attention；`alpha=0.0001, beta=0.00005, gamma=0.001, temperature=0.5`；无 legacy relation | `comparison/code/launch_formal_from_yolo_kd_job.sh fgd ...`、`comparison/code/launch_formal_transfer_kd_job.sh fgd ...`、`scripts/paper/run_paper_comparison_kd.sh fgd ...` | normalization sweep、assigner-mask fallback、batch relation、FGD env/CLI hyperparameter override |
| LD | `locked_ld_yolo_dfl_vlr_20260618`：YOLO DFL logits main LD + teacher-confidence/teacher-box-GT-IoU VLR-style candidate LD；`temperature=10.0, main_weight=0.25, vlr_weight=0.25` | 同上，方法参数为 `ld` | classification-logit KD、foreground-only LD、VLR/topk/weight/env override variants |
| HalluciDet-YOLO adaptation | `locked_hallucidet_yolo_official_unet_b64_20260618`：`replicate3` input + `segmentation_models_pytorch.Unet(resnet34, imagenet)` + frozen RGB YOLO11 detection loss | `comparison/hallucidet/train_hallucidet.py`、`scripts/paper/run_paper_hallucidet.sh` | old custom U-Net、`hallucidet_style` feature/response/margin KD profile、official_unet b16 diagnostic |

## CMDistill

CMDistill 仍在其他数据集和协议上继续实验，本轮不锁定、不删除、不归档 active 代码。

## 工程规则

- 主表和 paper wrapper 只能使用上表 locked implementation。
- `comparison_kd_profile=fgd|ld` 只选择方法，不再暴露 FGD/LD 内部实现参数。
- HalluciDet 训练入口只构建 official-style U-Net；custom U-Net 代码已从 active implementation 删除。
- 旧 FGD/HalluciDet 早期调试记录已归档到 `comparison/archive_legacy_comparison_20260618/` 和 `comparison/hallucidet/archive_legacy_20260618/`。
