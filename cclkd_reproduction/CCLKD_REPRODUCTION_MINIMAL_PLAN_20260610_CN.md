# CCLKD 最小复现计划 2026-06-10

## 当前目标

当前只复现 CCLKD。

不复现 CoLD，不复现 CMDistill，不继续扩大 YOLO11 component ablation。YOLO11 结果只保留为 extension/adaptation diagnostic，不再称为 CCLKD Table 4/5/12 的主复现。

## 为什么仍然需要 YOLOv5-X Baseline Gate

CCLKD Table 5 的 YOLOv5 baseline 是 AP50 80.9 / AP 46.3。如果本地 YOLOv5-X SAR baseline 远低于该值，则后续 CCLKD full 的结果无法解释：低结果可能来自数据 split、类别映射、YOLOv5 版本、评估协议或增强配置，而不一定来自 CCLKD 公式。

因此 baseline gate 是 CCLKD 复现的必要前置，不是 CoLD 复现。

## 并行第一批实验

推荐先并行启动：

| GPU | 实验 | 目的 |
|---:|---|---|
| 0 | YOLOv5x SAR baseline batch32 seed0 | Primary CCLKD Table-2-style baseline gate |
| 1 | YOLOv5x SAR baseline batch64 seed0 | Batch-size sanity |

可选第三条：

| GPU | 实验 | 目的 |
|---:|---|---|
| 2 | YOLOv5x RGB teacher baseline batch32 seed0 | CCLKD teacher-strength gate |

推荐命令：

```bash
LAUNCH=1 DRY_RUN=0 PARALLEL_LAUNCH=1 GPU_LIST=0,1 MAX_PARALLEL=2 \
  bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_sanity_matrix.sh gate
```

## CCLKD Full 复现优先级

等 YOLOv5-X baseline gate 跑到 50 epoch 后，如果趋势正常，立即并行启动 YOLOv5-X CCLKD full。CCLKD full 优先于所有消融。

如果还有多卡，再启动 ATKD-only 和 CCL-only。不要先跑 LLD/FLD/RLD 细消融。

## Gate 判断

50 epoch 若 AP50 异常低，例如低于 50，优先查数据/评估，不继续浪费 400 epoch。

400 epoch 若 batch32 或 batch64 任一接近 80.9/46.3，可继续解释 CCLKD。

如果 batch32 和 batch64 都远低于目标，优先查数据 split、类别映射、YOLOv5 版本、评估脚本和增强，而不是修改 CCLKD 公式。

## 当前 YOLO11 结果处理

当前 YOLO11 CCLKD v3 只作为 YOLO11 adaptation diagnostic。

不再把它称为 CCLKD Table 4/5/12 reproduction。不再为 YOLO11 CCLKD 扩大组件消融。

## Blocker

Correct CCLKD main-result reproduction still requires a YOLOv5-based online teacher-student CCLKD trainer.

Current `cclkd_reproduction/code/train_cclkd_online_hbb.py` is YOLO11-oriented and cannot be used as strict YOLOv5 Table 4/5 reproduction.

已搜索当前仓库，未发现严格 YOLOv5-based CCLKD online trainer。下一步应在 baseline gate 趋势正常后，单独实现或接入 YOLOv5 CCLKD trainer，再启动 CCLKD full / ATKD / CCL。
