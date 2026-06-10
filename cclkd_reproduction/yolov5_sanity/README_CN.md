# YOLOv5-X CCLKD Reproduction Gate

本目录只服务于 CCLKD 复现。它的目标不是复现 CoLD 或 CMDistill，而是先确认 YOLOv5-X SAR baseline 是否能对齐 CCLKD Table 5 中使用的 YOLOv5 baseline。只有 baseline gate 通过后，才继续解释 CCLKD full / ATKD / CCL 的复现结果。

CCLKD 原文主比较和消融是 YOLOv5 口径；YOLO11 是 extension comparison，不是 Table 4/5/12 的主复现目标。这里的 YOLOv5-X SAR baseline 是 CCLKD reproduction gate，不是其他方法的复现实验。

## 目标

Primary gate:

- model: YOLOv5x / CSPDarkNet-X
- modality: SAR
- imgsz: 256
- epochs: 400
- primary batch: 32
- batch sanity: 64
- optimizer: SGD
- lr: 0.01
- momentum: 0.937
- weight decay: 0.0005
- augmentation: YOLOv5 defaults plus explicit Mosaic 1.0 and MixUp 0.1

CCLKD Table 5 YOLOv5 baseline target:

- Params: about 86M
- AP50: 80.9
- AP: 46.3
- per-class AP50 reference: Oil Tank 57.7, Bridge 87.2, Harbor 97.9

Loose pass threshold:

- AP50 >= 78
- AP >= 44

CCLKD Table 2 writes OGSOD batch=32, so G1 uses batch32 as the primary gate. Batch64 is retained only as a batch-size sanity check because the shared YOLOv5 baseline may have benchmark batch64 provenance.

## 文件

```text
cclkd_reproduction/yolov5_sanity/
  README_CN.md
  scripts/
    prepare_yolov5_repo.sh
    launch_yolov5_ogsod_baseline.sh
    launch_yolov5_sanity_matrix.sh
  tools/
    check_yolov5_ogsod_dataset.py
    summarize_yolov5_sanity.py
  configs/
    hyp_cold_ogsod.yaml
  results/
    .gitkeep
```

## 准备 YOLOv5

```bash
bash cclkd_reproduction/yolov5_sanity/scripts/prepare_yolov5_repo.sh
```

默认 checkout `v7.0`，可用 `YOLOV5_REF=v6.2` 覆盖。脚本只打印依赖安装建议，不会自动 `pip install`。环境记录写到：

```text
cclkd_reproduction/yolov5_sanity/results/yolov5_env.txt
```

YOLOv5x v7.0 官方参数量约 86.7M，`nc=3` 后接近 CCLKD Table 5 的 YOLOv5 baseline 参数口径。因此默认 gate 使用 YOLOv5x。YOLOv5x6 不进入默认矩阵；只有在必要时才用于排查 139.99M 参数量矛盾。

## 数据集检查

```bash
python cclkd_reproduction/yolov5_sanity/tools/check_yolov5_ogsod_dataset.py \
  --data shared/configs/datasets_public/ogsod1_sar_detect.yaml \
  --teacher-data shared/configs/datasets_public/ogsod1_rgb_detect.yaml
```

输出：

- `results/dataset_sanity_sar.json`
- `results/dataset_sanity_sar.md`
- `results/dataset_sanity_rgb.json`
- `results/dataset_sanity_rgb.md`

检查项包括 `nc=3`、class order、train/test 数量、label 数量、每类 instance 数量、label 坐标合法性和抽样图像尺寸。CCLKD/共享表的 per-class 顺序常见为 Oil Tank / Bridge / Harbor；当前公开 yaml 常见顺序是 `bridge / harbor / storage_tank`，汇总时必须记录映射。

## Gate 矩阵

默认只打印命令，不启动：

```bash
DRY_RUN=1 LAUNCH=0 bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_sanity_matrix.sh gate
```

| ID | tag | modality | model | init | batch | seed | 作用 |
|---|---|---|---|---|---:|---:|---|
| G1 | cclkd_gate_sar_x_b32_pretrained | sar | x | pretrained | 32 | 0 | Primary CCLKD Table-2-style YOLOv5-X SAR baseline gate |
| G2 | cclkd_gate_sar_x_b64_pretrained | sar | x | pretrained | 64 | 0 | Batch-size sanity only |
| G3 | cclkd_gate_rgb_x_b32_pretrained | rgb | x | pretrained | 32 | 0 | Optional teacher-strength gate for CCLKD |

Scratch 和 x6 不再进入默认矩阵。Do not reproduce CoLD or CMDistill in this stage.

## 受控并行

默认设置：

- `DRY_RUN=1`
- `LAUNCH=0`
- `PARALLEL_LAUNCH=0`
- `MAX_PARALLEL=2`

只打印命令：

```bash
DRY_RUN=1 LAUNCH=0 bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_sanity_matrix.sh gate
```

受控并行启动前两条 gate：

```bash
LAUNCH=1 DRY_RUN=0 PARALLEL_LAUNCH=1 GPU_LIST=0,1 MAX_PARALLEL=2 \
  bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_sanity_matrix.sh gate
```

这会启动 G1/G2，并打印 G3 的剩余手动启动命令。若要同时启动 G1/G2/G3，设置 `GPU_LIST=0,1,2 MAX_PARALLEL=3`。

## 汇总

```bash
python cclkd_reproduction/yolov5_sanity/tools/summarize_yolov5_sanity.py \
  cclkd_reproduction/yolov5_sanity/results/runs
```

输出：

- `results/summary.csv`
- `results/summary.md`

## 决策规则

- 如果 G1 或 G2 接近 target：进入 YOLOv5 CCLKD full / ATKD / CCL 复现。
- 如果 G1 显著低于 G2：说明 batch32 可能影响 CCLKD Table-2-style baseline。
- 如果 G3 teacher 不强：不要期待大 KD 增益。
- 如果 G1/G2 都明显低：停止 KD，优先查数据 split、类别映射、YOLOv5 版本、评估脚本和增强，而不是修改 CCLKD 公式。
- 如果后续 x6 诊断接近 target 但 YOLOv5x 不接近：记录参数口径矛盾，并重新评估 backbone 口径。

50 epoch 若 AP50 异常低，例如低于 50，优先查数据/评估，不继续浪费 400 epoch。400 epoch 若 batch32 或 batch64 任一接近 80.9/46.3，可继续解释 CCLKD。

## 禁止写法

- “YOLO11 ablation reproduces CCLKD Table 12.”
- “CCLKD Table 12 failed on YOLO11.”
- “YOLOv5x6 is the paper backbone”，除非 x6 sanity 提供强证据。
- “This stage reproduces CoLD or CMDistill.”

## 推荐写法

- “We first verify the YOLOv5-X SAR baseline used by CCLKD.”
- “YOLOv5-X SAR baseline is the CCLKD reproduction gate.”
- “YOLO11 experiments are treated as adaptation diagnostics or Table-8-style extension comparisons.”

## 当前边界

暂时不要新增 YOLO11 CCLKD component ablation，不要继续修改 CCLKD YOLO11 公式，不要实现 CoLD/CMDistill-style baseline，不要跑大规模 YOLOv5 CCLKD full。只有 YOLOv5-X baseline gate 趋势正常后，才进入 YOLOv5 CCLKD full / ATKD / CCL 复现。
