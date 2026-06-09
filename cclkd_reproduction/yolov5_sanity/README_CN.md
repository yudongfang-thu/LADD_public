# YOLOv5 / CoLD-style Baseline Sanity

这组工具只服务于一个问题：在继续 CCLKD、CMDistill 或 CoLD 复现之前，先确认我们能否复现 CoLD 和 CCLKD 共同引用的 YOLOv5-X SAR baseline。

CCLKD Table 5 的 YOLOv5 baseline 与 CoLD Table I 完全一致；CCLKD 主对比和 Table 4/5 也是 YOLOv5 口径。YOLO11 component ablation 不能当作 CCLKD Table 12 reproduction，因此这里先验证 YOLOv5-X / CSPDarkNet-X 的基础结果是否站得住。

## 目标

Primary target:

- model: YOLOv5x / CSPDarkNet-X
- modality: SAR
- imgsz: 256
- epochs: 400
- batch: 64
- optimizer: SGD
- lr: 0.01
- momentum: 0.937
- weight decay: 0.0005
- augmentation: YOLOv5 defaults plus explicit Mosaic 1.0 and MixUp 0.1

CoLD Table I target:

- Params: 86.23M
- AP50: 80.9
- AP: 46.3
- per-class AP50: Oil Tank 57.7, Bridge 87.2, Harbor 97.9

Loose pass threshold:

- AP50 >= 78
- AP >= 44

CoLD 写 batch=64, imgsz=256, epochs=400, SGD lr=1e-2, momentum=0.937, weight decay=5e-4, Mosaic + MixUp。CCLKD Table 2 写 OGSOD batch=32，其余类似，所以 sanity 同时覆盖 batch64 和 batch32。

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

YOLOv5x v7.0 官方参数量约 86.7M，`nc=3` 后接近 CoLD Table I 的 86.23M。因此主 baseline 使用 YOLOv5x，不使用 YOLOv5x6。YOLOv5x6 只作为 139.99M 参数矛盾诊断。

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

检查项包括 `nc=3`、class order、train/test 数量、label 数量、每类 instance 数量、label 坐标合法性和抽样图像尺寸。CoLD 表常见顺序是 Oil Tank / Bridge / Harbor；当前公开 yaml 常见顺序是 `bridge / harbor / storage_tank`，汇总时必须记录映射。

## Smoke 矩阵

默认只打印命令，不启动：

```bash
DRY_RUN=1 LAUNCH=0 bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_sanity_matrix.sh smoke
```

| ID | tag | modality | model | init | batch | seed | 作用 |
|---|---|---|---|---|---:|---:|---|
| E1 | cold_b64_pretrained | sar | x | pretrained | 64 | 0 | CoLD Table I primary baseline sanity |
| E2 | cclkd_b32_pretrained | sar | x | pretrained | 32 | 0 | 检查 CCLKD batch=32 是否改变 baseline |
| E3 | cold_b64_scratch | sar | x | scratch | 64 | 0 | 检查论文是否可能依赖 COCO pretrained |
| E4 | rgb_teacher_b64_pretrained | rgb | x | pretrained | 64 | 0 | 估计 optical teacher 上界和 teacher-student gap |
| E5 | x6_b32_pretrained_diag | sar | x6 | pretrained | 32 | 0 | 仅诊断 139.99M 参数矛盾，不作为主 baseline |

手动启动第一条：

```bash
LAUNCH=1 DRY_RUN=0 bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_ogsod_baseline.sh \
  sar x pretrained 64 0 0 cold_b64_pretrained
```

E1 跑到至少 50 epoch 后先看趋势；如果 AP50 明显异常，例如低于 50 或 loss 不正常，立即停止，不要浪费 400 epoch。

## 汇总

```bash
python cclkd_reproduction/yolov5_sanity/tools/summarize_yolov5_sanity.py \
  cclkd_reproduction/yolov5_sanity/results/runs
```

输出：

- `results/summary.csv`
- `results/summary.md`

## 决策规则

- 如果 E1 接近 target：补 seed42/123，并进入 YOLOv5 CCLKD / CoLD / CMDistill 复现。
- 如果 E2 显著低于 E1：说明 batch32 会影响 CCLKD Table2-style baseline。
- 如果 E3 明显低于 E1：说明 pretrained 初始化很重要。
- 如果 E4 teacher 不强：不要期待大 KD 增益。
- 如果 E1/E2/E3 都明显低：停止 KD，优先查数据、评估、标签、增强、YOLOv5 版本和类别映射。
- 如果 E5 x6 接近 target 但 E1 不接近：记录 139.99M 参数口径矛盾，并重新评估 backbone。

## 禁止写法

- “YOLO11 ablation reproduces CCLKD Table 12.”
- “CCLKD Table 12 failed on YOLO11.”
- “YOLOv5x6 is the paper backbone”，除非 x6 sanity 提供强证据。

## 推荐写法

- “We first verify the YOLOv5-X SAR baseline used by CoLD/CCLKD.”
- “YOLO11 experiments are treated as adaptation diagnostics or Table-8-style extension comparisons.”

## 当前边界

暂时不要新增 YOLO11 CCLKD component ablation，不要继续修改 CCLKD YOLO11 公式，不要实现 CMDistill-style baseline，不要跑 CCLKD YOLOv5 full。只有 YOLOv5-X baseline 接近 80.9/46.3 后，才继续 CCLKD/CMDistill/CoLD 复现。
