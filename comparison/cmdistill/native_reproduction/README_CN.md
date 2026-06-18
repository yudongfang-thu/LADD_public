# CMDistill Native Benchmark 复现轨道

本目录用于回答一个独立问题：我们当前的 CMDistill-style 实现是否能在 CMDistill 原文自己的实验设定下复现到合理量级。这里的结果只用于支撑 CMDistill 作为可信对比方法，不直接混入 OGSOD formal 主表。

## 证据来源

- CMDistill 本地论文：`../paper/CMDistill__2025_JSTARS__Cross_Modal_Distillation_Framework_for_AAV_Image_Object_Detection.pdf`
- 论文 DOI：`10.1109/JSTARS.2024.3479717`
- VEDAI 官方下载页：`https://downloads.greyc.fr/vedai/`
- DroneVehicle 官方仓库：`https://github.com/VisDrone/DroneVehicle`

## 原文 native 设置

CMDistill 原文设置与我们 OGSOD 适配不同，必须分开记录：

| 项目 | CMDistill 原文 native 设置 | 当前 OGSOD 适配 |
|---|---|---|
| 数据集 | VEDAI、DroneVehicle | OGSOD-1.0 |
| 蒸馏方向 | IR teacher -> RGB student | RGB teacher -> SAR student |
| 推理输入 | RGB-only | SAR-only |
| 检测器 | YOLOv5 teacher/student | YOLO11 controlled comparison |
| 输入尺寸 | 640 x 640 | 256 x 256 |
| 优化器 | SGD, lr 0.01, momentum 0.937, weight decay 5e-4 | LADD formal protocol |
| batch | 64 | n/s=64, m/l=32, x=16 |
| LR 调度 | cosine decay | cosine decay |
| 增强 | random rotation, random crop, color dithering | formal OGSOD augmentation policy |

论文明确描述 teacher 为冻结的 IR detector，student 为 RGB detector；训练时使用 RGB/IR 成对输入，推理时移除 IR detector、PCCFD、SLRD、IBCLD，只保留 RGB student。

## 原文目标量级

从 CMDistill Table I / II 抽取的主目标：

| 数据集 | 方法 | 模态 | 论文 mAP |
|---|---|---|---|
| VEDAI | YOLOv5s baseline | RGB | 0.702 |
| VEDAI | CMDistill / Ours | IR -> RGB | 0.740 |
| DroneVehicle | YOLOv5s baseline | IR | 70.7 |
| DroneVehicle | CMDistill / Ours | IR -> RGB | 74.3 |

注意：VEDAI 表使用 0-1 小数；DroneVehicle 表使用百分数。DroneVehicle 上 CMDistill 强于单模态方法，但低于部分 RGB+IR 双模态融合方法，这一点和论文叙述一致。

## 当前目录内容

| 文件 | 作用 |
|---|---|
| `DATASETS.md` | 数据集入口、下载状态、未公开细节和建议落盘结构 |
| `LOCAL_DATA_STATUS.md` | 本机已下载数据和 tar sanity check |
| `VEDAI_PROTOCOL_AUDIT.md` | VEDAI 官方 DevKit、CMDistill 原文和同数据集论文的协议差异 |
| `REPRODUCTION_CHECKLIST.md` | 从下载到复现验收的任务清单 |
| `scripts/download_vedai.sh` | VEDAI 官方直链下载脚本，默认 512 release |
| `scripts/extract_vedai.sh` | 解压 VEDAI 官方 tar，不额外合并大 tar |
| `scripts/prepare_vedai_yolo_hbb.py` | VEDAI RGB/IR -> YOLO HBB 转换脚本 |
| `scripts/prepare_vedai_yolo_hbb.sh` | 解压 + 转换一键入口 |
| `scripts/setup_yolov5_v62.sh` | AutoDL/远端 YOLOv5 v6.2 环境准备 |
| `scripts/run_vedai_yolov5_baseline.sh` | VEDAI RGB/IR YOLOv5s baseline 启动脚本 |
| `scripts/queue_vedai_yolov5_baselines_autodl.sh` | 等 GPU 空闲后顺序跑 IR/RGB baseline |
| `scripts/train_vedai_yolov5_cmdistill_native.py` | 原生 YOLOv5 v6.2 CMDistill-style 训练入口 |
| `scripts/run_vedai_yolov5_cmdistill_native.sh` | VEDAI CMDistill native 启动脚本，默认 IR teacher -> RGB student |
| `scripts/check_dronevehicle_manual_download.sh` | DroneVehicle 百度网盘手动下载后的结构检查脚本 |

## 下一步

1. VEDAI 512 已在本机和 AutoDL 完成下载、解压、YOLO HBB 转换。
2. AutoDL 已准备 YOLOv5 v6.2 环境，并校验官方 `yolov5s.pt` 权重。
3. IR YOLOv5s baseline 已完成；RGB YOLOv5s baseline 已在 CMDistill Table I track 上复现到 `mAP@0.5=0.695`，接近论文 `0.702`。
4. VEDAI CMDistill native 已启动：冻结 IR teacher，以 RGB student 为推理模型，300 epoch、batch 64、640 输入，当前 formal screen 为 `cmdi_rgb_ir_e300_20260618_133714`。
5. 当前 native run 为 aligned no-geo 保守版本，用来保证 teacher/student 成对输入严格空间对齐；若结果未达论文 `0.740`，下一步实现同步 paired augmentation。
6. 后续扩展 DroneVehicle 74.3 mAP。
7. 由用户后续通过百度网盘下载 DroneVehicle Train/Validation/Test 到本目录 `data/raw/DroneVehicle/` 或外部 `CMDISTILL_NATIVE_DATA_ROOT`。
