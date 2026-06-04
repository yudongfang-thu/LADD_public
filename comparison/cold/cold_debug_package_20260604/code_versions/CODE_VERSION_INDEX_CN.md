# CoLD 实现版本索引

## 1. `legacy_20260409_non_mainline`

早期非主线 CoLD 实现，主要是 Ultralytics/YOLO11 风格下的 OBB/在线 KD 尝试。

包含：

- `src/cold_kd/`
- `src/cold_kd_online/`
- `tools/train_cold_kd.py`
- `tools/train_cold_kd_online.py`
- `scripts/run_cold_kd_clean_wait.sh`
- `scripts/run_cold_kd_online_clean.sh`

用途：追踪最早的 CoLD loss 设计，包括旧的 IoU weight / online teacher 思路。

## 2. `v5p0_hbb_current_local`

当前本地 YOLOv5-v5.0 HBB 复现主代码。

关键文件：

- `train_cold_v5p0_hbb.py`
- `run_cold_v5p0_hbb.sh`
- `queue_cold_terms_line_4090d.sh`
- `queue_cold_terms_line_5880ada.sh`
- `queue_cold_offline_terms_serial_90.sh`
- `audit_cold_teacher_grad.py`

用途：当前所有 90/117 CoLD 复现实验的基础实现。

## 3. `experiment_records/90_current_online_noiwm_20260604/extracted/code`

90 服务器上正在运行的精确代码快照。

当前实验：

| 实验 | GPU | 配置 |
| --- | ---: | --- |
| NCLD no-IWM | 3 | `candidate`, `cold_terms=ncld`, `IWM=none`, `batch=32`, `acc=64` |
| TCLD no-IWM | 4 | `candidate`, `cold_terms=tcld`, `IWM=none`, `batch=32`, `acc=64` |
| BOTH no-IWM | 5 | `candidate`, `cold_terms=both`, `IWM=none`, `batch=32`, `acc=64` |

## 4. `experiment_records/117_current_iwm_and_history_20260604/extracted/code`

117 服务器上正在运行的精确代码快照。

当前实验：

| 实验 | GPU | 配置 |
| --- | ---: | --- |
| BOTH + IWM | 0 | `candidate`, `cold_terms=both`, `IWM=mean`, `batch=64` |

## 5. YOLOv5 patch context

90 和 117 的 `extracted/yolov5_patch_context/` 保存了当时 YOLOv5-v5.0 的关键上下文文件，例如：

- `utils/datasets.py`
- `utils/loss.py`
- `utils/general.py`
- `utils/metrics.py`
- `models/yolo.py`
- `models/common.py`
- `test.py`

用途：检查 PyTorch 2.x 兼容 patch、cache 读取 patch、YOLO loss/评估逻辑是否影响结果。

