# CoLD 90 服务器迁移运行说明

日期：2026-06-03

## 目标

把 CoLD 复现工作从 117 服务器迁移到 90 服务器。117 后续只用于非 CoLD 的 LADD 主实验、消融和对比实验；90 作为 CoLD 独立诊断/复现服务器。

## 90 工作区

独立工作区：

```text
/mnt/dataY/ydf/projects/LADD_cold_v5p0_20260603
```

不要把 CoLD 新代码直接放进正在跑非 CoLD 任务的 `/mnt/dataY/ydf/projects/LADD_og` 主工作树。

复用 90 上已有资源：

```text
/mnt/dataY/ydf/projects/yolov5_cold_v5p0
/mnt/dataY/ydf/projects/LADD_ref/configs/datasets/ogsod_hbb_sar.yaml
/mnt/dataY/ydf/projects/LADD_ref/configs/datasets/ogsod_hbb_rgb.yaml
/mnt/dataY/ydf/projects/LADD_ref/yolov5x.pt
```

迁移后的 RGB teacher 权重：

```text
/mnt/dataY/ydf/projects/LADD_cold_v5p0_20260603/cold_anchor/weights/rgb_teacher_yolov5x_v5p0_coco_mixup010_e100_117_best.pt
```

## 启动原则

90 当前可能有非 CoLD 任务占用 GPU。启动前必须先检查：

```bash
nvidia-smi
pgrep -af 'train_ladd|train_cold|yolo|python'
```

不要默认抢占正在运行的非 CoLD 任务。等 GPU 空闲后再启动 CoLD。

## Smoke 测试

先跑 smoke，不直接开 400 epoch：

```bash
cd /mnt/dataY/ydf/projects/LADD_cold_v5p0_20260603
GPU_ID=0 \
EPOCHS=1 \
BATCH_SIZE=16 \
EFFECTIVE_BATCH_SIZE=16 \
MAX_BATCHES=2 \
TERMS=ncld \
RUN_TAG=smoke_$(date +%Y%m%d_%H%M%S) \
bash scripts/ogsod_public/cold_baseline_repro_20260528/tmux_launch_cold_90.sh
```

查看：

```bash
tmux ls
tmux attach -t <session>
tail -f cold_anchor/logs/*.log
```

Smoke 通过标准：

- 能读取 SAR/RGB YAML；
- 能加载 `yolov5x.pt` 和 RGB teacher `best.pt`；
- 能进入训练；
- `cold_stats.csv` 生成；
- `loc_cold` 非负；
- 无 CUDA OOM。

## 诊断实验建议

当前 117 离线 NCLD 50ep 前段明显落后 5090D no-cold baseline，不建议直接开 400ep。90 上先跑短诊断：

```bash
cd /mnt/dataY/ydf/projects/LADD_cold_v5p0_20260603
GPU_ID=0 \
EPOCHS=30 \
BATCH_SIZE=64 \
EFFECTIVE_BATCH_SIZE=64 \
TERMS="ncld tcld" \
CANDIDATE_TOPK=500 \
CANDIDATE_MIN_CONF=0.005 \
ALPHA_NON_TARGET=1.0 \
RUN_TAG=diag_topk500_conf005_alpha1_$(date +%Y%m%d_%H%M%S) \
bash scripts/ogsod_public/cold_baseline_repro_20260528/tmux_launch_cold_90.sh
```

只有短诊断趋势合理后，再考虑 `both` 或 400 epoch。
