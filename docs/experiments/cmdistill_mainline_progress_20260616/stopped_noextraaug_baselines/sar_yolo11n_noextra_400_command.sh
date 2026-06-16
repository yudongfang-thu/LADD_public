cd /root/autodl-tmp/LADD_public
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=4
python3 baseline/code/train_ogsod_baseline.py \
  --task hbb \
  --model yolo11n.pt \
  --data configs/datasets/ogsod_hbb_sar.yaml \
  --imgsz 256 \
  --epochs 400 \
  --batch 32 \
  --workers 4 \
  --device 0 \
  --patience 400 \
  --project runs_public/ogsod/hbb/cclkd_table2_noextraaug_20260616/baselines/sar \
  --name sar_yolo11n_hbb_400ep_sgd_table2_noextraaug_b32_s0 \
  --optimizer SGD \
  --lr0 0.01 \
  --lrf 0.01 \
  --momentum 0.937 \
  --mosaic 0.0 \
  --mixup 0.0 \
  --cutmix 0.0 \
  --close-mosaic 0 \
  --hsv-h 0.0 \
  --hsv-s 0.0 \
  --hsv-v 0.0 \
  --degrees 0.0 \
  --perspective 0.0 \
  --translate 0.0 \
  --scale 0.0 \
  --fliplr 0.0 \
  --flipud 0.0 \
  --erasing 0.0 \
  --disable-albumentations \
  --seed 0 \
  --deterministic \
  --save-period 100
