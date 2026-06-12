cd /mnt/dataY/ydf/projects/LADD_public
YOLOv5_AUTOINSTALL=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python3 external/yolov5/train.py \
  --img 256 \
  --epochs 80 \
  --batch-size 32 \
  --data configs/datasets/ogsod_hbb_sar.yaml \
  --hyp cclkd_reproduction/yolov5_sanity/configs/hyp_cold_ogsod.yaml \
  --device 0 \
  --project cclkd_reproduction/yolov5_sanity/results/runs \
  --name yolov5x_standard_train_b32_s0_standard_train_b32_e80_gpu0 \
  --weights external/yolov5/yolov5x.pt \
  --optimizer SGD \
  --patience 400 \
  --workers 4 \
  --seed 0 \
  --save-period 100 \
  --exist-ok
