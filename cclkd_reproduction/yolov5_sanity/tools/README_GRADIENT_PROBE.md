# YOLOv5 CCLKD Gradient Probe

`probe_yolov5_cclkd_gradients.py` is an offline diagnostic helper for the
YOLOv5x CCLKD reproduction audit. It runs one or a few paired SAR/RGB training
batches, computes CCLKD component losses, and reports student-parameter gradient
norms plus cosine similarity against the student detection-loss gradient.

It does not modify active training jobs:

- no optimizer step
- no checkpoint write
- no hook insertion into active 400epoch processes
- no change to `cclkd_yolov5_loss.py` or the main trainer

## Suggested Use

Run after a milestone checkpoint is available, preferably on server 90:

```bash
python cclkd_reproduction/yolov5_sanity/tools/probe_yolov5_cclkd_gradients.py \
  --weights cclkd_reproduction/yolov5_sanity/results/runs/<run>/weights/last.pt \
  --teacher-weights cclkd_reproduction/yolov5_sanity/results/runs/<run>/weights/last.pt \
  --student-ckpt-key model \
  --teacher-ckpt-key teacher \
  --data configs/datasets/ogsod_hbb_sar.yaml \
  --teacher-data configs/datasets/ogsod_hbb_rgb.yaml \
  --hyp cclkd_reproduction/yolov5_sanity/configs/hyp_cold_ogsod.yaml \
  --device 0 \
  --batch-size 8 \
  --max-batches 2 \
  --csv-out cclkd_reproduction/yolov5_sanity/results/<archive>/gradient_probe_batches.csv \
  --summary-out cclkd_reproduction/yolov5_sanity/results/<archive>/gradient_probe_summary.csv \
  --md-out cclkd_reproduction/yolov5_sanity/results/<archive>/gradient_probe_summary.md
```

Use a small batch first. The probe retains the computation graph while measuring
multiple component gradients, so memory use is higher than a normal single
backward pass.

## Reported Signals

Per batch and summary outputs include:

- student detection loss and teacher detection loss
- LLD, FLD, RLD, ATKD, CCL, and weighted KD losses
- gradient norms for detection, LLD, FLD, RLD, ATKD, CCL, and KD total
- cosine similarity of each KD component gradient against detection gradient
- COP valid/positive counts and positive ratio
- temperature mean/min/max
- feature-capture and NaN/Inf flags

These diagnostics support the final reproduction report, especially when a run
shows small positive AP deltas but weak paper-style component synergy.
