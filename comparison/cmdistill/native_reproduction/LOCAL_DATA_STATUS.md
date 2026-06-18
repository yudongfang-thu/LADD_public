# Local Data Status

Updated: 2026-06-18

## VEDAI 512

Status: downloaded, byte-size verified, extracted, and converted to YOLO HBB locally and on AutoDL.

Repo-relative location:

```text
comparison/cmdistill/native_reproduction/data/raw/VEDAI/512/
```

This directory is ignored by Git via `native_reproduction/.gitignore`.

AutoDL location:

```text
/root/autodl-tmp/LADD_public/comparison/cmdistill/native_reproduction/data/
```

Downloaded files:

| File | Local size |
|---|---:|
| `Annotations512.tar` | 1.7M |
| `Vehicules512.tar.001` | 667M |
| `Vehicules512.tar.002` | 566M |
| `DevKit.tar` | 531K |
| `TermsandConditionsofUseVeDAI2014.pdf` | 52K |

Tar structure sanity:

| Check | Count / result |
|---|---:|
| strict RGB images, `*_co.png` | 1246 |
| strict IR images, `*_ir.png` | 1246 |
| annotation txt files | 1267 |
| all PNG-like entries | 2573 |

Notes:

- The strict RGB/IR image pair count matches the CMDistill paper statement of 1246 aligned RGB-IR pairs.
- The official image archive includes extra historical duplicate-like IR files such as `*.png2.png` and one `copie` entry. Conversion scripts should ignore these and only consume strict `*_co.png` / `*_ir.png` pairs.
- The annotation archive has more txt files than strict image pairs; conversion should use paired image ids as the canonical sample list.
- `DevKit.tar` contains MATLAB evaluation/fold helper scripts. The official VEDAI protocol is 10-fold, with 1089 train ids and 121 test ids in each fold, but CMDistill reports only an approximate 8:2 split and does not disclose an exact split list.

Prepared YOLO HBB outputs:

```text
comparison/cmdistill/native_reproduction/data/processed/
  VEDAI512_paper8_hbb_paper80_seed0/
  VEDAI512_paper8_hbb_official_fold01/
```

`paper80_seed0` is the current main candidate because CMDistill says it used an approximate 8:2 train/test split:

| Split | RGB images | IR images | Label files | Empty labels |
|---|---:|---:|---:|---:|
| train | 997 | 997 | 997 | 9 |
| val | 249 | 249 | 249 | 2 |

`official_fold01` is retained as a VEDAI-devkit reference split. The official fold files cover 1210 ids rather than all 1246 strict RGB/IR pairs:

| Split | RGB images | IR images | Label files | Empty labels |
|---|---:|---:|---:|---:|
| train | 1089 | 1089 | 1089 | 8 |
| val | 121 | 121 | 121 | 2 |

Conversion policy:

- Use the 8 classes reported in CMDistill Table I: car, pickup, camper, truck, other, tractor, boat, van.
- Convert VEDAI quadrilateral boxes to HBB via min/max corner coordinates.
- Filter rare original class ids `7`, `8`, and `31` because CMDistill Table I does not report them; images containing only filtered classes are kept with empty labels.
- Use symlinks from processed images back to extracted official images to avoid duplicate image storage.

## AutoDL YOLOv5 Baselines

Status: IR baseline completed, and the CMDistill Table I RGB baseline track has reached the reported paper scale.

AutoDL environment:

| Item | Status |
|---|---|
| YOLOv5 code | `/root/autodl-tmp/yolov5-v6.2`, tag `v6.2` |
| Python | `/root/miniconda3/bin/python` |
| Python dependency check | passed |
| `train.py --help` smoke check | passed |
| Official `yolov5s.pt` | present |
| `yolov5s.pt` SHA256 | `8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee` |
| PyTorch >=2.6 checkpoint compatibility | `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in the launch script |
| NumPy >=1.24 YOLOv5 v6.2 compatibility | `np.int` / `np.float` aliases patched during setup |
| Early stopping alignment | baseline launcher now defaults `PATIENCE=EPOCHS` |

Parallel run status:

| Run | Setting | Status |
|---|---|---|
| IR + RGB both `batch=64` | parallel with LADD | OOM during first training epoch |
| IR + RGB both `batch=32` | parallel with LADD | stopped; used only as a feasibility probe |
| IR then RGB, both `batch=64` | parallel with LADD, one VEDAI run at a time | IR completed; RGB partial run stopped intentionally |
| RGB `batch=64`, Table I track | parallel with LADD | completed; best `mAP@0.5=0.695` after best.pt validation |

Completed IR run:

```text
/root/autodl-tmp/LADD_public/comparison/cmdistill/native_reproduction/runs/vedai_yolov5_baseline/paper80_seed0/ir/vedai512_ir_yolov5s_e300_b64_img640_s0_seq_b64_20260618_114037
```

IR final validation metrics at epoch 299:

| P | R | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|
| 0.69205 | 0.54195 | 0.59271 | 0.32596 |

Near-best IR metrics were about `mAP@0.5=0.601` and `mAP@0.5:0.95=0.331`.

Logs:

```text
/root/autodl-tmp/LADD_public/comparison/cmdistill/native_reproduction/logs/vedai_yolov5_seq_b64/20260618_114037_ir_then_rgb_b64_master.log
/root/autodl-tmp/LADD_public/comparison/cmdistill/native_reproduction/logs/vedai_yolov5_seq_b64/20260618_114037_ir_baseline_b64_seq_b64_20260618_114037.log
/root/autodl-tmp/LADD_public/comparison/cmdistill/native_reproduction/logs/vedai_yolov5_seq_b64/20260618_114037_rgb_baseline_b64_seq_b64_20260618_114037.log
```

The RGB log is partial and must not be treated as a result.

Completed RGB Table I baseline run:

```text
/root/autodl-tmp/LADD_public/comparison/cmdistill/native_reproduction/runs/vedai_yolov5_baseline/paper80_seed0/rgb/vedai512_rgb_yolov5s_e300_b64_img640_s0_table1_rgb_b64_20260618_124005
```

The run used the same CMDistill-like track as the IR baseline: VEDAI 512, `paper80_seed0`, YOLOv5s, `imgsz=640`, `batch=64`, SGD, cosine LR, seed 0. YOLOv5's default early stopping interrupted it after 231 epochs because the best epoch was 130; the final best.pt validation nevertheless matches CMDistill Table I closely. The launcher has since been updated so future runs default `PATIENCE=EPOCHS`.

RGB best.pt validation metrics:

| P | R | mAP@0.5 | mAP@0.5:0.95 | Paper Table I RGB YOLOv5s mAP | Gap |
|---:|---:|---:|---:|---:|---:|
| 0.700 | 0.649 | 0.695 | 0.390 | 0.702 | -0.007 |

RGB `results.csv` best row:

| Epoch | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|---:|
| 130 | 0.70047 | 0.64905 | 0.69191 | 0.38996 |

RGB logs:

```text
/root/autodl-tmp/LADD_public/comparison/cmdistill/native_reproduction/logs/vedai_yolov5_table1_rgb_b64/20260618_124005_rgb_baseline_table1_b64.log
```

## AutoDL CMDistill Native

Status: native YOLOv5 v6.2 training entry implemented, smoke-tested, and formal VEDAI run launched.

Implemented scripts:

```text
comparison/cmdistill/native_reproduction/scripts/train_vedai_yolov5_cmdistill_native.py
comparison/cmdistill/native_reproduction/scripts/run_vedai_yolov5_cmdistill_native.sh
```

Current CMDistill Table I direction follows the paper text: frozen IR teacher -> RGB student. The launcher defaults to `STUDENT_MODALITY=rgb` and `TEACHER_MODALITY=ir`; the reverse direction remains available only as an explicit diagnostic override.

Native smoke checks:

| Run | Direction | Setting | Result |
|---|---|---|---|
| `20260618_132703` | RGB student <- IR teacher | `epoch=1`, `batch=8`, aligned no-geo | completed, `mAP@0.5=0.682` |
| `20260618_132905` | RGB student <- IR teacher | `epoch=1`, `batch=64`, aligned no-geo | completed, `mAP@0.5=0.703`, peak total GPU memory about 20.3G with LADD also running |

Formal run launched:

```text
screen: cmdi_rgb_ir_e300_20260618_133714
log: /root/autodl-tmp/LADD_public/comparison/cmdistill/native_reproduction/logs/vedai_yolov5_cmdistill_native_formal/20260618_133714_rgb_ir_e300_b64_aligned_nogeo.log
run: /root/autodl-tmp/LADD_public/comparison/cmdistill/native_reproduction/runs/vedai_yolov5_cmdistill_native/paper80_seed0/vedai512_rgb_ir_cmdistill_yolov5s_e300_b64_img640_s0_rgb_ir_e300_b64_aligned_nogeo_20260618_133714
```

Formal settings:

| Item | Value |
|---|---|
| dataset | VEDAI512 `paper80_seed0` |
| student | RGB YOLOv5s, initialized from RGB Table I baseline best.pt |
| teacher | frozen IR YOLOv5s, initialized from completed IR baseline best.pt |
| epochs / batch / image size | 300 / 64 / 640 |
| optimizer | SGD, lr0 0.01, momentum 0.937, weight decay 5e-4, cosine LR |
| KD weights | feature=1, relation=1, logit=1 |
| paired augmentation mode | `ALIGNED_NO_GEO=1`; mosaic/geometric/flip/HSV disabled to keep RGB/IR teacher-student tensors spatially aligned |

Initial formal check:

| Epoch | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|---:|
| 0 | 0.737 | 0.646 | 0.703 | 0.406 |

Important caveat: the current formal run is a conservative aligned-pair variant, not yet the full paper augmentation variant (`random rotation`, `random crop`, `color dithering`). If final mAP does not reach the Table I `0.740` target, the next reproduction step is synchronized paired augmentation rather than changing the dataset split.

## DroneVehicle

Status: not downloaded locally.

Reason: official access is via BaiduYun Train/Validation/Test links. User action may be needed if browser login, quota, or membership is required.
