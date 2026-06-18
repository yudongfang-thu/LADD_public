# VEDAI Protocol Audit for CMDistill Native Reproduction

Updated: 2026-06-18

This note records the evidence trail before launching more VEDAI baseline runs. The goal is to avoid comparing numbers produced by different VEDAI protocols as if they were the same metric.

## Key conclusion

Do not use the completed IR-only YOLOv5s run on `paper80_seed0` as a pass/fail test for CMDistill Table I. It differs from the target result in at least three ways:

- CMDistill's reported baseline target for VEDAI is the RGB YOLOv5s baseline, not the IR teacher baseline.
- `paper80_seed0` is a random 997/249 split chosen to mimic CMDistill's "approximately 8:2" statement, but it is not the official VEDAI fold protocol.
- YOLOv5's default validation reports HBB IoU metrics (`mAP@0.5` and `mAP@0.5:0.95`), whereas the original VEDAI DevKit uses a point-in-ellipse matching protocol and 11-point AP.

The next experiment should therefore be chosen deliberately:

1. If the goal is to track CMDistill Table I, keep the CMDistill-like 8:2, `imgsz=640`, YOLOv5s, `batch=64`, SGD/cosine setting, but first run the RGB baseline and compare `mAP@0.5`, not `mAP@0.5:0.95`.
2. If the goal is to show a community-standard VEDAI reproduction, add official-fold runs, ideally all 10 folds or at least fold01 as a diagnostic, and report them separately from the CMDistill-like 8:2 track.

## Official VEDAI evidence

Official entry point: https://downloads.greyc.fr/vedai/

The official page describes VEDAI as a benchmark with images in multiple spectral bands and resolutions, and explicitly provides 512, 1024, and a development kit. The page also says a precise experimental protocol is provided for reproducible comparison.

Local DevKit evidence:

- `DevKit/evaluate_all_fold.m` evaluates files for the 10 folds, expecting result files named like `filename_%02d.txt`.
- `Annotations512/fold01.txt` ... `fold10.txt` and `fold01test.txt` ... `fold10test.txt` are present.
- Each official fold contains 1089 train ids and 121 test ids.
- The official fold files cover 1210 image ids, while the strict `*_co.png` / `*_ir.png` pair scan finds 1246 pairs. This explains why some papers describe VEDAI as 1210 images while CMDistill/SuperYOLO describe 1246 aligned pairs.

Official DevKit evaluator details:

- Input result format is per-class/per-fold point detections: `imgID X Y Score`.
- Matching is not IoU over HBB. For each ground-truth oriented object, the evaluator derives an ellipse from the OBB and counts a detection as positive if the predicted point lies inside the ellipse.
- AP is computed using the 11-point interpolation loop over recall thresholds `0:0.1:1`.
- Class IDs in the evaluator include `1 car`, `2 trucks`, `4 tractors`, `5 camping cars`, `9 vans`, `10 others`, `11 pickup`, `23 boats`, plus aggregate classes `201` and `301`.

Implication: official VEDAI DevKit AP is a different metric from YOLO HBB `mAP@0.5`.

## CMDistill native setting

Local source: `comparison/cmdistill/paper/CMDistill__2025_JSTARS__Cross_Modal_Distillation_Framework_for_AAV_Image_Object_Detection.pdf`

CMDistill states:

- VEDAI has 1246 aligned RGB-IR image pairs and 11 vehicle types.
- Train/test are split at an approximate 8:2 ratio.
- Teacher is an IR detector; student is an RGB detector.
- Detector backbone is YOLOv5 with FPN.
- Images are resized to 640 x 640.
- Augmentation mainly includes random rotation, random crop, and color dithering.
- Optimizer is SGD with initial lr 0.01, momentum 0.937, batch size 64, weight decay 5e-4, and cosine decay.
- During inference, only a single modality is used.

CMDistill does not make these items explicit enough for exact reproduction:

- 512 release or 1024 release.
- Exact split file list.
- Training epochs.
- Exact mAP threshold/evaluator.
- Whether VEDAI's official point-in-ellipse DevKit was used or whether the task was converted to HBB IoU evaluation.

Given that CMDistill compares generic detectors such as YOLOv5, YOLOv8, RT-DETR, FCOS, and TOOD with ordinary object-detection mAP formulas, its Table I is more consistent with a generic detector HBB/IoU mAP track than with the original VEDAI point-detection DevKit. This is still an inference, not an explicit statement in the paper.

## Same-dataset paper practices

| Paper / source | Split | Data / box handling | Metric | Training details most relevant to us |
|---|---|---|---|---|
| Original VEDAI DevKit | 10 folds | Point detections evaluated against OBB-derived ellipses | 11-point AP | Result files are `imgID X Y Score`; not YOLO HBB. |
| SuperYOLO, TGRS 2023 | 10-fold cross-validation; ablation on fold1; comparisons over 10 folds | 1024 and 512 releases; labels converted to YOLO-style relative boxes | `mAP50` | SGD, momentum 0.937, weight decay 5e-4, lr 0.01, 300 epochs; batch size 2. Reports YOLOv5s RGB 54.82, IR 49.94, multi 56.79, SuperYOLO RGB 72.49, multi 75.09. |
| YOLOrs-lite, IGARSS 2021 | 10 cross-validation folds | Rotated/remote-sensing YOLO setting | mAP averaged over 10 folds with IoU threshold 0.2 | Uses confidence threshold 0.7 and NMS threshold 0.1. Shows older VEDAI work may use IoU 0.2 because objects are tiny. |
| RT-YOLO, CMC 2023 | 4:1 split, 994 train / 248 test | HBB object detection | `mAP@0.5` | Input 640 x 640, COCO pretraining, Adam lr 0.001, batch 32, 300 epochs. |
| DAMSDet, ECCV 2024 | Public multispectral VEDAI benchmark track | Converts boxes to horizontal boxes following YOLOFusion | `mAP50` and COCO mAP | Uses 1024 x 1024 images and 50 epochs. Confirms recent multispectral VEDAI papers often leave the official ellipse metric and use HBB IoU metrics. |

Sources:

- VEDAI official page: https://downloads.greyc.fr/vedai/
- SuperYOLO arXiv page: https://arxiv.org/abs/2209.13351
- SuperYOLO GitHub: https://github.com/icey-zhang/SuperYOLO
- YOLOrs-lite PDF: https://par.nsf.gov/servlets/purl/10299509
- RT-YOLO full text: https://www.techscience.com/cmc/v75n1/51475/html
- DAMSDet ECCV page: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3984_ECCV_2024_paper.php

## Implications for our reproduction plan

Recommended reporting structure:

- `CMDistill-like VEDAI reproduction`: `paper80_seed0`, 640, YOLOv5s, batch 64, HBB mAP@0.5. This is the closest available track to CMDistill's text, but not officially exact because CMDistill omits the exact split and evaluator.
- `VEDAI official-fold sanity`: official fold01 first, then all 10 folds if needed. This answers whether the data conversion and YOLO training are in the same range as common VEDAI papers.
- `Original DevKit reproduction`: optional and separate. It requires converting YOLO detections into per-class `imgID X Y Score` point files and running the MATLAB/Octave DevKit, so it should not be mixed with YOLO validation metrics.

Current action recommendation:

- The CMDistill Table I alignment track has now been validated with RGB on `paper80_seed0`, evaluated primarily by `mAP@0.5`.
- The RGB YOLOv5s baseline reached best.pt validation `mAP@0.5=0.695`, close to CMDistill Table I's RGB YOLOv5s `0.702`.
- A native YOLOv5 CMDistill-style `IR teacher -> RGB student` formal run has been launched under the same Table I track: `cmdi_rgb_ir_e300_20260618_133714`.
- The active run deliberately disables unsynchronized geometric/color augmentation so the paired IR teacher input remains aligned with the RGB student input. This is a conservative first-pass reproduction; full random rotation/crop/color dithering requires synchronized paired augmentation.
- For stronger evidence independent of CMDistill's ambiguous split, prepare all official folds later and report them separately from this CMDistill-like track.
