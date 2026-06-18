# LADD Paper Protocol

最后更新：2026-06-18

本文定义论文主表唯一准入协议。任何结果进入 OGSOD-1.0 HBB 主表前，必须同时满足本文协议、`paper_results/` schema 和 paper launcher metadata 检查。历史 no-mosaic、A1-A2-B、BN-freeze、400 epoch、close@100、partial、smoke、diagnostic run 只保留为归档、诊断或附录证据，不能直接混入主表。

## 1. 主协议

```text
dataset: OGSOD-1.0 HBB
modalities: RGB teacher -> SAR student
inference: SAR-only
imgsz: 256
epochs: 800
A1 epochs: 10
B epochs: 800
optimizer: auto
lr0: 0.01
lrf: 0.01
cos_lr: true
warmup_epochs: 3.0
warmup_bias_lr: 0.1
mosaic: 1.0
close_mosaic: 700
mixup: 0.0
cutmix: 0.0
degrees: 0.0
perspective: 0.0
translate: 0.1
scale: 0.5
fliplr: 0.5
flipud: 0.0
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.0
erasing: 0.0
deterministic: true
seeds: 0, 42, 123
batch:
  yolo11n: 64
  yolo11s: 64
  yolo11m: 32
  yolo11l: 32
  yolo11x: 16
```

`mosaic=1.0, close_mosaic=700` 表示 800 epoch 中前 100 epoch 开启 mosaic，后 700 epoch 关闭 mosaic。本文统一称为 `mosaic100`。

机器可读版本位于 `configs/paper/ogsod_hbb_mosaic100.yaml`。Paper launcher 的共享常量位于 `scripts/paper/paper_common.sh`，每次修改协议后必须同步检查。

## 2. LADD 主方法准入

LADD 主表只接受：

```text
method: LADD Probe-A / LADD-clean A1B
method_key: clean_a1b_dynprobe
phase_chain: A1 -> B
LADD_A1B_MODE: dynamic_probe
RANK_D_NEG_CAP: 2.0
A1_MOSAIC: 1.0
A1_CLOSE_MOSAIC: 0
B_MOSAIC: 1.0
B_CLOSE_MOSAIC: 700
```

不允许：

```text
A2 as a main phase
static clean_a1b as mainline
dynamic clean_a1b_dyn as mainline
old A1-A2-B tags
BN-freeze repair runs as mainline
historical no-mosaic LADD rows as mainline
wrong nc / wrong yaml runs
```

Static 和 Dynamic clean A1B 只能进入 ablation 表；未标记 `clean_a1b_dynprobe` 的 LADD run 不能写作最终 LADD Probe-A。

## 3. Baseline 与对比方法准入

Baseline、LADD 和 comparison methods 必须按同容量、同 seed、同 dataset yaml、同增强协议配对。

对比方法主表只接受 paper wrapper 启动的受控适配：

```text
FGD-style / FGD-YOLO adaptation
LD
CMDistill-style / paper-aligned adaptation
HalluciDet-YOLO adaptation, only if its paper wrapper status is verified
CCLKD online comparison, optional and only if its online wrapper passes paper gate
```

Frozen-teacher KD comparison 必须满足：

```text
COMPARISON_KD_PROFILE = fgd / ld / cmdistill
PROFILE_KD_REPLACE_BASE = 1
STUDENT_BRANCH_MODE = raw
TEACHER_FEATURE_MODE = raw
LAMBDA_REACH = 0.0
LAMBDA_REC = 0.0
LAMBDA_TASKL = 0.0
ALPHA_S_REC = 0.0
```

CMDistill-style 必须满足：

```text
KD_CALIBRATION_MODE = affine
```

## 4. 结果准入状态

主表候选必须在 canonical CSV 中满足：

```text
status = verified
claim_usable = yes
protocol_id = ogsod_hbb_mosaic100_clean_a1b_probea_20260618
imgsz = 256
epochs = 800
mosaic = 1.0
close_mosaic = 700
git_commit is present
results.csv exists
args.yaml exists
manifest / paper_run_meta.env exists
```

下列结果只能标记为 `archive / diagnostic / smoke / partial / invalid / robustness_appendix`：

```text
no-mosaic runs
A1-A2-B runs
BN-freeze repair runs
400ep runs
close@100 runs
partial snapshots
smoke runs
diagnostic-only runs
runs missing results.csv / args.yaml / manifest
archived legacy LADD runs
CMDistill native VEDAI pending archive
frozen-teacher CCLKD loss profile
wrong nc / wrong yaml runs
```

## 5. 推荐入口

论文实验推荐只使用：

```bash
bash scripts/paper/run_paper_baseline.sh <sar|rgb> <n|s|m|l|x> <seed> <gpu_id>
bash scripts/paper/run_paper_ladd_probea.sh <n|s|m|l|x> <seed> <gpu_id>
bash scripts/paper/run_paper_comparison_kd.sh <fgd|ld|cmdistill> <n|s|m|l|x> <seed> <gpu_id>
bash scripts/paper/run_paper_hallucidet.sh <n|s|m> <seed> <gpu_id>
bash scripts/paper/run_paper_cclkd_online.sh <n|s> <seed> <gpu_id>
```

旧 launcher 保留为历史兼容或诊断入口，不作为论文主表直接入口。
