# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

LADD (Learnability-Aware Decomposition Distillation) — RGB-guided SAR object detection via teacher-student decomposition KD. This is a public debug/evidence package for reviewing LADD collapse issues, not a polished release. Chinese-language docs throughout.

## Repo layout

| Directory | Purpose |
|---|---|
| `shared/` | Vendored YOLO, shared TSKD scaffolding, dataset configs, CLI utilities |
| `ladd/code/` | Current LADD HBB training code (model/loss/trainer) |
| `ladd/code_versions/current_hbb/` | Snapshot of the current HBB code version with scripts |
| `ladd/scripts/` | Launch scripts for formal LADD jobs |
| `ladd/results/` | Compact LADD result summaries |
| `baseline/code/` | Single-modality baseline trainer |
| `baseline/scripts/` | Baseline launch and run scripts |
| `comparison/code/` | Formal launch scripts for FGD/LD/CCLKD/HalluciDet comparison jobs |
| `comparison/{fgd,ld,cclkd,hallucidet}/` | Method notes and compact result summaries for active comparison methods |
| `cclkd_reproduction/` | CCLKD paper-protocol online trainer, paper PDF, YOLO11n ablation plan, and compact diagnostics |
| `docs/` | Method overview, experiment plans/status, literature survey |
| `shared/configs/datasets_public/` | OGSOD-1.0 dataset YAMLs (SAR/RGB detect) |

No checkpoint weights (`.pt`/`.pth`) are included; they're in `.gitignore`.

## Python path

Training scripts in `ladd/code/` and `baseline/code/` inject `shared/` onto `sys.path` so the vendored `ultralytics` and shared modules are resolvable:

```python
sys.path.insert(0, str(REPO_ROOT / "shared"))
sys.path.insert(0, str(REPO_ROOT / "shared" / "yolo"))
```

## Training commands

All commands are run from the repo root.

### Baseline (single-modality SAR or RGB detector)

```
python baseline/code/train_ogsod_baseline.py \
  --task hbb --model yolo11n.pt \
  --data shared/configs/datasets_public/ogsod1_sar_detect.yaml \
  --imgsz 256 --epochs 800 --cos-lr --mosaic 0.0 --close-mosaic 0 \
  --seed 0 --deterministic --name my_baseline
```

Or via the launch script:
```
bash baseline/scripts/run_formal_baseline.sh sar n 0 <gpu_id>
```

### LADD HBB (manual single phase)

```
python ladd/code/train_ladd_hbb.py \
  --phase a1 --model <student.pt> \
  --data shared/configs/datasets_public/ogsod1_sar_detect.yaml \
  --teacher-data shared/configs/datasets_public/ogsod1_rgb_detect.yaml \
  --teacher-weights <rgb_teacher.pt> \
  --imgsz 256 --epochs 10
```

Phases: `a1` (teacher warmup), `a2` (decomposition pre-training), `b` (KD + detection), `c` (full fine-tuning).

### LADD automated chain

```
SAR_BASELINE=<path> RGB_TEACHER=<path> GPU_ID=0 SEED=0 \
  bash ladd/code_versions/current_hbb/scripts/ogsod_public/run_hbb_ladd_converged_chain.sh
```

Or via the launcher:
```
bash ladd/scripts/launch_formal_ladd_job.sh cap2 n 0 <gpu_id>
```

### Comparison methods

```
bash comparison/code/launch_formal_from_yolo_kd_job.sh fgd n 0 <gpu_id>
```

Formal launchers support `fgd`, `ld`, and `hallucidet` through frozen-teacher profiles. CCLKD uses the online trainer in `cclkd_reproduction/code/` for paper-protocol reproduction and `comparison/code/launch_formal_online_cclkd_job.sh` for controlled comparison after reproduction smoke.

### Formal protocol parameters (OGSOD-1.0 HBB)

`imgsz=256, epochs=800, cos_lr=True, mosaic=0.0, close_mosaic=0, default Albumentations, deterministic=True`.
Batch sizes: n/s=64, m/l=32, x=16.

## Architecture

### Shared scaffolding (`shared/`)

- `shared/yolo/ultralytics/` — vendored Ultralytics YOLO framework (DetectionModel, DetectionTrainer, v8DetectionLoss, dataset builders).
- `shared/teacher_student_decomposition_kd/` — base TSKD building blocks: `ConvNormAct`, `TeacherDecompositionBlock`, `StudentMimicResidualBlock`, `TeacherTaskHead`, `TeacherStudentDecompositionKDModel` (OBB-based), `TeacherStudentDecompositionKDTrainer`, `TeacherStudentDecompositionKDLoss`.
- `shared/train_cli_overrides.py` — exposes common Ultralytics training args (lr, augmentations, optimizer) as CLI flags.
- `shared/train_path_checks.py` — validates file paths for checkpoint arguments.

### LADD HBB method (`ladd/code/src/teacher_student_decomposition_kd_hbb/`)

Four files follow the standard pattern:

- **`model.py`** — `TeacherStudentDecompositionKDNRRLTeacherUAuxModelHBB` extends the OBB base with HBB-specific blocks:
  - `StudentResidualProjBlock` — bottleneck-projects f_s into z_s, with r_s = f_s - z_s (identity residual, 0 params).
  - `TeacherResidualDecompositionBlock` / `TeacherPrivateAwareDecompositionBlock` — decomposes teacher features into learnable z_t and unlearnable u_t.
  - `WeakTaskDecoder` / `ResidualForegroundHead` / `StudentSarAuxHead` — lightweight auxiliary heads for self-supervised signals on r_s.
  - Optional `r_obb_head` (independent OBB head on r_s) and `r_sar_head` (SAR high-frequency prediction head).
- **`loss.py`** — `TeacherStudentDecompositionKDNRRLTeacherUAuxLossHBB`: KD loss (MSE/contrastive/hybrid), NRRL (Normalized Reachability Ranking Loss), decorrelation loss, residual energy loss, teacher private auxiliary loss, mask losses.
- **`trainer.py`** — `ManualPhaseTeacherStudentDecompositionKDNRRLTeacherUAuxTrainer`: multi-phase trainer with phase-aware freezing, `PhaseMinEarlyStopping`, paired SAR/RGB dataset loading.
- **`base_hbb.py`** — `TeacherStudentDecompositionKDModelHBB`: HBB DetectionModel subclass that hooks feature decomposition into `_predict_once`.

### Core decomposition scheme

- **Teacher (RGB)**: f_t → z_t (learnable, transferable) + u_t (unlearnable, RGB-private). Two modes: `decomposed` (explicit learnable/unlearnable branches + reconstruction layer) and `residual` (bottleneck projection, u_t = f_t - z_t, zero-param reconstruction).
- **Student (SAR)**: f_s → z_s (bottleneck-projected common space) + r_s (residual = f_s - z_s, carries SAR-private structure).
- **KD**: z_s matches z_t via MSE, contrastive, or hybrid loss.
- **NRRL**: ranks teacher-student feature distances; rewards alignment with a softplus/hinge loss.
- **Residual aux**: weak self-supervised losses on r_s (foreground energy, object energy) to prevent collapse.

### Baseline trainer (`baseline/code/`)

`train_ogsod_baseline.py` — single-modality HBB/OBB training using `UnifiedAugOBBTrainer` with the formal augmentation policy. `--task hbb` maps to DetectionTrainer.

### Comparison methods (`comparison/code/`)

Launch scripts that invoke the LADD trainer with `--comparison-kd-profile` set to one of the four controlled methods. Formal launchers reject archived profiles.

## Known issues (current debugging focus)

1. **B-stage collapse**: LADD B phase shows late-training degradation on some seeds/machines — likely BN running stats pollution. Fix attempt: `FREEZE_BN_STATS=1`.
2. **CCLKD reproduction gap**: Old CCLKD runs before 2026-06-06 used an incorrect CCL formulation and are diagnostic only. Current CCL uses neck features and per-sampled negative similarity.
3. **4090D vs 90 divergence**: YOLO11s LADD results significantly lower on 4090D than on server 90 — protocol/implementation divergence under investigation.
