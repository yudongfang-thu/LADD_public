# CCLKD YOLOv5x running status

Update time: 2026-06-12 CST.

## Current pre-flight status

This local Codex session only completed code/status checks for the YOLOv5x CCLKD reproduction path. No new training job was launched from this machine.

## P0 implementation audit

| Item | Status | Evidence |
|---|---|---|
| Masked distribution excludes masked-out entries from softmax denominator | done | `cclkd_yolov5_loss.py::_masked_distribution` selects `values[mask.bool()]` before softmax |
| Spatial KL uses selected entries only and detaches teacher | done | `spatial_distribution_kl` uses `teacher_box_probs.detach()[mask]` |
| Feature KL is per-candidate feature-vector KL | done | `feature_kl` applies `log_softmax(..., dim=-1)` and `batchmean` |
| Explicit ATKD/CCL weights | done | `cclkd_paper_loss(..., atkd_weight, ccl_weight)` and trainer defaults by mode |
| raw_proxy/current_full blocked by default | done | Python and launcher require `--allow-raw-proxy` / `ALLOW_RAW_PROXY=1` |
| Diagnostics include weighted KD fields | done | `DIAG_FIELDS` includes `kd_scale` and `weighted_kd_to_student_det_ratio` |
| Smoke checker exists | done | `tools/check_yolov5_cclkd_smoke.py` |

## Static validation

Passed locally:

```bash
python3 -m py_compile \
  cclkd_reproduction/yolov5_sanity/code/cclkd_yolov5_loss.py \
  cclkd_reproduction/yolov5_sanity/code/train_yolov5_cclkd_full.py \
  cclkd_reproduction/yolov5_sanity/tools/check_yolov5_cclkd_smoke.py

bash -n cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_cclkd_full.sh
```

Dry-run command generation passed:

```bash
DRY_RUN=1 CCLKD_YOLOV5_MODE=paper_full SMOKE_EPOCHS=1 MAX_TRAIN_BATCHES=2 SKIP_VAL=1 \
  bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_cclkd_full.sh \
  64 0 0 smoke_paper_full
```

The generated command includes `--mode paper_full`, `--max-train-batches 2`, `--skip-val`, `--kd-weight`, and `--kd-warmup-epochs`.

raw proxy guard check passed:

```text
CCLKD_YOLOV5_MODE=current_full ... -> exit_code=2
```

## Local launch blocker

No real smoke or long experiment was launched locally because this machine is missing the runtime artifacts needed for YOLOv5x training:

| Requirement | Local status |
|---|---|
| `external/yolov5/.git` | missing |
| `external/yolov5/yolov5x.pt` | missing |
| `configs/datasets/ogsod_hbb_sar.yaml` | missing |
| `configs/datasets/ogsod_hbb_rgb.yaml` | missing |
| `nvidia-smi` / CUDA GPU | unavailable |

There is also no existing `cclkd_yolov5_diagnostics.csv` under `cclkd_reproduction/yolov5_sanity/results/`, so the smoke checker has no completed paper-mode run to validate yet.

## Running jobs

| run_name | mode | batch | seed | gpu | pid | epoch | latest AP50 | latest AP | status |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| none launched in this session | - | - | - | - | - | - | - | - | local pre-flight only |

## Recommendation

Do not start 80 epoch or 400 epoch runs until a real `paper_full` smoke completes and passes:

```bash
python cclkd_reproduction/yolov5_sanity/tools/check_yolov5_cclkd_smoke.py \
  --run-dir cclkd_reproduction/yolov5_sanity/results/runs/yolov5x_paper_full_b64_s0_smoke_paper_full_1ep2b
```

Once the server has YOLOv5, weights, dataset YAMLs, and GPU available, run the 1 epoch / 2 batch `paper_full` smoke first. If it passes, launch the 80 epoch mechanism checks before any 400 epoch reproduction.
