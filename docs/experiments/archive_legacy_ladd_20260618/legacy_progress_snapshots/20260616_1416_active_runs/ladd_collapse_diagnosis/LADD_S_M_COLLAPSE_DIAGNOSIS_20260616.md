# LADD s/m Collapse Diagnosis

Snapshot time: 2026-06-16 14:30 +08.

## Scope

This note diagnoses the active no-mosaic LADD A1->B A2-core runs:

- `n LADD`: AutoDL, YOLO11n, B800, currently healthy.
- `s LADD`: AutoDL, YOLO11s, B800, collapsed/lagging.
- `m LADD`: server 90 GPU5, YOLO11m, B800, early but clearly below same-epoch baseline.

Reference plots and tables:

- `../figures/active_runs_by_model_with_baselines.png`
- `../figures/active_runs_by_model_with_baselines_early180.png`
- `ladd_collapse_milestone_table.md`
- `diagnostic_figures/ladd_collapse_diagnostics.png`

## Key Evidence

The main failure signal is not ordinary slow convergence. It is loss-scale domination from LADD KD/private-feature terms.

| run | epoch | AP | SAR baseline same-epoch AP | KD loss | task loss | KD/task | u_aux_fg_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| n B | 384 | 0.52020 | n/a here | 0.01479 | 0.40472 | 0.0365 | 0.57625 |
| s B | 96 | 0.19192 | much higher | 84.113 | 0.75629 | 111 | 35.067 |
| m B | 40 | 0.19634 | much higher | 27.313 | 0.78405 | 34.8 | 113.54 |

A1 already shows the problem:

| run | A1 epoch | AP | task loss | nan flag | u_aux_fg_mean | u_aux_bg_mean |
|---|---:|---:|---:|---:|---:|---:|
| n A1 | 10 | 0.00023 | 1.09 | 0 | 0.64889 | 0.30398 |
| s A1 | 10 | 0.00011 | NaN | 1 | 39.994 | 36.648 |
| m A1 | 10 | 0.00011 | 1.3293 | 0 | 105.38 | 102.93 |

## Interpretation

1. The s/m failures are already seeded in A1.
   A1 AP is essentially zero for all models, so selecting `best.pt` by detector metric is not meaningful. For s, A1 has a nonfinite diagnostic flag by epoch 10. For m, A1 has no NaN but the private feature energy is already above 100.

2. B then applies full KD from epoch 1.
   The active runs use no KD warmup or decay. With abnormal A1 feature scale, B starts with KD/task ratios of roughly 4090x for s and 467x for m at epoch 1.

3. The likely implementation issue is unbounded feature scale in the LADD decomposition path.
   `_teacher_private_auxiliary_loss()` uses a margin energy objective:
   `relu(margin + ctx_mean - pos_mean)`.
   Once `pos_mean` exceeds `ctx_mean + margin`, the loss is zero or small even if both absolute energies explode. That allows s/m private feature magnitudes to grow to tens or hundreds, while n stays near 0.6.

4. Raw MSE KD is scale-sensitive.
   Current B uses `kd_calibration_mode=none` and raw MSE-like KD on foreground tokens. If the A1 decomposition creates high-scale targets, KD becomes huge and dominates detection.

5. This is not explained by the no-mosaic protocol alone.
   SAR baselines and CMDistill under the same no-mosaic protocol train normally. The collapse is specific to the LADD A1->B decomposition/KD path for s/m.

## Current Decision

The active s and m LADD runs should be treated as diagnostic, not claim-usable. They should not be used as evidence that LADD fails at s/m capacity under the protocol; they show that the current A1/B loss scaling is not capacity-stable.

## Recommended Next Checks

Before launching more long runs:

1. Run short B probes from the same A1 checkpoints:
   - `ALPHA_KD=0` or `LADD_B_DET_ONLY=1`
   - `ALPHA_KD=0.01`
   - KD warmup over 50-100 epochs

2. Run A1 probes for s/m with private-energy disabled or bounded:
   - `TEACHER_PRIVATE_AUX_MODE=none`
   - or add an explicit feature-energy regularizer/clamp for `u_t`

3. Prefer a scale-normalized KD probe:
   - `KD_CALIBRATION_MODE=norm_affine`
   - or normalize z_s/z_t before MSE for LADD KD

4. Avoid selecting A1 `best.pt` by detector AP.
   A1 detector AP is near zero. Use `last.pt`, a decomposition-stability metric, or a fixed short A1 epoch if A1 is retained.

## Do Not Conclude Yet

Do not use the current s/m LADD curves as final method comparison. They are valid diagnostics for a scaling/initialization issue, but not a valid final LADD capacity result.
