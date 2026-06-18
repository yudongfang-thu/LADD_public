#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_ROOT = REPO_ROOT / "shared"
YOLO_ROOT = SHARED_ROOT / "yolo"
SRC_ROOT = REPO_ROOT / "ladd" / "code" / "src"
for root in (SHARED_ROOT, YOLO_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from train_cli_overrides import add_common_detector_train_overrides, collect_common_detector_train_overrides  # noqa: E402
from train_path_checks import require_existing_file  # noqa: E402
from teacher_student_decomposition_kd_hbb import (  # noqa: E402
    ManualPhaseTeacherStudentDecompositionKDNRRLTeacherUAuxTrainer,
)
from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual single-phase LADD trainer for HBB/detect OGSOD.")
    parser.add_argument("--phase", choices=("a1", "a2", "b", "c", "b1", "b2"), required=True)
    parser.add_argument("--model", required=True, help="Student model weights or YAML.")
    parser.add_argument("--data", type=Path, required=True, help="Student SAR dataset YAML.")
    parser.add_argument("--teacher-data", type=Path, required=True, help="RGB teacher dataset YAML.")
    parser.add_argument("--teacher-weights", required=True, help="Frozen RGB teacher checkpoint.")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cache", default=False)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs_public" / "ogsod" / "hbb" / "ladd")
    parser.add_argument("--name", default="ogsod_hbb_ladd_phase")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from an Ultralytics last.pt checkpoint.")
    parser.add_argument("--validate-before-train", action="store_true")

    parser.add_argument("--phase-detect-mode", choices=("auto", "raw", "fused", "mimic", "recon"), default="raw")
    parser.add_argument("--det-loss-scale", type=float, default=None)
    parser.add_argument("--phase-min-epochs", type=int, default=None)
    parser.add_argument("--phase-stop-metric", choices=("default", "map", "a1_loss", "a1_task_reach"), default="default")
    parser.add_argument("--reach-target-mode", choices=("detach", "coupled"), default="detach")
    parser.add_argument("--kd-target-mode", choices=("detach", "coupled"), default="detach")
    parser.add_argument("--strict-batch-size", action="store_true")
    parser.add_argument(
        "--freeze-bn-stats",
        action="store_true",
        help="Keep BatchNorm layers in eval mode during training so running_mean/running_var are not updated.",
    )
    parser.add_argument(
        "--freeze-bn-after-epoch",
        type=int,
        default=-1,
        help=(
            "Delayed BatchNorm stats freeze for B-phase diagnostics. -1 disables delayed freeze; "
            "N keeps BN stats normal for zero-based epochs < N and freezes from epoch N onward."
        ),
    )
    parser.add_argument("--ladd-diag-log-bn", type=int, default=1, help="Write BN stats to ladd_diagnostics.csv.")
    parser.add_argument(
        "--ladd-diag-log-grad",
        type=int,
        default=0,
        help="Also record gradient norms in ladd_diagnostics.csv. Disabled by default to avoid overhead.",
    )
    parser.add_argument(
        "--ladd-grad-clip-norm",
        type=float,
        default=0.0,
        help="Explicit gradient clipping override. <=0 preserves Ultralytics default clipping.",
    )
    parser.add_argument(
        "--ladd-assert-phase-freeze",
        action="store_true",
        help="Assert B-phase frozen LADD modules remain requires_grad=False.",
    )
    parser.add_argument(
        "--ladd-diag-log-every",
        type=int,
        default=1,
        help="Record LADD diagnostics every N epochs. Defaults to every epoch.",
    )
    parser.add_argument(
        "--ladd-kd-decay-mode",
        choices=("none", "linear", "cosine", "step", "warmup", "warmup_linear", "linear_warmup", "ramp_linear"),
        default="none",
    )
    parser.add_argument("--ladd-kd-decay-start-epoch", type=int, default=-1)
    parser.add_argument("--ladd-kd-decay-end-epoch", type=int, default=-1)
    parser.add_argument("--ladd-kd-final-mult", type=float, default=1.0)
    parser.add_argument("--ladd-kd-stop-after-epoch", type=int, default=-1)
    parser.add_argument(
        "--ladd-b-loss-warmup-mode",
        choices=("none", "linear"),
        default="none",
        help="B-stage warmup for core LADD non-detection losses. Independent from ladd_kd_decay_mode.",
    )
    parser.add_argument("--ladd-b-loss-warmup-start-epoch", type=int, default=-1)
    parser.add_argument("--ladd-b-loss-warmup-end-epoch", type=int, default=-1)
    parser.add_argument("--ladd-b-loss-warmup-final-mult", type=float, default=1.0)
    parser.add_argument(
        "--ladd-b-loss-warmup-scope",
        choices=("core", "extended"),
        default="core",
        help=(
            "core scales alpha_kd and alpha_s_rec."
        ),
    )
    parser.add_argument(
        "--ladd-b-a2-core",
        action="store_true",
        help=(
            "In B phase, also enable A2 core teacher decomposition/reach losses and train the "
            "teacher decomposition, decoder, reachability, and teacher task-head modules."
        ),
    )
    parser.add_argument(
        "--ladd-b-frozen-reach-probe",
        action="store_true",
        help=(
            "In B phase with --ladd-b-a2-core, keep student_reachability frozen and detach q_s "
            "inside reach loss so reach updates teacher-side decomposition only."
        ),
    )
    parser.add_argument(
        "--ladd-b-det-only",
        action="store_true",
        help="In B phase, keep trainability unchanged but disable all non-detection LADD losses.",
    )
    parser.add_argument(
        "--ladd-a2-det-only",
        action="store_true",
        help="In A2 phase, keep trainability unchanged but disable all non-detection LADD losses.",
    )

    parser.add_argument("--lambda-rec", type=float, default=0.1)
    parser.add_argument("--lambda-taskL", type=float, default=1.0)
    parser.add_argument("--task-loss-fg-only", action="store_true")
    parser.add_argument("--alpha-kd", type=float, default=1.0)
    parser.add_argument("--alpha-s-rec", type=float, default=0.1)
    parser.add_argument("--lambda-reach", type=float, default=1.0)
    parser.add_argument("--lambda-match-inner", type=float, default=1.0)
    parser.add_argument("--lambda-rank-inner", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--reach-rank-mode", choices=("softplus", "hinge"), default="softplus")
    parser.add_argument("--reach-input-mode", choices=("adapter", "raw"), default="adapter")
    parser.add_argument("--rank-d-neg-cap", type=float, default=4.0)
    parser.add_argument("--use-fg-mask-for-reach", action="store_true")
    parser.add_argument("--use-fg-mask-for-rec", action="store_true")
    parser.add_argument("--use-mask", action="store_true")

    parser.add_argument("--student-detect-mode", choices=("fused", "mimic", "raw", "recon"), default="raw")
    parser.add_argument("--student-branch-mode", choices=("split", "raw", "single_proj"), default="split")
    parser.add_argument("--teacher-feature-mode", choices=("decomposed", "raw", "projected_raw"), default="decomposed")
    parser.add_argument("--kd-mechanism", choices=("mse", "contrastive", "hybrid"), default="mse")
    parser.add_argument("--contrastive-temperature", type=float, default=0.20)

    parser.add_argument("--kd-weight-mode", choices=("none", "teacher_task_conf", "reachability_gap"), default="none")
    parser.add_argument("--kd-weight-power", type=float, default=1.0)
    parser.add_argument("--kd-aggregation-mode", choices=("token", "score_weighted", "topk"), default="token")
    parser.add_argument("--kd-topk-ratio", type=float, default=0.5)
    parser.add_argument("--kd-calibration-mode", choices=("none", "affine", "norm_affine"), default="none")

    parser.add_argument("--unlearnable-hidden-ratio", type=float, default=1.0)

    parser.add_argument("--teacher-target-mode", choices=("static", "ema"), default="static")
    parser.add_argument("--teacher-ema-momentum", type=float, default=0.99)

    parser.add_argument(
        "--comparison-kd-profile",
        choices=("none", "fgd", "ld", "cmdistill", "cclkd"),
        default="none",
        help=(
            "Portable comparison KD profile for OGSOD HBB. "
            "fgd, ld, and cmdistill are generic detector KD transfers; "
            "cclkd is kept for existing online-comparison compatibility. "
            "HalluciDet now uses the standalone comparison/hallucidet trainer."
        ),
    )
    parser.add_argument("--profile-kd-weight", type=float, default=1.0)
    parser.add_argument(
        "--profile-kd-replace-base",
        action="store_true",
        help="Use the selected comparison profile instead of the base feature-KD term inside kd_loss.",
    )
    parser.add_argument("--fgd-alpha", type=float, default=0.0001)
    parser.add_argument("--fgd-beta", type=float, default=0.00005)
    parser.add_argument("--fgd-gamma", type=float, default=0.001)
    parser.add_argument("--fgd-lambda", type=float, default=0.0)
    parser.add_argument("--fgd-relation-weight", type=float, default=None, help="Deprecated alias for --fgd-lambda.")
    parser.add_argument("--fgd-normalization-mode", choices=("original", "channel_mean"), default="original")
    parser.add_argument("--fgd-temperature", type=float, default=0.5)
    parser.add_argument("--fgd-mask-mode", choices=("gt_box", "assigner"), default="gt_box")
    parser.add_argument("--fgd-bg-norm", type=int, default=1)
    parser.add_argument("--ld-temperature", type=float, default=10.0)
    parser.add_argument("--ld-use-vlr", type=int, default=1)
    parser.add_argument("--ld-quality-power", type=float, default=1.0)
    parser.add_argument("--ld-min-vlr-weight", type=float, default=0.0)
    parser.add_argument("--ld-vlr-topk", type=int, default=0)
    parser.add_argument("--ld-vlr-weight", type=float, default=0.25)
    parser.add_argument("--ld-main-weight", type=float, default=0.25)
    parser.add_argument("--ld-allow-empty-vlr", type=int, default=1)
    parser.add_argument("--cmdistill-feature-weight", type=float, default=1.0)
    parser.add_argument("--cmdistill-relation-weight", type=float, default=1.0)
    parser.add_argument("--cmdistill-logit-weight", type=float, default=1.0)
    parser.add_argument("--cmdistill-temperature", type=float, default=4.0)
    parser.add_argument("--cmdistill-max-tokens", type=int, default=512)
    parser.add_argument("--cmdistill-min-confidence", type=float, default=0.05)
    parser.add_argument("--cclkd-base-temperature", type=float, default=2.0)
    parser.add_argument("--cclkd-contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--cclkd-feat-weight", type=float, default=1.0)
    parser.add_argument("--cclkd-logit-weight", type=float, default=1.0)
    parser.add_argument("--cclkd-contrast-weight", type=float, default=0.5)
    parser.add_argument("--cclkd-bg-weight", type=float, default=0.1)
    parser.add_argument("--cclkd-min-confidence", type=float, default=0.1)
    parser.add_argument("--cclkd-max-tokens", type=int, default=512)
    parser.add_argument("--cclkd-temperature-min", type=float, default=0.5)
    parser.add_argument("--cclkd-temperature-max", type=float, default=5.0)
    parser.add_argument("--cclkd-entropy-scale", type=float, default=5.0)

    parser.add_argument("--c-weak-nrrl-scale", type=float, default=0.0)
    parser.add_argument("--c-weak-nrrl-detach-student", action="store_true")
    parser.add_argument("--reach-c-mode", choices=("none", "rank", "weight"), default="none")
    parser.add_argument("--lambda-reach-c", type=float, default=0.0)
    parser.add_argument("--b-reset-student-from-scratch", action="store_true")
    parser.add_argument(
        "--b-detector-source",
        type=Path,
        default=None,
        help="For B-phase split-load diagnostics, load detector weights from this checkpoint after model init.",
    )
    parser.add_argument(
        "--b-decomp-source",
        type=Path,
        default=None,
        help="For B-phase split-load diagnostics, load LADD decomposition/reach modules from this checkpoint.",
    )
    parser.add_argument(
        "--b-split-load-strict",
        action="store_true",
        help="Require strict split-load module matches for detector/decomposition checkpoints.",
    )
    parser.add_argument(
        "--b-load-student-split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In B split-load diagnostics, also load student_split from the decomposition checkpoint.",
    )
    parser.add_argument(
        "--b-load-student-reachability",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In B split-load diagnostics, load student_reachability from the decomposition checkpoint.",
    )
    parser.add_argument("--force-student-rec", action="store_true")

    add_common_detector_train_overrides(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher_weights = require_existing_file(args.teacher_weights, "--teacher-weights")
    model = YOLO(args.model)
    phase_detect_mode = None if args.phase_detect_mode == "auto" else args.phase_detect_mode
    train_kwargs = dict(
        trainer=ManualPhaseTeacherStudentDecompositionKDNRRLTeacherUAuxTrainer,
        phase=args.phase,
        phase_detect_mode=phase_detect_mode,
        det_loss_scale=args.det_loss_scale,
        phase_min_epochs=args.phase_min_epochs,
        phase_stop_metric=args.phase_stop_metric,
        strict_batch_size=args.strict_batch_size,
        freeze_bn_stats=args.freeze_bn_stats,
        freeze_bn_after_epoch=args.freeze_bn_after_epoch,
        ladd_diag_log_bn=args.ladd_diag_log_bn,
        ladd_diag_log_grad=args.ladd_diag_log_grad,
        ladd_grad_clip_norm=args.ladd_grad_clip_norm,
        ladd_assert_phase_freeze=args.ladd_assert_phase_freeze,
        ladd_diag_log_every=args.ladd_diag_log_every,
        ladd_kd_decay_mode=args.ladd_kd_decay_mode,
        ladd_kd_decay_start_epoch=args.ladd_kd_decay_start_epoch,
        ladd_kd_decay_end_epoch=args.ladd_kd_decay_end_epoch,
        ladd_kd_final_mult=args.ladd_kd_final_mult,
        ladd_kd_stop_after_epoch=args.ladd_kd_stop_after_epoch,
        ladd_b_loss_warmup_mode=args.ladd_b_loss_warmup_mode,
        ladd_b_loss_warmup_start_epoch=args.ladd_b_loss_warmup_start_epoch,
        ladd_b_loss_warmup_end_epoch=args.ladd_b_loss_warmup_end_epoch,
        ladd_b_loss_warmup_final_mult=args.ladd_b_loss_warmup_final_mult,
        ladd_b_loss_warmup_scope=args.ladd_b_loss_warmup_scope,
        ladd_b_a2_core=int(bool(args.ladd_b_a2_core)),
        ladd_b_frozen_reach_probe=int(bool(args.ladd_b_frozen_reach_probe)),
        ladd_b_det_only=int(bool(args.ladd_b_det_only)),
        ladd_a2_det_only=int(bool(args.ladd_a2_det_only)),
        data=str(args.data.resolve()),
        teacher_data=str(args.teacher_data.resolve()),
        teacher_weights=teacher_weights,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        cache=args.cache,
        patience=args.patience,
        fraction=args.fraction,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        resume=str(require_existing_file(args.resume, "--resume")) if args.resume is not None else None,
        validate_before_train=args.validate_before_train,
        lambda_rec=args.lambda_rec,
        lambda_taskL=args.lambda_taskL,
        task_loss_fg_only=args.task_loss_fg_only,
        alpha_kd=args.alpha_kd,
        alpha_s_rec=args.alpha_s_rec,
        lambda_reach=args.lambda_reach,
        lambda_match_inner=args.lambda_match_inner,
        lambda_rank_inner=args.lambda_rank_inner,
        delta=args.delta,
        use_soft_rank=(args.reach_rank_mode == "softplus"),
        rank_d_neg_cap=args.rank_d_neg_cap,
        reach_input_mode=args.reach_input_mode,
        use_fg_mask_for_reach=args.use_fg_mask_for_reach,
        use_fg_mask_for_rec=args.use_fg_mask_for_rec,
        reach_target_mode=args.reach_target_mode,
        kd_target_mode=args.kd_target_mode,
        use_mask=args.use_mask,
        student_detect_mode=args.student_detect_mode,
        student_branch_mode=args.student_branch_mode,
        teacher_feature_mode=args.teacher_feature_mode,
        kd_mechanism=args.kd_mechanism,
        contrastive_temperature=args.contrastive_temperature,
        kd_weight_mode=args.kd_weight_mode,
        kd_weight_power=args.kd_weight_power,
        kd_aggregation_mode=args.kd_aggregation_mode,
        kd_topk_ratio=args.kd_topk_ratio,
        kd_calibration_mode=args.kd_calibration_mode,
        unlearnable_hidden_ratio=args.unlearnable_hidden_ratio,
        teacher_target_mode=args.teacher_target_mode,
        teacher_ema_momentum=args.teacher_ema_momentum,
        comparison_kd_profile=args.comparison_kd_profile,
        profile_kd_weight=args.profile_kd_weight,
        profile_kd_replace_base=int(bool(args.profile_kd_replace_base)),
        fgd_alpha=args.fgd_alpha,
        fgd_beta=args.fgd_beta,
        fgd_gamma=args.fgd_gamma,
        fgd_lambda=args.fgd_lambda if args.fgd_relation_weight is None else args.fgd_relation_weight,
        fgd_normalization_mode=args.fgd_normalization_mode,
        fgd_temperature=args.fgd_temperature,
        fgd_mask_mode=args.fgd_mask_mode,
        fgd_bg_norm=int(bool(args.fgd_bg_norm)),
        ld_temperature=args.ld_temperature,
        ld_use_vlr=int(bool(args.ld_use_vlr)),
        ld_quality_power=args.ld_quality_power,
        ld_min_vlr_weight=args.ld_min_vlr_weight,
        ld_vlr_topk=args.ld_vlr_topk,
        ld_vlr_weight=args.ld_vlr_weight,
        ld_main_weight=args.ld_main_weight,
        ld_allow_empty_vlr=int(bool(args.ld_allow_empty_vlr)),
        cmdistill_feature_weight=args.cmdistill_feature_weight,
        cmdistill_relation_weight=args.cmdistill_relation_weight,
        cmdistill_logit_weight=args.cmdistill_logit_weight,
        cmdistill_temperature=args.cmdistill_temperature,
        cmdistill_max_tokens=args.cmdistill_max_tokens,
        cmdistill_min_confidence=args.cmdistill_min_confidence,
        cclkd_base_temperature=args.cclkd_base_temperature,
        cclkd_contrastive_temperature=args.cclkd_contrastive_temperature,
        cclkd_feat_weight=args.cclkd_feat_weight,
        cclkd_logit_weight=args.cclkd_logit_weight,
        cclkd_contrast_weight=args.cclkd_contrast_weight,
        cclkd_bg_weight=args.cclkd_bg_weight,
        cclkd_min_confidence=args.cclkd_min_confidence,
        cclkd_max_tokens=args.cclkd_max_tokens,
        cclkd_temperature_min=args.cclkd_temperature_min,
        cclkd_temperature_max=args.cclkd_temperature_max,
        cclkd_entropy_scale=args.cclkd_entropy_scale,
        c_weak_nrrl_scale=args.c_weak_nrrl_scale,
        c_weak_nrrl_detach_student=args.c_weak_nrrl_detach_student,
        reach_c_mode=args.reach_c_mode,
        lambda_reach_c=args.lambda_reach_c,
        b_reset_student_from_scratch=args.b_reset_student_from_scratch,
        b_detector_source=(
            str(require_existing_file(args.b_detector_source, "--b-detector-source"))
            if args.b_detector_source is not None
            else ""
        ),
        b_decomp_source=(
            str(require_existing_file(args.b_decomp_source, "--b-decomp-source"))
            if args.b_decomp_source is not None
            else ""
        ),
        b_split_load_strict=int(bool(args.b_split_load_strict)),
        b_load_student_split=int(bool(args.b_load_student_split)),
        b_load_student_reachability=int(bool(args.b_load_student_reachability)),
        force_student_rec=int(bool(args.force_student_rec)),
    )
    train_kwargs.update(collect_common_detector_train_overrides(args))
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
