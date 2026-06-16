from __future__ import annotations

import csv
from copy import copy, deepcopy
import math
import json
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, RANK
from ultralytics.utils.checks import check_file
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import unwrap_model

from d2ad_obb.aug_policy import apply_unified_paired_aug_policy
from d2ad_obb.paired_dataset import PairedOBBDataset

from .loss import TeacherStudentDecompositionKDNRRLTeacherUAuxLossHBB
from .model import TeacherStudentDecompositionKDNRRLTeacherUAuxModelHBB
from .schedule import (
    apply_det_only_phase_scales,
    compute_effective_ladd_weights,
)


class PhaseMinEarlyStopping:
    """Early stopping with a hard minimum epoch budget."""

    def __init__(self, patience: int = 30, min_epochs: int = 0):
        self.patience = patience or float("inf")
        self.min_epochs = max(int(min_epochs), 0)
        self.best_fitness = None
        self.best_epoch = 0
        self.possible_stop = False

    def __call__(self, epoch: int, fitness: float | None) -> bool:
        if fitness is None:
            return False
        if self.best_fitness is None or fitness > self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness
        if epoch < self.min_epochs:
            self.possible_stop = False
            return False
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= (self.patience - 1)
        if delta >= self.patience:
            LOGGER.info(
                f"PhaseMinEarlyStopping: no improvement in last {self.patience} epochs after minimum epoch "
                f"{self.min_epochs}. Best epoch was {self.best_epoch}."
            )
            return True
        return False


class TeacherStudentDecompositionKDNRRLTeacherUAuxTrainer(DetectionTrainer):
    """TSKD trainer with current residual-energy mainline plus teacher-side u_t controls."""

    def check_resume(self, overrides):
        """Resume from Ultralytics checkpoints while ignoring saved HBB custom trainer keys."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())
                ckpt_args = dict(load_checkpoint(last)[0].args)
                if not isinstance(ckpt_args.get("data"), dict) and not Path(ckpt_args.get("data", "")).exists():
                    ckpt_args["data"] = self.args.data

                dropped = sorted(k for k in ckpt_args if k not in DEFAULT_CFG_DICT)
                if dropped:
                    LOGGER.info(
                        "HBB resume: ignoring checkpoint-only custom args not in Ultralytics DEFAULT_CFG: "
                        + ", ".join(dropped)
                    )
                ckpt_args = {k: v for k, v in ckpt_args.items() if k in DEFAULT_CFG_DICT}
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                    "augmentations",
                    "save_period",
                    "workers",
                    "cache",
                    "patience",
                    "project",
                    "name",
                    "exist_ok",
                    "time",
                    "freeze",
                    "val",
                    "plots",
                ):
                    if k in overrides:
                        setattr(self.args, k, overrides[k])
            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found or is not compatible. "
                    "Please pass a valid checkpoint to --resume, e.g. path/to/last.pt"
                ) from e
        self.resume = resume

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        overrides = {} if overrides is None else dict(overrides)
        self.nrrl_cfg = {
            "lambda_reach": float(overrides.pop("lambda_reach", 1.0)),
            "lambda_match_inner": float(overrides.pop("lambda_match_inner", 1.0)),
            "lambda_rank_inner": float(overrides.pop("lambda_rank_inner", 1.0)),
            "delta": float(overrides.pop("delta", 0.2)),
            "use_soft_rank": overrides.pop("use_soft_rank", True),
            "use_fg_mask_for_reach": overrides.pop("use_fg_mask_for_reach", False),
            "use_fg_mask_for_rec": overrides.pop("use_fg_mask_for_rec", False),
            "normalize_reach": overrides.pop("normalize_reach", True),
            "rank_d_neg_cap": float(overrides.pop("rank_d_neg_cap", 4.0)),
            "lambda_anti_collapse": float(overrides.pop("lambda_anti_collapse", 0.0)),
            "anti_collapse_floor": float(overrides.pop("anti_collapse_floor", 0.0)),
            "reach_target_mode": overrides.pop("reach_target_mode", "detach"),
            "residual_aux_mode": overrides.pop("residual_aux_mode", "energy"),
            "lambda_residual_aux": float(overrides.pop("lambda_residual_aux", 0.25)),
            "energy_bg_weight": float(overrides.pop("energy_bg_weight", 1.0)),
            "energy_margin": float(overrides.pop("energy_margin", 0.2)),
            "mask_train_mode": overrides.pop("mask_train_mode", "none"),
            "mask_target_mode": overrides.pop("mask_target_mode", "none"),
            "lambda_mask_target": float(overrides.pop("lambda_mask_target", 0.0)),
            "reach_input_mode": overrides.pop("reach_input_mode", "adapter"),
            "kd_weight_mode": overrides.pop("kd_weight_mode", "none"),
            "kd_weight_power": float(overrides.pop("kd_weight_power", 1.0)),
            "kd_aggregation_mode": overrides.pop("kd_aggregation_mode", "token"),
            "kd_topk_ratio": float(overrides.pop("kd_topk_ratio", 0.5)),
            "instance_energy_radius": int(overrides.pop("instance_energy_radius", 1)),
        }
        self.teacher_u_cfg = {
            "teacher_private_aux_mode": overrides.pop("teacher_private_aux_mode", "none"),
            "lambda_teacher_private_aux": float(overrides.pop("lambda_teacher_private_aux", 0.25)),
            "teacher_private_bg_weight": float(overrides.pop("teacher_private_bg_weight", 1.0)),
            "teacher_private_margin": float(overrides.pop("teacher_private_margin", 0.2)),
            "unlearnable_hidden_ratio": float(overrides.pop("unlearnable_hidden_ratio", 1.0)),
        }
        self.explore_cfg = {
            "kd_calibration_mode": overrides.pop("kd_calibration_mode", "none"),
            "teacher_target_mode": overrides.pop("teacher_target_mode", "static"),
            "teacher_ema_momentum": float(overrides.pop("teacher_ema_momentum", 0.99)),
            "student_branch_mode": overrides.pop("student_branch_mode", "split"),
            "teacher_feature_mode": overrides.pop("teacher_feature_mode", "decomposed"),
            "kd_mechanism": overrides.pop("kd_mechanism", "mse"),
            "contrastive_temperature": float(overrides.pop("contrastive_temperature", 0.20)),
            "student_z_bottleneck_ratio": float(overrides.pop("student_z_bottleneck_ratio", 0.25)),
            "teacher_branch_mode": overrides.pop("teacher_branch_mode", "decomposed"),
            "teacher_z_bottleneck_ratio": float(overrides.pop("teacher_z_bottleneck_ratio", 0.25)),
            "force_student_rec": int(overrides.pop("force_student_rec", 0)) > 0,
        }
        self.v2_cfg = {
            "lambda_recon_task": float(overrides.pop("lambda_recon_task", 0.0)),
            "lambda_rs_complementary": float(overrides.pop("lambda_rs_complementary", 0.0)),
            "enable_recon_task": int(overrides.pop("enable_recon_task", 0)) > 0,
            "lambda_r_obb": float(overrides.pop("lambda_r_obb", 0.0)),
            "enable_r_obb_head": int(overrides.pop("enable_r_obb_head", 0)) > 0,
            # Path A (student reach mirror) and Path D (SAR self-supervised on
            # r_s) — pure opt-in; defaults preserve existing experiments.
            "lambda_s_repel": float(overrides.pop("lambda_s_repel", 0.0)),
            "s_repel_tau": float(overrides.pop("s_repel_tau", 0.0)),
            "s_repel_fg_only": int(overrides.pop("s_repel_fg_only", 0)) > 0,
            "lambda_path_b": float(overrides.pop("lambda_path_b", 0.0)),
            "path_b_margin": float(overrides.pop("path_b_margin", 0.2)),
            "path_b_fg_only": int(overrides.pop("path_b_fg_only", 0)) > 0,
            "lambda_r_sar": float(overrides.pop("lambda_r_sar", 0.0)),
            "r_sar_target": str(overrides.pop("r_sar_target", "sobel")),
            "r_sar_infonce_temperature": float(overrides.pop("r_sar_infonce_temperature", 0.1)),
            "r_sar_infonce_n_samples": int(overrides.pop("r_sar_infonce_n_samples", 256)),
            "r_sar_infonce_radius": int(overrides.pop("r_sar_infonce_radius", 1)),
            "r_sar_speckle_sigma": float(overrides.pop("r_sar_speckle_sigma", 0.3)),
            "enable_r_sar_head": int(overrides.pop("enable_r_sar_head", 0)) > 0,
        }
        self.dkd_cfg = {
            "enable_dkd": int(overrides.pop("enable_dkd", 0)) > 0,
            "dkd_temperature": float(overrides.pop("dkd_temperature", 4.0)),
            "lambda_tckd": float(overrides.pop("lambda_tckd", 0.5)),
            "lambda_nckd": float(overrides.pop("lambda_nckd", 1.0)),
            "dkd_fg_only": int(overrides.pop("dkd_fg_only", 1)) > 0,
            "teacher_fg_expand": int(overrides.pop("teacher_fg_expand", 0)) > 0,
            "teacher_fg_threshold": float(overrides.pop("teacher_fg_threshold", 0.3)),
            "lambda_proto_cls": float(overrides.pop("lambda_proto_cls", 0.0)),
            "comparison_kd_profile": overrides.pop("comparison_kd_profile", "none"),
            "profile_kd_weight": float(overrides.pop("profile_kd_weight", 1.0)),
            "profile_kd_replace_base": int(overrides.pop("profile_kd_replace_base", 0)) > 0,
            "fgd_alpha": float(overrides.pop("fgd_alpha", 0.0001)),
            "fgd_beta": float(overrides.pop("fgd_beta", 0.00005)),
            "fgd_gamma": float(overrides.pop("fgd_gamma", 0.001)),
            "fgd_lambda": float(overrides.pop("fgd_lambda", overrides.pop("fgd_relation_weight", 0.0))),
            "fgd_normalization_mode": str(overrides.pop("fgd_normalization_mode", "original")),
            "fgd_temperature": float(overrides.pop("fgd_temperature", 0.5)),
            "fgd_mask_mode": str(overrides.pop("fgd_mask_mode", "gt_box")),
            "fgd_bg_norm": int(overrides.pop("fgd_bg_norm", 1)) > 0,
            "ld_temperature": float(overrides.pop("ld_temperature", 10.0)),
            "ld_use_vlr": int(overrides.pop("ld_use_vlr", 1)) > 0,
            "ld_quality_power": float(overrides.pop("ld_quality_power", 1.0)),
            "ld_min_vlr_weight": float(overrides.pop("ld_min_vlr_weight", 0.0)),
            "ld_vlr_topk": int(overrides.pop("ld_vlr_topk", 0)),
            "ld_vlr_weight": float(overrides.pop("ld_vlr_weight", 0.25)),
            "ld_main_weight": float(overrides.pop("ld_main_weight", 0.25)),
            "ld_allow_empty_vlr": int(overrides.pop("ld_allow_empty_vlr", 1)) > 0,
            "cmdistill_feature_weight": float(overrides.pop("cmdistill_feature_weight", 1.0)),
            "cmdistill_relation_weight": float(overrides.pop("cmdistill_relation_weight", 1.0)),
            "cmdistill_logit_weight": float(overrides.pop("cmdistill_logit_weight", 1.0)),
            "cmdistill_temperature": float(overrides.pop("cmdistill_temperature", 4.0)),
            "cmdistill_max_tokens": int(overrides.pop("cmdistill_max_tokens", 512)),
            "cmdistill_min_confidence": float(overrides.pop("cmdistill_min_confidence", 0.05)),
            "cclkd_base_temperature": float(overrides.pop("cclkd_base_temperature", 2.0)),
            "cclkd_contrastive_temperature": float(overrides.pop("cclkd_contrastive_temperature", 0.1)),
            "cclkd_feat_weight": float(overrides.pop("cclkd_feat_weight", 1.0)),
            "cclkd_logit_weight": float(overrides.pop("cclkd_logit_weight", 1.0)),
            "cclkd_contrast_weight": float(overrides.pop("cclkd_contrast_weight", 0.5)),
            "cclkd_bg_weight": float(overrides.pop("cclkd_bg_weight", 0.1)),
            "cclkd_min_confidence": float(overrides.pop("cclkd_min_confidence", 0.1)),
            "cclkd_max_tokens": int(overrides.pop("cclkd_max_tokens", 512)),
            "cclkd_temperature_min": float(overrides.pop("cclkd_temperature_min", 0.5)),
            "cclkd_temperature_max": float(overrides.pop("cclkd_temperature_max", 5.0)),
            "cclkd_entropy_scale": float(overrides.pop("cclkd_entropy_scale", 5.0)),
        }
        self.tskd_cfg = {
            "teacher_data": overrides.pop("teacher_data", None),
            "teacher_weights": overrides.pop("teacher_weights", "yolo11s.pt"),
            "use_mask": bool(overrides.pop("use_mask", False)),
            "fusion_mode": overrides.pop("fusion_mode", "sum"),
            "student_detect_mode": overrides.pop("student_detect_mode", "raw"),
            "kd_target_mode": overrides.pop("kd_target_mode", "detach"),
            "lambda_rec": float(overrides.pop("lambda_rec", 0.1)),
            "lambda_sep": float(overrides.pop("lambda_sep", 0.05)),
            "lambda_taskL": float(overrides.pop("lambda_taskL", 1.0)),
            "task_loss_fg_only": bool(overrides.pop("task_loss_fg_only", False)),
            "alpha_kd": float(overrides.pop("alpha_kd", 1.0)),
            "alpha_s_rec": float(overrides.pop("alpha_s_rec", 0.1)),
            "alpha_sep": float(overrides.pop("alpha_sep", 0.05)),
            "lambda_mask_sparse": float(overrides.pop("lambda_mask_sparse", 0.0)),
            "lambda_mask_smooth": float(overrides.pop("lambda_mask_smooth", 0.0)),
        }
        self.diagnostic_cfg = {
            "validate_before_train": bool(overrides.pop("validate_before_train", False)),
            "freeze_bn_stats": bool(overrides.pop("freeze_bn_stats", False)),
            "freeze_bn_after_epoch": int(overrides.pop("freeze_bn_after_epoch", -1)),
            "ladd_diag_log_bn": int(overrides.pop("ladd_diag_log_bn", 1)) > 0,
            "ladd_diag_log_grad": int(overrides.pop("ladd_diag_log_grad", 0)) > 0,
            "ladd_grad_clip_norm": max(float(overrides.pop("ladd_grad_clip_norm", 0.0)), 0.0),
            "ladd_assert_phase_freeze": int(overrides.pop("ladd_assert_phase_freeze", 0)) > 0,
            "ladd_diag_log_every": max(int(overrides.pop("ladd_diag_log_every", 1)), 1),
            "ladd_kd_decay_mode": str(overrides.pop("ladd_kd_decay_mode", "none")),
            "ladd_kd_decay_start_epoch": int(overrides.pop("ladd_kd_decay_start_epoch", -1)),
            "ladd_kd_decay_end_epoch": int(overrides.pop("ladd_kd_decay_end_epoch", -1)),
            "ladd_kd_final_mult": float(overrides.pop("ladd_kd_final_mult", 1.0)),
            "ladd_kd_stop_after_epoch": int(overrides.pop("ladd_kd_stop_after_epoch", -1)),
            "ladd_b_loss_warmup_mode": str(overrides.pop("ladd_b_loss_warmup_mode", "none")),
            "ladd_b_loss_warmup_start_epoch": int(overrides.pop("ladd_b_loss_warmup_start_epoch", -1)),
            "ladd_b_loss_warmup_end_epoch": int(overrides.pop("ladd_b_loss_warmup_end_epoch", -1)),
            "ladd_b_loss_warmup_final_mult": float(overrides.pop("ladd_b_loss_warmup_final_mult", 1.0)),
            "ladd_b_loss_warmup_scope": str(overrides.pop("ladd_b_loss_warmup_scope", "core")),
            "ladd_b_a2_core": int(overrides.pop("ladd_b_a2_core", 0)) > 0,
            "ladd_b_det_only": int(overrides.pop("ladd_b_det_only", 0)) > 0,
            "ladd_a2_det_only": int(overrides.pop("ladd_a2_det_only", 0)) > 0,
        }
        self._last_grad_norms = None
        self._effective_ladd_weights = {}
        self._phase_freeze_assert_logged_contexts = set()
        self._phase_training_state_logged = False
        if self.tskd_cfg["teacher_data"] is None:
            raise ValueError("HBB LADD trainer requires 'teacher_data'.")
        overrides["task"] = "detect"
        super().__init__(cfg, overrides, _callbacks)
        self.teacher_data = check_det_dataset(str(self.tskd_cfg["teacher_data"]))
        self.teacher_model = None
        apply_unified_paired_aug_policy(self.args)
        self.teacher_target_ema = None
        if self.explore_cfg["teacher_target_mode"] == "ema":
            self.add_callback("optimizer_step", self._on_optimizer_step_update_teacher_target_ema)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        return PairedOBBDataset(
            img_path=img_path,
            teacher_img_path=self.teacher_data["train"],
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=(mode == "train"),
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        if hasattr(self, "manual_phase_cfg"):
            self._refresh_effective_ladd_weights()
        batch = super().preprocess_batch(batch)
        teacher_img = batch.get("teacher_img")
        if teacher_img is not None:
            batch["teacher_img"] = teacher_img.to(self.device, non_blocking=self.device.type == "cuda").float() / 255
        return batch

    def _load_teacher_model(self):
        teacher = YOLO(self.tskd_cfg["teacher_weights"]).model
        teacher = teacher.to(self.device)
        teacher.float()
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

    @staticmethod
    def _jsonify_metric_value(value):
        if isinstance(value, dict):
            return {str(k): TeacherStudentDecompositionKDNRRLTeacherUAuxTrainer._jsonify_metric_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [TeacherStudentDecompositionKDNRRLTeacherUAuxTrainer._jsonify_metric_value(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def _run_initial_validation_snapshot(self, tag: str = "initial_val") -> None:
        if not self.diagnostic_cfg.get("validate_before_train", False):
            return
        if RANK not in {-1, 0} or self.validator is None:
            return

        LOGGER.info("Running pre-train validation snapshot before epoch 1.")
        if not hasattr(self, "loss_items") or self.loss_items is None:
            loss_item_count = max(int(len(getattr(self, "loss_names", []))), 1)
            self.loss_items = torch.zeros(loss_item_count, device=self.device)
        if not hasattr(self, "epoch"):
            self.epoch = -1
        self._clear_memory(threshold=0.5)
        snapshot_model = self.ema.ema if getattr(self, "ema", None) is not None else self.model
        snapshot_model = deepcopy(unwrap_model(snapshot_model)).to(self.device).eval()
        metrics = self.validator(model=snapshot_model)
        if metrics is None:
            LOGGER.warning("Pre-train validation snapshot returned no metrics.")
            return

        payload = {str(k): self._jsonify_metric_value(v) for k, v in metrics.items()}
        payload["tag"] = tag
        payload["phase"] = getattr(self, "current_phase", None)
        payload["source_model"] = str(getattr(self.args, "model", ""))

        out_path = self.save_dir / f"{tag}_metrics.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        summary_bits = []
        for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness"):
            if key in payload:
                try:
                    summary_bits.append(f"{key}={float(payload[key]):.5f}")
                except Exception:
                    summary_bits.append(f"{key}={payload[key]}")
        LOGGER.info(f"Saved pre-train validation snapshot to {out_path}")
        if summary_bits:
            LOGGER.info("Pre-train validation summary: " + ", ".join(summary_bits))

    def get_validator(self):
        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
            "angle_loss",
            "t_rec_loss",
            "t_sep_loss",
            "reach_match_loss",
            "reach_rank_loss",
            "task_loss",
            "kd_loss",
            "s_rec_loss",
            "s_sep_loss",
            "r_aux_loss",
            "u_aux_loss",
            "mask_reg_loss",
            "recon_task_loss",
            "rs_comp_loss",
            "r_obb_loss",
            "s_repel_loss",
            "path_b_loss",
            "r_sar_loss",
            "dkd_loss",
            "proto_cls_loss",
            "d_pos_mean",
            "d_neg_mean",
            "rank_gap_mean",
            "r_aux_fg_mean",
            "r_aux_bg_mean",
            "u_aux_fg_mean",
            "u_aux_bg_mean",
            "mask_mean",
            "mask_std",
            "mask_fg_mean",
            "mask_bg_mean",
        )
        return yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = TeacherStudentDecompositionKDNRRLTeacherUAuxModelHBB(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            use_mask=self.tskd_cfg["use_mask"],
            fusion_mode=self.tskd_cfg["fusion_mode"],
            student_detect_mode=self.tskd_cfg["student_detect_mode"],
            unlearnable_hidden_ratio=self.teacher_u_cfg["unlearnable_hidden_ratio"],
            kd_calibration_mode=self.explore_cfg["kd_calibration_mode"],
            student_branch_mode=self.explore_cfg["student_branch_mode"],
            teacher_feature_mode=self.explore_cfg["teacher_feature_mode"],
            student_z_bottleneck_ratio=self.explore_cfg["student_z_bottleneck_ratio"],
            teacher_branch_mode=self.explore_cfg["teacher_branch_mode"],
            teacher_z_bottleneck_ratio=self.explore_cfg["teacher_z_bottleneck_ratio"],
            enable_recon_task=self.v2_cfg["enable_recon_task"],
            enable_r_obb_head=self.v2_cfg["enable_r_obb_head"],
            enable_r_sar_head=self.v2_cfg["enable_r_sar_head"],
        )
        if weights:
            model.load(weights)
        return model

    def _make_criterion(self, model, teacher_model):
        return TeacherStudentDecompositionKDNRRLTeacherUAuxLossHBB(
            model,
            teacher_model=teacher_model,
            lambda_rec=self.tskd_cfg["lambda_rec"],
            lambda_sep=self.tskd_cfg["lambda_sep"],
            lambda_reach=self.nrrl_cfg["lambda_reach"],
            lambda_match_inner=self.nrrl_cfg["lambda_match_inner"],
            lambda_rank_inner=self.nrrl_cfg["lambda_rank_inner"],
            lambda_taskL=self.tskd_cfg["lambda_taskL"],
            task_loss_fg_only=self.tskd_cfg.get("task_loss_fg_only", False),
            alpha_kd=self.tskd_cfg["alpha_kd"],
            alpha_s_rec=self.tskd_cfg["alpha_s_rec"],
            alpha_sep=self.tskd_cfg["alpha_sep"],
            lambda_mask_sparse=self.tskd_cfg["lambda_mask_sparse"],
            lambda_mask_smooth=self.tskd_cfg["lambda_mask_smooth"],
            delta=self.nrrl_cfg["delta"],
            use_soft_rank=bool(self.nrrl_cfg["use_soft_rank"]),
            use_fg_mask_for_reach=bool(self.nrrl_cfg["use_fg_mask_for_reach"]),
            use_fg_mask_for_rec=bool(self.nrrl_cfg["use_fg_mask_for_rec"]),
            normalize_reach=bool(self.nrrl_cfg["normalize_reach"]),
            rank_d_neg_cap=float(self.nrrl_cfg["rank_d_neg_cap"]),
            lambda_anti_collapse=float(self.nrrl_cfg["lambda_anti_collapse"]),
            anti_collapse_floor=float(self.nrrl_cfg["anti_collapse_floor"]),
            reach_target_mode=self.nrrl_cfg["reach_target_mode"],
            reach_input_mode=self.nrrl_cfg["reach_input_mode"],
            kd_target_mode=self.tskd_cfg["kd_target_mode"],
            residual_aux_mode=self.nrrl_cfg["residual_aux_mode"],
            lambda_residual_aux=self.nrrl_cfg["lambda_residual_aux"],
            energy_bg_weight=self.nrrl_cfg["energy_bg_weight"],
            energy_margin=self.nrrl_cfg["energy_margin"],
            teacher_private_aux_mode=self.teacher_u_cfg["teacher_private_aux_mode"],
            lambda_teacher_private_aux=self.teacher_u_cfg["lambda_teacher_private_aux"],
            teacher_private_bg_weight=self.teacher_u_cfg["teacher_private_bg_weight"],
            teacher_private_margin=self.teacher_u_cfg["teacher_private_margin"],
            mask_train_mode=self.nrrl_cfg["mask_train_mode"],
            mask_target_mode=self.nrrl_cfg["mask_target_mode"],
            lambda_mask_target=self.nrrl_cfg["lambda_mask_target"],
            kd_weight_mode=self.nrrl_cfg["kd_weight_mode"],
            kd_weight_power=self.nrrl_cfg["kd_weight_power"],
            kd_aggregation_mode=self.nrrl_cfg["kd_aggregation_mode"],
            kd_topk_ratio=self.nrrl_cfg["kd_topk_ratio"],
            kd_calibration_mode=self.explore_cfg["kd_calibration_mode"],
            instance_energy_radius=self.nrrl_cfg["instance_energy_radius"],
            student_branch_mode=self.explore_cfg["student_branch_mode"],
            teacher_feature_mode=self.explore_cfg["teacher_feature_mode"],
            kd_mechanism=self.explore_cfg["kd_mechanism"],
            contrastive_temperature=self.explore_cfg["contrastive_temperature"],
            lambda_recon_task=self.v2_cfg["lambda_recon_task"],
            lambda_rs_complementary=self.v2_cfg["lambda_rs_complementary"],
            lambda_r_obb=self.v2_cfg["lambda_r_obb"],
            lambda_s_repel=self.v2_cfg["lambda_s_repel"],
            s_repel_tau=self.v2_cfg["s_repel_tau"],
            s_repel_fg_only=self.v2_cfg["s_repel_fg_only"],
            lambda_path_b=self.v2_cfg["lambda_path_b"],
            path_b_margin=self.v2_cfg["path_b_margin"],
            path_b_fg_only=self.v2_cfg["path_b_fg_only"],
            lambda_r_sar=self.v2_cfg["lambda_r_sar"],
            r_sar_target=self.v2_cfg["r_sar_target"],
            r_sar_infonce_temperature=self.v2_cfg["r_sar_infonce_temperature"],
            r_sar_infonce_n_samples=self.v2_cfg["r_sar_infonce_n_samples"],
            r_sar_infonce_radius=self.v2_cfg["r_sar_infonce_radius"],
            r_sar_speckle_sigma=self.v2_cfg["r_sar_speckle_sigma"],
            enable_dkd=self.dkd_cfg["enable_dkd"],
            dkd_temperature=self.dkd_cfg["dkd_temperature"],
            lambda_tckd=self.dkd_cfg["lambda_tckd"],
            lambda_nckd=self.dkd_cfg["lambda_nckd"],
            dkd_fg_only=self.dkd_cfg["dkd_fg_only"],
            teacher_fg_expand=self.dkd_cfg["teacher_fg_expand"],
            teacher_fg_threshold=self.dkd_cfg["teacher_fg_threshold"],
            lambda_proto_cls=self.dkd_cfg["lambda_proto_cls"],
            comparison_kd_profile=self.dkd_cfg["comparison_kd_profile"],
            profile_kd_weight=self.dkd_cfg["profile_kd_weight"],
            profile_kd_replace_base=self.dkd_cfg["profile_kd_replace_base"],
            fgd_alpha=self.dkd_cfg["fgd_alpha"],
            fgd_beta=self.dkd_cfg["fgd_beta"],
            fgd_gamma=self.dkd_cfg["fgd_gamma"],
            fgd_lambda=self.dkd_cfg["fgd_lambda"],
            fgd_normalization_mode=self.dkd_cfg["fgd_normalization_mode"],
            fgd_temperature=self.dkd_cfg["fgd_temperature"],
            fgd_mask_mode=self.dkd_cfg["fgd_mask_mode"],
            fgd_bg_norm=self.dkd_cfg["fgd_bg_norm"],
            ld_temperature=self.dkd_cfg["ld_temperature"],
            ld_use_vlr=self.dkd_cfg["ld_use_vlr"],
            ld_quality_power=self.dkd_cfg["ld_quality_power"],
            ld_min_vlr_weight=self.dkd_cfg["ld_min_vlr_weight"],
            ld_vlr_topk=self.dkd_cfg["ld_vlr_topk"],
            ld_vlr_weight=self.dkd_cfg["ld_vlr_weight"],
            ld_main_weight=self.dkd_cfg["ld_main_weight"],
            ld_allow_empty_vlr=self.dkd_cfg["ld_allow_empty_vlr"],
            cmdistill_feature_weight=self.dkd_cfg["cmdistill_feature_weight"],
            cmdistill_relation_weight=self.dkd_cfg["cmdistill_relation_weight"],
            cmdistill_logit_weight=self.dkd_cfg["cmdistill_logit_weight"],
            cmdistill_temperature=self.dkd_cfg["cmdistill_temperature"],
            cmdistill_max_tokens=self.dkd_cfg["cmdistill_max_tokens"],
            cmdistill_min_confidence=self.dkd_cfg["cmdistill_min_confidence"],
            cclkd_base_temperature=self.dkd_cfg["cclkd_base_temperature"],
            cclkd_contrastive_temperature=self.dkd_cfg["cclkd_contrastive_temperature"],
            cclkd_feat_weight=self.dkd_cfg["cclkd_feat_weight"],
            cclkd_logit_weight=self.dkd_cfg["cclkd_logit_weight"],
            cclkd_contrast_weight=self.dkd_cfg["cclkd_contrast_weight"],
            cclkd_bg_weight=self.dkd_cfg["cclkd_bg_weight"],
            cclkd_min_confidence=self.dkd_cfg["cclkd_min_confidence"],
            cclkd_max_tokens=self.dkd_cfg["cclkd_max_tokens"],
            cclkd_temperature_min=self.dkd_cfg["cclkd_temperature_min"],
            cclkd_temperature_max=self.dkd_cfg["cclkd_temperature_max"],
            cclkd_entropy_scale=self.dkd_cfg["cclkd_entropy_scale"],
        )

    @staticmethod
    def _update_module_ema(target_module, source_module, momentum: float) -> None:
        with torch.no_grad():
            src_params = dict(source_module.named_parameters())
            for name, target_param in target_module.named_parameters():
                source_param = src_params[name]
                target_param.data.mul_(momentum).add_(source_param.data, alpha=1.0 - momentum)

            src_buffers = dict(source_module.named_buffers())
            for name, target_buffer in target_module.named_buffers():
                source_buffer = src_buffers[name]
                if torch.is_floating_point(target_buffer):
                    target_buffer.data.mul_(momentum).add_(source_buffer.data, alpha=1.0 - momentum)
                else:
                    target_buffer.data.copy_(source_buffer.data)

    def _setup_teacher_target_ema(self, model) -> None:
        if self.explore_cfg["teacher_target_mode"] != "ema":
            return

        self.teacher_target_ema = {
            "teacher_decomposition": deepcopy(model.teacher_decomposition).eval(),
            "teacher_decoder": deepcopy(model.teacher_decoder).eval(),
            "teacher_task_heads": deepcopy(model.teacher_task_heads).eval(),
        }
        for modules in self.teacher_target_ema.values():
            modules.to(self.device)
            for param in modules.parameters():
                param.requires_grad_(False)

        model.criterion.set_teacher_target_modules(self.teacher_target_ema)
        if self.ema and getattr(self.ema.ema, "criterion", None) is not None:
            self.ema.ema.criterion.set_teacher_target_modules(self.teacher_target_ema)

        if RANK in {-1, 0}:
            LOGGER.info(
                "Teacher target EMA enabled for KD target construction "
                f"(momentum={self.explore_cfg['teacher_ema_momentum']:.4f})."
            )

    def _on_optimizer_step_update_teacher_target_ema(self, trainer) -> None:
        if trainer is not self or self.teacher_target_ema is None:
            return

        student_model = unwrap_model(self.model)
        momentum = float(self.explore_cfg["teacher_ema_momentum"])
        for target_module, source_module in zip(
            self.teacher_target_ema["teacher_decomposition"],
            student_model.teacher_decomposition,
        ):
            self._update_module_ema(target_module, source_module, momentum)
        for target_module, source_module in zip(
            self.teacher_target_ema["teacher_decoder"],
            student_model.teacher_decoder,
        ):
            self._update_module_ema(target_module, source_module, momentum)
        for target_module, source_module in zip(
            self.teacher_target_ema["teacher_task_heads"],
            student_model.teacher_task_heads,
        ):
            self._update_module_ema(target_module, source_module, momentum)

    def _setup_train(self):
        super()._setup_train()
        self.teacher_model = self._load_teacher_model()
        student_model = unwrap_model(self.model)
        student_model.criterion = self._make_criterion(student_model, self.teacher_model)
        if self.ema:
            self.ema.ema.criterion = self._make_criterion(self.ema.ema, None)
        self._setup_teacher_target_ema(student_model)


class ManualPhaseTeacherStudentDecompositionKDNRRLTeacherUAuxTrainer(
    TeacherStudentDecompositionKDNRRLTeacherUAuxTrainer
):
    """Single-phase manual trainer for A1/A2/B/C with teacher-side u_t controls."""

    _DEFAULT_ULTRALYTICS_GRAD_CLIP_NORM = 10.0

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        overrides = {} if overrides is None else dict(overrides)
        phase = str(overrides.pop("phase", "")).lower()
        if phase not in {"a1", "a2", "b", "c", "b1", "b2"}:
            raise ValueError(
                "ManualPhaseTeacherStudentDecompositionKDNRRLTeacherUAuxTrainer requires phase in {'a1','a2','b','c','b1','b2'}."
            )
        phase_min_epochs = overrides.pop("phase_min_epochs", None)
        phase_stop_metric = overrides.pop("phase_stop_metric", "default")
        self.manual_phase_cfg = {
            "phase": phase,
            "phase_detect_mode": overrides.pop("phase_detect_mode", None),
            "det_loss_scale": overrides.pop("det_loss_scale", None),
            "phase_min_epochs": None if phase_min_epochs is None else int(phase_min_epochs),
            "phase_stop_metric": str(phase_stop_metric),
            "c_weak_nrrl_scale": float(overrides.pop("c_weak_nrrl_scale", 0.0)),
            "c_weak_nrrl_detach_student": bool(overrides.pop("c_weak_nrrl_detach_student", False)),
            "reach_c_mode": str(overrides.pop("reach_c_mode", "none")),
            "lambda_reach_c": float(overrides.pop("lambda_reach_c", 0.0)),
            "b_reset_student_from_scratch": bool(overrides.pop("b_reset_student_from_scratch", False)),
            "b_detector_source": str(overrides.pop("b_detector_source", "") or ""),
            "b_decomp_source": str(overrides.pop("b_decomp_source", "") or ""),
            "b_split_load_strict": int(overrides.pop("b_split_load_strict", 0)) > 0,
            "b_load_student_split": int(overrides.pop("b_load_student_split", 0)) > 0,
            "b_load_student_reachability": int(overrides.pop("b_load_student_reachability", 1)) > 0,
            "b_load_student_aux": int(overrides.pop("b_load_student_aux", 0)) > 0,
        }
        super().__init__(cfg, overrides, _callbacks)
        self.current_phase: str | None = None
        self.phase_stop_fitness: float | None = None

    def _resolve_phase_detect_mode(self) -> str:
        requested = self.manual_phase_cfg["phase_detect_mode"]
        if requested is not None:
            if requested not in {"raw", "fused", "mimic", "recon"}:
                raise ValueError(f"Unsupported phase_detect_mode: {requested}")
            return requested
        return "raw"

    def _resolve_det_loss_scale(self) -> float:
        det_scale = self.manual_phase_cfg["det_loss_scale"]
        if det_scale is not None:
            return float(det_scale)
        return 0.0 if self.manual_phase_cfg["phase"] in {"a1", "b1"} else 1.0

    def _resolve_phase_min_epochs(self) -> int:
        min_epochs = self.manual_phase_cfg["phase_min_epochs"]
        if min_epochs is not None:
            return max(int(min_epochs), 0)
        default_by_phase = {"a1": 50, "a2": 100, "b": 200, "c": 150, "b1": 50, "b2": 300}
        return default_by_phase[self.manual_phase_cfg["phase"]]

    def _compute_phase_stop_fitness(self, metrics: dict | None) -> float | None:
        phase = self.manual_phase_cfg["phase"]
        mode = self.manual_phase_cfg["phase_stop_metric"]
        if metrics is None:
            return None
        if mode == "default":
            mode = "a1_loss" if phase == "a1" else "map"
        if mode == "map":
            value = metrics.get("metrics/mAP50-95(B)")
            return None if value is None else float(value)
        if mode == "reach_loss":
            train_losses = self.label_loss_items(self.tloss)
            total = float(train_losses.get("train/reach_match_loss", 0.0)) + float(train_losses.get("train/reach_rank_loss", 0.0))
            return -total
        if mode == "kd_loss":
            train_losses = self.label_loss_items(self.tloss)
            return -float(train_losses.get("train/kd_loss", 0.0))
        if mode == "a1_task_reach":
            train_losses = self.label_loss_items(self.tloss)
            keys = (
                "train/reach_match_loss",
                "train/reach_rank_loss",
                "train/task_loss",
            )
            total = sum(float(train_losses.get(k, 0.0)) for k in keys)
            return -total
        if mode == "a1_loss":
            train_losses = self.label_loss_items(self.tloss)
            keys = (
                "train/t_rec_loss",
                "train/t_sep_loss",
                "train/reach_match_loss",
                "train/reach_rank_loss",
                "train/task_loss",
                "train/u_aux_loss",
                "train/mask_reg_loss",
            )
            total = sum(float(train_losses.get(k, 0.0)) for k in keys)
            return -total
        raise ValueError(f"Unsupported phase_stop_metric: {mode}")

    @staticmethod
    def _set_module_requires_grad(module, requires_grad: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def _set_phase_loss_scales(self, **scales: float) -> None:
        scales = apply_det_only_phase_scales(
            scales,
            phase=self.manual_phase_cfg["phase"],
            ladd_b_det_only=self.diagnostic_cfg.get("ladd_b_det_only", False),
            ladd_a2_det_only=self.diagnostic_cfg.get("ladd_a2_det_only", False),
        )
        model = unwrap_model(self.model)
        if getattr(model, "criterion", None) is not None:
            model.criterion.set_phase_loss_scales(**scales)
        if self.ema and getattr(self.ema.ema, "criterion", None) is not None:
            self.ema.ema.criterion.set_phase_loss_scales(**scales)

    def _current_phase_epoch_1based(self) -> int:
        return int(getattr(self, "epoch", 0)) + 1

    def _base_ladd_weight_dict(self) -> dict[str, float]:
        return {
            "alpha_kd": float(self.tskd_cfg["alpha_kd"]),
            "alpha_s_rec": float(self.tskd_cfg["alpha_s_rec"]),
            "alpha_sep": float(self.tskd_cfg["alpha_sep"]),
            "lambda_residual_aux": float(self.nrrl_cfg["lambda_residual_aux"]),
            "lambda_reach": float(self.nrrl_cfg["lambda_reach"]),
            "lambda_match_inner": float(self.nrrl_cfg["lambda_match_inner"]),
            "lambda_rank_inner": float(self.nrrl_cfg["lambda_rank_inner"]),
        }

    def _compute_effective_ladd_weights(self) -> dict[str, float]:
        return compute_effective_ladd_weights(
            phase=self.manual_phase_cfg["phase"],
            epoch_1based=self._current_phase_epoch_1based(),
            base_weights=self._base_ladd_weight_dict(),
            decay_mode=self.diagnostic_cfg.get("ladd_kd_decay_mode", "none"),
            decay_start_epoch=int(self.diagnostic_cfg.get("ladd_kd_decay_start_epoch", -1)),
            decay_end_epoch=int(self.diagnostic_cfg.get("ladd_kd_decay_end_epoch", -1)),
            final_mult=float(self.diagnostic_cfg.get("ladd_kd_final_mult", 1.0)),
            stop_after_epoch=int(self.diagnostic_cfg.get("ladd_kd_stop_after_epoch", -1)),
            ladd_b_det_only=bool(self.diagnostic_cfg.get("ladd_b_det_only", False)),
            ladd_a2_det_only=bool(self.diagnostic_cfg.get("ladd_a2_det_only", False)),
            b_loss_warmup_mode=self.diagnostic_cfg.get("ladd_b_loss_warmup_mode", "none"),
            b_loss_warmup_start_epoch=int(self.diagnostic_cfg.get("ladd_b_loss_warmup_start_epoch", -1)),
            b_loss_warmup_end_epoch=int(self.diagnostic_cfg.get("ladd_b_loss_warmup_end_epoch", -1)),
            b_loss_warmup_final_mult=float(self.diagnostic_cfg.get("ladd_b_loss_warmup_final_mult", 1.0)),
            b_loss_warmup_scope=self.diagnostic_cfg.get("ladd_b_loss_warmup_scope", "core"),
        )

    @staticmethod
    def _apply_effective_weights_to_criterion(criterion, weights: dict[str, float]) -> None:
        if criterion is None:
            return
        criterion.alpha_kd = float(weights["alpha_kd"])
        criterion.alpha_s_rec = float(weights["alpha_s_rec"])
        criterion.alpha_sep = float(weights["alpha_sep"])
        criterion.lambda_residual_aux = float(weights["lambda_residual_aux"])
        criterion.lambda_reach = float(weights["lambda_reach"])
        criterion.lambda_match_inner = float(weights["lambda_match_inner"])
        criterion.lambda_rank_inner = float(weights["lambda_rank_inner"])

    def _refresh_effective_ladd_weights(self) -> dict[str, float]:
        weights = self._compute_effective_ladd_weights()
        self._effective_ladd_weights = weights
        model = unwrap_model(self.model)
        self._apply_effective_weights_to_criterion(getattr(model, "criterion", None), weights)
        if self.ema and getattr(self.ema.ema, "criterion", None) is not None:
            self._apply_effective_weights_to_criterion(self.ema.ema.criterion, weights)
        return weights

    def _set_reachability_enabled(self, enabled: bool) -> None:
        model = unwrap_model(self.model)
        if getattr(model, "criterion", None) is not None:
            model.criterion.set_reachability_enabled(enabled)
        if self.ema and getattr(self.ema.ema, "criterion", None) is not None:
            self.ema.ema.criterion.set_reachability_enabled(enabled)

    def _set_reach_student_detach(self, enabled: bool) -> None:
        model = unwrap_model(self.model)
        if getattr(model, "criterion", None) is not None:
            model.criterion.set_reach_student_detach(enabled)
        if self.ema and getattr(self.ema.ema, "criterion", None) is not None:
            self.ema.ema.criterion.set_reach_student_detach(enabled)

    def _set_phase_target_modes(self, *, reach_target_mode: str, kd_target_mode: str) -> None:
        model = unwrap_model(self.model)
        if getattr(model, "criterion", None) is not None:
            model.criterion.set_target_modes(
                match_target_mode=reach_target_mode,
                kd_target_mode=kd_target_mode,
            )
        if self.ema and getattr(self.ema.ema, "criterion", None) is not None:
            self.ema.ema.criterion.set_target_modes(
                match_target_mode=reach_target_mode,
                kd_target_mode=kd_target_mode,
            )

    def _set_detect_mode(self, detect_mode: str) -> None:
        model = unwrap_model(self.model)
        model.student_detect_mode = detect_mode
        if self.ema and hasattr(self.ema, "ema"):
            self.ema.ema.student_detect_mode = detect_mode

    def _maybe_reset_student_from_scratch_for_phase_b(self) -> None:
        if self.manual_phase_cfg["phase"] != "b" or not self.manual_phase_cfg["b_reset_student_from_scratch"]:
            return

        model = unwrap_model(self.model)
        fresh_model = TeacherStudentDecompositionKDNRRLTeacherUAuxModelHBB(
            cfg=deepcopy(model.yaml),
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=False,
            use_mask=self.tskd_cfg["use_mask"],
            fusion_mode=self.tskd_cfg["fusion_mode"],
            student_detect_mode=self.tskd_cfg["student_detect_mode"],
            unlearnable_hidden_ratio=self.teacher_u_cfg["unlearnable_hidden_ratio"],
            kd_calibration_mode=self.explore_cfg["kd_calibration_mode"],
            student_branch_mode=self.explore_cfg["student_branch_mode"],
            teacher_feature_mode=self.explore_cfg["teacher_feature_mode"],
            student_z_bottleneck_ratio=self.explore_cfg["student_z_bottleneck_ratio"],
            teacher_branch_mode=self.explore_cfg["teacher_branch_mode"],
            teacher_z_bottleneck_ratio=self.explore_cfg["teacher_z_bottleneck_ratio"],
        )
        reset_module_names = (
            "model",
            "student_split",
            "student_reachability",
            "student_r_aux_decoder",
            "student_r_fg_heads",
        )
        for name in reset_module_names:
            getattr(model, name).load_state_dict(getattr(fresh_model, name).state_dict())
            if self.ema and hasattr(self.ema, "ema"):
                getattr(self.ema.ema, name).load_state_dict(getattr(fresh_model, name).state_dict())

        if RANK in {-1, 0}:
            LOGGER.info(
                "Manual phase 'b': reset student-side modules from scratch before distillation "
                "(student detector/split/reachability/residual aux)."
            )

    @staticmethod
    def _load_state_with_report(target, source, *, strict: bool) -> dict:
        source_state = source.state_dict()
        if strict:
            result = target.load_state_dict(source_state, strict=True)
            return {
                "missing_keys": list(getattr(result, "missing_keys", [])),
                "unexpected_keys": list(getattr(result, "unexpected_keys", [])),
                "num_tensors": len(source_state),
                "num_loaded_tensors": len(source_state),
                "num_shape_mismatch": 0,
                "shape_mismatch_keys": [],
            }

        target_state = target.state_dict()
        compatible_state = {}
        unexpected = []
        shape_mismatch = []
        for key, value in source_state.items():
            if key not in target_state:
                unexpected.append(key)
                continue
            if tuple(value.shape) != tuple(target_state[key].shape):
                shape_mismatch.append(
                    {
                        "key": key,
                        "source_shape": list(value.shape),
                        "target_shape": list(target_state[key].shape),
                    }
                )
                continue
            compatible_state[key] = value

        result = target.load_state_dict(compatible_state, strict=False)
        missing = list(getattr(result, "missing_keys", []))
        unexpected = sorted(set(unexpected + list(getattr(result, "unexpected_keys", []))))
        return {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "num_tensors": len(source_state),
            "num_loaded_tensors": len(compatible_state),
            "num_shape_mismatch": len(shape_mismatch),
            "shape_mismatch_keys": shape_mismatch,
        }

    def _maybe_apply_b_split_load(self) -> None:
        if self.manual_phase_cfg["phase"] != "b":
            return
        detector_source = self.manual_phase_cfg.get("b_detector_source", "")
        decomp_source = self.manual_phase_cfg.get("b_decomp_source", "")
        if not detector_source and not decomp_source:
            return
        if not detector_source or not decomp_source:
            raise ValueError(
                "B split-load diagnostics require both --b-detector-source and --b-decomp-source."
            )

        strict = bool(self.manual_phase_cfg.get("b_split_load_strict", False))
        model = unwrap_model(self.model)
        report = {
            "b_detector_source": detector_source,
            "b_decomp_source": decomp_source,
            "b_split_load_strict": strict,
            "b_load_student_split": bool(self.manual_phase_cfg.get("b_load_student_split", False)),
            "b_load_student_reachability": bool(
                self.manual_phase_cfg.get("b_load_student_reachability", True)
            ),
            "b_load_student_aux": bool(self.manual_phase_cfg.get("b_load_student_aux", False)),
            "selected_decomp_modules": [],
            "note": (
                "Split-load overwrites only selected modules from b_decomp_source; "
                "unselected modules retain the START_MODEL/MODEL initialization unless separately reset."
            ),
            "modules": {},
        }

        detector_model = YOLO(detector_source).model
        detector_report = self._load_state_with_report(model.model, detector_model.model, strict=strict)
        report["modules"]["detector.model"] = detector_report
        if self.ema and hasattr(self.ema, "ema"):
            self._load_state_with_report(self.ema.ema.model, detector_model.model, strict=strict)

        decomp_model = YOLO(decomp_source).model
        module_names = [
            "teacher_decomposition",
            "teacher_decoder",
            "teacher_task_heads",
        ]
        if self.manual_phase_cfg.get("b_load_student_split", False):
            module_names.append("student_split")
        if self.manual_phase_cfg.get("b_load_student_reachability", True):
            module_names.append("student_reachability")
        if self.manual_phase_cfg.get("b_load_student_aux", False):
            module_names.extend(("student_r_aux_decoder", "student_r_fg_heads"))
        report["selected_decomp_modules"] = module_names
        for name in module_names:
            if not hasattr(model, name):
                continue
            if not hasattr(decomp_model, name):
                if strict:
                    raise AttributeError(f"B split-load source missing module: {name}")
                report["modules"][name] = {"missing_module": True}
                continue
            module_report = self._load_state_with_report(
                getattr(model, name),
                getattr(decomp_model, name),
                strict=strict,
            )
            report["modules"][name] = module_report
            if self.ema and hasattr(self.ema, "ema") and hasattr(self.ema.ema, name):
                self._load_state_with_report(
                    getattr(self.ema.ema, name),
                    getattr(decomp_model, name),
                    strict=strict,
                )

        report_path = self.save_dir / "b_split_load_manifest.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if RANK in {-1, 0}:
            LOGGER.info(
                "Manual phase 'b': applied split-load initialization "
                f"detector_source={detector_source} decomp_source={decomp_source} "
                f"strict={strict} manifest={report_path}"
            )

    def _apply_manual_phase(self, announce: bool = True) -> None:
        model = unwrap_model(self.model)
        phase = self.manual_phase_cfg["phase"]
        detect_mode = self._resolve_phase_detect_mode()
        det_scale = self._resolve_det_loss_scale()
        student_branch_split = self.explore_cfg["student_branch_mode"] == "split"
        student_branch_use_zs = self.explore_cfg["student_branch_mode"] in {"split", "single_proj", "residual"}
        force_student_rec = self.explore_cfg.get("force_student_rec", False)
        enable_student_rec = student_branch_split or force_student_rec
        teacher_decomposed = self.explore_cfg["teacher_feature_mode"] == "decomposed"
        teacher_projected_raw = self.explore_cfg["teacher_feature_mode"] == "projected_raw"
        if phase in {"a1", "a2"} and (not student_branch_use_zs or not teacher_decomposed):
            if self.explore_cfg["student_branch_mode"] != "raw":
                raise ValueError(
                    "A1/A2 only support student_branch_mode in {'split', 'single_proj', 'residual', 'raw'} and "
                    "teacher_feature_mode='decomposed'."
                )
        use_residual_aux_head = student_branch_split and self.nrrl_cfg["residual_aux_mode"] == "fg"
        use_reach_adapter = self.nrrl_cfg["reach_input_mode"] == "adapter"
        c_weak_nrrl_scale = max(float(self.manual_phase_cfg["c_weak_nrrl_scale"]), 0.0)
        reach_c_mode = self.manual_phase_cfg["reach_c_mode"]
        if reach_c_mode not in {"none", "rank", "weight"}:
            raise ValueError(f"Unsupported reach_c_mode: {reach_c_mode}")
        lambda_reach_c = max(float(self.manual_phase_cfg["lambda_reach_c"]), 0.0)
        c_enable_weak_nrrl = phase == "c" and teacher_decomposed and c_weak_nrrl_scale > 0
        c_enable_frozen_reach_rank = phase == "c" and teacher_decomposed and reach_c_mode == "rank" and lambda_reach_c > 0
        self._set_detect_mode(detect_mode)
        self._set_reach_student_detach(False)

        recon_task_enabled = getattr(model, "recon_task_enabled", False)
        b_a2_core = bool(self.diagnostic_cfg.get("ladd_b_a2_core", False))
        if phase == "a1":
            self._set_module_requires_grad(model.model, False)
            self._set_module_requires_grad(model.student_split, False)
            self._set_module_requires_grad(model.teacher_decomposition, True)
            self._set_module_requires_grad(model.teacher_decoder, True)
            self._set_module_requires_grad(model.student_r_aux_decoder, False)
            self._set_module_requires_grad(model.student_reachability, use_reach_adapter)
            self._set_module_requires_grad(model.teacher_task_heads, True)
            self._set_module_requires_grad(model.student_r_fg_heads, False)
            if recon_task_enabled:
                self._set_module_requires_grad(model.teacher_recon_decoder, True)
                self._set_module_requires_grad(model.teacher_recon_task_heads, True)
            self._set_reachability_enabled(True)
            self._set_phase_target_modes(reach_target_mode="coupled", kd_target_mode="detach")
            self._set_phase_loss_scales(
                det=det_scale, rec=1.0, teacher_sep=1.0, match=1.0, unmatch=1.0, task=1.0,
                kd=0.0, student_rec=0.0, student_sep=0.0, residual_aux=0.0,
                teacher_private_aux=1.0, mask=1.0,
                recon_task=(1.0 if recon_task_enabled else 0.0), rs_comp=0.0,
            )
        elif phase == "a2":
            self._set_module_requires_grad(model.model, True)
            self._set_module_requires_grad(model.student_split, False)
            self._set_module_requires_grad(model.teacher_decomposition, True)
            self._set_module_requires_grad(model.teacher_decoder, True)
            self._set_module_requires_grad(model.student_r_aux_decoder, False)
            self._set_module_requires_grad(model.student_reachability, use_reach_adapter)
            self._set_module_requires_grad(model.teacher_task_heads, True)
            self._set_module_requires_grad(model.student_r_fg_heads, False)
            if recon_task_enabled:
                self._set_module_requires_grad(model.teacher_recon_decoder, True)
                self._set_module_requires_grad(model.teacher_recon_task_heads, True)
            self._set_reachability_enabled(True)
            self._set_phase_target_modes(reach_target_mode="coupled", kd_target_mode="detach")
            self._set_phase_loss_scales(
                det=det_scale, rec=1.0, teacher_sep=1.0, match=1.0, unmatch=1.0, task=1.0,
                kd=0.0, student_rec=0.0, student_sep=0.0, residual_aux=0.0,
                teacher_private_aux=1.0, mask=1.0,
                recon_task=(1.0 if recon_task_enabled else 0.0), rs_comp=0.0,
            )
        elif phase == "b":
            self._set_module_requires_grad(model.model, True)
            self._set_module_requires_grad(model.student_split, student_branch_use_zs)
            self._set_module_requires_grad(model.teacher_decomposition, b_a2_core)
            self._set_module_requires_grad(model.teacher_decoder, b_a2_core or teacher_projected_raw)
            self._set_module_requires_grad(model.student_r_aux_decoder, use_residual_aux_head)
            self._set_module_requires_grad(model.student_reachability, b_a2_core and use_reach_adapter)
            self._set_module_requires_grad(model.teacher_task_heads, b_a2_core)
            self._set_module_requires_grad(model.student_r_fg_heads, use_residual_aux_head)
            if recon_task_enabled:
                self._set_module_requires_grad(model.teacher_recon_decoder, b_a2_core)
                self._set_module_requires_grad(model.teacher_recon_task_heads, b_a2_core)
            self._set_reachability_enabled(b_a2_core)
            self._set_phase_target_modes(
                reach_target_mode=("coupled" if b_a2_core else "detach"),
                kd_target_mode="detach",
            )
            self._set_phase_loss_scales(
                det=det_scale,
                rec=(1.0 if b_a2_core else 0.0),
                teacher_sep=(1.0 if b_a2_core else 0.0),
                match=(1.0 if b_a2_core else 0.0),
                unmatch=(1.0 if b_a2_core else 0.0),
                task=(1.0 if b_a2_core else 0.0),
                kd=1.0, student_rec=(1.0 if enable_student_rec else 0.0),
                student_sep=(1.0 if student_branch_split else 0.0),
                residual_aux=(1.0 if student_branch_split else 0.0),
                teacher_private_aux=(1.0 if b_a2_core else 0.0),
                mask=(1.0 if b_a2_core else 0.0),
                recon_task=(1.0 if b_a2_core and recon_task_enabled else 0.0),
                rs_comp=1.0,
                # Path A / D — enabled when student_split is trainable.
                s_repel=(1.0 if student_branch_split else 0.0),
                path_b=(1.0 if student_branch_split else 0.0),
                r_sar=(1.0 if student_branch_split else 0.0),
            )
        elif phase == "c":
            self._set_module_requires_grad(model.model, True)
            self._set_module_requires_grad(model.student_split, student_branch_use_zs)
            self._set_module_requires_grad(model.teacher_decomposition, teacher_decomposed)
            self._set_module_requires_grad(model.teacher_decoder, teacher_decomposed or teacher_projected_raw)
            self._set_module_requires_grad(model.student_r_aux_decoder, use_residual_aux_head)
            self._set_module_requires_grad(model.student_reachability, False)
            self._set_module_requires_grad(model.teacher_task_heads, teacher_decomposed)
            self._set_module_requires_grad(model.student_r_fg_heads, use_residual_aux_head)
            if recon_task_enabled:
                self._set_module_requires_grad(model.teacher_recon_decoder, True)
                self._set_module_requires_grad(model.teacher_recon_task_heads, True)
            self._set_reachability_enabled(c_enable_weak_nrrl or c_enable_frozen_reach_rank)
            self._set_reach_student_detach(
                (c_enable_weak_nrrl and bool(self.manual_phase_cfg["c_weak_nrrl_detach_student"]))
                or reach_c_mode in {"rank", "weight"}
            )
            self._set_phase_target_modes(
                reach_target_mode="coupled" if (c_enable_weak_nrrl or c_enable_frozen_reach_rank) else self.nrrl_cfg["reach_target_mode"],
                kd_target_mode=self.tskd_cfg["kd_target_mode"],
            )
            self._set_phase_loss_scales(
                det=det_scale,
                rec=(1.0 if teacher_decomposed else 0.0),
                teacher_sep=(1.0 if teacher_decomposed else 0.0),
                match=c_weak_nrrl_scale,
                unmatch=(c_weak_nrrl_scale if c_enable_weak_nrrl else lambda_reach_c if c_enable_frozen_reach_rank else 0.0),
                task=(1.0 if teacher_decomposed else 0.0),
                kd=1.0, student_rec=(1.0 if enable_student_rec else 0.0),
                student_sep=(1.0 if student_branch_split else 0.0),
                residual_aux=(1.0 if student_branch_split else 0.0),
                teacher_private_aux=(1.0 if teacher_decomposed else 0.0),
                mask=(1.0 if teacher_decomposed else 0.0),
                recon_task=(1.0 if recon_task_enabled else 0.0), rs_comp=1.0,
                r_obb=(1.0 if (student_branch_split and self.v2_cfg["enable_r_obb_head"]) else 0.0),
                # Path A / D — enabled when student_split is trainable.
                s_repel=(1.0 if student_branch_split else 0.0),
                path_b=(1.0 if student_branch_split else 0.0),
                r_sar=(1.0 if student_branch_split else 0.0),
                dkd=(1.0 if self.dkd_cfg["enable_dkd"] else 0.0),
                proto_cls=(1.0 if self.dkd_cfg["lambda_proto_cls"] > 0 else 0.0),
            )
        elif phase == "b1":
            # B1: freeze student backbone, train student_split with KD (+rec)
            # Detection loss set to 0 (backbone frozen → no gradient from det)
            self._set_module_requires_grad(model.model, False)
            self._set_module_requires_grad(model.student_split, student_branch_use_zs)
            self._set_module_requires_grad(model.teacher_decomposition, False)
            self._set_module_requires_grad(model.teacher_decoder, False)
            self._set_module_requires_grad(model.student_r_aux_decoder, False)
            self._set_module_requires_grad(model.student_reachability, False)
            self._set_module_requires_grad(model.teacher_task_heads, False)
            self._set_module_requires_grad(model.student_r_fg_heads, False)
            if recon_task_enabled:
                self._set_module_requires_grad(model.teacher_recon_decoder, False)
                self._set_module_requires_grad(model.teacher_recon_task_heads, False)
            self._set_reachability_enabled(False)
            self._set_phase_target_modes(reach_target_mode="detach", kd_target_mode="detach")
            self._set_phase_loss_scales(
                det=det_scale, rec=(1.0 if student_branch_split else 0.0),
                teacher_sep=0.0, match=0.0, unmatch=0.0, task=0.0,
                kd=1.0, student_rec=(1.0 if enable_student_rec else 0.0),
                student_sep=0.0, residual_aux=0.0, teacher_private_aux=0.0, mask=0.0,
                recon_task=0.0, rs_comp=1.0,
            )
        elif phase == "b2":
            # B2: joint training (same as old C phase)
            self._set_module_requires_grad(model.model, True)
            self._set_module_requires_grad(model.student_split, student_branch_use_zs)
            self._set_module_requires_grad(model.teacher_decomposition, teacher_decomposed)
            self._set_module_requires_grad(model.teacher_decoder, teacher_decomposed or teacher_projected_raw)
            self._set_module_requires_grad(model.student_r_aux_decoder, use_residual_aux_head)
            self._set_module_requires_grad(model.student_reachability, False)
            self._set_module_requires_grad(model.teacher_task_heads, teacher_decomposed)
            self._set_module_requires_grad(model.student_r_fg_heads, use_residual_aux_head)
            if recon_task_enabled:
                self._set_module_requires_grad(model.teacher_recon_decoder, True)
                self._set_module_requires_grad(model.teacher_recon_task_heads, True)
            self._set_reachability_enabled(c_enable_weak_nrrl or c_enable_frozen_reach_rank)
            self._set_reach_student_detach(
                (c_enable_weak_nrrl and bool(self.manual_phase_cfg["c_weak_nrrl_detach_student"]))
                or reach_c_mode in {"rank", "weight"}
            )
            self._set_phase_target_modes(
                reach_target_mode="coupled" if (c_enable_weak_nrrl or c_enable_frozen_reach_rank) else self.nrrl_cfg["reach_target_mode"],
                kd_target_mode=self.tskd_cfg["kd_target_mode"],
            )
            self._set_phase_loss_scales(
                det=det_scale,
                rec=(1.0 if teacher_decomposed else 0.0),
                teacher_sep=(1.0 if teacher_decomposed else 0.0),
                match=c_weak_nrrl_scale,
                unmatch=(c_weak_nrrl_scale if c_enable_weak_nrrl else lambda_reach_c if c_enable_frozen_reach_rank else 0.0),
                task=(1.0 if teacher_decomposed else 0.0),
                kd=1.0, student_rec=(1.0 if enable_student_rec else 0.0),
                student_sep=(1.0 if student_branch_split else 0.0),
                residual_aux=(1.0 if student_branch_split else 0.0),
                teacher_private_aux=(1.0 if teacher_decomposed else 0.0),
                mask=(1.0 if teacher_decomposed else 0.0),
                recon_task=(1.0 if recon_task_enabled else 0.0), rs_comp=1.0,
                # Path A / D — enabled when student_split is trainable.
                s_repel=(1.0 if student_branch_split else 0.0),
                path_b=(1.0 if student_branch_split else 0.0),
                r_sar=(1.0 if student_branch_split else 0.0),
                dkd=(1.0 if self.dkd_cfg["enable_dkd"] else 0.0),
                proto_cls=(1.0 if self.dkd_cfg["lambda_proto_cls"] > 0 else 0.0),
            )
        else:
            raise ValueError(f"Unsupported manual phase: {phase}")

        self.current_phase = phase
        if announce and RANK in {-1, 0}:
            LOGGER.info(
                f"Manual TSKD-NRRL-teacher-u-aux phase '{phase}' configured with detect_mode='{detect_mode}' "
                f"and det_loss_scale={det_scale:.3f}"
            )

    def _warn_conflicting_warmup_modes(self) -> None:
        if RANK not in {-1, 0}:
            return
        if str(self.manual_phase_cfg.get("phase", "")).lower() != "b":
            return

        kd_mode = str(self.diagnostic_cfg.get("ladd_kd_decay_mode", "none") or "none").lower()
        b_loss_mode = str(self.diagnostic_cfg.get("ladd_b_loss_warmup_mode", "none") or "none").lower()

        kd_warmup_modes = {"warmup", "warmup_linear", "linear_warmup", "ramp_linear"}
        if b_loss_mode != "none" and kd_mode in kd_warmup_modes:
            LOGGER.warning(
                "Both KD warmup and core B loss warmup are enabled. "
                "alpha_kd will be multiplied by both kd_multiplier and b_loss_warmup_multiplier. "
                "For core LADD warmup experiments, prefer LADD_KD_DECAY_MODE=none."
            )

    def _setup_train(self):
        super()._setup_train()
        self.stopper = PhaseMinEarlyStopping(
            patience=int(self.args.patience),
            min_epochs=self._resolve_phase_min_epochs(),
        )
        self._maybe_reset_student_from_scratch_for_phase_b()
        self._maybe_apply_b_split_load()
        self._apply_manual_phase(announce=True)
        self._refresh_effective_ladd_weights()
        self._warn_conflicting_warmup_modes()
        if self._should_freeze_bn_stats():
            self._set_bn_stats_eval(unwrap_model(self.model))
        self._log_phase_training_state()
        self._assert_b_phase_frozen_modules(unwrap_model(self.model), context="after_apply_manual_phase")
        self._log_bn_stats_strategy()
        self._run_initial_validation_snapshot()

    def validate(self):
        self._refresh_effective_ladd_weights()
        metrics = self.validator(self)
        if metrics is None:
            return None, None
        phase_fitness = self._compute_phase_stop_fitness(metrics)
        self.phase_stop_fitness = phase_fitness
        if phase_fitness is None:
            fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())
        else:
            metrics["phase_stop_fitness"] = phase_fitness
            metrics.pop("fitness", None)
            fitness = phase_fitness
        if self.best_fitness is None or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def _model_train(self):
        super()._model_train()
        model = unwrap_model(self.model)
        phase = self.manual_phase_cfg["phase"]
        if phase in {"a1", "a2", "b1"}:
            model.model.eval()
            model.student_split.eval()
            model.student_r_aux_decoder.eval()
            model.student_r_fg_heads.eval()
        elif phase == "b":
            if self.diagnostic_cfg.get("ladd_b_a2_core", False):
                model.teacher_decomposition.train()
                model.teacher_decoder.train()
                model.student_reachability.train()
                model.teacher_task_heads.train()
                if getattr(model, "recon_task_enabled", False):
                    model.teacher_recon_decoder.train()
                    model.teacher_recon_task_heads.train()
            else:
                model.teacher_decomposition.eval()
                model.teacher_decoder.eval()
                model.student_reachability.eval()
                model.teacher_task_heads.eval()
        elif phase == "c":
            train_residual_aux = self.explore_cfg["student_branch_mode"] == "split" and self.nrrl_cfg["residual_aux_mode"] == "fg"
            model.student_r_aux_decoder.train(train_residual_aux)
            model.student_r_fg_heads.train(train_residual_aux)
        if self._should_freeze_bn_stats():
            self._set_bn_stats_eval(model)
        self._refresh_effective_ladd_weights()
        self._assert_b_phase_frozen_modules(model, context="after_model_train_and_bn_freeze")

    def optimizer_step(self):
        log_grad = bool(self.diagnostic_cfg.get("ladd_diag_log_grad", False))
        grad_clip_norm = float(self.diagnostic_cfg.get("ladd_grad_clip_norm", 0.0) or 0.0)
        if not log_grad and grad_clip_norm <= 0.0:
            return super().optimizer_step()

        self.scaler.unscale_(self.optimizer)
        if log_grad:
            self._last_grad_norms = self._collect_grad_norms()
        else:
            self._last_grad_norms = None
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self._DEFAULT_ULTRALYTICS_GRAD_CLIP_NORM,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def save_metrics(self, metrics):
        super().save_metrics(metrics)
        self._append_ladd_diagnostics(metrics)

    @staticmethod
    def _set_bn_stats_eval(model):
        """Freeze BatchNorm running statistics without changing trainability."""
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    def _count_bn_modules(self) -> tuple[int, int]:
        model = unwrap_model(self.model)
        total = 0
        eval_count = 0
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                total += 1
                if not module.training:
                    eval_count += 1
        return total, eval_count

    def _log_phase_training_state(self) -> None:
        if self._phase_training_state_logged or RANK not in {-1, 0}:
            return
        self._phase_training_state_logged = True
        model = unwrap_model(self.model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        bn_total, bn_eval = self._count_bn_modules()
        grad_clip_norm = float(self.diagnostic_cfg.get("ladd_grad_clip_norm", 0.0) or 0.0)
        if grad_clip_norm > 0.0:
            grad_clip_label = "EXPLICIT"
            effective_grad_clip_norm = grad_clip_norm
        else:
            grad_clip_label = "ULTRALYTICS_DEFAULT"
            effective_grad_clip_norm = self._DEFAULT_ULTRALYTICS_GRAD_CLIP_NORM
        weights = self._refresh_effective_ladd_weights()
        LOGGER.info(
            "ladd_phase_diag "
            f"phase={self.manual_phase_cfg.get('phase', '')} "
            f"trainable_params={trainable_params}/{total_params} "
            f"bn_eval={bn_eval}/{bn_total} "
            f"freeze_bn_stats={bool(self.diagnostic_cfg.get('freeze_bn_stats', False))} "
            f"freeze_bn_after_epoch={int(self.diagnostic_cfg.get('freeze_bn_after_epoch', -1))} "
            f"ladd_assert_phase_freeze={bool(self.diagnostic_cfg.get('ladd_assert_phase_freeze', False))} "
            f"alpha_kd={float(self.tskd_cfg['alpha_kd'])} "
            f"base_alpha_kd={float(weights.get('base_alpha_kd', self.tskd_cfg['alpha_kd']))} "
            f"effective_alpha_kd={float(weights.get('alpha_kd', 0.0))} "
            f"kd_multiplier={float(weights.get('kd_multiplier', 1.0))} "
            f"kd_warmup_active={bool(weights.get('kd_warmup_active', 0.0))} "
            f"ladd_kd_decay_mode={self.diagnostic_cfg.get('ladd_kd_decay_mode', 'none')} "
            f"ladd_kd_decay_start_epoch={int(self.diagnostic_cfg.get('ladd_kd_decay_start_epoch', -1))} "
            f"ladd_kd_decay_end_epoch={int(self.diagnostic_cfg.get('ladd_kd_decay_end_epoch', -1))} "
            f"ladd_kd_final_mult={float(self.diagnostic_cfg.get('ladd_kd_final_mult', 1.0))} "
            f"ladd_kd_stop_after_epoch={int(self.diagnostic_cfg.get('ladd_kd_stop_after_epoch', -1))} "
            f"b_loss_warmup_mode={self.diagnostic_cfg.get('ladd_b_loss_warmup_mode', 'none')} "
            f"b_loss_warmup_start_epoch={int(self.diagnostic_cfg.get('ladd_b_loss_warmup_start_epoch', -1))} "
            f"b_loss_warmup_end_epoch={int(self.diagnostic_cfg.get('ladd_b_loss_warmup_end_epoch', -1))} "
            f"b_loss_warmup_final_mult={float(self.diagnostic_cfg.get('ladd_b_loss_warmup_final_mult', 1.0))} "
            f"b_loss_warmup_scope={self.diagnostic_cfg.get('ladd_b_loss_warmup_scope', 'core')} "
            f"b_loss_warmup_multiplier={float(weights.get('b_loss_warmup_multiplier', 1.0))} "
            f"b_loss_warmup_active={bool(weights.get('b_loss_warmup_active', 0.0))} "
            f"ladd_b_a2_core={bool(self.diagnostic_cfg.get('ladd_b_a2_core', False))} "
            f"ladd_b_det_only={bool(self.diagnostic_cfg.get('ladd_b_det_only', False))} "
            f"ladd_a2_det_only={bool(self.diagnostic_cfg.get('ladd_a2_det_only', False))} "
            f"ladd_diag_log_grad={bool(self.diagnostic_cfg.get('ladd_diag_log_grad', False))} "
            f"ladd_grad_clip={grad_clip_label} "
            f"ladd_grad_clip_norm={grad_clip_norm} "
            f"effective_grad_clip_norm={effective_grad_clip_norm}"
        )

    def _assert_modules_frozen(self, model, module_names: tuple[str, ...], *, context: str) -> None:
        if not self.diagnostic_cfg.get("ladd_assert_phase_freeze", False):
            return
        if self.manual_phase_cfg.get("phase") != "b":
            return
        missing = []
        violations = []
        checked = []
        for module_name in module_names:
            module = getattr(model, module_name, None)
            if module is None:
                missing.append(module_name)
                continue
            checked.append(module_name)
            for parameter_name, parameter in module.named_parameters(recurse=True):
                if parameter.requires_grad:
                    violations.append(f"{module_name}.{parameter_name}")
        if missing and RANK in {-1, 0} and context not in self._phase_freeze_assert_logged_contexts:
            LOGGER.warning(f"[LADD phase-freeze assert] context={context}: missing modules: {missing}")
        if violations:
            preview = ", ".join(violations[:20])
            extra = "" if len(violations) <= 20 else f" ... (+{len(violations) - 20} more)"
            raise RuntimeError(
                "[LADD phase-freeze assert] B phase frozen modules have trainable parameters "
                f"after {context}: {preview}{extra}"
            )
        if RANK in {-1, 0}:
            if context not in self._phase_freeze_assert_logged_contexts:
                LOGGER.info(f"[LADD phase-freeze assert] PASS context={context} checked={checked}")
        self._phase_freeze_assert_logged_contexts.add(context)

    def _assert_b_phase_frozen_modules(self, model, *, context: str) -> None:
        if self.diagnostic_cfg.get("ladd_b_a2_core", False):
            return
        self._assert_modules_frozen(
            model,
            ("teacher_decomposition", "student_reachability", "teacher_task_heads"),
            context=context,
        )

    def _bn_stats_mode_label(self) -> str:
        if self.diagnostic_cfg.get("freeze_bn_stats", False):
            return "always_freeze"
        freeze_after = int(self.diagnostic_cfg.get("freeze_bn_after_epoch", -1))
        if self.manual_phase_cfg.get("phase") == "b" and freeze_after >= 0:
            return f"delayed_freeze@{freeze_after}"
        return "normal"

    def _should_freeze_bn_stats(self) -> bool:
        if self.diagnostic_cfg.get("freeze_bn_stats", False):
            return True
        freeze_after = int(self.diagnostic_cfg.get("freeze_bn_after_epoch", -1))
        if self.manual_phase_cfg.get("phase") != "b" or freeze_after < 0:
            return False
        return int(getattr(self, "epoch", 0)) >= freeze_after

    def _log_bn_stats_strategy(self) -> None:
        if RANK not in {-1, 0}:
            return
        freeze_after = int(self.diagnostic_cfg.get("freeze_bn_after_epoch", -1))
        if self.diagnostic_cfg.get("freeze_bn_stats", False):
            if freeze_after >= 0:
                LOGGER.warning("B_FREEZE_BN_STATS=1 overrides B_FREEZE_BN_AFTER_EPOCH.")
            LOGGER.info("bn_stats_mode=always_freeze")
        elif self.manual_phase_cfg.get("phase") == "b" and freeze_after >= 0:
            LOGGER.info(f"bn_stats_mode=delayed_freeze@{freeze_after}")
        else:
            LOGGER.info("bn_stats_mode=normal")

    def _collect_bn_stats(self) -> dict[str, float | int]:
        if not self.diagnostic_cfg.get("ladd_diag_log_bn", True):
            nan = float("nan")
            return {
                "bn_running_var_max": nan,
                "bn_running_var_mean": nan,
                "bn_running_var_p95": nan,
                "bn_running_mean_abs_max": nan,
                "bn_num_layers": 0,
            }
        model = unwrap_model(self.model)
        running_vars = []
        running_means = []
        num_layers = 0
        with torch.no_grad():
            for module in model.modules():
                if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    continue
                if module.running_var is None or module.running_mean is None:
                    continue
                num_layers += 1
                running_vars.append(module.running_var.detach().float().flatten().cpu())
                running_means.append(module.running_mean.detach().float().abs().flatten().cpu())
        if not running_vars:
            nan = float("nan")
            return {
                "bn_running_var_max": nan,
                "bn_running_var_mean": nan,
                "bn_running_var_p95": nan,
                "bn_running_mean_abs_max": nan,
                "bn_num_layers": 0,
            }
        var = torch.cat(running_vars)
        mean_abs = torch.cat(running_means)
        return {
            "bn_running_var_max": float(var.max().item()),
            "bn_running_var_mean": float(var.mean().item()),
            "bn_running_var_p95": float(torch.quantile(var, 0.95).item()),
            "bn_running_mean_abs_max": float(mean_abs.max().item()),
            "bn_num_layers": int(num_layers),
        }

    def _collect_grad_norms(self) -> dict[str, float]:
        groups = {
            "total": [],
            "backbone": [],
            "neck": [],
            "head": [],
        }
        model = unwrap_model(self.model)
        for name, parameter in model.named_parameters():
            grad = parameter.grad
            if grad is None:
                continue
            norm = grad.detach().float().norm(2)
            groups["total"].append(norm)
            lname = name.lower()
            if "backbone" in lname:
                groups["backbone"].append(norm)
            elif "neck" in lname:
                groups["neck"].append(norm)
            elif "head" in lname or "detect" in lname or ".dfl" in lname:
                groups["head"].append(norm)

        def combine(values):
            if not values:
                return float("nan")
            stacked = torch.stack(values)
            return float(torch.sqrt(torch.sum(stacked * stacked)).item())

        return {
            "grad_norm_total": combine(groups["total"]),
            "grad_norm_backbone": combine(groups["backbone"]),
            "grad_norm_neck": combine(groups["neck"]),
            "grad_norm_head": combine(groups["head"]),
        }

    @staticmethod
    def _metric_value(metrics: dict, key: str) -> float:
        value = metrics.get(key, float("nan"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    @staticmethod
    def _has_nonfinite(values) -> bool:
        for value in values:
            try:
                if not math.isfinite(float(value)):
                    return True
            except (TypeError, ValueError):
                return True
        return False

    def _collect_cmdistill_stats(self) -> dict:
        criterion = getattr(unwrap_model(self.model), "criterion", None)
        raw_stats = getattr(criterion, "_cmdistill_last_stats", {}) if criterion is not None else {}
        keys = (
            "cmdistill_pcc_levels",
            "cmdistill_slrd_tokens",
            "cmdistill_ibcld_candidate_ratio",
            "cmdistill_ibcld_fg_count",
            "cmdistill_ibcld_teacher_conf_added_count",
            "cmdistill_ibcld_cls_loss",
            "cmdistill_ibcld_box_loss",
            "cmdistill_pcc_loss",
            "cmdistill_relation_loss",
            "cmdistill_ibcld_loss",
            "cmdistill_total_loss",
        )
        return {key: raw_stats.get(key, 0.0) for key in keys}

    def _append_ladd_diagnostics(self, metrics: dict) -> None:
        if RANK not in {-1, 0}:
            return
        every = int(self.diagnostic_cfg.get("ladd_diag_log_every", 1))
        epoch_1based = int(getattr(self, "epoch", 0)) + 1
        if epoch_1based != 1 and epoch_1based % every != 0 and epoch_1based != int(self.epochs):
            return
        weights = self._refresh_effective_ladd_weights()
        bn_stats = self._collect_bn_stats()
        cmdistill_stats = self._collect_cmdistill_stats()
        nonfinite_metrics_or_cmdistill = int(
            self._has_nonfinite(list(metrics.values()) + list(cmdistill_stats.values()))
        )
        nonfinite_bn_stats = int(self._has_nonfinite(list(bn_stats.values())))
        grad_stats = self._last_grad_norms if self.diagnostic_cfg.get("ladd_diag_log_grad", False) else None
        if grad_stats is None:
            nan = float("nan")
            grad_stats = {
                "grad_norm_total": nan,
                "grad_norm_backbone": nan,
                "grad_norm_neck": nan,
                "grad_norm_head": nan,
            }
        row = {
            "epoch": epoch_1based,
            "stage": self.manual_phase_cfg.get("phase", ""),
            "lr_pg0": self._metric_value(metrics, "lr/pg0"),
            "lr_pg1": self._metric_value(metrics, "lr/pg1"),
            "lr_pg2": self._metric_value(metrics, "lr/pg2"),
            "train_box_loss": self._metric_value(metrics, "train/box_loss"),
            "train_cls_loss": self._metric_value(metrics, "train/cls_loss"),
            "train_dfl_loss": self._metric_value(metrics, "train/dfl_loss"),
            "kd_loss": self._metric_value(metrics, "train/kd_loss"),
            "reach_match_loss": self._metric_value(metrics, "train/reach_match_loss"),
            "reach_rank_loss": self._metric_value(metrics, "train/reach_rank_loss"),
            **cmdistill_stats,
            **bn_stats,
            "bn_stats_mode": self._bn_stats_mode_label(),
            "bn_stats_frozen_this_epoch": int(self._should_freeze_bn_stats()),
            "nonfinite_metrics_or_cmdistill": nonfinite_metrics_or_cmdistill,
            "nonfinite_bn_stats": nonfinite_bn_stats,
            "nan_or_inf_detected": int(nonfinite_metrics_or_cmdistill or nonfinite_bn_stats),
            **grad_stats,
            "base_alpha_kd": float(weights.get("base_alpha_kd", self.tskd_cfg["alpha_kd"])),
            "effective_alpha_kd": float(weights.get("alpha_kd", 0.0)),
            "kd_multiplier": float(weights.get("kd_multiplier", 1.0)),
            "kd_warmup_active": int(bool(weights.get("kd_warmup_active", 0.0))),
            "ladd_kd_decay_mode": self.diagnostic_cfg.get("ladd_kd_decay_mode", "none"),
            "ladd_kd_decay_start_epoch": int(self.diagnostic_cfg.get("ladd_kd_decay_start_epoch", -1)),
            "ladd_kd_decay_end_epoch": int(self.diagnostic_cfg.get("ladd_kd_decay_end_epoch", -1)),
            "ladd_kd_final_mult": float(self.diagnostic_cfg.get("ladd_kd_final_mult", 1.0)),
            "ladd_kd_stop_after_epoch": int(self.diagnostic_cfg.get("ladd_kd_stop_after_epoch", -1)),
            "b_loss_warmup_mode": self.diagnostic_cfg.get("ladd_b_loss_warmup_mode", "none"),
            "b_loss_warmup_start_epoch": int(self.diagnostic_cfg.get("ladd_b_loss_warmup_start_epoch", -1)),
            "b_loss_warmup_end_epoch": int(self.diagnostic_cfg.get("ladd_b_loss_warmup_end_epoch", -1)),
            "b_loss_warmup_final_mult": float(self.diagnostic_cfg.get("ladd_b_loss_warmup_final_mult", 1.0)),
            "b_loss_warmup_scope": self.diagnostic_cfg.get("ladd_b_loss_warmup_scope", "core"),
            "b_loss_warmup_multiplier": float(weights.get("b_loss_warmup_multiplier", 1.0)),
            "b_loss_warmup_active": int(bool(weights.get("b_loss_warmup_active", 0.0))),
            "ladd_b_a2_core": int(self.diagnostic_cfg.get("ladd_b_a2_core", False)),
            "ladd_b_det_only": int(self.diagnostic_cfg.get("ladd_b_det_only", False)),
            "ladd_a2_det_only": int(self.diagnostic_cfg.get("ladd_a2_det_only", False)),
            "effective_alpha_s_rec": float(weights.get("alpha_s_rec", 0.0)),
            "effective_alpha_sep": float(weights.get("alpha_sep", 0.0)),
            "effective_lambda_residual_aux": float(weights.get("lambda_residual_aux", 0.0)),
            "effective_lambda_reach": float(weights.get("lambda_reach", 0.0)),
            "effective_lambda_match_inner": float(weights.get("lambda_match_inner", 0.0)),
            "effective_lambda_rank_inner": float(weights.get("lambda_rank_inner", 0.0)),
        }
        path = self.save_dir / "ladd_diagnostics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        if self.dkd_cfg.get("comparison_kd_profile") == "cmdistill" and epoch_1based == 1:
            LOGGER.info(
                "cmdistill_smoke_stats "
                f"candidate_ratio={float(cmdistill_stats['cmdistill_ibcld_candidate_ratio']):.6f} "
                f"fg_count={int(cmdistill_stats['cmdistill_ibcld_fg_count'])} "
                f"teacher_conf_added={int(cmdistill_stats['cmdistill_ibcld_teacher_conf_added_count'])} "
                f"pcc_loss={float(cmdistill_stats['cmdistill_pcc_loss']):.6f} "
                f"relation_loss={float(cmdistill_stats['cmdistill_relation_loss']):.6f} "
                f"ibcld_loss={float(cmdistill_stats['cmdistill_ibcld_loss']):.6f} "
                f"total={float(cmdistill_stats['cmdistill_total_loss']):.6f}"
            )
