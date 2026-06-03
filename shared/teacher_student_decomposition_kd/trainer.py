from __future__ import annotations

from copy import copy, deepcopy
import json
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models import yolo
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

from d2ad_obb.aug_policy import apply_unified_paired_aug_policy
from d2ad_obb.paired_dataset import PairedOBBDataset

from .loss import TeacherStudentDecompositionKDLoss
from .model import TeacherStudentDecompositionKDModel


class TeacherStudentDecompositionKDTrainer(OBBTrainer):
    """Trainer for teacher learnable/unlearnable + student mimic/residual KD."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        overrides = {} if overrides is None else dict(overrides)
        self.tskd_cfg = {
            "teacher_data": overrides.pop("teacher_data", None),
            "teacher_weights": overrides.pop("teacher_weights", "yolo11n-obb.pt"),
            "use_mask": bool(overrides.pop("use_mask", False)),
            "fusion_mode": overrides.pop("fusion_mode", "sum"),
            "student_detect_mode": overrides.pop("student_detect_mode", "fused"),
            "match_target_mode": overrides.pop("match_target_mode", "detach"),
            "kd_target_mode": overrides.pop("kd_target_mode", "detach"),
            "lambda_rec": float(overrides.pop("lambda_rec", 0.1)),
            "lambda_sep": float(overrides.pop("lambda_sep", 0.05)),
            "lambda_match": float(overrides.pop("lambda_match", 1.0)),
            "lambda_unmatch": float(overrides.pop("lambda_unmatch", 0.5)),
            "lambda_taskL": float(overrides.pop("lambda_taskL", 1.0)),
            "task_loss_fg_only": bool(overrides.pop("task_loss_fg_only", False)),
            "alpha_kd": float(overrides.pop("alpha_kd", 1.0)),
            "alpha_s_rec": float(overrides.pop("alpha_s_rec", 0.1)),
            "alpha_sep": float(overrides.pop("alpha_sep", 0.05)),
            "lambda_mask_sparse": float(overrides.pop("lambda_mask_sparse", 0.0)),
            "lambda_mask_smooth": float(overrides.pop("lambda_mask_smooth", 0.0)),
            "margin": float(overrides.pop("margin", 3.0)),
        }
        self.diagnostic_cfg = {
            "validate_before_train": bool(overrides.pop("validate_before_train", False)),
        }
        if self.tskd_cfg["teacher_data"] is None:
            raise ValueError("TeacherStudentDecompositionKDTrainer requires 'teacher_data'.")
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)
        self.teacher_data = check_det_dataset(str(self.tskd_cfg["teacher_data"]))
        self.teacher_model = None

        apply_unified_paired_aug_policy(self.args)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        teacher_path = self.teacher_data["train"]
        return PairedOBBDataset(
            img_path=img_path,
            teacher_img_path=teacher_path,
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

    def get_model(self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True):
        model = TeacherStudentDecompositionKDModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            use_mask=self.tskd_cfg["use_mask"],
            fusion_mode=self.tskd_cfg["fusion_mode"],
            student_detect_mode=self.tskd_cfg["student_detect_mode"],
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
            "angle_loss",
            "t_rec_loss",
            "t_sep_loss",
            "match_loss",
            "unmatch_loss",
            "task_loss",
            "kd_loss",
            "s_rec_loss",
            "s_sep_loss",
            "mask_reg_loss",
            "unmatch_active_ratio",
        )
        return yolo.obb.OBBValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        teacher_img = batch.get("teacher_img")
        if teacher_img is not None:
            batch["teacher_img"] = teacher_img.to(self.device, non_blocking=self.device.type == "cuda").float() / 255
        return batch

    def _setup_train(self):
        super()._setup_train()
        self.teacher_model = self._load_teacher_model()
        student_model = unwrap_model(self.model)
        student_model.criterion = TeacherStudentDecompositionKDLoss(
            student_model,
            teacher_model=self.teacher_model,
            lambda_rec=self.tskd_cfg["lambda_rec"],
            lambda_sep=self.tskd_cfg["lambda_sep"],
            lambda_match=self.tskd_cfg["lambda_match"],
            lambda_unmatch=self.tskd_cfg["lambda_unmatch"],
            lambda_taskL=self.tskd_cfg["lambda_taskL"],
            alpha_kd=self.tskd_cfg["alpha_kd"],
            alpha_s_rec=self.tskd_cfg["alpha_s_rec"],
            alpha_sep=self.tskd_cfg["alpha_sep"],
            lambda_mask_sparse=self.tskd_cfg["lambda_mask_sparse"],
            lambda_mask_smooth=self.tskd_cfg["lambda_mask_smooth"],
            margin=self.tskd_cfg["margin"],
            match_target_mode=self.tskd_cfg["match_target_mode"],
            kd_target_mode=self.tskd_cfg["kd_target_mode"],
        )
        if self.ema:
            self.ema.ema.criterion = TeacherStudentDecompositionKDLoss(
                self.ema.ema,
                teacher_model=None,
                lambda_rec=self.tskd_cfg["lambda_rec"],
                lambda_sep=self.tskd_cfg["lambda_sep"],
                lambda_match=self.tskd_cfg["lambda_match"],
                lambda_unmatch=self.tskd_cfg["lambda_unmatch"],
                lambda_taskL=self.tskd_cfg["lambda_taskL"],
                alpha_kd=self.tskd_cfg["alpha_kd"],
                alpha_s_rec=self.tskd_cfg["alpha_s_rec"],
                alpha_sep=self.tskd_cfg["alpha_sep"],
                lambda_mask_sparse=self.tskd_cfg["lambda_mask_sparse"],
                lambda_mask_smooth=self.tskd_cfg["lambda_mask_smooth"],
                margin=self.tskd_cfg["margin"],
                match_target_mode=self.tskd_cfg["match_target_mode"],
                kd_target_mode=self.tskd_cfg["kd_target_mode"],
            )

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
            return {str(k): TeacherStudentDecompositionKDTrainer._jsonify_metric_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [TeacherStudentDecompositionKDTrainer._jsonify_metric_value(v) for v in value]
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
        # AutoBackend(validation) may fuse Conv+BN in-place for nn.Module inputs. Validate a deep copy so EMA/model
        # state_dict keys remain unchanged for subsequent optimizer/EMA steps.
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


class StagedTeacherStudentDecompositionKDTrainer(TeacherStudentDecompositionKDTrainer):
    """Three-phase variant: TDN pretrain -> student train -> joint finetune."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        overrides = {} if overrides is None else dict(overrides)
        self.stage_cfg = {
            "phase_a_epochs": int(overrides.pop("phase_a_epochs", 0)),
            "phase_b_epochs": int(overrides.pop("phase_b_epochs", 0)),
            "phase_c_epochs": overrides.pop("phase_c_epochs", None),
            "joint_match_target_mode": overrides.pop("joint_match_target_mode", "coupled"),
            "joint_kd_target_mode": overrides.pop("joint_kd_target_mode", "coupled"),
        }
        super().__init__(cfg, overrides, _callbacks)
        self.current_phase: str | None = None
        self._validate_stage_schedule()
        self.add_callback("on_train_epoch_start", self._on_train_epoch_start)

    def _validate_stage_schedule(self) -> None:
        phase_a = self.stage_cfg["phase_a_epochs"]
        phase_b = self.stage_cfg["phase_b_epochs"]
        phase_c = self.stage_cfg["phase_c_epochs"]
        total = self.epochs
        if phase_a < 0 or phase_b < 0 or (phase_c is not None and int(phase_c) < 0):
            raise ValueError("Phase epoch counts must be non-negative.")
        if phase_c is None:
            phase_c = total - phase_a - phase_b
        else:
            phase_c = int(phase_c)
        if phase_a + phase_b + phase_c != total:
            raise ValueError(
                f"Staged schedule must sum to total epochs={total}, got "
                f"{phase_a}+{phase_b}+{phase_c}={phase_a + phase_b + phase_c}."
            )
        self.stage_cfg["phase_c_epochs"] = phase_c

    def _phase_for_epoch(self, epoch: int) -> str:
        phase_a = self.stage_cfg["phase_a_epochs"]
        phase_b = self.stage_cfg["phase_b_epochs"]
        if epoch < phase_a:
            return "tdn"
        if epoch < phase_a + phase_b:
            return "student"
        return "joint"

    @staticmethod
    def _set_module_requires_grad(module, requires_grad: bool) -> None:
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def _set_phase_loss_scales(self, **scales: float) -> None:
        model = unwrap_model(self.model)
        if getattr(model, "criterion", None) is not None:
            model.criterion.set_phase_loss_scales(**scales)
        if self.ema and getattr(self.ema.ema, "criterion", None) is not None:
            self.ema.ema.criterion.set_phase_loss_scales(**scales)

    def _set_phase_target_modes(self, *, match_target_mode: str, kd_target_mode: str) -> None:
        model = unwrap_model(self.model)
        if getattr(model, "criterion", None) is not None:
            model.criterion.set_target_modes(
                match_target_mode=match_target_mode,
                kd_target_mode=kd_target_mode,
            )
        if self.ema and getattr(self.ema.ema, "criterion", None) is not None:
            self.ema.ema.criterion.set_target_modes(
                match_target_mode=match_target_mode,
                kd_target_mode=kd_target_mode,
            )

    def _apply_training_phase(self, phase: str, announce: bool = True) -> None:
        model = unwrap_model(self.model)
        epoch = getattr(self, "epoch", self.start_epoch)

        if phase == "tdn":
            self._set_module_requires_grad(model.model, False)
            self._set_module_requires_grad(model.student_split, False)
            self._set_module_requires_grad(model.teacher_decomposition, True)
            self._set_module_requires_grad(model.student_reachability, True)
            self._set_module_requires_grad(model.teacher_task_heads, True)
            self._set_phase_target_modes(match_target_mode="coupled", kd_target_mode="detach")
            self._set_phase_loss_scales(
                rec=1.0,
                teacher_sep=1.0,
                match=1.0,
                unmatch=1.0,
                task=1.0,
                kd=0.0,
                student_rec=0.0,
                student_sep=0.0,
                mask=1.0,
            )
        elif phase == "student":
            self._set_module_requires_grad(model.model, True)
            self._set_module_requires_grad(model.student_split, True)
            self._set_module_requires_grad(model.teacher_decomposition, False)
            self._set_module_requires_grad(model.student_reachability, True)
            self._set_module_requires_grad(model.teacher_task_heads, False)
            self._set_phase_target_modes(match_target_mode="detach", kd_target_mode="detach")
            self._set_phase_loss_scales(
                rec=0.0,
                teacher_sep=0.0,
                match=1.0,
                unmatch=1.0,
                task=0.0,
                kd=1.0,
                student_rec=1.0,
                student_sep=1.0,
                mask=0.0,
            )
        elif phase == "joint":
            self._set_module_requires_grad(model.model, True)
            self._set_module_requires_grad(model.student_split, True)
            self._set_module_requires_grad(model.teacher_decomposition, True)
            self._set_module_requires_grad(model.student_reachability, True)
            self._set_module_requires_grad(model.teacher_task_heads, True)
            self._set_phase_target_modes(
                match_target_mode=self.stage_cfg["joint_match_target_mode"],
                kd_target_mode=self.stage_cfg["joint_kd_target_mode"],
            )
            self._set_phase_loss_scales(
                rec=1.0,
                teacher_sep=1.0,
                match=1.0,
                unmatch=1.0,
                task=1.0,
                kd=1.0,
                student_rec=1.0,
                student_sep=1.0,
                mask=1.0,
            )
        else:
            raise ValueError(f"Unsupported phase: {phase}")

        self.current_phase = phase
        if announce and RANK in {-1, 0}:
            LOGGER.info(f"Stage schedule switch: epoch {epoch + 1}/{self.epochs} -> phase '{phase}'")

    def _on_train_epoch_start(self, trainer) -> None:
        phase = self._phase_for_epoch(trainer.epoch)
        if phase != self.current_phase:
            self._apply_training_phase(phase, announce=True)

    def _setup_train(self):
        super()._setup_train()
        self._apply_training_phase(self._phase_for_epoch(self.start_epoch), announce=True)

    def _model_train(self):
        super()._model_train()
        model = unwrap_model(self.model)
        if self.current_phase == "tdn":
            model.model.eval()
            model.student_split.eval()
        elif self.current_phase == "student":
            model.teacher_decomposition.eval()
            model.teacher_task_heads.eval()


class ManualPhaseTeacherStudentDecompositionKDTrainer(TeacherStudentDecompositionKDTrainer):
    """Single-phase manual trainer for A1/A2/B/C workflows."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        overrides = {} if overrides is None else dict(overrides)
        phase = str(overrides.pop("phase", "")).lower()
        if phase not in {"a1", "a2", "b", "c"}:
            raise ValueError("ManualPhaseTeacherStudentDecompositionKDTrainer requires phase in {'a1','a2','b','c'}.")
        self.manual_phase_cfg = {
            "phase": phase,
            "phase_detect_mode": overrides.pop("phase_detect_mode", None),
            "det_loss_scale": overrides.pop("det_loss_scale", None),
        }
        super().__init__(cfg, overrides, _callbacks)
        self.current_phase: str | None = None

    def _resolve_phase_detect_mode(self) -> str:
        requested = self.manual_phase_cfg["phase_detect_mode"]
        if requested is not None:
            if requested not in {"raw", "fused", "mimic", "recon"}:
                raise ValueError(f"Unsupported phase_detect_mode: {requested}")
            return requested
        return "raw" if self.manual_phase_cfg["phase"] in {"a1", "a2"} else self.tskd_cfg["student_detect_mode"]

    def _resolve_det_loss_scale(self) -> float:
        det_scale = self.manual_phase_cfg["det_loss_scale"]
        if det_scale is not None:
            return float(det_scale)
        return 0.0 if self.manual_phase_cfg["phase"] == "a1" else 1.0

    @staticmethod
    def _set_module_requires_grad(module, requires_grad: bool) -> None:
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def _set_phase_loss_scales(self, **scales: float) -> None:
        model = unwrap_model(self.model)
        if getattr(model, "criterion", None) is not None:
            model.criterion.set_phase_loss_scales(**scales)
        if self.ema and getattr(self.ema.ema, "criterion", None) is not None:
            self.ema.ema.criterion.set_phase_loss_scales(**scales)

    def _set_phase_target_modes(self, *, match_target_mode: str, kd_target_mode: str) -> None:
        model = unwrap_model(self.model)
        if getattr(model, "criterion", None) is not None:
            model.criterion.set_target_modes(
                match_target_mode=match_target_mode,
                kd_target_mode=kd_target_mode,
            )
        if self.ema and getattr(self.ema.ema, "criterion", None) is not None:
            self.ema.ema.criterion.set_target_modes(
                match_target_mode=match_target_mode,
                kd_target_mode=kd_target_mode,
            )

    def _set_detect_mode(self, detect_mode: str) -> None:
        model = unwrap_model(self.model)
        model.student_detect_mode = detect_mode
        if self.ema and hasattr(self.ema, "ema"):
            self.ema.ema.student_detect_mode = detect_mode

    def _apply_manual_phase(self, announce: bool = True) -> None:
        model = unwrap_model(self.model)
        phase = self.manual_phase_cfg["phase"]
        detect_mode = self._resolve_phase_detect_mode()
        det_scale = self._resolve_det_loss_scale()
        self._set_detect_mode(detect_mode)

        if phase == "a1":
            self._set_module_requires_grad(model.model, False)
            self._set_module_requires_grad(model.student_split, False)
            self._set_module_requires_grad(model.teacher_decomposition, True)
            self._set_module_requires_grad(model.student_reachability, True)
            self._set_module_requires_grad(model.teacher_task_heads, True)
            self._set_phase_target_modes(match_target_mode="coupled", kd_target_mode="detach")
            self._set_phase_loss_scales(
                det=det_scale,
                rec=1.0,
                teacher_sep=1.0,
                match=1.0,
                unmatch=1.0,
                task=1.0,
                kd=0.0,
                student_rec=0.0,
                student_sep=0.0,
                mask=1.0,
            )
        elif phase == "a2":
            self._set_module_requires_grad(model.model, True)
            self._set_module_requires_grad(model.student_split, False)
            self._set_module_requires_grad(model.teacher_decomposition, True)
            self._set_module_requires_grad(model.student_reachability, True)
            self._set_module_requires_grad(model.teacher_task_heads, True)
            self._set_phase_target_modes(match_target_mode="coupled", kd_target_mode="detach")
            self._set_phase_loss_scales(
                det=det_scale,
                rec=1.0,
                teacher_sep=1.0,
                match=1.0,
                unmatch=1.0,
                task=1.0,
                kd=0.0,
                student_rec=0.0,
                student_sep=0.0,
                mask=1.0,
            )
        elif phase == "b":
            self._set_module_requires_grad(model.model, True)
            self._set_module_requires_grad(model.student_split, True)
            self._set_module_requires_grad(model.teacher_decomposition, False)
            self._set_module_requires_grad(model.student_reachability, True)
            self._set_module_requires_grad(model.teacher_task_heads, False)
            self._set_phase_target_modes(match_target_mode="detach", kd_target_mode="detach")
            self._set_phase_loss_scales(
                det=det_scale,
                rec=0.0,
                teacher_sep=0.0,
                match=1.0,
                unmatch=1.0,
                task=0.0,
                kd=1.0,
                student_rec=1.0,
                student_sep=1.0,
                mask=0.0,
            )
        elif phase == "c":
            self._set_module_requires_grad(model.model, True)
            self._set_module_requires_grad(model.student_split, True)
            self._set_module_requires_grad(model.teacher_decomposition, True)
            self._set_module_requires_grad(model.student_reachability, True)
            self._set_module_requires_grad(model.teacher_task_heads, True)
            self._set_phase_target_modes(
                match_target_mode=self.tskd_cfg["match_target_mode"],
                kd_target_mode=self.tskd_cfg["kd_target_mode"],
            )
            self._set_phase_loss_scales(
                det=det_scale,
                rec=1.0,
                teacher_sep=1.0,
                match=1.0,
                unmatch=1.0,
                task=1.0,
                kd=1.0,
                student_rec=1.0,
                student_sep=1.0,
                mask=1.0,
            )
        else:
            raise ValueError(f"Unsupported manual phase: {phase}")

        self.current_phase = phase
        if announce and RANK in {-1, 0}:
            LOGGER.info(
                f"Manual TSKD phase '{phase}' configured with detect_mode='{detect_mode}' and det_loss_scale={det_scale:.3f}"
            )

    def _setup_train(self):
        super()._setup_train()
        self._apply_manual_phase(announce=True)

    def _model_train(self):
        super()._model_train()
        model = unwrap_model(self.model)
        if self.current_phase == "a1":
            model.model.eval()
            model.student_split.eval()
        elif self.current_phase == "a2":
            model.student_split.eval()
        elif self.current_phase == "b":
            model.teacher_decomposition.eval()
            model.teacher_task_heads.eval()
