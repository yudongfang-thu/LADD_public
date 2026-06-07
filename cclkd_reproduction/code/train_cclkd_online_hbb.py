#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_ROOT = REPO_ROOT / "shared"
YOLO_ROOT = SHARED_ROOT / "yolo"
for root in (SHARED_ROOT, YOLO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from d2ad_obb.paired_dataset import PairedOBBDataset  # noqa: E402
from train_cli_overrides import add_common_detector_train_overrides, collect_common_detector_train_overrides  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.data import build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.models import yolo  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.nn.tasks import DetectionModel, load_checkpoint  # noqa: E402
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK  # noqa: E402
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402
from ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16, unwrap_model  # noqa: E402


def _parse_pred_dict(preds: Any) -> dict[str, torch.Tensor]:
    if isinstance(preds, tuple):
        preds = preds[1]
    if isinstance(preds, dict) and "one2many" in preds:
        preds = preds["one2many"]
    if not isinstance(preds, dict) or "boxes" not in preds or "scores" not in preds or "feats" not in preds:
        raise TypeError("CCLKD online loss expects YOLO HBB raw prediction dicts with boxes/scores/feats.")
    return preds


def _flatten_feat(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


class CCLKDOnlineReproLoss(nn.Module):
    """Online CCLKD reproduction loss for HBB YOLO11.

    The student receives SAR images. The trainable teacher receives paired RGB
    images, has its own detection loss, and produces KD targets in the same
    forward/backward step. Teacher tensors are detached inside KD terms so the
    teacher is optimized by its RGB detection loss rather than by chasing the
    student.
    """

    def __init__(
        self,
        student_model: DetectionModel,
        teacher_model: DetectionModel,
        teacher_det_weight: float = 1.0,
        kd_weight: float = 1.0,
        lld_weight: float = 1.0,
        fld_weight: float = 1.0,
        rld_weight: float = 1.0,
        ccl_weight: float = 1.0,
        temperature_min: float = 0.5,
        temperature_max: float = 5.0,
        entropy_scale: float = 5.0,
        contrastive_temperature: float = 0.1,
        min_confidence: float = 0.1,
        max_tokens: int = 512,
    ):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_det = student_model.init_criterion()
        self.teacher_det = teacher_model.init_criterion()
        self.assigner_loss = v8DetectionLoss(student_model)
        self.teacher_det_weight = float(teacher_det_weight)
        self.kd_weight = float(kd_weight)
        self.lld_weight = float(lld_weight)
        self.fld_weight = float(fld_weight)
        self.rld_weight = float(rld_weight)
        self.ccl_weight = float(ccl_weight)
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        self.entropy_scale = float(entropy_scale)
        self.contrastive_temperature = float(contrastive_temperature)
        self.min_confidence = float(min_confidence)
        self.max_tokens = int(max_tokens)

    def update(self) -> None:
        for criterion in (self.student_det, self.teacher_det):
            if hasattr(criterion, "update"):
                criterion.update()

    def __call__(self, student_preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if "teacher_img" not in batch:
            raise RuntimeError("CCLKD online reproduction requires paired RGB 'teacher_img' in each batch.")

        teacher_preds = self.teacher_model(batch["teacher_img"])
        student_det_loss, student_items = self.student_det(student_preds, batch)
        teacher_det_loss, teacher_items = self.teacher_det(teacher_preds, batch)

        student_main = _parse_pred_dict(student_preds)
        teacher_main = _parse_pred_dict(teacher_preds)
        (fg_mask, _, _, _, _), _, _ = self.assigner_loss.get_assigned_targets_and_loss(student_main, batch)
        target_scores = self._target_scores(student_main, batch, fg_mask)
        kd_loss = self._cclkd_loss(student_main, teacher_main, fg_mask, target_scores)

        total = student_det_loss + self.teacher_det_weight * teacher_det_loss + self.kd_weight * kd_loss
        items = torch.cat(
            (
                student_items.detach(),
                teacher_items.detach(),
                kd_loss.detach().reshape(1),
            )
        )
        return total, items

    def _target_scores(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = self.assigner_loss.get_assigned_targets_and_loss(preds, batch)[0][3:]
        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=pred_scores.device, dtype=dtype) * self.assigner_loss.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.assigner_loss.preprocess(targets.to(pred_scores.device), pred_scores.shape[0], scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.assigner_loss.bbox_decode(anchor_points, pred_distri)
        _, _, target_scores, _, _ = self.assigner_loss.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        if target_scores.shape[:2] != fg_mask.shape:
            raise RuntimeError("Internal target score shape mismatch in CCLKD online loss.")
        return target_scores

    def _cclkd_loss(
        self,
        student: dict[str, torch.Tensor],
        teacher: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> torch.Tensor:
        student_distri = student["boxes"].permute(0, 2, 1).contiguous()
        teacher_distri = teacher["boxes"].detach().permute(0, 2, 1).contiguous()
        student_scores = student["scores"].permute(0, 2, 1).contiguous()
        teacher_scores = teacher["scores"].detach().permute(0, 2, 1).contiguous()
        if student_distri.shape != teacher_distri.shape or student_scores.shape != teacher_scores.shape:
            raise RuntimeError("CCLKD online reproduction requires same-capacity student/teacher logits.")

        total = student_distri.new_zeros(())
        levels = 0
        offset = 0
        for student_feat, teacher_feat in zip(student["feats"], teacher["feats"]):
            tokens = student_feat.shape[2] * student_feat.shape[3]
            total = total + self._cclkd_level_loss(
                student_feat,
                teacher_feat.detach(),
                fg_mask[:, offset : offset + tokens],
                target_scores[:, offset : offset + tokens],
                student_distri[:, offset : offset + tokens],
                teacher_distri[:, offset : offset + tokens],
                student_scores[:, offset : offset + tokens],
                teacher_scores[:, offset : offset + tokens],
            )
            offset += tokens
            levels += 1
        return total / max(levels, 1)

    def _cclkd_level_loss(
        self,
        student_map: torch.Tensor,
        teacher_map: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores: torch.Tensor,
        student_distri: torch.Tensor,
        teacher_distri: torch.Tensor,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> torch.Tensor:
        zero = student_map.new_zeros(())
        fg = fg_mask.bool()
        valid = fg & (target_scores.amax(dim=-1) > 0)
        if not valid.any():
            return zero

        teacher_probs = teacher_scores.sigmoid()
        teacher_conf, teacher_label = teacher_probs.max(dim=-1)
        target_label = target_scores.argmax(dim=-1)
        cop = valid & teacher_label.eq(target_label) & (teacher_conf >= self.min_confidence)
        if not cop.any():
            return zero

        labels = target_label.reshape(-1)
        valid_flat = valid.reshape(-1)
        cop_flat = cop.reshape(-1)
        teacher_conf_flat = teacher_conf.reshape(-1)
        teacher_probs_flat = teacher_probs.reshape(-1, teacher_probs.shape[-1])
        student_feat_flat = _flatten_feat(student_map).reshape(-1, student_map.shape[1])
        teacher_feat_flat = _flatten_feat(teacher_map).reshape(-1, teacher_map.shape[1])
        student_distri_flat = student_distri.reshape(-1, student_distri.shape[-1])
        teacher_distri_flat = teacher_distri.reshape(-1, teacher_distri.shape[-1])

        classes = labels[cop_flat].unique(sorted=True)
        inv_freq = torch.stack([
            1.0 / (cop_flat & labels.eq(class_id)).sum().clamp_min(1).to(student_map.dtype)
            for class_id in classes
        ])
        class_weights = inv_freq / inv_freq.sum().clamp_min(1e-6)

        lld = zero
        fld = zero
        rld = zero
        ccl = zero
        used = 0
        for class_weight, class_id in zip(class_weights, classes):
            pos_idx = torch.where(cop_flat & labels.eq(class_id))[0]
            if pos_idx.numel() == 0:
                continue
            if pos_idx.numel() > self.max_tokens:
                pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[: self.max_tokens]]

            neg_idx = torch.where(valid_flat & ~labels.eq(class_id))[0]
            if neg_idx.numel() == 0:
                neg_idx = torch.where((~valid_flat) & (teacher_conf_flat >= self.min_confidence))[0]
            if neg_idx.numel() > self.max_tokens:
                neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[: self.max_tokens]]

            temperature = self._adaptive_temperature(teacher_probs_flat[pos_idx, class_id])
            reg_max = student_distri_flat.shape[-1] // 4
            s_box = student_distri_flat[pos_idx].reshape(-1, 4, reg_max)
            t_box = teacher_distri_flat[pos_idx].reshape(-1, 4, reg_max)
            box_lld = F.kl_div(
                F.log_softmax(s_box / temperature, dim=-1),
                F.softmax(t_box / temperature, dim=-1),
                reduction="batchmean",
            ) * temperature.pow(2)
            lld = lld + class_weight * box_lld
            fld = fld + class_weight * F.mse_loss(student_feat_flat[pos_idx], teacher_feat_flat[pos_idx])

            if pos_idx.numel() > 1:
                s_rel = F.normalize(student_feat_flat[pos_idx], dim=-1, eps=1e-6)
                t_rel = F.normalize(teacher_feat_flat[pos_idx], dim=-1, eps=1e-6)
                n_pos = float(pos_idx.numel())
                rld = rld + class_weight * F.mse_loss(s_rel.T @ s_rel / n_pos, t_rel.T @ t_rel / n_pos)

            if neg_idx.numel() > 0:
                if neg_idx.numel() >= pos_idx.numel():
                    sampled_neg = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[: pos_idx.numel()]]
                else:
                    sampled_neg = neg_idx
                min_n = min(pos_idx.numel(), sampled_neg.numel())
                if min_n == 0:
                    used += 1
                    continue
                s_pos = F.normalize(student_feat_flat[pos_idx[:min_n]], dim=-1, eps=1e-6)
                t_pos = F.normalize(teacher_feat_flat[pos_idx[:min_n]], dim=-1, eps=1e-6)
                s_neg = F.normalize(student_feat_flat[sampled_neg[:min_n]], dim=-1, eps=1e-6)
                t_neg = F.normalize(teacher_feat_flat[sampled_neg[:min_n]], dim=-1, eps=1e-6)
                pos_sim = (s_pos * t_pos).sum(dim=-1) / self.contrastive_temperature
                neg_sim = (s_neg * t_neg).sum(dim=-1) / self.contrastive_temperature
                logits = torch.stack((pos_sim, neg_sim), dim=-1)
                ccl = ccl + class_weight * (-F.log_softmax(logits, dim=-1)[:, 0].mean())
            used += 1

        if used == 0:
            return zero
        return self.lld_weight * lld + self.fld_weight * fld + self.rld_weight * rld + self.ccl_weight * ccl

    def _adaptive_temperature(self, class_scores: torch.Tensor) -> torch.Tensor:
        class_scores = class_scores.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(class_scores * class_scores.log() + (1.0 - class_scores) * (1.0 - class_scores).log())
        entropy = (entropy.mean() / math.log(2.0)).clamp(0.0, 1.0)
        temperature = self.temperature_min + (self.temperature_max - self.temperature_min) * torch.sigmoid(
            self.entropy_scale * (entropy - 0.5)
        )
        return temperature.clamp(self.temperature_min, self.temperature_max)


class CCLKDOnlineHBBTrainer(DetectionTrainer):
    """Paper-protocol CCLKD trainer: online RGB teacher + SAR student."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        overrides = {} if overrides is None else dict(overrides)
        self.cclkd_cfg = {
            "teacher_data": overrides.pop("teacher_data", None),
            "teacher_weights": overrides.pop("teacher_weights", None),
            "teacher_det_weight": float(overrides.pop("teacher_det_weight", 1.0)),
            "kd_weight": float(overrides.pop("kd_weight", 1.0)),
            "lld_weight": float(overrides.pop("lld_weight", 1.0)),
            "fld_weight": float(overrides.pop("fld_weight", 1.0)),
            "rld_weight": float(overrides.pop("rld_weight", 1.0)),
            "ccl_weight": float(overrides.pop("ccl_weight", 1.0)),
            "temperature_min": float(overrides.pop("cclkd_temperature_min", 0.5)),
            "temperature_max": float(overrides.pop("cclkd_temperature_max", 5.0)),
            "entropy_scale": float(overrides.pop("cclkd_entropy_scale", 5.0)),
            "contrastive_temperature": float(overrides.pop("cclkd_contrastive_temperature", 0.1)),
            "min_confidence": float(overrides.pop("cclkd_min_confidence", 0.1)),
            "max_tokens": int(overrides.pop("cclkd_max_tokens", 512)),
        }
        if self.cclkd_cfg["teacher_data"] is None or self.cclkd_cfg["teacher_weights"] is None:
            raise ValueError("CCLKDOnlineHBBTrainer requires teacher_data and teacher_weights.")
        overrides["task"] = "detect"
        super().__init__(cfg, overrides, _callbacks)
        if self.world_size > 1:
            raise RuntimeError("Use one CCLKD reproduction process per GPU; this trainer does not wrap teacher DDP.")
        self.teacher_data = check_det_dataset(str(self.cclkd_cfg["teacher_data"]))
        self.teacher_model: DetectionModel | None = None

    def setup_model(self):
        ckpt = super().setup_model()
        teacher_weights, _ = load_checkpoint(str(self.cclkd_cfg["teacher_weights"]), device=self.device)
        self.teacher_model = DetectionModel(
            cfg=teacher_weights.yaml,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=RANK == -1,
        )
        self.teacher_model.load(teacher_weights)
        self.teacher_model = self.teacher_model.to(self.device).float()
        self.teacher_model.nc = self.data["nc"]
        self.teacher_model.names = self.data["names"]
        self.teacher_model.args = self.args
        for p in self.teacher_model.parameters():
            p.requires_grad_(True)
        return ckpt

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        return PairedOBBDataset(
            img_path=img_path,
            teacher_img_path=self.teacher_data["train"],
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True,
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
        batch = super().preprocess_batch(batch)
        batch["teacher_img"] = batch["teacher_img"].to(self.device, non_blocking=self.device.type == "cuda").float() / 255
        return batch

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        if self.teacher_model is None:
            return super().build_optimizer(model, name=name, lr=lr, momentum=momentum, decay=decay, iterations=iterations)
        joint = nn.ModuleList([unwrap_model(model), self.teacher_model])
        return super().build_optimizer(joint, name=name, lr=lr, momentum=momentum, decay=decay, iterations=iterations)

    def _setup_train(self):
        super()._setup_train()
        student = unwrap_model(self.model)
        assert self.teacher_model is not None
        student.criterion = CCLKDOnlineReproLoss(student, self.teacher_model, **self.cclkd_cfg_without_paths())
        if self.ema:
            self.ema.ema.criterion = student.init_criterion()
        LOGGER.info("CCLKD online reproduction: teacher is trainable and optimized with RGB detection loss.")

    def cclkd_cfg_without_paths(self) -> dict[str, float | int]:
        return {k: v for k, v in self.cclkd_cfg.items() if k not in {"teacher_data", "teacher_weights"}}

    def _model_train(self):
        super()._model_train()
        if self.teacher_model is not None:
            self.teacher_model.train()

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        if self.teacher_model is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            torch.nn.utils.clip_grad_norm_(self.teacher_model.parameters(), max_norm=10.0)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def get_validator(self):
        self.loss_names = (
            "s_box_loss",
            "s_cls_loss",
            "s_dfl_loss",
            "t_box_loss",
            "t_cls_loss",
            "t_dfl_loss",
            "cclkd_loss",
        )
        return yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=deepcopy(self.args),
            _callbacks=self.callbacks,
        )

    def validate(self):
        """Validate the SAR student as a detector without accumulating online loss.

        The online CCLKD training criterion returns seven loss items
        (student/teacher detection losses plus KD). Ultralytics' in-training
        validator expects the validation loss vector to match detector-only
        losses, so we run validation in inference mode and keep mAP/fitness as
        the monitored quantity. This does not change training or reported
        detection metrics.
        """
        # Standalone validation may fuse or otherwise mutate the model it
        # receives. Validate a copy so EMA state remains update-compatible with
        # the live student on the next training epoch.
        model = deepcopy(self.ema.ema if self.ema else unwrap_model(self.model))
        metrics = self.validator(model=model)
        if metrics is None:
            return None, None
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())
        metrics.update({key: float("nan") for key in self.label_loss_items(prefix="val")})
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def save_model(self):
        super().save_model()
        if self.teacher_model is None or RANK not in {-1, 0}:
            return
        teacher_ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(unwrap_model(self.teacher_model)).half(),
            "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            "train_args": vars(self.args),
        }
        teacher_last = self.wdir / "teacher_last.pt"
        torch.save(teacher_ckpt, teacher_last)
        if self.best_fitness == self.fitness:
            torch.save(teacher_ckpt, self.wdir / "teacher_best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CCLKD paper-protocol online HBB reproduction trainer.")
    parser.add_argument("--model-size", choices=("n", "s"), required=True)
    parser.add_argument("--model", required=True, help="Student YOLO11n/s COCO pretrained checkpoint.")
    parser.add_argument("--teacher-weights", required=True, help="Teacher YOLO11n/s COCO pretrained checkpoint.")
    parser.add_argument("--data", type=Path, required=True, help="SAR OGSOD HBB YAML, nc=3.")
    parser.add_argument("--teacher-data", type=Path, required=True, help="RGB OGSOD HBB YAML, nc=3.")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cache", default=False)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs_public" / "ogsod" / "hbb" / "cclkd_reproduction")
    parser.add_argument("--name", required=True)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--teacher-det-weight", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--lld-weight", type=float, default=1.0)
    parser.add_argument("--fld-weight", type=float, default=1.0)
    parser.add_argument("--rld-weight", type=float, default=1.0)
    parser.add_argument("--ccl-weight", type=float, default=1.0)
    parser.add_argument("--cclkd-temperature-min", type=float, default=0.5)
    parser.add_argument("--cclkd-temperature-max", type=float, default=5.0)
    parser.add_argument("--cclkd-entropy-scale", type=float, default=5.0)
    parser.add_argument("--cclkd-contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--cclkd-min-confidence", type=float, default=0.1)
    parser.add_argument("--cclkd-max-tokens", type=int, default=512)
    add_common_detector_train_overrides(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = f"yolo11{args.model_size}.pt"
    for role, value in (("student", args.model), ("teacher", args.teacher_weights)):
        if Path(value).name != expected:
            raise SystemExit(f"{role} weights must be {expected} for CCLKD paper reproduction, got {value}.")

    model = YOLO(args.model)
    train_kwargs = dict(
        trainer=CCLKDOnlineHBBTrainer,
        data=str(args.data.resolve()),
        teacher_data=str(args.teacher_data.resolve()),
        teacher_weights=str(Path(args.teacher_weights).resolve()),
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
        teacher_det_weight=args.teacher_det_weight,
        kd_weight=args.kd_weight,
        lld_weight=args.lld_weight,
        fld_weight=args.fld_weight,
        rld_weight=args.rld_weight,
        ccl_weight=args.ccl_weight,
        cclkd_temperature_min=args.cclkd_temperature_min,
        cclkd_temperature_max=args.cclkd_temperature_max,
        cclkd_entropy_scale=args.cclkd_entropy_scale,
        cclkd_contrastive_temperature=args.cclkd_contrastive_temperature,
        cclkd_min_confidence=args.cclkd_min_confidence,
        cclkd_max_tokens=args.cclkd_max_tokens,
    )
    train_kwargs.update(collect_common_detector_train_overrides(args))
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
