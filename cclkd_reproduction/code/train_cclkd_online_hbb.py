#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from ultralytics.data import build_dataloader, build_yolo_dataset  # noqa: E402
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


def _detach_pred_dict(preds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    out: dict[str, torch.Tensor | list[torch.Tensor]] = {}
    for key, value in preds.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach()
        elif isinstance(value, list):
            out[key] = [v.detach() if isinstance(v, torch.Tensor) else v for v in value]
        else:
            out[key] = value
    return out


def _cclkd_pretrain_path(model_size: str) -> Path:
    if model_size not in {"n", "s"}:
        raise ValueError(f"CCLKD online reproduction only supports YOLO11n/s, got {model_size!r}.")
    return REPO_ROOT / f"yolo11{model_size}.pt"


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
        fld_temperature: float = 1.0,
        fld_temperature_mode: str = "patm",
        min_confidence: float = 0.1,
        max_tokens: int = 512,
        formulation: str = "adapted",
        ccl_mode: str = "paper_pair",
        ccl_source: str = "box_distribution",
        rld_mode: str = "paper_instance",
        roi_grid_size: int = 3,
    ):
        super().__init__()
        if formulation not in {"adapted", "paper"}:
            raise ValueError(f"formulation must be 'adapted' or 'paper', got {formulation!r}.")
        if ccl_mode not in {"paper_pair", "anchor_teacher_neg"}:
            raise ValueError(
                "ccl_mode must be 'paper_pair' or 'anchor_teacher_neg', "
                f"got {ccl_mode!r}."
            )
        if ccl_source not in {"box_distribution", "roi_feature"}:
            raise ValueError(
                "ccl_source must be 'box_distribution' or 'roi_feature', "
                f"got {ccl_source!r}."
            )
        if rld_mode not in {"paper_instance", "channel"}:
            raise ValueError(
                "rld_mode must be 'paper_instance' or 'channel', "
                f"got {rld_mode!r}."
            )
        if fld_temperature <= 0:
            raise ValueError(f"fld_temperature must be positive, got {fld_temperature}.")
        if fld_temperature_mode not in {"fixed", "patm"}:
            raise ValueError(f"fld_temperature_mode must be 'fixed' or 'patm', got {fld_temperature_mode!r}.")
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
        self.fld_temperature = float(fld_temperature)
        self.fld_temperature_mode = fld_temperature_mode
        self.min_confidence = float(min_confidence)
        self.max_tokens = int(max_tokens)
        self.formulation = formulation
        self.ccl_mode = ccl_mode
        self.ccl_source = ccl_source
        self.rld_mode = rld_mode
        self.roi_grid_size = int(roi_grid_size)
        self._kd_component_levels: list[torch.Tensor] | None = None
        self._diag_accumulator = self._new_diag_accumulator()

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
        assign_source = teacher_main if self.formulation == "paper" else student_main
        fg_mask, target_scores, target_bboxes = self._assigned_targets(assign_source, batch)
        self._kd_component_levels = []
        kd_loss = self._cclkd_loss(student_main, teacher_main, fg_mask, target_scores, target_bboxes)
        if self._kd_component_levels:
            kd_components = torch.stack(self._kd_component_levels).mean(dim=0)
        else:
            kd_components = kd_loss.detach().new_zeros(4)
        self._kd_component_levels = None
        self._last_batch_diagnostics = {
            "student_box_loss": float(student_items.detach().flatten()[0].cpu()) if student_items.numel() > 0 else math.nan,
            "student_cls_loss": float(student_items.detach().flatten()[1].cpu()) if student_items.numel() > 1 else math.nan,
            "student_dfl_loss": float(student_items.detach().flatten()[2].cpu()) if student_items.numel() > 2 else math.nan,
            "teacher_box_loss": float(teacher_items.detach().flatten()[0].cpu()) if teacher_items.numel() > 0 else math.nan,
            "teacher_cls_loss": float(teacher_items.detach().flatten()[1].cpu()) if teacher_items.numel() > 1 else math.nan,
            "teacher_dfl_loss": float(teacher_items.detach().flatten()[2].cpu()) if teacher_items.numel() > 2 else math.nan,
            "cclkd_loss": float(kd_loss.detach().cpu()),
            "cclkd_lld_loss": float(kd_components.detach().flatten()[0].cpu()) if kd_components.numel() > 0 else math.nan,
            "cclkd_fld_loss": float(kd_components.detach().flatten()[1].cpu()) if kd_components.numel() > 1 else math.nan,
            "cclkd_rld_loss": float(kd_components.detach().flatten()[2].cpu()) if kd_components.numel() > 2 else math.nan,
            "cclkd_ccl_loss": float(kd_components.detach().flatten()[3].cpu()) if kd_components.numel() > 3 else math.nan,
        }

        total = student_det_loss + self.teacher_det_weight * teacher_det_loss + self.kd_weight * kd_loss
        items = torch.cat(
            (
                student_items.detach(),
                teacher_items.detach(),
                kd_loss.detach().reshape(1),
                kd_components.detach(),
            )
        )
        return total, items

    def _assigned_targets(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preds = _detach_pred_dict(preds)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        from ultralytics.utils.tal import make_anchors

        anchor_points, stride_tensor = make_anchors(preds["feats"], self.assigner_loss.stride, 0.5)
        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=pred_scores.device, dtype=dtype) * self.assigner_loss.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.assigner_loss.preprocess(targets.to(pred_scores.device), pred_scores.shape[0], scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.assigner_loss.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner_loss.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        return fg_mask, target_scores, target_bboxes

    def _cclkd_loss(
        self,
        student: dict[str, torch.Tensor],
        teacher: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        target_scores: torch.Tensor,
        target_bboxes: torch.Tensor,
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
            level_stride = float(self.assigner_loss.stride[levels])
            total = total + self._cclkd_level_loss(
                student_feat,
                teacher_feat.detach(),
                level_stride,
                fg_mask[:, offset : offset + tokens],
                target_scores[:, offset : offset + tokens],
                student_distri[:, offset : offset + tokens],
                teacher_distri[:, offset : offset + tokens],
                student_scores[:, offset : offset + tokens],
                teacher_scores[:, offset : offset + tokens],
                target_bboxes[:, offset : offset + tokens],
            )
            offset += tokens
            levels += 1
        return total / max(levels, 1)

    def _cclkd_level_loss(
        self,
        student_map: torch.Tensor,
        teacher_map: torch.Tensor,
        level_stride: float,
        fg_mask: torch.Tensor,
        target_scores: torch.Tensor,
        student_distri: torch.Tensor,
        teacher_distri: torch.Tensor,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        target_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        zero = student_map.new_zeros(())
        fg = fg_mask.bool()
        valid = fg & (target_scores.amax(dim=-1) > 0)
        target_label = target_scores.argmax(dim=-1)
        if not valid.any():
            self._record_cop_diagnostics(valid, valid & False, target_label, [], [])
            self._record_kd_components(zero, zero, zero, zero)
            return zero

        teacher_probs = teacher_scores.sigmoid()
        teacher_conf, teacher_label = teacher_probs.max(dim=-1)
        cop = valid & teacher_label.eq(target_label) & (teacher_conf >= self.min_confidence)
        if not cop.any():
            self._record_cop_diagnostics(valid, cop, target_label, [], [])
            self._record_kd_components(zero, zero, zero, zero)
            return zero

        labels = target_label.reshape(-1)
        valid_flat = valid.reshape(-1)
        cop_flat = cop.reshape(-1)
        teacher_conf_flat = teacher_conf.reshape(-1)
        teacher_probs_flat = teacher_probs.reshape(-1, teacher_probs.shape[-1])
        target_bboxes_flat = target_bboxes.reshape(-1, target_bboxes.shape[-1])
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
        neg_token_counts: list[int] = []
        temperatures: list[float] = []
        for class_weight, class_id in zip(class_weights, classes):
            pos_idx = torch.where(cop_flat & labels.eq(class_id))[0]
            if pos_idx.numel() == 0:
                continue
            if pos_idx.numel() > self.max_tokens:
                pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[: self.max_tokens]]

            neg_idx = torch.where(valid_flat & ~labels.eq(class_id))[0]
            if neg_idx.numel() == 0 and self.formulation != "paper":
                neg_idx = torch.where((~valid_flat) & (teacher_conf_flat >= self.min_confidence))[0]
            if neg_idx.numel() > self.max_tokens:
                neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[: self.max_tokens]]
            neg_token_counts.append(int(neg_idx.numel()))

            if self.formulation == "paper":
                temperature = self._adaptive_temperature_class(teacher_probs_flat[pos_idx, class_id])
            else:
                temperature = self._adaptive_temperature(teacher_probs_flat[pos_idx, class_id])
            temperatures.extend([float(x) for x in temperature.detach().reshape(-1).float().cpu().tolist()])
            reg_max = student_distri_flat.shape[-1] // 4
            s_box = student_distri_flat[pos_idx].reshape(-1, 4, reg_max)
            t_box = teacher_distri_flat[pos_idx].reshape(-1, 4, reg_max)
            if self.formulation == "paper":
                box_lld = self._dfl_kl(s_box, t_box, temperature)
                if neg_idx.numel() > 0:
                    neg_for_lld = neg_idx
                    if neg_for_lld.numel() > pos_idx.numel():
                        neg_for_lld = neg_for_lld[
                            torch.randperm(neg_for_lld.numel(), device=neg_for_lld.device)[: pos_idx.numel()]
                        ]
                    s_neg_box = student_distri_flat[neg_for_lld].reshape(-1, 4, reg_max)
                    t_neg_box = teacher_distri_flat[neg_for_lld].reshape(-1, 4, reg_max)
                    box_lld = box_lld + self._dfl_kl(s_neg_box, t_neg_box, temperature)
            else:
                box_lld = F.kl_div(
                    F.log_softmax(s_box / temperature.view(-1, 1, 1), dim=-1),
                    F.softmax(t_box / temperature.view(-1, 1, 1), dim=-1),
                    reduction="none",
                ).sum(dim=(-1, -2))
                box_lld = (box_lld * temperature.pow(2)).mean()
            lld = lld + class_weight * box_lld

            if self.formulation == "paper":
                pos_boxes = target_bboxes_flat[pos_idx]
                s_pos_feat = self._sample_box_features(student_map, pos_idx, pos_boxes, level_stride)
                t_pos_feat = self._sample_box_features(teacher_map, pos_idx, pos_boxes, level_stride)
                fld_temperature = (
                    temperature
                    if self.fld_temperature_mode == "patm"
                    else student_map.new_tensor(self.fld_temperature)
                )
                fld = fld + class_weight * self._feature_kl(s_pos_feat, t_pos_feat, fld_temperature)
            else:
                s_pos_feat = student_feat_flat[pos_idx]
                t_pos_feat = teacher_feat_flat[pos_idx]
                fld = fld + class_weight * F.mse_loss(s_pos_feat, t_pos_feat)

            if pos_idx.numel() > 1:
                rld_temperature = temperature if self.formulation == "paper" else student_map.new_tensor(1.0)
                rld = rld + class_weight * self._relationship_loss(s_pos_feat, t_pos_feat, rld_temperature)

            if neg_idx.numel() > 0:
                if neg_idx.numel() >= pos_idx.numel():
                    sampled_neg = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[: pos_idx.numel()]]
                else:
                    sampled_neg = neg_idx
                min_n = min(pos_idx.numel(), sampled_neg.numel())
                if min_n == 0:
                    used += 1
                    continue
                if self.formulation == "paper":
                    if self.ccl_source == "box_distribution":
                        s_pos = self._dfl_distribution_vector(student_distri_flat[pos_idx[:min_n]], temperature)
                        t_pos = self._dfl_distribution_vector(teacher_distri_flat[pos_idx[:min_n]], temperature)
                        s_neg = self._dfl_distribution_vector(student_distri_flat[sampled_neg[:min_n]], temperature)
                        t_neg = self._dfl_distribution_vector(teacher_distri_flat[sampled_neg[:min_n]], temperature)
                        s_pos = F.normalize(s_pos, dim=-1, eps=1e-6)
                        t_pos = F.normalize(t_pos, dim=-1, eps=1e-6)
                        s_neg = F.normalize(s_neg, dim=-1, eps=1e-6)
                        t_neg = F.normalize(t_neg, dim=-1, eps=1e-6)
                    else:
                        pos_boxes = target_bboxes_flat[pos_idx[:min_n]]
                        neg_boxes = target_bboxes_flat[sampled_neg[:min_n]]
                        s_pos = F.normalize(
                            self._sample_box_features(student_map, pos_idx[:min_n], pos_boxes, level_stride),
                            dim=-1,
                            eps=1e-6,
                        )
                        t_pos = F.normalize(
                            self._sample_box_features(teacher_map, pos_idx[:min_n], pos_boxes, level_stride),
                            dim=-1,
                            eps=1e-6,
                        )
                        s_neg = F.normalize(
                            self._sample_box_features(student_map, sampled_neg[:min_n], neg_boxes, level_stride),
                            dim=-1,
                            eps=1e-6,
                        )
                        t_neg = F.normalize(
                            self._sample_box_features(teacher_map, sampled_neg[:min_n], neg_boxes, level_stride),
                            dim=-1,
                            eps=1e-6,
                        )
                    pos_sim = (s_pos * t_pos).sum(dim=-1) / self.contrastive_temperature
                    if self.ccl_mode == "paper_pair":
                        neg_sim = (s_neg * t_neg).sum(dim=-1) / self.contrastive_temperature
                    else:
                        neg_sim = (s_pos * t_neg).sum(dim=-1) / self.contrastive_temperature
                else:
                    s_anchor = F.normalize(student_feat_flat[pos_idx[:min_n]], dim=-1, eps=1e-6)
                    t_pos = F.normalize(teacher_feat_flat[pos_idx[:min_n]], dim=-1, eps=1e-6)
                    t_neg = F.normalize(teacher_feat_flat[sampled_neg[:min_n]], dim=-1, eps=1e-6)
                    pos_sim = (s_anchor * t_pos).sum(dim=-1) / self.contrastive_temperature
                    neg_sim = (s_anchor * t_neg).sum(dim=-1) / self.contrastive_temperature
                logits = torch.stack((pos_sim, neg_sim), dim=-1)
                ccl = ccl + class_weight * (-F.log_softmax(logits, dim=-1)[:, 0].mean())
            used += 1

        if used == 0:
            self._record_cop_diagnostics(valid, cop, labels.reshape_as(cop), neg_token_counts, temperatures)
            self._record_kd_components(zero, zero, zero, zero)
            return zero
        self._record_cop_diagnostics(valid, cop, labels.reshape_as(cop), neg_token_counts, temperatures)
        self._record_kd_components(lld, fld, rld, ccl)
        return self.lld_weight * lld + self.fld_weight * fld + self.rld_weight * rld + self.ccl_weight * ccl

    @staticmethod
    def _new_diag_accumulator() -> dict[str, float]:
        return {
            "levels": 0.0,
            "cop_valid_tokens": 0.0,
            "cop_positive_tokens": 0.0,
            "cop_class0_count": 0.0,
            "cop_class1_count": 0.0,
            "cop_class2_count": 0.0,
            "neg_tokens_sum": 0.0,
            "neg_tokens_count": 0.0,
            "temperature_sum": 0.0,
            "temperature_count": 0.0,
            "temperature_min": math.nan,
            "temperature_max": math.nan,
        }

    def _record_cop_diagnostics(
        self,
        valid: torch.Tensor,
        cop: torch.Tensor,
        labels: torch.Tensor,
        neg_token_counts: list[int],
        temperatures: list[float],
    ) -> None:
        acc = self._diag_accumulator
        valid_detached = valid.detach()
        cop_detached = cop.detach()
        labels_detached = labels.detach()
        acc["levels"] += 1.0
        acc["cop_valid_tokens"] += float(valid_detached.sum().cpu())
        acc["cop_positive_tokens"] += float(cop_detached.sum().cpu())
        for class_id in range(3):
            acc[f"cop_class{class_id}_count"] += float((cop_detached & labels_detached.eq(class_id)).sum().cpu())
        if neg_token_counts:
            acc["neg_tokens_sum"] += float(sum(neg_token_counts))
            acc["neg_tokens_count"] += float(len(neg_token_counts))
        for temperature in temperatures:
            if not math.isfinite(temperature):
                continue
            acc["temperature_sum"] += float(temperature)
            acc["temperature_count"] += 1.0
            acc["temperature_min"] = temperature if math.isnan(acc["temperature_min"]) else min(acc["temperature_min"], temperature)
            acc["temperature_max"] = temperature if math.isnan(acc["temperature_max"]) else max(acc["temperature_max"], temperature)

    def pop_epoch_diagnostics(self) -> dict[str, float]:
        acc = self._diag_accumulator
        valid = acc["cop_valid_tokens"]
        positive = acc["cop_positive_tokens"]
        out = {
            "cop_valid_tokens": valid,
            "cop_positive_tokens": positive,
            "cop_positive_ratio": positive / valid if valid > 0 else math.nan,
            "cop_class0_count": acc["cop_class0_count"],
            "cop_class1_count": acc["cop_class1_count"],
            "cop_class2_count": acc["cop_class2_count"],
            "neg_tokens_mean": acc["neg_tokens_sum"] / acc["neg_tokens_count"] if acc["neg_tokens_count"] > 0 else math.nan,
            "temperature_mean": acc["temperature_sum"] / acc["temperature_count"] if acc["temperature_count"] > 0 else math.nan,
            "temperature_min": acc["temperature_min"],
            "temperature_max": acc["temperature_max"],
        }
        self._diag_accumulator = self._new_diag_accumulator()
        return out

    def _record_kd_components(
        self,
        lld: torch.Tensor,
        fld: torch.Tensor,
        rld: torch.Tensor,
        ccl: torch.Tensor,
    ) -> None:
        if self._kd_component_levels is None:
            return
        self._kd_component_levels.append(
            torch.stack(
                (
                    (self.lld_weight * lld).detach(),
                    (self.fld_weight * fld).detach(),
                    (self.rld_weight * rld).detach(),
                    (self.ccl_weight * ccl).detach(),
                )
            )
        )

    @staticmethod
    def _dfl_kl(student_box: torch.Tensor, teacher_box: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        if student_box.numel() == 0:
            return student_box.new_zeros(())
        if temperature.ndim == 0:
            t = temperature.view(1, 1, 1)
            scale = temperature.pow(2)
        else:
            t = temperature.view(-1, 1, 1)
            scale = temperature.pow(2)
        loss = F.kl_div(
            F.log_softmax(student_box / t, dim=-1),
            F.softmax(teacher_box / t, dim=-1),
            reduction="none",
        ).sum(dim=(-1, -2))
        return (loss * scale).mean()

    @staticmethod
    def _dfl_distribution_vector(box_logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        if box_logits.numel() == 0:
            return box_logits
        reg_max = box_logits.shape[-1] // 4
        if reg_max <= 0 or reg_max * 4 != box_logits.shape[-1]:
            raise RuntimeError(
                f"Expected YOLO DFL logits with 4*reg_max channels, got shape={tuple(box_logits.shape)}."
            )
        box = box_logits.reshape(-1, 4, reg_max)
        if temperature.ndim == 0 or temperature.numel() == 1:
            t = temperature.reshape(1, 1, 1)
        elif temperature.numel() >= box.shape[0]:
            t = temperature[: box.shape[0]].reshape(-1, 1, 1)
        else:
            t = temperature.mean().reshape(1, 1, 1)
        prob = F.softmax(box / t, dim=-1)
        return prob.flatten(1)

    @staticmethod
    def _feature_kl(student_feat: torch.Tensor, teacher_feat: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        if student_feat.numel() == 0:
            return student_feat.new_zeros(())
        if temperature.ndim == 0:
            t = temperature
            scale = temperature.pow(2)
        else:
            t = temperature.view(-1, 1)
            scale = temperature.pow(2)
        loss = F.kl_div(
            F.log_softmax(student_feat / t, dim=-1),
            F.softmax(teacher_feat / t, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        return (loss * scale).mean()

    def _relationship_loss(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        if student_feat.numel() == 0 or student_feat.shape[0] <= 1:
            return student_feat.new_zeros(())
        s_rel = F.normalize(student_feat, dim=-1, eps=1e-6)
        t_rel = F.normalize(teacher_feat, dim=-1, eps=1e-6)
        if self.rld_mode == "paper_instance":
            norm = math.sqrt(float(student_feat.shape[-1]))
            s_corr = (s_rel @ s_rel.T) / max(norm, 1.0)
            t_corr = (t_rel @ t_rel.T) / max(norm, 1.0)
        else:
            n_pos = float(student_feat.shape[0])
            s_corr = s_rel.T @ s_rel / max(n_pos, 1.0)
            t_corr = t_rel.T @ t_rel / max(n_pos, 1.0)
        scale = temperature.pow(2) if temperature.ndim == 0 else temperature.pow(2).mean()
        return (s_corr - t_corr).pow(2).mean() * scale

    def _sample_box_features(
        self,
        feature_map: torch.Tensor,
        flat_idx: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        level_stride: float,
    ) -> torch.Tensor:
        """Paper-style region feature extraction via bilinear sampling inside each assigned box."""
        if flat_idx.numel() == 0:
            return feature_map.new_zeros((0, feature_map.shape[1]))
        bsz, channels, height, width = feature_map.shape
        tokens = height * width
        batch_idx = torch.div(flat_idx, tokens, rounding_mode="floor")
        boxes = boxes_xyxy.to(device=feature_map.device, dtype=feature_map.dtype)
        image_w = float(width * level_stride)
        image_h = float(height * level_stride)
        x1 = boxes[:, 0].clamp(0, image_w - 1)
        y1 = boxes[:, 1].clamp(0, image_h - 1)
        x2 = boxes[:, 2].clamp(0, image_w - 1)
        y2 = boxes[:, 3].clamp(0, image_h - 1)
        x2 = torch.maximum(x2, x1 + 1.0)
        y2 = torch.maximum(y2, y1 + 1.0)

        grid_size = max(self.roi_grid_size, 1)
        lin = torch.linspace(0.5 / grid_size, 1.0 - 0.5 / grid_size, grid_size, device=feature_map.device, dtype=feature_map.dtype)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        xs = x1[:, None, None] + (x2 - x1)[:, None, None] * xx
        ys = y1[:, None, None] + (y2 - y1)[:, None, None] * yy
        x_norm = xs / max(image_w - 1.0, 1.0) * 2.0 - 1.0
        y_norm = ys / max(image_h - 1.0, 1.0) * 2.0 - 1.0
        grid = torch.stack((x_norm, y_norm), dim=-1)

        sampled = F.grid_sample(feature_map[batch_idx], grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.flatten(2).mean(dim=-1).reshape(flat_idx.numel(), channels)

    def _adaptive_temperature(self, class_scores: torch.Tensor) -> torch.Tensor:
        class_scores = class_scores.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(class_scores * class_scores.log() + (1.0 - class_scores) * (1.0 - class_scores).log())
        entropy = (entropy / math.log(2.0)).clamp(0.0, 1.0)
        temperature = self.temperature_min + (self.temperature_max - self.temperature_min) * torch.sigmoid(
            self.entropy_scale * (entropy - 0.5)
        )
        return temperature.clamp(self.temperature_min, self.temperature_max)

    def _adaptive_temperature_class(self, class_scores: torch.Tensor) -> torch.Tensor:
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
        if "teacher_weights" in overrides:
            raise ValueError(
                "CCLKD online trainer no longer accepts arbitrary teacher_weights; "
                "teacher and student both initialize from yolo11{model_size}.pt."
            )
        self.cclkd_cfg = {
            "teacher_data": overrides.pop("teacher_data", None),
            "model_size": overrides.pop("model_size", None),
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
            "fld_temperature": float(overrides.pop("cclkd_fld_temperature", 1.0)),
            "fld_temperature_mode": overrides.pop("cclkd_fld_temperature_mode", "patm"),
            "min_confidence": float(overrides.pop("cclkd_min_confidence", 0.1)),
            "max_tokens": int(overrides.pop("cclkd_max_tokens", 512)),
            "formulation": overrides.pop("cclkd_formulation", "adapted"),
            "ccl_mode": overrides.pop("cclkd_ccl_mode", "paper_pair"),
            "ccl_source": overrides.pop("cclkd_ccl_source", "box_distribution"),
            "rld_mode": overrides.pop("cclkd_rld_mode", "paper_instance"),
            "roi_grid_size": int(overrides.pop("cclkd_roi_grid_size", 3)),
        }
        self.cclkd_validate_teacher_every = int(overrides.pop("cclkd_validate_teacher_every", 0))
        if self.cclkd_cfg["model_size"] not in {"n", "s"}:
            raise ValueError("CCLKDOnlineHBBTrainer requires model_size to be one of {'n', 's'}.")
        if self.cclkd_cfg["teacher_data"] is None:
            raise ValueError("CCLKDOnlineHBBTrainer requires teacher_data.")
        overrides["task"] = "detect"
        super().__init__(cfg, overrides, _callbacks)
        if self.world_size > 1:
            raise RuntimeError("Use one CCLKD reproduction process per GPU; this trainer does not wrap teacher DDP.")
        self.teacher_data = check_det_dataset(str(self.cclkd_cfg["teacher_data"]))
        self.teacher_model: DetectionModel | None = None
        self._teacher_val_loader = None
        self._last_teacher_val_metrics = {"teacher_val_map50": math.nan, "teacher_val_map": math.nan}

    def setup_model(self):
        ckpt = super().setup_model()
        pretrain_path = _cclkd_pretrain_path(str(self.cclkd_cfg["model_size"]))
        if not pretrain_path.is_file():
            raise FileNotFoundError(f"Missing CCLKD online pretrain checkpoint: {pretrain_path}")
        teacher_weights, _ = load_checkpoint(str(pretrain_path), device=self.device)
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
        LOGGER.info(
            "CCLKD online reproduction: teacher is trainable and optimized with RGB detection loss "
            f"(formulation={self.cclkd_cfg['formulation']}, "
            f"ccl_mode={self.cclkd_cfg['ccl_mode']}, "
            f"ccl_source={self.cclkd_cfg['ccl_source']}, "
            f"rld_mode={self.cclkd_cfg['rld_mode']})."
        )

    def cclkd_cfg_without_paths(self) -> dict[str, Any]:
        return {k: v for k, v in self.cclkd_cfg.items() if k not in {"teacher_data", "model_size"}}

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

    @staticmethod
    def _metric_value(metrics: dict, key: str) -> float:
        value = metrics.get(key, float("nan"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def save_metrics(self, metrics):
        super().save_metrics(metrics)
        self._append_cclkd_diagnostics(metrics)

    def get_validator(self):
        self.loss_names = (
            "s_box_loss",
            "s_cls_loss",
            "s_dfl_loss",
            "t_box_loss",
            "t_cls_loss",
            "t_dfl_loss",
            "cclkd_loss",
            "cclkd_lld_loss",
            "cclkd_fld_loss",
            "cclkd_rld_loss",
            "cclkd_ccl_loss",
        )
        return yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=deepcopy(self.args),
            _callbacks=self.callbacks,
        )

    def _get_teacher_val_loader(self):
        if self._teacher_val_loader is not None:
            return self._teacher_val_loader
        if self.teacher_model is None:
            return None
        val_path = self.teacher_data.get("val")
        if not val_path:
            LOGGER.warning("CCLKD teacher validation requested but teacher_data has no val split.")
            return None
        gs = max(int(unwrap_model(self.teacher_model).stride.max()), 32)
        dataset = build_yolo_dataset(self.args, val_path, self.args.batch, self.teacher_data, mode="val", rect=True, stride=gs)
        self._teacher_val_loader = build_dataloader(
            dataset,
            batch=self.args.batch,
            workers=self.args.workers * 2,
            shuffle=False,
            rank=-1,
            drop_last=False,
            seed=self.args.seed,
        )
        return self._teacher_val_loader

    def _validate_teacher_if_due(self, epoch_1based: int) -> dict[str, float]:
        every = int(self.cclkd_validate_teacher_every)
        if every <= 0 or epoch_1based % every != 0:
            return self._last_teacher_val_metrics
        if self.teacher_model is None:
            return self._last_teacher_val_metrics
        loader = self._get_teacher_val_loader()
        if loader is None:
            return self._last_teacher_val_metrics
        LOGGER.info(f"Running optional CCLKD teacher RGB validation at epoch {epoch_1based}.")
        teacher_copy = deepcopy(unwrap_model(self.teacher_model))
        validator = yolo.detect.DetectionValidator(
            loader,
            save_dir=self.save_dir / "teacher_val",
            args=deepcopy(self.args),
            _callbacks=self.callbacks,
        )
        metrics = validator(model=teacher_copy) or {}
        teacher_metrics = {
            "teacher_val_map50": self._metric_value(metrics, "metrics/mAP50(B)"),
            "teacher_val_map": self._metric_value(metrics, "metrics/mAP50-95(B)"),
        }
        self._last_teacher_val_metrics = teacher_metrics
        return teacher_metrics

    def _append_cclkd_diagnostics(self, metrics: dict) -> None:
        if RANK not in {-1, 0}:
            return
        epoch_1based = int(getattr(self, "epoch", 0)) + 1
        criterion = getattr(unwrap_model(self.model), "criterion", None)
        cop_diag = criterion.pop_epoch_diagnostics() if hasattr(criterion, "pop_epoch_diagnostics") else {}
        teacher_val_metrics = self._validate_teacher_if_due(epoch_1based)
        s_det = (
            self._metric_value(metrics, "train/s_box_loss")
            + self._metric_value(metrics, "train/s_cls_loss")
            + self._metric_value(metrics, "train/s_dfl_loss")
        )
        kd_loss = self._metric_value(metrics, "train/cclkd_loss")
        kd_to_student_det_ratio = (float(self.cclkd_cfg["kd_weight"]) * kd_loss / s_det) if math.isfinite(kd_loss) and math.isfinite(s_det) and s_det > 0 else math.nan
        row = {
            "epoch": epoch_1based,
            "formulation": self.cclkd_cfg["formulation"],
            "ccl_mode": self.cclkd_cfg["ccl_mode"],
            "ccl_source": self.cclkd_cfg["ccl_source"],
            "rld_mode": self.cclkd_cfg["rld_mode"],
            "lld_weight": self.cclkd_cfg["lld_weight"],
            "fld_weight": self.cclkd_cfg["fld_weight"],
            "rld_weight": self.cclkd_cfg["rld_weight"],
            "ccl_weight": self.cclkd_cfg["ccl_weight"],
            "kd_weight": self.cclkd_cfg["kd_weight"],
            "student_box_loss": self._metric_value(metrics, "train/s_box_loss"),
            "student_cls_loss": self._metric_value(metrics, "train/s_cls_loss"),
            "student_dfl_loss": self._metric_value(metrics, "train/s_dfl_loss"),
            "teacher_box_loss": self._metric_value(metrics, "train/t_box_loss"),
            "teacher_cls_loss": self._metric_value(metrics, "train/t_cls_loss"),
            "teacher_dfl_loss": self._metric_value(metrics, "train/t_dfl_loss"),
            "cclkd_loss": kd_loss,
            "cclkd_lld_loss": self._metric_value(metrics, "train/cclkd_lld_loss"),
            "cclkd_fld_loss": self._metric_value(metrics, "train/cclkd_fld_loss"),
            "cclkd_rld_loss": self._metric_value(metrics, "train/cclkd_rld_loss"),
            "cclkd_ccl_loss": self._metric_value(metrics, "train/cclkd_ccl_loss"),
            "kd_to_student_det_ratio": kd_to_student_det_ratio,
            "lld_raw_or_weighted": self._metric_value(metrics, "train/cclkd_lld_loss"),
            "fld_raw_or_weighted": self._metric_value(metrics, "train/cclkd_fld_loss"),
            "rld_raw_or_weighted": self._metric_value(metrics, "train/cclkd_rld_loss"),
            "ccl_raw_or_weighted": self._metric_value(metrics, "train/cclkd_ccl_loss"),
            "component_values_are_weighted": 1,
            "teacher_val_map50": teacher_val_metrics.get("teacher_val_map50", math.nan),
            "teacher_val_map": teacher_val_metrics.get("teacher_val_map", math.nan),
            **cop_diag,
        }
        path = self.save_dir / "cclkd_diagnostics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

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
    parser.add_argument("--cclkd-fld-temperature", type=float, default=1.0)
    parser.add_argument(
        "--cclkd-fld-temperature-mode",
        choices=("fixed", "patm"),
        default="patm",
        help="'fixed' uses --cclkd-fld-temperature; 'patm' reuses the class-wise PATM temperature for FLD.",
    )
    parser.add_argument("--cclkd-min-confidence", type=float, default=0.1)
    parser.add_argument("--cclkd-max-tokens", type=int, default=512)
    parser.add_argument(
        "--cclkd-formulation",
        choices=("adapted", "paper"),
        default="paper",
        help="'paper' is the default CCLKD reproduction path: teacher-side COP, target/non-target LLD, "
        "class-wise temperature, paper-pair box-distribution CCL, and box-aligned FLD feature sampling. "
        "'adapted' is kept only for legacy YOLO11 token-level diagnostic runs.",
    )
    parser.add_argument(
        "--cclkd-ccl-mode",
        choices=("paper_pair", "anchor_teacher_neg"),
        default="paper_pair",
        help="'paper_pair' uses sim(T_pos,S_pos) vs sim(T_neg,S_neg), matching CCLKD Algorithm 2 more closely; "
        "'anchor_teacher_neg' keeps the previous student-anchor approximation.",
    )
    parser.add_argument(
        "--cclkd-ccl-source",
        choices=("box_distribution", "roi_feature"),
        default="box_distribution",
        help="'box_distribution' uses per-side YOLO DFL localization distributions for CCL, closer to CCLKD Eq.17-18; "
        "'roi_feature' keeps the previous box-aligned feature CCL.",
    )
    parser.add_argument(
        "--cclkd-rld-mode",
        choices=("paper_instance", "channel"),
        default="paper_instance",
        help="'paper_instance' computes R=F F^T over instances within each class, closer to CCLKD Eq.13-14; "
        "'channel' keeps the previous F^T F channel-correlation diagnostic.",
    )
    parser.add_argument("--cclkd-roi-grid-size", type=int, default=3)
    parser.add_argument(
        "--cclkd-validate-teacher-every",
        type=int,
        default=0,
        help="Optionally validate the online RGB teacher every N epochs and log teacher_val_map50/map. "
        "Default 0 disables this expensive diagnostic.",
    )
    add_common_detector_train_overrides(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cclkd_formulation == "adapted":
        LOGGER.warning(
            "CCLKD formulation is set to 'adapted'. This is a legacy YOLO11 token-level diagnostic mode, "
            "not the paper-aligned CCLKD reproduction path. Use --cclkd-formulation paper for formal experiments."
        )
    pretrain_path = _cclkd_pretrain_path(args.model_size)
    if not pretrain_path.is_file():
        raise SystemExit(f"Missing CCLKD online pretrain checkpoint: {pretrain_path}")

    model = YOLO(str(pretrain_path))
    train_kwargs = dict(
        trainer=CCLKDOnlineHBBTrainer,
        model_size=args.model_size,
        data=str(args.data.resolve()),
        teacher_data=str(args.teacher_data.resolve()),
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
        cclkd_fld_temperature=args.cclkd_fld_temperature,
        cclkd_fld_temperature_mode=args.cclkd_fld_temperature_mode,
        cclkd_min_confidence=args.cclkd_min_confidence,
        cclkd_max_tokens=args.cclkd_max_tokens,
        cclkd_formulation=args.cclkd_formulation,
        cclkd_ccl_mode=args.cclkd_ccl_mode,
        cclkd_ccl_source=args.cclkd_ccl_source,
        cclkd_rld_mode=args.cclkd_rld_mode,
        cclkd_roi_grid_size=args.cclkd_roi_grid_size,
        cclkd_validate_teacher_every=args.cclkd_validate_teacher_every,
    )
    train_kwargs.update(collect_common_detector_train_overrides(args))
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
