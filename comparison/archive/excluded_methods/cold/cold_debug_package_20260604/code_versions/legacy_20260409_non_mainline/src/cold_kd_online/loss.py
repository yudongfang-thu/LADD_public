from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8OBBLoss
from ultralytics.utils.metrics import probiou
from ultralytics.utils.tal import make_anchors

from cold_kd.loss import _decoupled_kl, _unwrap_teacher_preds


class OnlineCoLDOBBLoss(v8OBBLoss):
    """Online CoLD for YOLO11-OBB: trainable RGB teacher + SAR student."""

    def __init__(
        self,
        model,
        teacher_model=None,
        lambda_kd: float = 1.0,
        lambda_cls_cold: float = 1.0,
        lambda_loc_cold: float = 1.0,
        alpha_non_target: float = 2.0,
        temperature: float = 20.0,
        kd_region: str = "positive",
        teacher_det_weight: float = 0.1,
    ):
        super().__init__(model)
        self.teacher_model = teacher_model
        self.lambda_kd = float(lambda_kd)
        self.lambda_cls_cold = float(lambda_cls_cold)
        self.lambda_loc_cold = float(lambda_loc_cold)
        self.alpha_non_target = float(alpha_non_target)
        self.temperature = float(temperature)
        self.teacher_det_weight = float(teacher_det_weight)
        if kd_region not in {"positive", "all"}:
            raise ValueError(f"kd_region must be 'positive' or 'all', got {kd_region!r}.")
        self.kd_region = kd_region
        self.teacher_det_criterion = v8OBBLoss(teacher_model) if teacher_model is not None else None

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        loss = torch.zeros(7, device=self.device)  # box, cls, dfl, angle, cls_cold, loc_cold, teacher_det
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR ❌ OBB dataset incorrectly formatted or not an OBB dataset.") from e

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)
        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        cls_cold_loss = torch.zeros((), device=self.device)
        loc_cold_loss = torch.zeros((), device=self.device)
        teacher_det_loss = torch.zeros((), device=self.device)
        mean_iou_weight = torch.zeros((), device=self.device)
        mean_teacher_top_conf = torch.zeros((), device=self.device)

        if fg_mask.sum():
            target_bboxes_abs = target_bboxes.clone()
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum)
            cls_cold_loss, loc_cold_loss, teacher_det_loss, mean_iou_weight, mean_teacher_top_conf = (
                self._compute_online_cold_loss(
                    pred_distri=pred_distri,
                    pred_scores=pred_scores,
                    pred_angle=pred_angle,
                    batch=batch,
                    fg_mask=fg_mask,
                    target_bboxes_abs=target_bboxes_abs,
                    anchor_points=anchor_points,
                    stride_tensor=stride_tensor,
                )
            )
            loss[4] = cls_cold_loss * self.lambda_kd * self.lambda_cls_cold
            loss[5] = loc_cold_loss * self.lambda_kd * self.lambda_loc_cold
            loss[6] = teacher_det_loss * self.teacher_det_weight
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[3] *= self.hyp.angle

        log_items = torch.cat(
            (
                loss.detach(),
                torch.stack((mean_iou_weight.detach(), mean_teacher_top_conf.detach())),
            )
        )
        return loss * batch_size, log_items

    def _compute_teacher_det_loss(self, teacher_preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        zero = next(iter(teacher_preds.values())).new_zeros(())
        if self.teacher_det_criterion is None:
            return zero
        teacher_det_vec, _ = self.teacher_det_criterion(teacher_preds, batch)
        batch_size = max(int(batch["img"].shape[0]), 1)
        return teacher_det_vec.sum() / batch_size

    def _compute_online_cold_loss(
        self,
        pred_distri: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_angle: torch.Tensor,
        batch: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        target_bboxes_abs: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = torch.zeros((), device=self.device)
        if self.teacher_model is None or "teacher_img" not in batch:
            return zero, zero, zero, zero, zero

        teacher_outputs = self.teacher_model(batch["teacher_img"])
        teacher_preds = _unwrap_teacher_preds(teacher_outputs)
        teacher_scores = teacher_preds.get("scores")
        teacher_distri = teacher_preds.get("boxes")
        teacher_angle = teacher_preds.get("angle")
        if teacher_scores is None or teacher_distri is None or teacher_angle is None:
            return zero, zero, zero, zero, zero

        teacher_det_loss = self._compute_teacher_det_loss(teacher_preds, batch)

        student_logits = pred_scores
        teacher_logits = teacher_scores.permute(0, 2, 1).contiguous()
        teacher_distri = teacher_distri.permute(0, 2, 1).contiguous()
        teacher_angle = teacher_angle.permute(0, 2, 1).contiguous()

        temperature = max(self.temperature, 1e-6)

        if self.kd_region == "positive":
            mask = fg_mask.bool()
            if not mask.any():
                return zero, zero, teacher_det_loss, zero, zero
            s_cls_logits = student_logits[mask]
            t_cls_logits = teacher_logits[mask]
            s_box_dist = pred_distri[mask]
            t_box_dist = teacher_distri[mask]
            teacher_pred_boxes = self.bbox_decode(anchor_points, teacher_distri, teacher_angle)
            teacher_pred_boxes[..., :4] *= stride_tensor
            teacher_boxes_pos = teacher_pred_boxes[mask]
            target_boxes_pos = target_bboxes_abs[mask]
        else:
            s_cls_logits = student_logits.reshape(-1, student_logits.shape[-1])
            t_cls_logits = teacher_logits.reshape(-1, teacher_logits.shape[-1])
            s_box_dist = pred_distri.reshape(-1, pred_distri.shape[-1])
            t_box_dist = teacher_distri.reshape(-1, teacher_distri.shape[-1])
            teacher_pred_boxes = self.bbox_decode(anchor_points, teacher_distri, teacher_angle)
            teacher_pred_boxes[..., :4] *= stride_tensor
            teacher_boxes_pos = teacher_pred_boxes.reshape(-1, teacher_pred_boxes.shape[-1])
            target_boxes_pos = target_bboxes_abs.reshape(-1, target_bboxes_abs.shape[-1])

        teacher_cls_probs = F.softmax(t_cls_logits / temperature, dim=-1)
        student_cls_probs = F.softmax(s_cls_logits / temperature, dim=-1)
        cls_cold = _decoupled_kl(
            teacher_cls_probs,
            student_cls_probs,
            alpha_non_target=self.alpha_non_target,
        ) * (temperature**2)

        reg_max = self.reg_max
        s_box_probs = F.softmax(s_box_dist.view(-1, 4, reg_max) / temperature, dim=-1)
        t_box_probs = F.softmax(t_box_dist.view(-1, 4, reg_max) / temperature, dim=-1)
        iou_weight = probiou(teacher_boxes_pos, target_boxes_pos).squeeze(-1).clamp_(0.0, 1.0)
        if iou_weight.numel() == 0:
            loc_cold = zero
            mean_iou_weight = zero
        else:
            per_anchor_side_losses = []
            for side in range(4):
                t_side = t_box_probs[:, side, :]
                s_side = s_box_probs[:, side, :]
                top_idx = t_side.argmax(dim=-1, keepdim=True)
                p_t = t_side.gather(-1, top_idx).squeeze(-1)
                q_t = s_side.gather(-1, top_idx).squeeze(-1)
                eps = 1e-8
                teacher_bin = torch.stack((p_t, 1.0 - p_t), dim=-1).clamp_min(eps)
                student_bin = torch.stack((q_t, 1.0 - q_t), dim=-1).clamp_min(eps)
                tpd = (teacher_bin * (teacher_bin.log() - student_bin.log())).sum(dim=-1)
                mask_nt = F.one_hot(top_idx.squeeze(-1), num_classes=reg_max).bool()
                teacher_rest = t_side.masked_select(~mask_nt).view(-1, reg_max - 1).clamp_min(eps)
                student_rest = s_side.masked_select(~mask_nt).view(-1, reg_max - 1).clamp_min(eps)
                teacher_rest = teacher_rest / teacher_rest.sum(dim=-1, keepdim=True).clamp_min(eps)
                student_rest = student_rest / student_rest.sum(dim=-1, keepdim=True).clamp_min(eps)
                npd = (teacher_rest * (teacher_rest.log() - student_rest.log())).sum(dim=-1)
                per_anchor_side_losses.append(tpd + self.alpha_non_target * npd)
            per_anchor_loc = torch.stack(per_anchor_side_losses, dim=1).mean(dim=1) * (temperature**2)
            loc_cold = (per_anchor_loc * iou_weight).sum() / iou_weight.sum().clamp_min(1e-6)
            mean_iou_weight = iou_weight.mean()

        teacher_top_conf = teacher_cls_probs.max(dim=-1).values.mean() if teacher_cls_probs.numel() else zero
        return cls_cold, loc_cold, teacher_det_loss, mean_iou_weight, teacher_top_conf
