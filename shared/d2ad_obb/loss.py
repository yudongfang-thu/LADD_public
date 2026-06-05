from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8OBBLoss
from ultralytics.utils.tal import make_anchors


class D2ADOBBLoss(v8OBBLoss):
    """v8OBBLoss plus train-time D2AD losses on positive anchor features."""

    def __init__(
        self,
        model,
        teacher_model=None,
        lambda_inv: float = 1.0,
        lambda_orth: float = 0.05,
        lambda_rec: float = 0.0,
    ):
        super().__init__(model)
        self.student_model = model
        self.teacher_model = teacher_model
        self.lambda_inv = lambda_inv
        self.lambda_orth = lambda_orth
        self.lambda_rec = lambda_rec

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate OBB loss plus D2AD losses."""
        loss = torch.zeros(7, device=self.device)  # box, cls, dfl, angle, inv, orth, rec
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
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not an OBB dataset."
            ) from e

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

        if fg_mask.sum():
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
            d2ad_inv, d2ad_orth, d2ad_rec = self._compute_d2ad_losses(preds["feats"], batch, fg_mask)
            loss[4] = d2ad_inv * self.lambda_inv
            loss[5] = d2ad_orth * self.lambda_orth
            loss[6] = d2ad_rec * self.lambda_rec
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[3] *= self.hyp.angle
        return loss * batch_size, loss.detach()

    def _compute_d2ad_losses(
        self,
        student_feats: list[torch.Tensor],
        batch: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = torch.zeros((), device=self.device)
        if self.teacher_model is None or "teacher_img" not in batch or fg_mask.sum() == 0:
            return zero, zero, zero

        with torch.no_grad():
            teacher_outputs = self.teacher_model(batch["teacher_img"])
            teacher_preds = self._unwrap_teacher_preds(teacher_outputs)
            teacher_feats = teacher_preds.get("feats")
        if teacher_feats is None:
            return zero, zero, zero
        if len(student_feats) != len(teacher_feats):
            raise ValueError(
                "D2AD minimal implementation expects teacher and student to expose the same number of FPN levels."
            )

        inv_loss = zero
        orth_loss = zero
        rec_loss = zero
        used_levels = 0
        offset = 0
        teacher_target_mode = getattr(self.student_model, "teacher_target_mode", "projected")
        for i, (student_feat, teacher_feat) in enumerate(zip(student_feats, teacher_feats)):
            student_flat = student_feat.permute(0, 2, 3, 1).reshape(student_feat.shape[0], -1, student_feat.shape[1])
            teacher_flat = teacher_feat.permute(0, 2, 3, 1).reshape(teacher_feat.shape[0], -1, teacher_feat.shape[1])
            student_inv = self.student_model.d2ad_student_inv[i](student_flat)
            student_res = self.student_model.d2ad_student_res[i](student_flat)
            if teacher_target_mode == "projected":
                teacher_inv = self.student_model.d2ad_teacher_inv[i](teacher_flat)
            else:
                if student_inv.shape[-1] != teacher_flat.shape[-1]:
                    raise ValueError(
                        "Raw teacher targets require student invariant dim to match teacher feat dim, "
                        f"but got student {student_inv.shape[-1]} and teacher {teacher_flat.shape[-1]} at level {i}."
                    )
                teacher_inv = teacher_flat

            num_tokens = student_flat.shape[1]
            level_mask = fg_mask[:, offset : offset + num_tokens].bool()
            offset += num_tokens
            if not level_mask.any():
                continue

            s_inv_pos = student_inv[level_mask]
            s_res_pos = student_res[level_mask]
            t_inv_pos = teacher_inv[level_mask]
            inv_loss = inv_loss + F.mse_loss(s_inv_pos, t_inv_pos, reduction="mean")
            orth_loss = orth_loss + (F.normalize(s_inv_pos, dim=-1) * F.normalize(s_res_pos, dim=-1)).sum(-1).pow(2).mean()

            if self.lambda_rec > 0:
                recon = self.student_model.d2ad_reconstruct[i](torch.cat((student_inv, student_res), dim=-1))
                rec_loss = rec_loss + F.mse_loss(recon[level_mask], student_flat[level_mask], reduction="mean")
            used_levels += 1

        if used_levels == 0:
            return zero, zero, zero
        return inv_loss / used_levels, orth_loss / used_levels, rec_loss / used_levels

    @staticmethod
    def _unwrap_teacher_preds(outputs: Any) -> dict[str, torch.Tensor]:
        if isinstance(outputs, tuple):
            if len(outputs) >= 2 and isinstance(outputs[1], dict):
                return outputs[1]
            return outputs[0] if isinstance(outputs[0], dict) else {}
        return outputs if isinstance(outputs, dict) else {}
