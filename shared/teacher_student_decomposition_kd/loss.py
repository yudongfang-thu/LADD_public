from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8OBBLoss
from ultralytics.utils.tal import make_anchors


def _flatten_feat(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


def _unwrap_teacher_preds(outputs: Any) -> dict[str, torch.Tensor]:
    if isinstance(outputs, tuple):
        if len(outputs) >= 2 and isinstance(outputs[1], dict):
            return outputs[1]
        return outputs[0] if isinstance(outputs[0], dict) else {}
    return outputs if isinstance(outputs, dict) else {}


def _decorrelation_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.numel() == 0 or y.numel() == 0:
        return x.new_zeros(())
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])
    if x.shape[0] < 2:
        return x.new_zeros(())
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    x = x / (x.std(dim=0, keepdim=True, unbiased=False) + eps)
    y = y / (y.std(dim=0, keepdim=True, unbiased=False) + eps)
    corr = (x.transpose(0, 1) @ y) / x.shape[0]
    return corr.pow(2).mean()


def _margin_separation_loss(q_s: torch.Tensor, u_t: torch.Tensor, margin: float) -> tuple[torch.Tensor, torch.Tensor]:
    if q_s.numel() == 0 or u_t.numel() == 0:
        zero = q_s.new_zeros(())
        return zero, zero
    dist = (q_s - u_t).norm(dim=-1)
    active_ratio = (dist < margin).float().mean()
    return F.relu(margin - dist).mean(), active_ratio


def _total_variation_loss(mask: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0:
        return mask.new_zeros(())
    tv_h = (mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs().mean()
    tv_w = (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


class TeacherStudentDecompositionKDLoss(v8OBBLoss):
    """OBB loss plus teacher/student learnable decomposition KD objectives."""

    def __init__(
        self,
        model,
        teacher_model=None,
        lambda_rec: float = 0.1,
        lambda_sep: float = 0.05,
        lambda_match: float = 1.0,
        lambda_unmatch: float = 0.5,
        lambda_taskL: float = 1.0,
        alpha_kd: float = 1.0,
        alpha_s_rec: float = 0.1,
        alpha_sep: float = 0.05,
        lambda_mask_sparse: float = 0.0,
        lambda_mask_smooth: float = 0.0,
        margin: float = 1.0,
        match_target_mode: str = "detach",
        kd_target_mode: str = "detach",
    ):
        super().__init__(model)
        self.student_model = model
        self.teacher_model = teacher_model
        self.lambda_rec = lambda_rec
        self.lambda_sep = lambda_sep
        self.lambda_match = lambda_match
        self.lambda_unmatch = lambda_unmatch
        self.lambda_taskL = lambda_taskL
        self.alpha_kd = alpha_kd
        self.alpha_s_rec = alpha_s_rec
        self.alpha_sep = alpha_sep
        self.lambda_mask_sparse = lambda_mask_sparse
        self.lambda_mask_smooth = lambda_mask_smooth
        self.margin = margin
        self.match_target_mode = self._validate_target_mode(match_target_mode, "match_target_mode")
        self.kd_target_mode = self._validate_target_mode(kd_target_mode, "kd_target_mode")
        self.phase_loss_scales = {
            "det": 1.0,
            "rec": 1.0,
            "teacher_sep": 1.0,
            "match": 1.0,
            "unmatch": 1.0,
            "task": 1.0,
            "kd": 1.0,
            "student_rec": 1.0,
            "student_sep": 1.0,
            "mask": 1.0,
        }

    @staticmethod
    def _validate_target_mode(mode: str, name: str) -> str:
        if mode not in {"detach", "coupled"}:
            raise ValueError(f"{name} must be 'detach' or 'coupled', got {mode!r}.")
        return mode

    def set_phase_loss_scales(self, **scales: float) -> None:
        for name, value in scales.items():
            if name not in self.phase_loss_scales:
                raise KeyError(f"Unknown phase loss scale: {name}")
            self.phase_loss_scales[name] = float(value)

    def set_target_modes(self, *, match_target_mode: str | None = None, kd_target_mode: str | None = None) -> None:
        if match_target_mode is not None:
            self.match_target_mode = self._validate_target_mode(match_target_mode, "match_target_mode")
        if kd_target_mode is not None:
            self.kd_target_mode = self._validate_target_mode(kd_target_mode, "kd_target_mode")

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        loss = torch.zeros(13, device=self.device)
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

        target_scores_sum = target_scores.sum().clamp(min=1.0)
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

        extra_losses = self._compute_decomposition_losses(preds, batch, fg_mask, target_scores, target_scores_sum)
        loss[4:] = torch.stack(extra_losses[:-1])

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[3] *= self.hyp.angle
        loss[:4] *= self.phase_loss_scales["det"]
        log_items = torch.cat((loss.detach(), extra_losses[-1].detach().reshape(1)))
        return loss * batch_size, log_items

    def _compute_decomposition_losses(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        zero = torch.zeros((), device=self.device)
        if (
            self.teacher_model is None
            or "teacher_img" not in batch
            or "student_base_feats" not in preds
            or "z_s_feats" not in preds
            or "r_s_feats" not in preds
        ):
            return (zero, zero, zero, zero, zero, zero, zero, zero, zero, zero)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(batch["teacher_img"])
            teacher_feats = _unwrap_teacher_preds(teacher_outputs).get("feats")
        if teacher_feats is None:
            return (zero, zero, zero, zero, zero, zero, zero, zero, zero, zero)

        student_raw_feats = preds["student_base_feats"]
        student_z_feats = preds["z_s_feats"]
        student_r_feats = preds["r_s_feats"]
        student_recon_feats = preds.get("student_recon_feats")
        if student_recon_feats is None:
            return (zero, zero, zero, zero, zero, zero, zero, zero, zero, zero)
        if not (
            len(teacher_feats)
            == len(student_raw_feats)
            == len(student_z_feats)
            == len(student_r_feats)
            == len(student_recon_feats)
        ):
            raise ValueError("Teacher/student decomposition KD expects matching multi-scale P3/P4/P5 feature counts.")

        rec_loss = zero
        teacher_sep_loss = zero
        match_loss = zero
        unmatch_loss = zero
        task_loss = zero
        kd_loss = zero
        student_rec_loss = zero
        student_sep_loss = zero
        mask_reg_loss = zero
        unmatch_active_ratio = zero
        pos_levels = 0
        all_levels = 0
        offset = 0

        for i, (teacher_feat, student_raw, z_s_map, r_s_map, student_recon) in enumerate(
            zip(teacher_feats, student_raw_feats, student_z_feats, student_r_feats, student_recon_feats)
        ):
            z_t_map, u_t_map, mask_map, recon_map = self.student_model.teacher_decomposition[i](teacher_feat)
            q_s_map = self.student_model.student_reachability[i](student_raw)
            task_logits = self.student_model.teacher_task_heads[i](z_t_map)

            z_t = _flatten_feat(z_t_map)
            u_t = _flatten_feat(u_t_map)
            z_s = _flatten_feat(z_s_map)
            r_s = _flatten_feat(r_s_map)
            q_s = _flatten_feat(q_s_map)
            task_pred = _flatten_feat(task_logits)
            mask_tokens = _flatten_feat(mask_map) if mask_map is not None else None

            rec_loss = rec_loss + F.l1_loss(recon_map, teacher_feat, reduction="mean")
            student_rec_loss = student_rec_loss + F.l1_loss(student_recon, student_raw, reduction="mean")

            n_tokens = z_t.shape[1]
            level_fg_mask = fg_mask[:, offset : offset + n_tokens].bool()
            level_target_scores = target_scores[:, offset : offset + n_tokens].to(task_pred.dtype)
            offset += n_tokens

            task_loss = task_loss + self.bce(task_pred, level_target_scores).sum() / target_scores_sum

            if mask_map is not None and (self.lambda_mask_sparse > 0 or self.lambda_mask_smooth > 0):
                sparse = mask_map.mean()
                smooth = _total_variation_loss(mask_map)
                mask_reg_loss = mask_reg_loss + self.lambda_mask_sparse * sparse + self.lambda_mask_smooth * smooth

            if not level_fg_mask.any():
                all_levels += 1
                continue

            z_t_pos = z_t[level_fg_mask]
            u_t_pos = u_t[level_fg_mask]
            z_s_pos = z_s[level_fg_mask]
            r_s_pos = r_s[level_fg_mask]
            q_s_pos = q_s[level_fg_mask]
            match_target = z_t_pos.detach() if self.match_target_mode == "detach" else z_t_pos
            kd_target = z_t * mask_tokens if mask_tokens is not None else z_t
            kd_target_pos = kd_target[level_fg_mask].detach() if self.kd_target_mode == "detach" else kd_target[level_fg_mask]

            teacher_sep_loss = teacher_sep_loss + _decorrelation_loss(z_t_pos, u_t_pos)
            match_loss = match_loss + F.mse_loss(q_s_pos, match_target, reduction="mean")
            unmatch_term, unmatch_active = _margin_separation_loss(q_s_pos, u_t_pos, margin=self.margin)
            unmatch_loss = unmatch_loss + unmatch_term
            unmatch_active_ratio = unmatch_active_ratio + unmatch_active
            kd_loss = kd_loss + F.mse_loss(z_s_pos, kd_target_pos, reduction="mean")
            student_sep_loss = student_sep_loss + _decorrelation_loss(z_s_pos, r_s_pos)

            pos_levels += 1
            all_levels += 1

        if all_levels == 0:
            return (zero, zero, zero, zero, zero, zero, zero, zero, zero, zero)

        rec_loss = rec_loss / all_levels
        student_rec_loss = student_rec_loss / all_levels
        task_loss = task_loss / all_levels
        if self.lambda_mask_sparse > 0 or self.lambda_mask_smooth > 0:
            mask_reg_loss = mask_reg_loss / all_levels
        else:
            mask_reg_loss = zero

        if pos_levels > 0:
            teacher_sep_loss = teacher_sep_loss / pos_levels
            match_loss = match_loss / pos_levels
            unmatch_loss = unmatch_loss / pos_levels
            unmatch_active_ratio = unmatch_active_ratio / pos_levels
            kd_loss = kd_loss / pos_levels
            student_sep_loss = student_sep_loss / pos_levels
        else:
            teacher_sep_loss = zero
            match_loss = zero
            unmatch_loss = zero
            unmatch_active_ratio = zero
            kd_loss = zero
            student_sep_loss = zero

        return (
            rec_loss * self.lambda_rec * self.phase_loss_scales["rec"],
            teacher_sep_loss * self.lambda_sep * self.phase_loss_scales["teacher_sep"],
            match_loss * self.lambda_match * self.phase_loss_scales["match"],
            unmatch_loss * self.lambda_unmatch * self.phase_loss_scales["unmatch"],
            task_loss * self.lambda_taskL * self.phase_loss_scales["task"],
            kd_loss * self.alpha_kd * self.phase_loss_scales["kd"],
            student_rec_loss * self.alpha_s_rec * self.phase_loss_scales["student_rec"],
            student_sep_loss * self.alpha_sep * self.phase_loss_scales["student_sep"],
            mask_reg_loss * self.phase_loss_scales["mask"],
            unmatch_active_ratio,
        )
