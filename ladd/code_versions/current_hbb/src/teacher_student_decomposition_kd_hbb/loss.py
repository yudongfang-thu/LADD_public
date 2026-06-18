from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


def _flatten_feat(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


def _standardize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    return (x - x.mean(dim=dims, keepdim=True)) / x.std(dim=dims, keepdim=True, unbiased=False).clamp_min(eps)


def _pkd_channel_standardize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PKD-style channel-wise zero-mean/unit-variance normalization."""
    if x.dim() != 4:
        raise RuntimeError(f"PKD normalization expects [N, C, H, W], got {tuple(x.shape)}.")
    n, c, h, w = x.shape
    y = x.permute(1, 0, 2, 3).reshape(c, -1)
    mean = y.mean(dim=-1, keepdim=True)
    std = y.std(dim=-1, keepdim=True)
    y = (y - mean) / (std + eps)
    return y.reshape(c, n, h, w).permute(1, 0, 2, 3)


def _unwrap_teacher_preds(outputs: Any) -> dict[str, torch.Tensor]:
    if isinstance(outputs, tuple):
        if len(outputs) >= 2 and isinstance(outputs[1], dict):
            return outputs[1]
        return outputs[0] if isinstance(outputs[0], dict) else {}
    return outputs if isinstance(outputs, dict) else {}


def _pairwise_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU for xyxy boxes in the same coordinate unit."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    return inter / (area1[:, None] + area2[None, :] - inter).clamp_min(eps)


def _aligned_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """IoU for aligned xyxy box tensors with shape [..., 4]."""
    if boxes1.shape != boxes2.shape or boxes1.shape[-1] != 4:
        raise RuntimeError(f"Aligned IoU expects matching [..., 4] boxes, got {tuple(boxes1.shape)} and {tuple(boxes2.shape)}.")
    lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp_min(0) * (boxes1[..., 3] - boxes1[..., 1]).clamp_min(0)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp_min(0) * (boxes2[..., 3] - boxes2[..., 1]).clamp_min(0)
    return inter / (area1 + area2 - inter).clamp_min(eps)


def _contrastive_alignment_loss(student: torch.Tensor, teacher: torch.Tensor, temperature: float = 0.20) -> torch.Tensor:
    if student.numel() == 0 or teacher.numel() == 0:
        return student.new_zeros(())
    student = student.reshape(-1, student.shape[-1])
    teacher = teacher.reshape(-1, teacher.shape[-1])
    if student.shape[0] < 2:
        return student.new_zeros(())
    student = F.normalize(student, dim=-1, eps=1e-6)
    teacher = F.normalize(teacher, dim=-1, eps=1e-6)
    logits_st = (student @ teacher.transpose(0, 1)) / max(float(temperature), 1e-6)
    logits_ts = logits_st.transpose(0, 1)
    labels = torch.arange(student.shape[0], device=student.device)
    return 0.5 * (F.cross_entropy(logits_st, labels) + F.cross_entropy(logits_ts, labels))


def spatial_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1, eps=eps)


def squared_l2_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).pow(2).sum(dim=1, keepdim=True)


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros(())
    if mask is None:
        return x.mean()
    num = (x * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den


def _masked_l1_loss(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    if x.numel() == 0 or y.numel() == 0:
        return x.new_zeros(())
    diff = (x - y).abs()
    if mask is None:
        return diff.mean()
    mask = mask.to(diff.dtype)
    if mask.shape != diff.shape:
        mask = mask.expand_as(diff)
    den = mask.sum().clamp_min(eps)
    return (diff * mask).sum() / den


class TeacherStudentDecompositionKDNRRLTeacherUAuxLossHBB(v8DetectionLoss):
    """LADD HBB loss surface: detector, reconstruction, reachability, taskL, and KD."""

    def __init__(
        self,
        model,
        teacher_model=None,
        lambda_rec: float = 0.1,
        lambda_reach: float = 1.0,
        lambda_match_inner: float = 1.0,
        lambda_rank_inner: float = 1.0,
        lambda_taskL: float = 1.0,
        task_loss_fg_only: bool = False,
        alpha_kd: float = 1.0,
        alpha_s_rec: float = 0.1,
        delta: float = 0.2,
        use_soft_rank: bool = True,
        use_fg_mask_for_reach: bool = False,
        use_fg_mask_for_rec: bool = False,
        normalize_reach: bool = True,
        rank_d_neg_cap: float = 4.0,
        reach_target_mode: str = "detach",
        kd_target_mode: str = "detach",
        reach_input_mode: str = "adapter",
        kd_weight_mode: str = "none",
        kd_weight_power: float = 1.0,
        kd_aggregation_mode: str = "token",
        kd_topk_ratio: float = 0.5,
        kd_calibration_mode: str = "none",
        reach_student_detach: bool = False,
        student_branch_mode: str = "split",
        teacher_feature_mode: str = "decomposed",
        kd_mechanism: str = "mse",
        contrastive_temperature: float = 0.20,
        comparison_kd_profile: str = "none",
        profile_kd_weight: float = 1.0,
        profile_kd_replace_base: bool = False,
        fgd_alpha: float = 0.0001,
        fgd_beta: float = 0.00005,
        fgd_gamma: float = 0.001,
        fgd_lambda: float = 0.0,
        fgd_normalization_mode: str = "original",
        fgd_temperature: float = 0.5,
        fgd_mask_mode: str = "gt_box",
        fgd_bg_norm: bool = True,
        ld_temperature: float = 10.0,
        ld_use_vlr: bool = True,
        ld_quality_power: float = 1.0,
        ld_min_vlr_weight: float = 0.0,
        ld_vlr_topk: int = 0,
        ld_vlr_weight: float = 0.25,
        ld_main_weight: float = 0.25,
        ld_allow_empty_vlr: bool = True,
        cmdistill_feature_weight: float = 1.0,
        cmdistill_relation_weight: float = 1.0,
        cmdistill_logit_weight: float = 1.0,
        cmdistill_temperature: float = 4.0,
        cmdistill_max_tokens: int = 512,
        cmdistill_min_confidence: float = 0.05,
        cclkd_base_temperature: float = 2.0,
        cclkd_contrastive_temperature: float = 0.1,
        cclkd_feat_weight: float = 1.0,
        cclkd_logit_weight: float = 1.0,
        cclkd_contrast_weight: float = 0.5,
        cclkd_bg_weight: float = 0.1,
        cclkd_min_confidence: float = 0.1,
        cclkd_max_tokens: int = 512,
        cclkd_temperature_min: float = 0.5,
        cclkd_temperature_max: float = 5.0,
        cclkd_entropy_scale: float = 5.0,
    ):
        super().__init__(model)
        self.comparison_kd_profile = self._validate_comparison_kd_profile(comparison_kd_profile)
        self.profile_kd_weight = float(profile_kd_weight)
        self.profile_kd_replace_base = bool(profile_kd_replace_base)
        self.fgd_alpha = float(fgd_alpha)
        self.fgd_beta = float(fgd_beta)
        self.fgd_gamma = float(fgd_gamma)
        self.fgd_lambda = max(float(fgd_lambda), 0.0)
        self.fgd_temperature = max(float(fgd_temperature), 1e-6)
        self.fgd_mask_mode = self._validate_fgd_mask_mode(fgd_mask_mode)
        self.fgd_bg_norm = bool(fgd_bg_norm)
        self.fgd_normalization_mode = self._validate_fgd_normalization_mode(fgd_normalization_mode)
        self.ld_temperature = max(float(ld_temperature), 1e-6)
        self.ld_use_vlr = bool(ld_use_vlr)
        self.ld_quality_power = max(float(ld_quality_power), 0.0)
        self.ld_min_vlr_weight = max(float(ld_min_vlr_weight), 0.0)
        self.ld_vlr_topk = max(int(ld_vlr_topk), 0)
        self.ld_vlr_weight = max(float(ld_vlr_weight), 0.0)
        self.ld_main_weight = max(float(ld_main_weight), 0.0)
        self.ld_allow_empty_vlr = bool(ld_allow_empty_vlr)
        self._ld_warned_missing_teacher_scores = False
        self.cmdistill_feature_weight = max(float(cmdistill_feature_weight), 0.0)
        self.cmdistill_relation_weight = max(float(cmdistill_relation_weight), 0.0)
        self.cmdistill_logit_weight = max(float(cmdistill_logit_weight), 0.0)
        self.cmdistill_temperature = max(float(cmdistill_temperature), 1e-6)
        self.cmdistill_max_tokens = max(int(cmdistill_max_tokens), 16)
        self.cmdistill_min_confidence = min(max(float(cmdistill_min_confidence), 0.0), 1.0)
        self.cclkd_base_temperature = max(float(cclkd_base_temperature), 1e-6)
        self.cclkd_contrastive_temperature = max(float(cclkd_contrastive_temperature), 1e-6)
        self.cclkd_feat_weight = max(float(cclkd_feat_weight), 0.0)
        self.cclkd_logit_weight = max(float(cclkd_logit_weight), 0.0)
        self.cclkd_contrast_weight = max(float(cclkd_contrast_weight), 0.0)
        self.cclkd_bg_weight = max(float(cclkd_bg_weight), 0.0)
        self.cclkd_min_confidence = min(max(float(cclkd_min_confidence), 1e-6), 1.0)
        self.cclkd_max_tokens = max(int(cclkd_max_tokens), 16)
        self.cclkd_temperature_min = max(float(cclkd_temperature_min), 1e-6)
        self.cclkd_temperature_max = max(float(cclkd_temperature_max), self.cclkd_temperature_min)
        self.cclkd_entropy_scale = max(float(cclkd_entropy_scale), 1e-6)
        self.student_model = model
        self.teacher_model = teacher_model
        self.lambda_rec = lambda_rec
        self.lambda_reach = lambda_reach
        self.lambda_match_inner = lambda_match_inner
        self.lambda_rank_inner = lambda_rank_inner
        self.lambda_taskL = lambda_taskL
        self.task_loss_fg_only = bool(task_loss_fg_only)
        self.alpha_kd = alpha_kd
        self.alpha_s_rec = alpha_s_rec
        self.delta = delta
        self.use_soft_rank = use_soft_rank
        self.use_fg_mask_for_reach = use_fg_mask_for_reach
        self.use_fg_mask_for_rec = use_fg_mask_for_rec
        self.normalize_reach = normalize_reach
        self.rank_d_neg_cap = float(rank_d_neg_cap)
        self.reachability_enabled = True
        self.reach_target_mode = self._validate_target_mode(reach_target_mode, "reach_target_mode")
        self.kd_target_mode = self._validate_target_mode(kd_target_mode, "kd_target_mode")
        self.reach_input_mode = self._validate_reach_input_mode(reach_input_mode)
        self.kd_weight_mode = self._validate_kd_weight_mode(kd_weight_mode)
        self.kd_weight_power = float(kd_weight_power)
        self.kd_aggregation_mode = self._validate_kd_aggregation_mode(kd_aggregation_mode)
        self.kd_topk_ratio = float(kd_topk_ratio)
        self.kd_calibration_mode = self._validate_kd_calibration_mode(kd_calibration_mode)
        if self.comparison_kd_profile == "cmdistill" and self.kd_calibration_mode != "affine":
            LOGGER.warning(
                "CMDistill expects KD_CALIBRATION_MODE=affine for the adaptive 1x1 layer; "
                f"got {self.kd_calibration_mode!r}."
            )
        self.reach_student_detach = bool(reach_student_detach)
        self.student_branch_mode = self._validate_student_branch_mode(student_branch_mode)
        self.teacher_feature_mode = self._validate_teacher_feature_mode(teacher_feature_mode)
        self.kd_mechanism = self._validate_kd_mechanism(kd_mechanism)
        self.contrastive_temperature = float(contrastive_temperature)
        self.teacher_target_modules = None
        self._cmdistill_last_stats: dict[str, float | int] = {}
        self.phase_loss_scales = {
            "det": 1.0,
            "rec": 1.0,
            "match": 1.0,
            "unmatch": 1.0,
            "task": 1.0,
            "kd": 1.0,
            "student_rec": 1.0,
        }

    @staticmethod
    def _validate_target_mode(mode: str, name: str) -> str:
        if mode not in {"detach", "coupled"}:
            raise ValueError(f"{name} must be 'detach' or 'coupled', got {mode!r}.")
        return mode

    @staticmethod
    def _validate_reach_input_mode(mode: str) -> str:
        if mode not in {"adapter", "raw"}:
            raise ValueError(f"reach_input_mode must be 'adapter' or 'raw', got {mode!r}.")
        return mode

    @staticmethod
    def _validate_kd_weight_mode(mode: str) -> str:
        if mode not in {"none", "teacher_task_conf", "reachability_gap"}:
            raise ValueError(
                f"kd_weight_mode must be 'none', 'teacher_task_conf', or 'reachability_gap', got {mode!r}."
            )
        return mode

    @staticmethod
    def _validate_kd_aggregation_mode(mode: str) -> str:
        if mode not in {"token", "score_weighted", "topk"}:
            raise ValueError(f"kd_aggregation_mode must be 'token', 'score_weighted', or 'topk', got {mode!r}.")
        return mode

    @staticmethod
    def _validate_kd_calibration_mode(mode: str) -> str:
        if mode not in {"none", "affine", "norm_affine"}:
            raise ValueError(f"kd_calibration_mode must be 'none', 'affine', or 'norm_affine', got {mode!r}.")
        return mode

    @staticmethod
    def _validate_student_branch_mode(mode: str) -> str:
        if mode not in {"split", "raw", "single_proj"}:
            raise ValueError(
                f"student_branch_mode must be 'split', 'raw', or 'single_proj', got {mode!r}."
            )
        return mode

    @staticmethod
    def _validate_teacher_feature_mode(mode: str) -> str:
        if mode not in {"decomposed", "raw", "projected_raw"}:
            raise ValueError(
                f"teacher_feature_mode must be 'decomposed', 'raw', or 'projected_raw', got {mode!r}."
            )
        return mode

    @staticmethod
    def _validate_kd_mechanism(mode: str) -> str:
        if mode not in {"mse", "contrastive", "hybrid"}:
            raise ValueError(f"kd_mechanism must be 'mse', 'contrastive', or 'hybrid', got {mode!r}.")
        return mode

    @staticmethod
    def _validate_comparison_kd_profile(mode: str) -> str:
        if mode not in {"none", "fgd", "ld", "cmdistill", "cclkd"}:
            raise ValueError(
                "comparison_kd_profile must be one of "
                "{'none', 'fgd', 'ld', 'cmdistill', 'cclkd'}, got "
                f"{mode!r}."
            )
        return mode

    @staticmethod
    def _validate_fgd_mask_mode(mode: str) -> str:
        if mode not in {"gt_box", "assigner"}:
            raise ValueError(f"fgd_mask_mode must be 'gt_box' or 'assigner', got {mode!r}.")
        return mode

    @staticmethod
    def _validate_fgd_normalization_mode(mode: str) -> str:
        if mode not in {"original", "channel_mean"}:
            raise ValueError(f"fgd_normalization_mode must be 'original' or 'channel_mean', got {mode!r}.")
        return mode

    def set_phase_loss_scales(self, **scales: float) -> None:
        for name, value in scales.items():
            if name not in self.phase_loss_scales:
                raise KeyError(f"Unknown phase loss scale: {name}")
            self.phase_loss_scales[name] = float(value)

    def set_target_modes(self, *, match_target_mode: str | None = None, kd_target_mode: str | None = None) -> None:
        if match_target_mode is not None:
            self.reach_target_mode = self._validate_target_mode(match_target_mode, "reach_target_mode")
        if kd_target_mode is not None:
            self.kd_target_mode = self._validate_target_mode(kd_target_mode, "kd_target_mode")

    def set_reachability_enabled(self, enabled: bool) -> None:
        self.reachability_enabled = bool(enabled)

    def set_reach_student_detach(self, enabled: bool) -> None:
        self.reach_student_detach = bool(enabled)

    def set_teacher_target_modules(self, modules: dict[str, Any] | None) -> None:
        self.teacher_target_modules = modules

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        loss = torch.zeros(10, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_scores.shape[0]

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR ❌ HBB dataset incorrectly formatted or not an HBB dataset.") from e

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner *= stride_tensor
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

        # `gt_bboxes` stays in input-image pixel xyxy coordinates for FGD
        # GT-box masks. `target_bboxes` is converted below to stride units for
        # the normal YOLO bbox/DFL loss and for LD candidate-region IoU logic.
        if fg_mask.sum():
            target_bboxes /= stride_tensor
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

        extra_losses = self._compute_decomposition_losses(
            preds,
            batch,
            fg_mask,
            target_scores,
            target_scores_sum,
            pred_distri,
            pred_scores,
            anchor_points,
            stride_tensor,
            target_bboxes,
            pred_bboxes,
            gt_bboxes,
            mask_gt,
            imgsz,
        )
        loss[4:] = torch.stack(extra_losses[:6])

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[3] *= getattr(self.hyp, "angle", 0.0)
        loss[:4] *= self.phase_loss_scales["det"]

        log_items = torch.cat((loss.detach(), torch.stack(extra_losses[6:]).detach()))
        return loss * batch_size, log_items

    def normalized_reachability_loss(
        self,
        z_t_map: torch.Tensor,
        u_t_map: torch.Tensor,
        q_s_map: torch.Tensor,
        fg_mask_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_ref = z_t_map.detach() if self.reach_target_mode == "detach" else z_t_map
        u_ref = u_t_map.detach() if self.reach_target_mode == "detach" else u_t_map
        q_ref = q_s_map.detach() if self.reach_student_detach else q_s_map

        if self.normalize_reach:
            z_cmp = spatial_normalize(z_ref)
            u_cmp = spatial_normalize(u_ref)
            q_cmp = spatial_normalize(q_ref)
        else:
            z_cmp = z_ref
            u_cmp = u_ref
            q_cmp = q_ref

        d_pos = squared_l2_map(q_cmp, z_cmp)
        d_neg = squared_l2_map(q_cmp, u_cmp)
        match_loss = masked_mean(d_pos, fg_mask_map)

        if self.normalize_reach and self.rank_d_neg_cap < 4.0:
            d_neg_eff = d_neg.clamp(max=self.rank_d_neg_cap)
        else:
            d_neg_eff = d_neg

        rank_logits = self.delta + d_pos - d_neg_eff
        rank_map = F.softplus(rank_logits) if self.use_soft_rank else F.relu(rank_logits)
        rank_loss = masked_mean(rank_map, fg_mask_map)
        rank_gap_mean = masked_mean(d_neg - d_pos, fg_mask_map)
        d_pos_mean = masked_mean(d_pos, fg_mask_map)
        d_neg_mean = masked_mean(d_neg, fg_mask_map)
        return match_loss, rank_loss, d_pos_mean, d_neg_mean, rank_gap_mean

    def _compute_kd_pos_weights(
        self,
        task_pred_pos: torch.Tensor | None,
        target_scores_pos: torch.Tensor,
        q_s_pos: torch.Tensor | None = None,
        z_t_pos: torch.Tensor | None = None,
        u_t_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.kd_weight_mode == "none":
            return None

        if self.kd_weight_mode == "teacher_task_conf":
            if task_pred_pos is None:
                return None
            teacher_probs = task_pred_pos.sigmoid()
            target_mass = target_scores_pos.sum(dim=-1)
            assigned_conf = (teacher_probs * target_scores_pos).sum(dim=-1)
            fallback_conf = teacher_probs.amax(dim=-1)
            conf = torch.where(target_mass > 0, assigned_conf / target_mass.clamp_min(1e-6), fallback_conf)
        else:
            if q_s_pos is None or z_t_pos is None or u_t_pos is None:
                return None
            q_ref = q_s_pos.detach() if self.reach_student_detach else q_s_pos
            if self.normalize_reach:
                q_cmp = F.normalize(q_ref, dim=-1, eps=1e-6)
                z_cmp = F.normalize(z_t_pos.detach(), dim=-1, eps=1e-6)
                u_cmp = F.normalize(u_t_pos.detach(), dim=-1, eps=1e-6)
            else:
                q_cmp = q_ref
                z_cmp = z_t_pos.detach()
                u_cmp = u_t_pos.detach()
            d_pos = (q_cmp - z_cmp).pow(2).mean(dim=-1)
            d_neg = (q_cmp - u_cmp).pow(2).mean(dim=-1)
            gap = d_neg - d_pos
            gap_scale = gap.detach().abs().mean().clamp_min(1e-6)
            conf = torch.sigmoid(gap / gap_scale)
        conf = conf.clamp_min(1e-6)
        if self.kd_weight_power != 1.0:
            conf = conf.pow(self.kd_weight_power)
        conf = conf / conf.mean().clamp_min(1e-6)
        return conf.detach()

    def _compute_kd_loss(
        self,
        student_pos: torch.Tensor,
        target_pos: torch.Tensor,
        target_scores_pos: torch.Tensor,
        kd_pos_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if student_pos.numel() == 0:
            return student_pos.new_zeros(())

        if self.kd_calibration_mode == "norm_affine" and self.kd_mechanism != "contrastive":
            student_pos = F.normalize(student_pos, dim=-1, eps=1e-6)
            target_pos = F.normalize(target_pos, dim=-1, eps=1e-6)

        mse_loss = (student_pos - target_pos).pow(2).mean(dim=-1)
        contrastive_loss = _contrastive_alignment_loss(
            student_pos,
            target_pos,
            temperature=self.contrastive_temperature,
        )
        if self.kd_mechanism == "contrastive":
            return contrastive_loss

        token_loss = mse_loss
        token_weights = None

        if self.kd_aggregation_mode == "score_weighted":
            token_weights = target_scores_pos.sum(dim=-1).detach()
        elif self.kd_aggregation_mode == "topk":
            scores = target_scores_pos.sum(dim=-1).detach()
            keep = max(1, int(round(scores.numel() * self.kd_topk_ratio)))
            keep = min(keep, scores.numel())
            topk_scores, topk_idx = torch.topk(scores, k=keep, largest=True, sorted=False)
            token_loss = token_loss[topk_idx]
            token_weights = topk_scores
            if kd_pos_weights is not None:
                kd_pos_weights = kd_pos_weights[topk_idx]

        if kd_pos_weights is not None:
            token_weights = kd_pos_weights if token_weights is None else token_weights * kd_pos_weights

        if token_weights is None:
            mse_term = token_loss.mean()
        else:
            mse_term = (token_loss * token_weights).sum() / token_weights.sum().clamp_min(1e-6)

        if self.kd_mechanism == "hybrid":
            return mse_term + contrastive_loss
        return mse_term

    @staticmethod
    def _fgd_get_attention(feat: torch.Tensor, temp: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Official FGD-style spatial/channel attention from absolute feature response."""
        bsz, channels, height, width = feat.shape
        value = feat.abs()
        spatial = value.mean(dim=1).reshape(bsz, -1)
        spatial = F.softmax(spatial / max(float(temp), 1e-6), dim=1).reshape(bsz, height, width) * (height * width)
        channel = value.mean(dim=(2, 3))
        channel = F.softmax(channel / max(float(temp), 1e-6), dim=1) * channels
        return spatial, channel

    def _build_fgd_assigner_masks(self, fg_mask: torch.Tensor, height: int, width: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = fg_mask.shape[0]
        mask_fg = fg_mask.reshape(bsz, height, width).to(dtype)
        mask_bg = torch.where(mask_fg > 0, torch.zeros_like(mask_fg), torch.ones_like(mask_fg))
        if self.fgd_bg_norm:
            bg_sum = mask_bg.flatten(1).sum(dim=1).clamp_min(1e-6).view(bsz, 1, 1)
            mask_bg = mask_bg / bg_sum
        return mask_fg, mask_bg

    def _build_fgd_gt_box_masks(
        self,
        gt_bboxes: torch.Tensor | None,
        mask_gt: torch.Tensor | None,
        output_size: tuple[int, int],
        imgsz: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
        bsz: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build FGD foreground/background masks from pixel xyxy GT boxes.

        `v8DetectionLoss.preprocess()` converts dataset xywh labels to pixel
        xyxy coordinates using `imgsz`; these pixel boxes are projected to the
        current feature grid and filled with 1 / projected_box_area, matching
        the official FGD mask convention more closely than assigner tokens.
        """
        height, width = output_size
        mask_fg = torch.zeros((bsz, height, width), device=device, dtype=dtype)
        if gt_bboxes is None or mask_gt is None or imgsz is None:
            raise RuntimeError("FGD gt_box mask requires gt_bboxes, mask_gt, and imgsz; use fgd_mask_mode='assigner' for fallback masks.")
        else:
            if gt_bboxes.dim() != 3 or gt_bboxes.shape[-1] < 4:
                raise RuntimeError(f"FGD gt_box mask expects gt_bboxes [B, M, 4], got {tuple(gt_bboxes.shape)}.")
            if mask_gt.shape[:2] != gt_bboxes.shape[:2]:
                raise RuntimeError(
                    f"FGD gt_box mask expects mask_gt shape compatible with gt_bboxes, got "
                    f"mask_gt={tuple(mask_gt.shape)} gt_bboxes={tuple(gt_bboxes.shape)}."
                )
            img_h = float(imgsz[0].detach().item())
            img_w = float(imgsz[1].detach().item())
            if img_h <= 0 or img_w <= 0:
                raise RuntimeError(f"FGD gt_box mask received invalid imgsz={tuple(imgsz.shape)} values.")
            valid = mask_gt.squeeze(-1).bool()
            for b in range(bsz):
                boxes = gt_bboxes[b, valid[b], :4]
                if boxes.numel() == 0:
                    continue
                boxes = boxes.to(device=device, dtype=dtype)
                x1 = torch.floor((boxes[:, 0] / img_w * width).clamp(0, width - 1)).long()
                y1 = torch.floor((boxes[:, 1] / img_h * height).clamp(0, height - 1)).long()
                x2 = torch.ceil((boxes[:, 2] / img_w * width).clamp(0, width - 1)).long()
                y2 = torch.ceil((boxes[:, 3] / img_h * height).clamp(0, height - 1)).long()
                for j in range(boxes.shape[0]):
                    if x2[j] < x1[j] or y2[j] < y1[j]:
                        continue
                    area = float((x2[j] - x1[j] + 1) * (y2[j] - y1[j] + 1))
                    value = mask_fg.new_tensor(1.0 / max(area, 1.0))
                    region = mask_fg[b, y1[j] : y2[j] + 1, x1[j] : x2[j] + 1]
                    mask_fg[b, y1[j] : y2[j] + 1, x1[j] : x2[j] + 1] = torch.maximum(region, value)
            mask_bg = torch.where(mask_fg > 0, torch.zeros_like(mask_fg), torch.ones_like(mask_fg))
        if self.fgd_bg_norm:
            bg_sum = mask_bg.flatten(1).sum(dim=1).clamp_min(1e-6).view(bsz, 1, 1)
            mask_bg = mask_bg / bg_sum
        return mask_fg, mask_bg

    @staticmethod
    def _fgd_batch_relation_legacy_loss(student_map: torch.Tensor, teacher_map: torch.Tensor) -> torch.Tensor:
        """Legacy batch-wise relation approximation; disabled by default.

        This is not the official FGD trainable global context relation module.
        It remains opt-in through `fgd_lambda` only for ablation continuity.
        """
        bsz, channels, _, _ = student_map.shape
        if bsz <= 1:
            return student_map.new_zeros(())
        student_flat = student_map.permute(0, 2, 3, 1).reshape(bsz, -1, channels)
        teacher_flat = teacher_map.detach().permute(0, 2, 3, 1).reshape(bsz, -1, channels)
        student_global = F.normalize(student_flat.mean(dim=1), dim=-1, eps=1e-6)
        teacher_global = F.normalize(teacher_flat.mean(dim=1), dim=-1, eps=1e-6)
        return F.mse_loss(student_global @ student_global.transpose(0, 1), teacher_global @ teacher_global.transpose(0, 1))

    def _fgd_style_loss(
        self,
        student_map: torch.Tensor,
        teacher_map: torch.Tensor,
        assigner_fg_mask: torch.Tensor,
        gt_bboxes: torch.Tensor | None = None,
        mask_gt: torch.Tensor | None = None,
        imgsz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """FGD-YOLO adaptation: focal fg/bg feature loss + attention mask loss."""
        if student_map.shape != teacher_map.shape:
            raise RuntimeError(
                "FGD requires matching student/teacher feature maps; explicit model-owned align conv is not implemented, got "
                f"student={tuple(student_map.shape)} teacher={tuple(teacher_map.shape)}."
            )
        bsz, _, height, width = student_map.shape
        teacher_map = teacher_map.detach()
        s_t, c_t = self._fgd_get_attention(teacher_map, self.fgd_temperature)
        s_s, c_s = self._fgd_get_attention(student_map, self.fgd_temperature)
        if self.fgd_mask_mode == "gt_box":
            mask_fg, mask_bg = self._build_fgd_gt_box_masks(
                gt_bboxes, mask_gt, (height, width), imgsz, student_map.dtype, student_map.device, bsz
            )
        else:
            mask_fg, mask_bg = self._build_fgd_assigner_masks(assigner_fg_mask, height, width, student_map.dtype)

        s_t_sqrt = s_t.clamp_min(0).sqrt().unsqueeze(1)
        c_t_sqrt = c_t.clamp_min(0).sqrt().view(bsz, -1, 1, 1)
        fea_t = teacher_map * s_t_sqrt * c_t_sqrt
        fea_s = student_map * s_t_sqrt * c_t_sqrt
        fg_w = mask_fg.clamp_min(0).sqrt().unsqueeze(1)
        bg_w = mask_bg.clamp_min(0).sqrt().unsqueeze(1)

        fg_raw = F.mse_loss(fea_s * fg_w, fea_t * fg_w, reduction="sum")
        bg_raw = F.mse_loss(fea_s * bg_w, fea_t * bg_w, reduction="sum")
        if self.fgd_normalization_mode == "original":
            # FGD-style mask values already encode spatial normalization.
            fg_loss = fg_raw / max(bsz, 1)
            bg_loss = bg_raw / max(bsz, 1)
        else:
            # Optional YOLO adaptation: keep spatial mask semantics, average only across channels.
            _, channels, _, _ = student_map.shape
            denom = max(bsz, 1) * max(channels, 1)
            fg_loss = fg_raw / denom
            bg_loss = bg_raw / denom
        mask_loss = (c_s - c_t).abs().sum() / max(bsz, 1) + (s_s - s_t).abs().sum() / max(bsz, 1)
        relation_loss = (
            self._fgd_batch_relation_legacy_loss(student_map, teacher_map) if self.fgd_lambda > 0 else student_map.new_zeros(())
        )
        return (
            self.fgd_alpha * fg_loss
            + self.fgd_beta * bg_loss
            + self.fgd_gamma * mask_loss
            + self.fgd_lambda * relation_loss
        )

    def _ld_style_loss(
        self,
        student_distri: torch.Tensor | None,
        teacher_distri: torch.Tensor | None,
        fg_mask: torch.Tensor,
        target_scores: torch.Tensor,
        teacher_scores: torch.Tensor | None,
        teacher_bboxes: torch.Tensor | None = None,
        gt_bboxes: torch.Tensor | None = None,
        mask_gt: torch.Tensor | None = None,
        level_stride_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Localization Distillation on YOLO DFL logits with main + VLR-style regions."""
        if student_distri is None or teacher_distri is None:
            raise RuntimeError("LD requires raw student and teacher DFL logits; received a missing distribution tensor.")
        if student_distri.shape != teacher_distri.shape:
            raise RuntimeError(
                "LD requires matching raw student/teacher DFL logits, got "
                f"student={tuple(student_distri.shape)} teacher={tuple(teacher_distri.shape)}."
            )
        if student_distri.numel() == 0 or student_distri.shape[-1] % 4 != 0:
            raise RuntimeError(f"LD received an invalid DFL-logit shape: {tuple(student_distri.shape)}.")
        if target_scores.shape[:2] != student_distri.shape[:2]:
            raise RuntimeError(
                f"LD requires target_scores [B, N, C] aligned with DFL logits, got "
                f"target_scores={tuple(target_scores.shape)} logits={tuple(student_distri.shape)}."
            )

        temperature = self.ld_temperature
        reg_max = student_distri.shape[-1] // 4
        fg = fg_mask.reshape(student_distri.shape[:2]).bool()
        zero = student_distri.new_zeros(())

        def weighted_dfl_kl(mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            if not mask.any():
                return zero
            s = student_distri[mask].reshape(-1, 4, reg_max)
            t = teacher_distri.detach()[mask].reshape(-1, 4, reg_max)
            kl = F.kl_div(
                F.log_softmax(s / temperature, dim=-1),
                F.softmax(t / temperature, dim=-1),
                reduction="none",
            ).mean(dim=-1) * (temperature**2)
            per_anchor = kl.mean(dim=-1)
            w = weights[mask].to(per_anchor.dtype).clamp_min(0)
            return (per_anchor * w).sum() / w.sum().clamp_min(1e-6)

        assigned_cls = target_scores.argmax(dim=-1)
        target_quality = target_scores.amax(dim=-1).clamp_min(0)
        if teacher_scores is not None and teacher_scores.shape[:2] == student_distri.shape[:2]:
            gather_idx = assigned_cls.clamp(0, teacher_scores.shape[-1] - 1).unsqueeze(-1)
            teacher_assigned_conf = teacher_scores.detach().sigmoid().gather(-1, gather_idx).squeeze(-1)
            teacher_any_conf = teacher_scores.detach().sigmoid().amax(dim=-1)
        else:
            if not self._ld_warned_missing_teacher_scores:
                LOGGER.warning("LD did not receive aligned teacher class logits; falling back to target_scores for main weights.")
                self._ld_warned_missing_teacher_scores = True
            teacher_assigned_conf = torch.ones_like(target_quality)
            teacher_any_conf = torch.zeros_like(target_quality)

        main_weights = (target_quality * teacher_assigned_conf).clamp_min(0)
        if self.ld_quality_power != 1.0:
            main_weights = main_weights.pow(self.ld_quality_power)
        main_loss = weighted_dfl_kl(fg, main_weights) if fg.any() else zero

        vlr_loss = zero
        if self.ld_use_vlr and self.ld_vlr_weight > 0:
            vlr_weights = teacher_any_conf.clone()
            if teacher_bboxes is not None and gt_bboxes is not None and mask_gt is not None and level_stride_tensor is not None:
                if teacher_bboxes.shape[:2] != fg.shape:
                    raise RuntimeError(
                        f"LD VLR teacher_bboxes must align with logits, got teacher_bboxes={tuple(teacher_bboxes.shape)} "
                        f"logits={tuple(student_distri.shape)}."
                    )
                if gt_bboxes.dim() != 3 or gt_bboxes.shape[-1] < 4 or mask_gt.shape[:2] != gt_bboxes.shape[:2]:
                    raise RuntimeError(
                        f"LD VLR expects gt_bboxes [B, M, 4] and compatible mask_gt, got "
                        f"gt_bboxes={tuple(gt_bboxes.shape)} mask_gt={tuple(mask_gt.shape)}."
                    )
                stride = level_stride_tensor.reshape(1, -1, 1).to(teacher_bboxes.dtype)
                teacher_boxes_px = teacher_bboxes.detach() * stride
                iou_weights = torch.zeros_like(vlr_weights)
                valid_gt = mask_gt.squeeze(-1).bool()
                for b in range(student_distri.shape[0]):
                    boxes_gt = gt_bboxes[b, valid_gt[b], :4].to(device=student_distri.device, dtype=teacher_boxes_px.dtype)
                    if boxes_gt.numel() == 0:
                        continue
                    ious = _pairwise_iou_xyxy(teacher_boxes_px[b], boxes_gt)
                    iou_weights[b] = ious.max(dim=1).values
                if teacher_scores is not None:
                    vlr_weights = vlr_weights * iou_weights
                else:
                    vlr_weights = iou_weights

            vlr_mask = (~fg) & (vlr_weights > self.ld_min_vlr_weight)
            if self.ld_vlr_topk > 0 and vlr_mask.any():
                topk_mask = torch.zeros_like(vlr_mask)
                for b in range(vlr_mask.shape[0]):
                    idx = torch.where(vlr_mask[b])[0]
                    if idx.numel() == 0:
                        continue
                    k = min(int(self.ld_vlr_topk), idx.numel())
                    selected = idx[torch.topk(vlr_weights[b, idx], k=k, largest=True, sorted=False).indices]
                    topk_mask[b, selected] = True
                vlr_mask = topk_mask
            if vlr_mask.any():
                vlr_loss = weighted_dfl_kl(vlr_mask, vlr_weights)
            elif not self.ld_allow_empty_vlr:
                raise RuntimeError("LD VLR found no candidate anchors and ld_allow_empty_vlr=False.")

        return self.ld_main_weight * main_loss + self.ld_vlr_weight * vlr_loss

    def _cmdistill_style_loss(
        self,
        student_map: torch.Tensor,
        teacher_map: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores: torch.Tensor,
        student_distri: torch.Tensor | None,
        teacher_distri: torch.Tensor | None,
        student_scores: torch.Tensor | None,
        teacher_scores: torch.Tensor | None,
        student_bboxes: torch.Tensor | None = None,
        teacher_bboxes: torch.Tensor | None = None,
        level_index: int | None = None,
        num_levels: int | None = None,
    ) -> torch.Tensor:
        """CMDistill feature-side KD: PCCFD + SLRD.

        This is a paper-aligned adaptation because no official CMDistill code is
        available in the project. It follows the paper components:
        PCC feature distillation on selected FPN maps and semantic relation
        distillation on the deepest map. IBCLD is computed once on the full
        concatenated detector outputs in `_compute_decomposition_losses`.
        """
        if student_map.shape != teacher_map.shape:
            raise RuntimeError(
                "CMDistill requires matching student/teacher feature maps, got "
                f"student={tuple(student_map.shape)} teacher={tuple(teacher_map.shape)}."
            )
        teacher_map = teacher_map.detach()
        zero = student_map.new_zeros(())
        is_first_level = level_index is None or int(level_index) == 0
        is_last_level = level_index is None or num_levels is None or int(level_index) == int(num_levels) - 1

        feature_loss = zero
        if self.cmdistill_feature_weight > 0 and (is_first_level or is_last_level):
            feature_loss = self._cmdistill_pcc_feature_loss(student_map, teacher_map)

        relation_loss = zero
        if self.cmdistill_relation_weight > 0 and is_last_level:
            relation_loss = self._cmdistill_relation_loss(student_map, teacher_map)

        return (
            self.cmdistill_feature_weight * feature_loss
            + self.cmdistill_relation_weight * relation_loss
        )

    def _cmdistill_update_stats(self, **stats: float | int) -> None:
        current = getattr(self, "_cmdistill_last_stats", {})
        current.update(stats)
        self._cmdistill_last_stats = current

    def _cmdistill_combine_components(
        self,
        pcc_loss: torch.Tensor,
        pcc_levels: int,
        relation_loss: torch.Tensor,
        relation_levels: int,
        output_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = pcc_loss.new_zeros(())
        feature_term = pcc_loss / pcc_levels if pcc_levels > 0 else zero
        relation_term = relation_loss / relation_levels if relation_levels > 0 else zero
        output_term = output_loss if output_loss is not None else zero
        total = (
            self.cmdistill_feature_weight * feature_term
            + self.cmdistill_relation_weight * relation_term
            + self.cmdistill_logit_weight * output_term
        )
        return total, feature_term, relation_term, output_term

    def _cmdistill_pcc_feature_loss(self, student_map: torch.Tensor, teacher_map: torch.Tensor) -> torch.Tensor:
        """PCCFD: CMDistill Pearson-correlation feature distillation.

        CMDistill defines PCCFD as normalized feature imitation. The concrete
        channel-wise normalization follows the open PKD implementation for the
        under-specified tensor reduction detail.
        """
        student_norm = _pkd_channel_standardize_map(student_map)
        teacher_norm = _pkd_channel_standardize_map(teacher_map.detach())
        return 0.5 * F.mse_loss(student_norm, teacher_norm)

    def _cmdistill_relation_loss(
        self,
        student_map: torch.Tensor,
        teacher_map: torch.Tensor,
    ) -> torch.Tensor:
        student_flat = _flatten_feat(student_map)
        teacher_flat = _flatten_feat(teacher_map.detach())
        if student_flat.shape[1] < 2:
            return student_map.new_zeros(())

        candidate_idx = torch.arange(student_flat.shape[1], device=student_map.device)
        if candidate_idx.numel() > self.cmdistill_max_tokens:
            perm = torch.randperm(candidate_idx.numel(), device=student_map.device)[: self.cmdistill_max_tokens]
            candidate_idx = candidate_idx[perm]
        if candidate_idx.numel() < 2:
            return student_map.new_zeros(())

        self._cmdistill_update_stats(cmdistill_slrd_tokens=int(candidate_idx.numel()))
        student_tokens = F.normalize(student_flat[:, candidate_idx, :], dim=-1, eps=1e-6)
        teacher_tokens = F.normalize(teacher_flat[:, candidate_idx, :], dim=-1, eps=1e-6)
        student_relation = torch.bmm(student_tokens, student_tokens.transpose(1, 2))
        teacher_relation = torch.bmm(teacher_tokens, teacher_tokens.transpose(1, 2))
        return F.l1_loss(student_relation, teacher_relation)

    def _cmdistill_output_loss(
        self,
        student_distri: torch.Tensor | None,
        teacher_distri: torch.Tensor | None,
        student_scores: torch.Tensor | None,
        teacher_scores: torch.Tensor | None,
        fg_mask: torch.Tensor,
        target_scores: torch.Tensor,
        student_bboxes: torch.Tensor | None,
        teacher_bboxes: torch.Tensor | None,
    ) -> torch.Tensor:
        if student_scores is None or teacher_scores is None or student_scores.shape != teacher_scores.shape:
            raise RuntimeError("CMDistill IBCLD requires matching student/teacher class logits.")
        if student_bboxes is None or teacher_bboxes is None or student_bboxes.shape != teacher_bboxes.shape:
            raise RuntimeError("CMDistill IBCLD requires matching decoded student/teacher boxes.")
        if target_scores.shape[:2] != student_scores.shape[:2]:
            raise RuntimeError(
                "CMDistill requires target_scores aligned with class logits, got "
                f"target_scores={tuple(target_scores.shape)} scores={tuple(student_scores.shape)}."
            )

        teacher_scores = teacher_scores.detach()
        teacher_bboxes = teacher_bboxes.detach()
        teacher_conf = teacher_scores.sigmoid().amax(dim=-1)
        fg = fg_mask.reshape(student_scores.shape[:2]).bool()
        teacher_conf_candidate = torch.zeros_like(fg)
        if self.cmdistill_min_confidence > 0:
            teacher_conf_candidate = teacher_conf >= self.cmdistill_min_confidence
        candidate = fg | teacher_conf_candidate
        teacher_conf_added = teacher_conf_candidate & ~fg
        if not candidate.any():
            self._cmdistill_update_stats(
                cmdistill_ibcld_candidate_ratio=0.0,
                cmdistill_ibcld_fg_count=int(fg.sum().detach().cpu()),
                cmdistill_ibcld_teacher_conf_added_count=int(teacher_conf_added.sum().detach().cpu()),
                cmdistill_ibcld_cls_loss=0.0,
                cmdistill_ibcld_box_loss=0.0,
            )
            return student_scores.new_zeros(())

        cls_loss = F.binary_cross_entropy_with_logits(
            student_scores[candidate],
            teacher_scores[candidate].sigmoid(),
            reduction="mean",
        )
        iou = _aligned_iou_xyxy(student_bboxes[candidate], teacher_bboxes[candidate])
        box_loss = (1.0 - iou).mean()
        self._cmdistill_update_stats(
            cmdistill_ibcld_candidate_ratio=float(candidate.float().mean().detach().cpu()),
            cmdistill_ibcld_fg_count=int(fg.sum().detach().cpu()),
            cmdistill_ibcld_teacher_conf_added_count=int(teacher_conf_added.sum().detach().cpu()),
            cmdistill_ibcld_cls_loss=float(cls_loss.detach().cpu()),
            cmdistill_ibcld_box_loss=float(box_loss.detach().cpu()),
        )

        return cls_loss + box_loss

    def _cclkd_style_loss(
        self,
        student_map: torch.Tensor,
        teacher_map: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores: torch.Tensor,
        student_distri: torch.Tensor | None,
        teacher_distri: torch.Tensor | None,
        student_scores: torch.Tensor | None,
        teacher_scores: torch.Tensor | None,
    ) -> torch.Tensor:
        """CCLKD paper-structured ATKD + CCL adaptation for YOLO11 HBB.

        This follows the paper's component structure: COP category mask,
        entropy-mapped class temperature, LLD/FLD/RLD and class-balanced CCL.
        YOLO11 has DFL regression distributions instead of YOLOv5 objectness
        anchors, so raw DFL logits are used as the spatial distribution.
        """
        bsz = student_map.shape[0]
        student_flat = _flatten_feat(student_map)
        teacher_flat = _flatten_feat(teacher_map.detach())
        fg = fg_mask.reshape(bsz, -1).bool()
        zero = student_map.new_zeros(())
        if not fg.any() or target_scores.shape[:2] != fg.shape:
            return zero

        valid_target = fg & (target_scores.amax(dim=-1) > 0)
        if not valid_target.any():
            return zero

        if (
            student_scores is None
            or teacher_scores is None
            or student_scores.shape != teacher_scores.shape
            or teacher_scores.shape[:2] != fg.shape
        ):
            raise RuntimeError("CCLKD requires matching student/teacher class logits for COP and LLD.")

        teacher_probs = teacher_scores.detach().sigmoid()
        teacher_conf, teacher_label = teacher_probs.max(dim=-1)
        target_label = target_scores.argmax(dim=-1)
        cop_pos = valid_target & teacher_label.eq(target_label) & (teacher_conf >= self.cclkd_min_confidence)
        if not cop_pos.any():
            return zero

        labels_flat = target_label.reshape(-1)
        cop_pos_flat = cop_pos.reshape(-1)
        valid_flat = valid_target.reshape(-1)
        teacher_probs_flat = teacher_probs.reshape(-1, teacher_probs.shape[-1])
        student_feat_flat = student_flat.reshape(-1, student_flat.shape[-1])
        teacher_feat_flat = teacher_flat.reshape(-1, teacher_flat.shape[-1])

        student_distri_flat = None
        teacher_distri_flat = None
        if (
            student_distri is not None
            and teacher_distri is not None
            and student_distri.shape == teacher_distri.shape
            and student_distri.shape[-1] % 4 == 0
        ):
            student_distri_flat = student_distri.reshape(-1, student_distri.shape[-1])
            teacher_distri_flat = teacher_distri.detach().reshape(-1, teacher_distri.shape[-1])

        classes = labels_flat[cop_pos_flat].unique(sorted=True)
        inv_freq = []
        class_masks = []
        for class_id in classes:
            pos_mask = cop_pos_flat & labels_flat.eq(class_id)
            class_masks.append((class_id, pos_mask))
            inv_freq.append(1.0 / float(pos_mask.sum().clamp_min(1).item()))
        inv_freq_tensor = torch.tensor(inv_freq, device=student_map.device, dtype=student_map.dtype)
        class_weights = inv_freq_tensor / inv_freq_tensor.sum().clamp_min(1e-6)

        lld_loss = zero
        fld_loss = zero
        rld_loss = zero
        ccl_loss = zero
        used_classes = 0

        for class_weight, (class_id, pos_mask) in zip(class_weights, class_masks):
            if not pos_mask.any():
                continue
            pos_idx = torch.where(pos_mask)[0]
            if pos_idx.numel() > self.cclkd_max_tokens:
                pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[: self.cclkd_max_tokens]]

            neg_mask = valid_flat & ~labels_flat.eq(class_id)
            if not neg_mask.any():
                neg_mask = (~valid_flat) & (teacher_conf.reshape(-1) >= self.cclkd_min_confidence)
            neg_idx = torch.where(neg_mask)[0]
            if neg_idx.numel() > self.cclkd_max_tokens:
                neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[: self.cclkd_max_tokens]]

            class_scores = teacher_probs_flat[pos_idx, class_id].clamp(1e-6, 1.0 - 1e-6)
            entropy = -(class_scores * class_scores.log() + (1.0 - class_scores) * (1.0 - class_scores).log())
            entropy = (entropy / 0.6931471805599453).clamp(0.0, 1.0)
            temperature = self.cclkd_temperature_min + (
                self.cclkd_temperature_max - self.cclkd_temperature_min
            ) * torch.sigmoid(self.cclkd_entropy_scale * (entropy - 0.5))
            temperature = temperature.clamp(self.cclkd_temperature_min, self.cclkd_temperature_max)

            # LLD: YOLO11 DFL spatial-distribution KD. Classification logits
            # are deliberately excluded because CCLKD LLD is localization-only.
            box_lld = zero
            if student_distri_flat is not None and teacher_distri_flat is not None:
                reg_max = student_distri_flat.shape[-1] // 4
                if reg_max > 0:
                    s_box = student_distri_flat[pos_idx].reshape(-1, 4, reg_max)
                    t_box = teacher_distri_flat[pos_idx].reshape(-1, 4, reg_max)
                    box_lld = F.kl_div(
                        F.log_softmax(s_box / temperature.view(-1, 1, 1), dim=-1),
                        F.softmax(t_box / temperature.view(-1, 1, 1), dim=-1),
                        reduction="none",
                    ).sum(dim=(-1, -2))
                    box_lld = (box_lld * temperature.pow(2)).mean()
            lld_loss = lld_loss + class_weight * box_lld

            # FLD: category-masked feature MSE, not probability KL.
            fld_loss = fld_loss + class_weight * F.mse_loss(student_feat_flat[pos_idx], teacher_feat_flat[pos_idx])

            # RLD: feature-dimension correlation alignment, C = R^T R / n.
            if pos_idx.numel() > 1:
                s_rel = F.normalize(student_feat_flat[pos_idx], dim=-1, eps=1e-6)
                t_rel = F.normalize(teacher_feat_flat[pos_idx], dim=-1, eps=1e-6)
                n_pos = float(pos_idx.numel())
                s_corr = s_rel.transpose(0, 1) @ s_rel / n_pos
                t_corr = t_rel.transpose(0, 1) @ t_rel / n_pos
                rld_loss = rld_loss + class_weight * F.mse_loss(s_corr, t_corr)

            # CCL: class-balanced target/non-target teacher-student alignment.
            if neg_idx.numel() > 0:
                if neg_idx.numel() >= pos_idx.numel():
                    sampled_neg = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[: pos_idx.numel()]]
                else:
                    sampled_neg = neg_idx
                min_n = min(pos_idx.numel(), sampled_neg.numel())
                if min_n == 0:
                    used_classes += 1
                    continue
                s_anchor = F.normalize(student_feat_flat[pos_idx[:min_n]], dim=-1, eps=1e-6)
                t_pos = F.normalize(teacher_feat_flat[pos_idx[:min_n]], dim=-1, eps=1e-6)
                t_neg = F.normalize(teacher_feat_flat[sampled_neg[:min_n]], dim=-1, eps=1e-6)
                pos_sim = (s_anchor * t_pos).sum(dim=-1) / self.cclkd_contrastive_temperature
                neg_sim = (s_anchor * t_neg).sum(dim=-1) / self.cclkd_contrastive_temperature
                ccl_loss = ccl_loss + class_weight * (-torch.log_softmax(torch.stack((pos_sim, neg_sim), dim=-1), dim=-1)[:, 0].mean())

            used_classes += 1

        if used_classes == 0:
            return zero
        return self.cclkd_logit_weight * lld_loss + self.cclkd_feat_weight * (fld_loss + rld_loss) + self.cclkd_contrast_weight * ccl_loss

    def _compute_profile_kd_loss(
        self,
        student_map: torch.Tensor,
        teacher_map: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores: torch.Tensor,
        student_distri: torch.Tensor | None,
        teacher_distri: torch.Tensor | None,
        student_scores: torch.Tensor | None,
        teacher_scores: torch.Tensor | None,
        anchor_points: torch.Tensor | None = None,
        stride_tensor: torch.Tensor | None = None,
        target_bboxes: torch.Tensor | None = None,
        student_bboxes: torch.Tensor | None = None,
        teacher_bboxes: torch.Tensor | None = None,
        gt_bboxes: torch.Tensor | None = None,
        mask_gt: torch.Tensor | None = None,
        imgsz: torch.Tensor | None = None,
        level_index: int | None = None,
        num_levels: int | None = None,
    ) -> torch.Tensor:
        if self.comparison_kd_profile == "none" or self.profile_kd_weight <= 0:
            return student_map.new_zeros(())
        if self.comparison_kd_profile == "fgd":
            return self._fgd_style_loss(student_map, teacher_map, fg_mask, gt_bboxes=gt_bboxes, mask_gt=mask_gt, imgsz=imgsz)
        if self.comparison_kd_profile == "ld":
            return self._ld_style_loss(
                student_distri,
                teacher_distri,
                fg_mask,
                target_scores,
                teacher_scores,
                teacher_bboxes=teacher_bboxes,
                gt_bboxes=gt_bboxes,
                mask_gt=mask_gt,
                level_stride_tensor=stride_tensor,
            )
        if self.comparison_kd_profile == "cmdistill":
            return self._cmdistill_style_loss(
                student_map,
                teacher_map,
                fg_mask,
                target_scores,
                student_distri,
                teacher_distri,
                student_scores,
                teacher_scores,
                student_bboxes,
                teacher_bboxes,
                level_index,
                num_levels,
            )
        if self.comparison_kd_profile == "cclkd":
            return self._cclkd_style_loss(
                student_map,
                teacher_map,
                fg_mask,
                target_scores,
                student_distri,
                teacher_distri,
                student_scores,
                teacher_scores,
            )
        raise AssertionError(f"Unexpected comparison_kd_profile: {self.comparison_kd_profile}")

    def _compute_decomposition_losses(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        student_pred_distri: torch.Tensor,
        student_pred_scores: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        student_pred_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        zero = torch.zeros((), device=self.device)
        self._cmdistill_last_stats = {}
        if (
            self.teacher_model is None
            or "teacher_img" not in batch
            or "student_base_feats" not in preds
            or "z_s_feats" not in preds
            or "r_s_feats" not in preds
        ):
            return (zero,) * 13

        with torch.no_grad():
            teacher_outputs = self.teacher_model(batch["teacher_img"])
        teacher_preds = _unwrap_teacher_preds(teacher_outputs)
        teacher_feats = teacher_preds.get("feats")
        teacher_distri_all = teacher_preds.get("boxes")
        if isinstance(teacher_distri_all, torch.Tensor) and teacher_distri_all.dim() == 3:
            if teacher_distri_all.shape[1] == student_pred_distri.shape[-1]:
                teacher_distri_all = teacher_distri_all.permute(0, 2, 1).contiguous()
            if teacher_distri_all.shape != student_pred_distri.shape:
                teacher_distri_all = None
        if self.comparison_kd_profile in {"ld", "cmdistill"} and (
            not isinstance(teacher_distri_all, torch.Tensor)
            or teacher_distri_all.shape != student_pred_distri.shape
        ):
            raw_shape = (
                tuple(teacher_preds["boxes"].shape)
                if isinstance(teacher_preds.get("boxes"), torch.Tensor)
                else None
            )
            raise RuntimeError(
                f"{self.comparison_kd_profile} could not obtain raw teacher DFL logits matching the student distribution. "
                f"raw_teacher_boxes={raw_shape}, student={tuple(student_pred_distri.shape)}. "
                "The teacher eval forward must return (decoded_predictions, raw_predictions_dict)."
            )
        teacher_pred_bboxes = None
        if self.comparison_kd_profile in {"ld", "cmdistill"} and isinstance(teacher_distri_all, torch.Tensor):
            # Decoded boxes are in the same stride/grid unit as `anchor_points`.
            # Per-level LD converts them to pixel xyxy via `stride_tensor` only
            # when comparing to pixel-space GT boxes for VLR-style weights.
            teacher_pred_bboxes = self.bbox_decode(anchor_points, teacher_distri_all)
        teacher_scores_all = teacher_preds.get("scores")
        if isinstance(teacher_scores_all, torch.Tensor) and teacher_scores_all.dim() == 3:
            if teacher_scores_all.shape[1] == student_pred_scores.shape[-1]:
                teacher_scores_all = teacher_scores_all.permute(0, 2, 1).contiguous()
            if teacher_scores_all.shape[:2] != student_pred_scores.shape[:2]:
                teacher_scores_all = None
        if teacher_feats is None:
            return (zero,) * 13

        student_raw_feats = preds["student_base_feats"]
        student_z_feats = preds["z_s_feats"]
        student_r_feats = preds["r_s_feats"]
        student_recon_feats = preds.get("student_recon_feats")
        if student_recon_feats is None:
            return (zero,) * 13
        if not (
            len(teacher_feats)
            == len(student_raw_feats)
            == len(student_z_feats)
            == len(student_r_feats)
            == len(student_recon_feats)
        ):
            raise ValueError("Teacher/student decomposition KD expects matching multi-scale P3/P4/P5 feature counts.")

        target_modules = self.teacher_target_modules or {}
        target_teacher_decomposition = target_modules.get("teacher_decomposition", self.student_model.teacher_decomposition)
        target_teacher_decoder = target_modules.get("teacher_decoder", self.student_model.teacher_decoder)
        target_teacher_task_heads = target_modules.get("teacher_task_heads", self.student_model.teacher_task_heads)
        student_branch_split = self.student_branch_mode == "split"
        student_branch_use_zs = self.student_branch_mode in {"split", "single_proj"}
        teacher_decomposed = self.teacher_feature_mode == "decomposed"
        teacher_projected_raw = self.teacher_feature_mode == "projected_raw"

        rec_loss = zero
        reach_match_loss = zero
        reach_rank_loss = zero
        task_loss = zero
        kd_loss = zero
        student_rec_loss = zero
        d_pos_mean = zero
        d_neg_mean = zero
        rank_gap_mean = zero
        mask_mean = zero
        mask_std = zero
        mask_fg_mean = zero
        mask_bg_mean = zero
        pos_levels = 0
        all_levels = 0
        reach_levels = 0
        mask_levels = 0
        mask_fg_levels = 0
        rec_levels = 0
        student_rec_levels = 0
        profile_levels = 0
        profile_kd_loss = zero
        cmd_pcc_loss = zero
        cmd_pcc_levels = 0
        cmd_relation_loss = zero
        cmd_relation_levels = 0
        offset = 0

        for i, (teacher_feat, student_raw, z_s_map, r_s_map, student_recon) in enumerate(
            zip(teacher_feats, student_raw_feats, student_z_feats, student_r_feats, student_recon_feats)
        ):
            if teacher_decomposed:
                z_t_map, u_t_map, mask_map, recon_map = self.student_model.teacher_decomposition[i](teacher_feat)
                decoded_z_t = self.student_model.teacher_decoder[i](z_t_map)
                task_logits = self.student_model.teacher_task_heads[i](decoded_z_t)
            else:
                z_t_map = self.student_model.teacher_decoder[i](teacher_feat) if teacher_projected_raw else teacher_feat
                u_t_map = torch.zeros_like(z_t_map)
                mask_map = None
                recon_map = None
                task_logits = None

            need_q_s = teacher_decomposed and (
                self.reachability_enabled
                or self.kd_weight_mode == "reachability_gap"
            )
            if need_q_s:
                if self.reach_input_mode == "adapter":
                    q_s_map = self.student_model.student_reachability[i](student_raw)
                else:
                    q_s_map = student_raw
            else:
                q_s_map = None

            if teacher_decomposed and self.teacher_target_modules is not None:
                with torch.no_grad():
                    target_z_t_map, target_u_t_map, _, _ = target_teacher_decomposition[i](teacher_feat)
                    target_task_logits = target_teacher_task_heads[i](target_teacher_decoder[i](target_z_t_map))
            else:
                target_z_t_map, target_u_t_map, target_task_logits = (
                    z_t_map,
                    u_t_map,
                    task_logits,
                )

            z_t = _flatten_feat(z_t_map)
            z_s_source_map = z_s_map if student_branch_use_zs else student_raw
            if self.kd_calibration_mode == "none":
                z_s_kd_map = z_s_source_map
            else:
                z_s_kd_map = self.student_model.student_kd_calibration[i](z_s_source_map)
            z_s_kd = _flatten_feat(z_s_kd_map)
            task_pred = _flatten_feat(task_logits) if task_logits is not None else None
            task_pred_target = _flatten_feat(target_task_logits) if target_task_logits is not None else None
            q_s = _flatten_feat(q_s_map) if q_s_map is not None else None
            target_z_t = _flatten_feat(target_z_t_map)
            target_u_t = _flatten_feat(target_u_t_map)

            batch_size, _, height, width = z_t_map.shape
            n_tokens = z_t.shape[1]
            level_fg_mask = fg_mask[:, offset : offset + n_tokens].bool()
            level_target_scores = target_scores[:, offset : offset + n_tokens].to(z_t.dtype)
            level_student_distri = student_pred_distri[:, offset : offset + n_tokens].to(z_t.dtype)
            level_teacher_distri = (
                teacher_distri_all[:, offset : offset + n_tokens].to(z_t.dtype)
                if isinstance(teacher_distri_all, torch.Tensor)
                else None
            )
            level_anchor_points = anchor_points[offset : offset + n_tokens].to(z_t.dtype)
            level_stride_tensor = stride_tensor[offset : offset + n_tokens].to(z_t.dtype)
            level_target_bboxes = target_bboxes[:, offset : offset + n_tokens].to(z_t.dtype)
            level_student_bboxes = student_pred_bboxes[:, offset : offset + n_tokens].to(z_t.dtype)
            level_teacher_bboxes = (
                teacher_pred_bboxes[:, offset : offset + n_tokens].to(z_t.dtype)
                if isinstance(teacher_pred_bboxes, torch.Tensor)
                else None
            )
            level_student_scores = student_pred_scores[:, offset : offset + n_tokens].to(z_t.dtype)
            level_teacher_scores = (
                teacher_scores_all[:, offset : offset + n_tokens].to(z_t.dtype)
                if isinstance(teacher_scores_all, torch.Tensor)
                else None
            )
            offset += n_tokens
            if self.use_fg_mask_for_rec and level_fg_mask.any():
                rec_mask_map = level_fg_mask.reshape(batch_size, height, width).unsqueeze(1).to(teacher_feat.dtype)
            elif self.use_fg_mask_for_rec:
                rec_mask_map = None
            else:
                rec_mask_map = None

            if teacher_decomposed and (not self.use_fg_mask_for_rec or rec_mask_map is not None):
                rec_loss = rec_loss + _masked_l1_loss(recon_map, teacher_feat, rec_mask_map)
                rec_levels += 1
            if student_branch_split and (not self.use_fg_mask_for_rec or rec_mask_map is not None):
                student_rec_loss = student_rec_loss + _masked_l1_loss(student_recon, student_raw, rec_mask_map)
                student_rec_levels += 1

            if task_pred is not None:
                if self.task_loss_fg_only:
                    # FG-only BCE: 仅在 yolo TaskAlignedAssigner 标记为 foreground 的 spatial token 上施加。
                    # 目的:防止 task loss 对 background token 强制压 0,保护 SAR student 在"语义模糊但 SAR 强响应"
                    # 位置上的 confidence calibration(避免 Es1 type Q3 dConf 暴跌)。
                    bce_full = F.binary_cross_entropy_with_logits(
                        task_pred, level_target_scores, reduction="none"
                    )
                    fg_w = level_fg_mask.to(bce_full.dtype).unsqueeze(-1)
                    denom = fg_w.sum() * level_target_scores.shape[-1]
                    if denom > 0:
                        task_loss = task_loss + (bce_full * fg_w).sum() / denom
                else:
                    task_loss = task_loss + self.bce(task_pred, level_target_scores).sum() / target_scores_sum

            if mask_map is not None:
                mask_mean = mask_mean + mask_map.mean()
                mask_std = mask_std + mask_map.std(unbiased=False)
                mask_levels += 1

            if teacher_decomposed and self.reachability_enabled:
                if self.use_fg_mask_for_reach:
                    if level_fg_mask.any():
                        reach_mask_map = level_fg_mask.reshape(batch_size, height, width).unsqueeze(1).to(z_t_map.dtype)
                    else:
                        reach_mask_map = None
                else:
                    reach_mask_map = None

                if not self.use_fg_mask_for_reach or level_fg_mask.any():
                    level_match_loss, level_rank_loss, level_d_pos, level_d_neg, level_gap = self.normalized_reachability_loss(
                        z_t_map=z_t_map,
                        u_t_map=u_t_map,
                        q_s_map=q_s_map,
                        fg_mask_map=reach_mask_map,
                    )
                    reach_match_loss = reach_match_loss + level_match_loss
                    reach_rank_loss = reach_rank_loss + level_rank_loss
                    d_pos_mean = d_pos_mean + level_d_pos
                    d_neg_mean = d_neg_mean + level_d_neg
                    rank_gap_mean = rank_gap_mean + level_gap
                    reach_levels += 1

            if self.comparison_kd_profile == "cmdistill" and self.profile_kd_weight > 0:
                is_first_level = i == 0
                is_last_level = i == len(teacher_feats) - 1
                cmd_teacher_map = target_z_t_map.detach() if self.kd_target_mode == "detach" else target_z_t_map
                if self.cmdistill_feature_weight > 0 and (is_first_level or is_last_level):
                    cmd_pcc_loss = cmd_pcc_loss + self._cmdistill_pcc_feature_loss(z_s_kd_map, cmd_teacher_map)
                    cmd_pcc_levels += 1
                if self.cmdistill_relation_weight > 0 and is_last_level:
                    cmd_relation_loss = cmd_relation_loss + self._cmdistill_relation_loss(z_s_kd_map, cmd_teacher_map)
                    cmd_relation_levels += 1
            elif self.comparison_kd_profile != "none" and self.profile_kd_weight > 0:
                level_profile_loss = self._compute_profile_kd_loss(
                    z_s_kd_map,
                    target_z_t_map.detach() if self.kd_target_mode == "detach" else target_z_t_map,
                    level_fg_mask,
                    level_target_scores,
                    level_student_distri,
                    level_teacher_distri,
                    level_student_scores,
                    level_teacher_scores,
                    level_anchor_points,
                    level_stride_tensor,
                    level_target_bboxes,
                    level_student_bboxes,
                    level_teacher_bboxes,
                    gt_bboxes.to(z_t.dtype),
                    mask_gt,
                    imgsz.to(z_t.dtype),
                    i,
                    len(teacher_feats),
                )
                profile_kd_loss = profile_kd_loss + level_profile_loss
                profile_levels += 1

            if not level_fg_mask.any():
                all_levels += 1
                continue

            fg_mask_map = level_fg_mask.reshape(batch_size, height, width).unsqueeze(1).to(r_s_map.dtype)
            if mask_map is not None:
                fg_mask_bool = fg_mask_map > 0
                bg_mask_bool = ~fg_mask_bool
                mask_fg_mean = mask_fg_mean + (mask_map[fg_mask_bool].mean() if fg_mask_bool.any() else zero)
                mask_bg_mean = mask_bg_mean + (mask_map[bg_mask_bool].mean() if bg_mask_bool.any() else zero)
                mask_fg_levels += 1

            z_s_kd_pos = z_s_kd[level_fg_mask]
            task_pred_pos = task_pred_target[level_fg_mask] if task_pred_target is not None else None
            level_target_scores_pos = level_target_scores[level_fg_mask]
            q_s_pos = q_s[level_fg_mask] if q_s is not None else None

            kd_target_pos = target_z_t[level_fg_mask].detach() if self.kd_target_mode == "detach" else target_z_t[level_fg_mask]
            kd_pos_weights = self._compute_kd_pos_weights(
                task_pred_pos,
                level_target_scores_pos,
                q_s_pos=q_s_pos,
                z_t_pos=(target_z_t[level_fg_mask] if teacher_decomposed else None),
                u_t_pos=(target_u_t[level_fg_mask] if teacher_decomposed else None),
            )
            if not self.profile_kd_replace_base:
                kd_loss = kd_loss + self._compute_kd_loss(
                    z_s_kd_pos,
                    kd_target_pos,
                    level_target_scores_pos,
                    kd_pos_weights,
                )

            pos_levels += 1
            all_levels += 1

        if all_levels == 0:
            return (zero,) * 13

        rec_loss = rec_loss / rec_levels if rec_levels > 0 else zero
        student_rec_loss = student_rec_loss / student_rec_levels if student_rec_levels > 0 else zero
        task_loss = task_loss / all_levels
        if mask_levels > 0:
            mask_mean = mask_mean / mask_levels
            mask_std = mask_std / mask_levels
        else:
            mask_mean = zero
            mask_std = zero
        if mask_fg_levels > 0:
            mask_fg_mean = mask_fg_mean / mask_fg_levels
            mask_bg_mean = mask_bg_mean / mask_fg_levels
        else:
            mask_fg_mean = zero
            mask_bg_mean = zero

        if reach_levels > 0:
            reach_match_loss = reach_match_loss / reach_levels
            reach_rank_loss = reach_rank_loss / reach_levels
            d_pos_mean = d_pos_mean / reach_levels
            d_neg_mean = d_neg_mean / reach_levels
            rank_gap_mean = rank_gap_mean / reach_levels
        else:
            reach_match_loss = zero
            reach_rank_loss = zero
            d_pos_mean = zero
            d_neg_mean = zero
            rank_gap_mean = zero

        if pos_levels > 0:
            kd_loss = kd_loss / pos_levels
        else:
            kd_loss = zero

        if profile_levels > 0:
            kd_loss = kd_loss + self.profile_kd_weight * (profile_kd_loss / profile_levels)

        if self.comparison_kd_profile == "cmdistill" and self.profile_kd_weight > 0:
            cmd_output_loss = zero
            if self.cmdistill_logit_weight > 0:
                cmd_output_loss = self._cmdistill_output_loss(
                    student_pred_distri,
                    teacher_distri_all,
                    student_pred_scores,
                    teacher_scores_all,
                    fg_mask,
                    target_scores,
                    student_pred_bboxes,
                    teacher_pred_bboxes,
                )
            cmd_total, cmd_feature_term, cmd_relation_term, cmd_output_term = self._cmdistill_combine_components(
                cmd_pcc_loss,
                cmd_pcc_levels,
                cmd_relation_loss,
                cmd_relation_levels,
                cmd_output_loss,
            )
            kd_loss = kd_loss + self.profile_kd_weight * cmd_total
            self._cmdistill_update_stats(
                cmdistill_pcc_levels=int(cmd_pcc_levels),
                cmdistill_pcc_loss=float(cmd_feature_term.detach().cpu()),
                cmdistill_relation_loss=float(cmd_relation_term.detach().cpu()),
                cmdistill_ibcld_loss=float(cmd_output_term.detach().cpu()),
                cmdistill_total_loss=float(cmd_total.detach().cpu()),
            )

        reach_match_loss = (
            reach_match_loss
            * self.lambda_reach
            * self.lambda_match_inner
            * self.phase_loss_scales["match"]
        )
        reach_rank_loss = (
            reach_rank_loss
            * self.lambda_reach
            * self.lambda_rank_inner
            * self.phase_loss_scales["unmatch"]
        )

        return (
            rec_loss * self.lambda_rec * self.phase_loss_scales["rec"],
            reach_match_loss,
            reach_rank_loss,
            task_loss * self.lambda_taskL * self.phase_loss_scales["task"],
            kd_loss * self.alpha_kd * self.phase_loss_scales["kd"],
            student_rec_loss * self.alpha_s_rec * self.phase_loss_scales["student_rec"],
            d_pos_mean,
            d_neg_mean,
            rank_gap_mean,
            mask_mean,
            mask_std,
            mask_fg_mean,
            mask_bg_mean,
        )
