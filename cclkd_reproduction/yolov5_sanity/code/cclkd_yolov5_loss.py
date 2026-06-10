#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class PositiveCandidates:
    vectors: torch.Tensor
    labels: torch.Tensor
    indices: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    levels: torch.Tensor
    batch_indices: torch.Tensor
    grid_x: torch.Tensor
    grid_y: torch.Tensor


class YoloV5FeatureCapture:
    """Capture YOLOv5 Detect input feature maps for box-aligned KD."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._features: list[torch.Tensor] = []
        self._handles: list[Any] = []

    def clear(self):
        self._features = []

    def install(self):
        for module in self.model.modules():
            if module.__class__.__name__ == "Detect":
                self._handles.append(module.register_forward_pre_hook(self._hook))
        if not self._handles:
            raise RuntimeError("Could not find YOLOv5 Detect module for feature capture")
        return self

    @property
    def features(self) -> list[torch.Tensor]:
        return self._features

    def _hook(self, _module, inputs):
        if not inputs:
            self._features = []
            return
        x = inputs[0]
        if isinstance(x, (list, tuple)):
            self._features = list(x)
        else:
            self._features = [x]


def positive_vectors(preds, indices):
    out = []
    for level, pi in enumerate(preds):
        b, a, gj, gi = indices[level]
        if b.numel():
            out.append(pi[b, a, gj, gi])
    if out:
        return torch.cat(out, 0)
    return preds[0].new_zeros((0, preds[0].shape[-1]))


def collect_yolov5_positive_candidates(preds, targets, compute_loss) -> PositiveCandidates:
    with torch.no_grad():
        tcls, _tbox, indices, _anchors = compute_loss.build_targets(preds, targets)
    vectors = positive_vectors(preds, indices)
    device = preds[0].device
    if vectors.numel() == 0:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return PositiveCandidates(vectors, empty, indices, empty, empty, empty, empty)

    labels = torch.cat([x.to(device=device, dtype=torch.long) for x in tcls], 0)
    levels, batch_indices, grid_x, grid_y = [], [], [], []
    for level, (b, _a, gj, gi) in enumerate(indices):
        if b.numel() == 0:
            continue
        levels.append(torch.full_like(b, level, device=device, dtype=torch.long))
        batch_indices.append(b.to(device=device, dtype=torch.long))
        grid_x.append(gi.to(device=device, dtype=torch.long))
        grid_y.append(gj.to(device=device, dtype=torch.long))
    return PositiveCandidates(
        vectors=vectors,
        labels=labels,
        indices=indices,
        levels=torch.cat(levels, 0),
        batch_indices=torch.cat(batch_indices, 0),
        grid_x=torch.cat(grid_x, 0),
        grid_y=torch.cat(grid_y, 0),
    )


def build_teacher_cop(teacher_vectors, labels, nc: int):
    device = teacher_vectors.device
    n = teacher_vectors.shape[0]
    if n == 0:
        empty_bool = torch.empty(0, device=device, dtype=torch.bool)
        empty_float = torch.empty(0, device=device)
        return {
            "valid_mask": empty_bool,
            "positive_mask": empty_bool,
            "teacher_scores": empty_float,
            "teacher_labels": torch.empty(0, device=device, dtype=torch.long),
            "class_counts": torch.zeros(nc, device=device),
        }

    cls_probs = teacher_vectors[:, 5:].detach().sigmoid()
    teacher_labels = cls_probs.argmax(dim=1)
    safe_labels = labels.clamp(min=0, max=max(nc - 1, 0))
    teacher_scores = cls_probs.gather(1, safe_labels.view(-1, 1)).squeeze(1)
    valid_mask = torch.ones(n, device=device, dtype=torch.bool)
    positive_mask = teacher_labels == safe_labels
    class_counts = torch.zeros(nc, device=device)
    for cls in range(nc):
        class_counts[cls] = ((safe_labels == cls) & positive_mask).sum()
    return {
        "valid_mask": valid_mask,
        "positive_mask": positive_mask,
        "teacher_scores": teacher_scores,
        "teacher_labels": teacher_labels,
        "class_counts": class_counts,
    }


def adaptive_temperature_from_teacher_scores(
    teacher_scores,
    mask,
    t_min: float = 0.5,
    t_max: float = 5.0,
    entropy_scale: float = 5.0,
):
    if mask.sum() == 0:
        return teacher_scores.new_tensor((t_min + t_max) * 0.5)
    scores = teacher_scores[mask].detach().clamp(min=1e-6, max=1.0)
    entropy = -(scores * scores.log()).mean()
    return t_min + (t_max - t_min) * torch.sigmoid(entropy_scale * entropy)


def _masked_distribution(values, mask, temperature):
    weights = mask.to(values.dtype).view(-1, 1)
    logits = (values * weights).reshape(-1) / temperature.clamp_min(1e-6)
    return F.softmax(logits, dim=0)


def spatial_distribution_kl(student_box_probs, teacher_box_probs, mask, temperature):
    if mask.sum() == 0:
        return student_box_probs.new_zeros(())
    weights = mask.to(student_box_probs.dtype).view(-1, 1)
    t = temperature.clamp_min(1e-6)
    s_logits = (student_box_probs * weights).reshape(-1) / t
    t_logits = (teacher_box_probs.detach() * weights).reshape(-1) / t
    return F.kl_div(F.log_softmax(s_logits, dim=0), F.softmax(t_logits, dim=0), reduction="sum") * t * t


def sample_box_features(
    features: list[torch.Tensor],
    levels: torch.Tensor,
    batch_indices: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    roi_grid_size: int = 3,
):
    """Sample local Detect-input feature vectors.

    This implementation uses direct sampled feature vectors. The 1x1
    projection layer from the paper is intentionally deferred to keep the
    smoke/audit optimizer surface unchanged.
    """

    n = levels.numel()
    if n == 0:
        device = features[0].device if features else levels.device
        return torch.empty(0, 0, device=device), False
    if not features:
        return torch.empty(n, 0, device=levels.device), False

    max_dim = max(int(f.shape[1]) * roi_grid_size * roi_grid_size for f in features)
    sampled_all = features[0].new_zeros((n, max_dim))
    ok = True
    offsets = torch.linspace(
        -(roi_grid_size // 2),
        roi_grid_size // 2,
        roi_grid_size,
        device=levels.device,
        dtype=features[0].dtype,
    )

    for level in levels.unique(sorted=True):
        level_int = int(level.item())
        mask = levels == level
        if level_int >= len(features):
            ok = False
            continue
        feat = features[level_int]
        _, _c, h, w = feat.shape
        rows = torch.where(mask)[0]
        b = batch_indices[mask].long().clamp_(0, feat.shape[0] - 1)
        gx = grid_x[mask].to(dtype=feat.dtype).clamp(0, max(w - 1, 0))
        gy = grid_y[mask].to(dtype=feat.dtype).clamp(0, max(h - 1, 0))
        xx = (gx[:, None, None] + offsets[None, None, :]).clamp(0, max(w - 1, 0))
        yy = (gy[:, None, None] + offsets[None, :, None]).clamp(0, max(h - 1, 0))
        if w > 1:
            xx = 2.0 * xx / (w - 1) - 1.0
        else:
            xx = torch.zeros_like(xx)
        if h > 1:
            yy = 2.0 * yy / (h - 1) - 1.0
        else:
            yy = torch.zeros_like(yy)
        grid = torch.stack(
            (
                xx.expand(-1, roi_grid_size, -1),
                yy.expand(-1, -1, roi_grid_size),
            ),
            dim=-1,
        )
        sampled = F.grid_sample(feat[b], grid, mode="bilinear", padding_mode="border", align_corners=True)
        flat = sampled.flatten(1)
        sampled_all[rows, : flat.shape[1]] = flat
    return sampled_all, ok


def feature_kl(student_features, teacher_features, mask, temperature):
    if mask.sum() == 0 or student_features.shape[1] == 0 or teacher_features.shape[1] == 0:
        return student_features.new_zeros(())
    t = temperature.clamp_min(1e-6)
    s = student_features[mask].reshape(-1) / t
    tt = teacher_features[mask].detach().reshape(-1) / t
    return F.kl_div(F.log_softmax(s, dim=0), F.softmax(tt, dim=0), reduction="sum") * t * t


def relationship_loss(student_features, teacher_features, mask, temperature):
    if mask.sum() < 2 or student_features.shape[1] == 0 or teacher_features.shape[1] == 0:
        return student_features.new_zeros(())
    s = F.normalize(student_features[mask], dim=1, eps=1e-6)
    t = F.normalize(teacher_features[mask].detach(), dim=1, eps=1e-6)
    return F.mse_loss(s @ s.T, t @ t.T) * temperature * temperature


def contrastive_loss(student_box_probs, teacher_box_probs, pos_mask, neg_mask, temperature, tau: float = 0.1):
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return student_box_probs.new_zeros(())
    b_s = _masked_distribution(student_box_probs, pos_mask, temperature)
    b_t = _masked_distribution(teacher_box_probs.detach(), pos_mask, temperature)
    hat_s = _masked_distribution(student_box_probs, neg_mask, temperature)
    hat_t = _masked_distribution(teacher_box_probs.detach(), neg_mask, temperature)
    pos_sim = F.cosine_similarity(b_t, b_s, dim=0) / tau
    neg_sim = F.cosine_similarity(hat_t, hat_s, dim=0) / tau
    logits = torch.stack((pos_sim, neg_sim), dim=0)
    return -F.log_softmax(logits, dim=0)[0]


def _zero_outputs(device_tensor, nc: int, student_feature_levels: int, teacher_feature_levels: int, mode: str):
    zero = device_tensor.new_zeros(())
    diagnostics = {
        "mode": mode,
        "cop_valid_candidates": 0.0,
        "cop_positive_candidates": 0.0,
        "cop_positive_ratio": 0.0,
        "cop_class0_count": 0.0,
        "cop_class1_count": 0.0,
        "cop_class2_count": 0.0,
        "neg_candidates_mean": 0.0,
        "temperature_mean": 0.0,
        "temperature_min": 0.0,
        "temperature_max": 0.0,
        "feature_capture_ok": 0.0,
        "student_feature_levels": float(student_feature_levels),
        "teacher_feature_levels": float(teacher_feature_levels),
        "nan_or_inf_detected": 0.0,
    }
    return zero, torch.stack((zero, zero, zero, zero)), diagnostics


def cclkd_paper_loss(
    student_preds,
    teacher_preds,
    targets,
    student_loss,
    student_features: list[torch.Tensor],
    teacher_features: list[torch.Tensor],
    mode: str,
    nc: int,
    t_min: float = 0.5,
    t_max: float = 5.0,
    entropy_scale: float = 5.0,
    contrastive_temperature: float = 0.1,
    roi_grid_size: int = 3,
):
    candidates = collect_yolov5_positive_candidates(student_preds, targets, student_loss)
    teacher_vectors = positive_vectors(teacher_preds, candidates.indices).detach()
    student_feature_levels = len(student_features)
    teacher_feature_levels = len(teacher_features)
    if candidates.vectors.numel() == 0:
        return _zero_outputs(student_preds[0], nc, student_feature_levels, teacher_feature_levels, mode)

    cop = build_teacher_cop(teacher_vectors, candidates.labels, nc)
    valid_mask = cop["valid_mask"]
    positive_mask = cop["positive_mask"]
    student_box_probs = candidates.vectors[:, :4].sigmoid()
    teacher_box_probs = teacher_vectors[:, :4].sigmoid()

    detached_teacher_features = [x.detach() for x in teacher_features]
    student_sampled, student_features_ok = sample_box_features(
        student_features,
        candidates.levels,
        candidates.batch_indices,
        candidates.grid_x,
        candidates.grid_y,
        roi_grid_size=roi_grid_size,
    )
    teacher_sampled, teacher_features_ok = sample_box_features(
        detached_teacher_features,
        candidates.levels,
        candidates.batch_indices,
        candidates.grid_x,
        candidates.grid_y,
        roi_grid_size=roi_grid_size,
    )

    lld = candidates.vectors.new_zeros(())
    fld = candidates.vectors.new_zeros(())
    rld = candidates.vectors.new_zeros(())
    ccl = candidates.vectors.new_zeros(())
    temperatures = []
    class_weights = []
    ccl_terms = []
    neg_counts = []

    include_atkd = mode in {"paper_atkd_only", "paper_full"}
    include_ccl = mode in {"paper_ccl_only", "paper_full"}
    safe_labels = candidates.labels.clamp(min=0, max=max(nc - 1, 0))

    for cls in range(nc):
        class_pos = positive_mask & (safe_labels == cls)
        if class_pos.sum() == 0:
            continue
        class_neg = valid_mask & ~class_pos
        temperature = adaptive_temperature_from_teacher_scores(
            cop["teacher_scores"], class_pos, t_min=t_min, t_max=t_max, entropy_scale=entropy_scale
        )
        temperatures.append(temperature.detach())
        neg_counts.append(class_neg.sum().detach().float())
        if include_atkd:
            lld = lld + spatial_distribution_kl(student_box_probs, teacher_box_probs, class_pos, temperature)
            lld = lld + spatial_distribution_kl(student_box_probs, teacher_box_probs, class_neg, temperature)
            fld = fld + feature_kl(student_sampled, teacher_sampled, class_pos, temperature)
            rld = rld + relationship_loss(student_sampled, teacher_sampled, class_pos, temperature)
        if include_ccl and class_neg.sum() > 0:
            ccl_terms.append(
                contrastive_loss(
                    student_box_probs,
                    teacher_box_probs,
                    class_pos,
                    class_neg,
                    temperature,
                    tau=contrastive_temperature,
                )
            )
            class_weights.append(1.0 / class_pos.sum().detach().float().clamp_min(1.0))

    class_count = max(len(temperatures), 1)
    if include_atkd:
        lld = lld / class_count
        fld = fld / class_count
        rld = rld / class_count
    if include_ccl and ccl_terms:
        weights = torch.stack(class_weights)
        weights = weights / weights.sum().clamp_min(1e-6)
        ccl = torch.stack(ccl_terms).mul(weights.to(ccl_terms[0].device)).sum()

    total = lld + fld + rld + ccl
    temp_tensor = torch.stack(temperatures) if temperatures else candidates.vectors.new_zeros(1)
    finite_tensors = torch.stack((total.detach(), lld.detach(), fld.detach(), rld.detach(), ccl.detach()))
    nan_or_inf = float((~torch.isfinite(finite_tensors)).any().item())
    class_counts = cop["class_counts"].detach()
    positive_count = float(positive_mask.sum().detach().item())
    valid_count = float(valid_mask.sum().detach().item())
    diagnostics = {
        "mode": mode,
        "cop_valid_candidates": valid_count,
        "cop_positive_candidates": positive_count,
        "cop_positive_ratio": positive_count / max(valid_count, 1.0),
        "cop_class0_count": float(class_counts[0].item()) if nc > 0 else 0.0,
        "cop_class1_count": float(class_counts[1].item()) if nc > 1 else 0.0,
        "cop_class2_count": float(class_counts[2].item()) if nc > 2 else 0.0,
        "neg_candidates_mean": float(torch.stack(neg_counts).mean().item()) if neg_counts else 0.0,
        "temperature_mean": float(temp_tensor.detach().mean().item()),
        "temperature_min": float(temp_tensor.detach().min().item()),
        "temperature_max": float(temp_tensor.detach().max().item()),
        "feature_capture_ok": float(bool(student_features_ok and teacher_features_ok and student_feature_levels and teacher_feature_levels)),
        "student_feature_levels": float(student_feature_levels),
        "teacher_feature_levels": float(teacher_feature_levels),
        "nan_or_inf_detected": nan_or_inf,
    }
    return total, torch.stack((lld.detach(), fld.detach(), rld.detach(), ccl.detach())), diagnostics
