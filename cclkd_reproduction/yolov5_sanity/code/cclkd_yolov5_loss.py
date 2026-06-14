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


def _masked_distribution(values: torch.Tensor, mask: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """Return a distribution over only selected candidate-box position entries."""

    selected = values[mask.bool()]
    if selected.numel() == 0:
        return values.new_zeros((0,))
    logits = selected.reshape(-1) / temperature.clamp_min(1e-6)
    return F.softmax(logits, dim=0)


def spatial_distribution_kl(
    student_box_probs: torch.Tensor,
    teacher_box_probs: torch.Tensor,
    mask: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    mask = mask.bool()
    if mask.sum() == 0:
        return student_box_probs.new_zeros(())
    t = temperature.clamp_min(1e-6)
    s_logits = student_box_probs[mask].reshape(-1) / t
    t_logits = teacher_box_probs.detach()[mask].reshape(-1) / t
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
    mask = mask.bool()
    if mask.sum() == 0 or student_features.shape[1] == 0 or teacher_features.shape[1] == 0:
        return student_features.new_zeros(())
    t = temperature.clamp_min(1e-6)
    s = student_features[mask] / t
    tt = teacher_features.detach()[mask] / t
    return F.kl_div(F.log_softmax(s, dim=-1), F.softmax(tt, dim=-1), reduction="batchmean") * t * t


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


def class_conditioned_box_vectors(vectors: torch.Tensor, class_id: int) -> torch.Tensor:
    """Class-conditioned candidate-box representation for YOLOv5 CCL.

    The legacy CCL path used only xywh logits, so positive and negative sets
    could have nearly identical cosine similarity. This representation injects
    objectness and the class-j probability into the localization vector, which
    better matches the paper's category-constrained candidate distributions.
    """

    box = vectors[:, :4].sigmoid()
    obj = vectors[:, 4:5].sigmoid() if vectors.shape[1] > 4 else torch.ones_like(box[:, :1])
    if vectors.shape[1] > 5 + class_id:
        cls_prob = vectors[:, 5 + class_id : 6 + class_id].sigmoid()
    else:
        cls_prob = torch.ones_like(obj)
    return torch.cat((box * obj * cls_prob, obj * cls_prob, cls_prob), dim=1)


def contrastive_pair_loss(
    student_vectors: torch.Tensor,
    teacher_vectors: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    tau: float = 0.1,
    pair_mode: str = "anchor_teacher_neg",
):
    """InfoNCE-style CCL over candidate vectors.

    `paper_pair` follows the extracted Algorithm 2 literally: positive
    teacher-student candidates compete against non-target teacher-student
    candidates. `anchor_teacher_neg` is the practical category-discriminative
    variant suggested by the paper text: student positives are pulled toward
    teacher positives and pushed away from teacher negatives.
    """

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        zero = student_vectors.new_zeros(())
        return zero, zero.detach(), zero.detach()
    s = F.normalize(student_vectors, dim=-1, eps=1e-6)
    t = F.normalize(teacher_vectors.detach(), dim=-1, eps=1e-6)
    pos_sim_each = (s[pos_mask] * t[pos_mask]).sum(dim=-1)
    if pair_mode == "paper_pair":
        neg_sim_each = (s[neg_mask] * t[neg_mask]).sum(dim=-1)
        pos_sim = pos_sim_each.mean()
        neg_sim = neg_sim_each.mean()
        logits = torch.stack((pos_sim, neg_sim), dim=0) / max(float(tau), 1e-6)
        return -F.log_softmax(logits, dim=0)[0], pos_sim.detach(), neg_sim.detach()
    if pair_mode != "anchor_teacher_neg":
        raise ValueError(f"Unknown CCL pair_mode: {pair_mode!r}")

    n = min(int(pos_mask.sum().item()), int(neg_mask.sum().item()), 256)
    if n <= 0:
        zero = student_vectors.new_zeros(())
        return zero, zero.detach(), zero.detach()
    pos_idx = torch.where(pos_mask)[0]
    neg_idx = torch.where(neg_mask)[0]
    pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[:n]]
    neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[:n]]
    pos_sim = (s[pos_idx] * t[pos_idx]).sum(dim=-1)
    neg_sim = (s[pos_idx] * t[neg_idx]).sum(dim=-1)
    logits = torch.stack((pos_sim, neg_sim), dim=-1) / max(float(tau), 1e-6)
    return -F.log_softmax(logits, dim=-1)[:, 0].mean(), pos_sim.mean().detach(), neg_sim.mean().detach()


def _zero_outputs(
    device_tensor,
    nc: int,
    student_feature_levels: int,
    teacher_feature_levels: int,
    mode: str,
    atkd_weight: float = 0.0,
    ccl_weight: float = 0.0,
):
    zero = device_tensor.new_zeros(())
    diagnostics = {
        "mode": mode,
        "atkd_weight": float(atkd_weight),
        "ccl_weight": float(ccl_weight),
        "atkd_loss": 0.0,
        "weighted_atkd_loss": 0.0,
        "weighted_ccl_loss": 0.0,
        "cop_valid_candidates": 0.0,
        "cop_positive_candidates": 0.0,
        "cop_positive_ratio": 0.0,
        "cop_class0_count": 0.0,
        "cop_class1_count": 0.0,
        "cop_class2_count": 0.0,
        "neg_candidates_mean": 0.0,
        "ccl_pos_sim": 0.0,
        "ccl_neg_sim": 0.0,
        "ccl_margin": 0.0,
        "ccl_valid_classes": 0.0,
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
    atkd_weight: float = 1.0,
    ccl_weight: float = 1.0,
    ccl_source: str = "box_class",
    ccl_pair_mode: str = "anchor_teacher_neg",
):
    if ccl_source not in {"box_proxy", "box_class", "roi_feature"}:
        raise ValueError(f"Unknown CCL source: {ccl_source!r}")
    if ccl_pair_mode not in {"paper_pair", "anchor_teacher_neg"}:
        raise ValueError(f"Unknown CCL pair mode: {ccl_pair_mode!r}")
    candidates = collect_yolov5_positive_candidates(student_preds, targets, student_loss)
    teacher_vectors = positive_vectors(teacher_preds, candidates.indices).detach()
    student_feature_levels = len(student_features)
    teacher_feature_levels = len(teacher_features)
    if candidates.vectors.numel() == 0:
        return _zero_outputs(
            student_preds[0],
            nc,
            student_feature_levels,
            teacher_feature_levels,
            mode,
            atkd_weight=atkd_weight,
            ccl_weight=ccl_weight,
        )

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
    ccl_pos_sims = []
    ccl_neg_sims = []
    neg_counts = []

    include_atkd = atkd_weight != 0.0
    include_ccl = ccl_weight != 0.0
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
            if ccl_source == "box_proxy":
                ccl_term = contrastive_loss(
                    student_box_probs,
                    teacher_box_probs,
                    class_pos,
                    class_neg,
                    temperature,
                    tau=contrastive_temperature,
                )
                zero_sim = ccl_term.detach().new_zeros(())
                pos_sim = zero_sim
                neg_sim = zero_sim
            else:
                if ccl_source == "box_class":
                    ccl_student_vectors = class_conditioned_box_vectors(candidates.vectors, cls)
                    ccl_teacher_vectors = class_conditioned_box_vectors(teacher_vectors, cls)
                else:
                    ccl_student_vectors = student_sampled
                    ccl_teacher_vectors = teacher_sampled
                ccl_term, pos_sim, neg_sim = contrastive_pair_loss(
                    ccl_student_vectors,
                    ccl_teacher_vectors,
                    class_pos,
                    class_neg,
                    tau=contrastive_temperature,
                    pair_mode=ccl_pair_mode,
                )
            ccl_terms.append(
                ccl_term
            )
            ccl_pos_sims.append(pos_sim)
            ccl_neg_sims.append(neg_sim)
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

    atkd = lld + fld + rld
    weighted_atkd = float(atkd_weight) * atkd
    weighted_ccl = float(ccl_weight) * ccl
    total = weighted_atkd + weighted_ccl
    temp_tensor = torch.stack(temperatures) if temperatures else candidates.vectors.new_zeros(1)
    finite_tensors = torch.stack((total.detach(), lld.detach(), fld.detach(), rld.detach(), ccl.detach()))
    nan_or_inf = float((~torch.isfinite(finite_tensors)).any().item())
    class_counts = cop["class_counts"].detach()
    ccl_pos_tensor = torch.stack(ccl_pos_sims) if ccl_pos_sims else candidates.vectors.new_zeros(1)
    ccl_neg_tensor = torch.stack(ccl_neg_sims) if ccl_neg_sims else candidates.vectors.new_zeros(1)
    ccl_margin_tensor = ccl_pos_tensor - ccl_neg_tensor
    positive_count = float(positive_mask.sum().detach().item())
    valid_count = float(valid_mask.sum().detach().item())
    diagnostics = {
        "mode": mode,
        "atkd_weight": float(atkd_weight),
        "ccl_weight": float(ccl_weight),
        "atkd_loss": float(atkd.detach().item()),
        "weighted_atkd_loss": float(weighted_atkd.detach().item()),
        "weighted_ccl_loss": float(weighted_ccl.detach().item()),
        "cop_valid_candidates": valid_count,
        "cop_positive_candidates": positive_count,
        "cop_positive_ratio": positive_count / max(valid_count, 1.0),
        "cop_class0_count": float(class_counts[0].item()) if nc > 0 else 0.0,
        "cop_class1_count": float(class_counts[1].item()) if nc > 1 else 0.0,
        "cop_class2_count": float(class_counts[2].item()) if nc > 2 else 0.0,
        "neg_candidates_mean": float(torch.stack(neg_counts).mean().item()) if neg_counts else 0.0,
        "ccl_pos_sim": float(ccl_pos_tensor.detach().mean().item()),
        "ccl_neg_sim": float(ccl_neg_tensor.detach().mean().item()),
        "ccl_margin": float(ccl_margin_tensor.detach().mean().item()),
        "ccl_valid_classes": float(len(ccl_terms)),
        "temperature_mean": float(temp_tensor.detach().mean().item()),
        "temperature_min": float(temp_tensor.detach().min().item()),
        "temperature_max": float(temp_tensor.detach().max().item()),
        "feature_capture_ok": float(bool(student_features_ok and teacher_features_ok and student_feature_levels and teacher_feature_levels)),
        "student_feature_levels": float(student_feature_levels),
        "teacher_feature_levels": float(teacher_feature_levels),
        "nan_or_inf_detected": nan_or_inf,
    }
    return total, torch.stack((lld.detach(), fld.detach(), rld.detach(), ccl.detach())), diagnostics
