#!/usr/bin/env python3
"""Synthetic smoke checks for non-CCLKD comparison loss profiles.

This does not launch training. It instantiates the loss object through
`__new__` and exercises only the profile helper methods with synthetic tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (
    REPO_ROOT / "shared",
    REPO_ROOT / "shared" / "yolo",
    REPO_ROOT / "ladd" / "code" / "src",
    REPO_ROOT / "ladd" / "code_versions" / "current_hbb" / "src",
):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from teacher_student_decomposition_kd_hbb.loss import (  # noqa: E402
    TeacherStudentDecompositionKDNRRLTeacherUAuxLossHBB,
    _pkd_channel_standardize_map,
)


def make_loss(**overrides):
    obj = TeacherStudentDecompositionKDNRRLTeacherUAuxLossHBB.__new__(
        TeacherStudentDecompositionKDNRRLTeacherUAuxLossHBB
    )
    defaults = {
        "fgd_alpha": 0.0001,
        "fgd_beta": 0.00005,
        "fgd_gamma": 0.001,
        "fgd_lambda": 0.0,
        "fgd_normalization_mode": "original",
        "fgd_temperature": 0.5,
        "fgd_mask_mode": "gt_box",
        "fgd_bg_norm": True,
        "ld_temperature": 10.0,
        "ld_use_vlr": True,
        "ld_quality_power": 1.0,
        "ld_min_vlr_weight": 0.0,
        "ld_vlr_topk": 0,
        "ld_vlr_weight": 0.25,
        "ld_main_weight": 0.25,
        "ld_allow_empty_vlr": True,
        "_ld_warned_missing_teacher_scores": False,
        "cmdistill_feature_weight": 1.0,
        "cmdistill_relation_weight": 1.0,
        "cmdistill_logit_weight": 1.0,
        "cmdistill_temperature": 4.0,
        "cmdistill_max_tokens": 64,
        "cmdistill_min_confidence": 0.05,
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        setattr(obj, key, value)
    return obj


def check_fgd():
    torch.manual_seed(1)
    criterion = make_loss()
    student = torch.randn(2, 16, 8, 8, requires_grad=True)
    teacher = torch.randn(2, 16, 8, 8, requires_grad=True)
    gt_bboxes = student.new_tensor(
        [
            [[8.0, 8.0, 32.0, 32.0], [0.0, 0.0, 0.0, 0.0]],
            [[16.0, 16.0, 48.0, 56.0], [4.0, 4.0, 12.0, 12.0]],
        ]
    )
    mask_gt = student.new_tensor([[[1.0], [0.0]], [[1.0], [1.0]]]).bool()
    assigner_fg = torch.zeros(2, 64, dtype=torch.bool)
    assigner_fg[:, :4] = True
    imgsz = student.new_tensor([64.0, 64.0])
    loss = criterion._fgd_style_loss(student, teacher, assigner_fg, gt_bboxes, mask_gt, imgsz)
    assert torch.isfinite(loss), loss
    assert loss.item() > 0, loss.item()
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None

    criterion_no_mask = make_loss(fgd_gamma=0.0)
    loss_no_mask = criterion_no_mask._fgd_style_loss(
        student.detach().clone().requires_grad_(True),
        teacher.detach().clone().requires_grad_(True),
        assigner_fg,
        gt_bboxes,
        mask_gt,
        imgsz,
    )
    assert torch.isfinite(loss_no_mask)
    assert not torch.isclose(loss.detach(), loss_no_mask.detach())

    criterion_assigner = make_loss(fgd_mask_mode="assigner")
    loss_assigner = criterion_assigner._fgd_style_loss(
        student.detach().clone().requires_grad_(True),
        teacher.detach().clone(),
        assigner_fg,
    )
    assert torch.isfinite(loss_assigner) and loss_assigner.item() > 0

    criterion_channel = make_loss(fgd_normalization_mode="channel_mean")
    loss_channel = criterion_channel._fgd_style_loss(
        student.detach().clone().requires_grad_(True),
        teacher.detach().clone(),
        assigner_fg,
        gt_bboxes,
        mask_gt,
        imgsz,
    )
    assert torch.isfinite(loss_channel) and loss_channel.item() > 0


def check_ld():
    torch.manual_seed(2)
    bsz, n_tokens, reg_max, n_cls = 2, 12, 8, 3
    student = torch.randn(bsz, n_tokens, 4 * reg_max, requires_grad=True)
    teacher = torch.randn(bsz, n_tokens, 4 * reg_max)
    fg = torch.zeros(bsz, n_tokens, dtype=torch.bool)
    fg[0, :3] = True
    fg[1, 4:6] = True
    target_scores = torch.zeros(bsz, n_tokens, n_cls)
    target_scores[..., 1][fg] = 0.8
    teacher_scores = torch.randn(bsz, n_tokens, n_cls)
    teacher_scores[..., 1] += 2.0
    gt_bboxes = student.new_tensor(
        [
            [[2.0, 2.0, 10.0, 10.0]],
            [[4.0, 4.0, 11.0, 11.0]],
        ]
    )
    mask_gt = torch.ones(bsz, 1, 1, dtype=torch.bool)
    teacher_bboxes = student.new_zeros(bsz, n_tokens, 4)
    teacher_bboxes[:, :, :] = student.new_tensor([2.0, 2.0, 10.0, 10.0])
    stride = student.new_ones(n_tokens, 1)

    main_only = make_loss(ld_use_vlr=False)
    loss_main = main_only._ld_style_loss(student, teacher, fg, target_scores, teacher_scores)
    assert torch.isfinite(loss_main) and loss_main.item() > 0
    loss_main.backward(retain_graph=True)
    assert student.grad is not None and torch.isfinite(student.grad).all()

    student.grad.zero_()
    with_vlr = make_loss(ld_use_vlr=True)
    loss_vlr = with_vlr._ld_style_loss(
        student,
        teacher,
        fg,
        target_scores,
        teacher_scores,
        teacher_bboxes=teacher_bboxes,
        gt_bboxes=gt_bboxes,
        mask_gt=mask_gt,
        level_stride_tensor=stride,
    )
    assert torch.isfinite(loss_vlr) and loss_vlr.item() > 0
    loss_vlr.backward()
    assert torch.isfinite(student.grad).all()

    try:
        with_vlr._ld_style_loss(student, teacher[:, :-1], fg, target_scores, teacher_scores)
    except RuntimeError:
        pass
    else:
        raise AssertionError("LD shape mismatch did not raise RuntimeError")


def check_cmdistill():
    torch.manual_seed(3)
    bsz, channels, height, width, reg_max, n_cls = 2, 16, 8, 8, 8, 4
    n_tokens = height * width
    criterion = make_loss()
    student_map = torch.randn(bsz, channels, height, width, requires_grad=True)
    teacher_map = torch.randn(bsz, channels, height, width, requires_grad=True)
    student_distri = torch.randn(bsz, n_tokens, 4 * reg_max, requires_grad=True)
    teacher_distri = torch.randn(bsz, n_tokens, 4 * reg_max)
    student_scores = torch.randn(bsz, n_tokens, n_cls, requires_grad=True)
    teacher_scores = torch.randn(bsz, n_tokens, n_cls)
    student_bboxes = torch.rand(bsz, n_tokens, 4, requires_grad=True)
    teacher_bboxes = torch.rand(bsz, n_tokens, 4)
    student_bboxes_xyxy = torch.cat(
        [
            torch.minimum(student_bboxes[..., :2], student_bboxes[..., 2:]),
            torch.maximum(student_bboxes[..., :2], student_bboxes[..., 2:]) + 0.1,
        ],
        dim=-1,
    )
    teacher_bboxes_xyxy = torch.cat(
        [
            torch.minimum(teacher_bboxes[..., :2], teacher_bboxes[..., 2:]),
            torch.maximum(teacher_bboxes[..., :2], teacher_bboxes[..., 2:]) + 0.1,
        ],
        dim=-1,
    )
    fg = torch.zeros(bsz, n_tokens, dtype=torch.bool)
    fg[0, :5] = True
    fg[1, 10:16] = True
    target_scores = torch.zeros(bsz, n_tokens, n_cls)
    target_scores[..., 2][fg] = 1.0

    loss = criterion._cmdistill_style_loss(
        student_map,
        teacher_map,
        fg,
        target_scores,
        student_distri,
        teacher_distri,
        student_scores,
        teacher_scores,
        student_bboxes_xyxy,
        teacher_bboxes_xyxy,
        level_index=2,
        num_levels=3,
    )
    assert torch.isfinite(loss), loss
    assert loss.item() > 0, loss.item()
    loss.backward()
    assert student_map.grad is not None and torch.isfinite(student_map.grad).all()
    assert student_distri.grad is None
    assert student_scores.grad is None
    assert student_bboxes.grad is None
    assert teacher_map.grad is None

    output_loss = criterion._cmdistill_output_loss(
        student_distri,
        teacher_distri,
        student_scores,
        teacher_scores,
        fg,
        target_scores,
        student_bboxes_xyxy,
        teacher_bboxes_xyxy,
    )
    assert torch.isfinite(output_loss), output_loss
    assert output_loss.item() > 0, output_loss.item()
    output_loss.backward()
    assert student_distri.grad is None
    assert student_scores.grad is not None and torch.isfinite(student_scores.grad).all()
    assert student_bboxes.grad is not None and torch.isfinite(student_bboxes.grad).all()
    assert teacher_scores.grad is None

    criterion_no_sampling = make_loss(cmdistill_max_tokens=n_tokens)
    batch_relation = criterion_no_sampling._cmdistill_relation_loss(
        student_map.detach(),
        teacher_map.detach(),
    )
    img0_relation = criterion_no_sampling._cmdistill_relation_loss(
        student_map.detach()[:1],
        teacher_map.detach()[:1],
    )
    img1_relation = criterion_no_sampling._cmdistill_relation_loss(
        student_map.detach()[1:],
        teacher_map.detach()[1:],
    )
    assert torch.allclose(batch_relation, 0.5 * (img0_relation + img1_relation), atol=1e-6)

    logit_only_style = make_loss(
        cmdistill_feature_weight=0.0,
        cmdistill_relation_weight=0.0,
        cmdistill_logit_weight=1.0,
    )._cmdistill_style_loss(
        student_map.detach().clone().requires_grad_(True),
        teacher_map.detach().clone(),
        fg,
        target_scores,
        student_distri.detach().clone(),
        teacher_distri,
        student_scores.detach().clone(),
        teacher_scores,
        student_bboxes_xyxy.detach().clone(),
        teacher_bboxes_xyxy,
        level_index=2,
        num_levels=3,
    )
    assert torch.isfinite(logit_only_style)
    assert logit_only_style.item() == 0.0

    feature_only = make_loss(cmdistill_relation_weight=0.0, cmdistill_logit_weight=0.0)
    feature_loss = feature_only._cmdistill_style_loss(
        student_map.detach().clone().requires_grad_(True),
        teacher_map.detach().clone(),
        fg,
        target_scores,
        None,
        None,
        student_scores.detach().clone(),
        teacher_scores,
        None,
        None,
        level_index=0,
        num_levels=3,
    )
    assert torch.isfinite(feature_loss) and feature_loss.item() > 0

    middle_level_feature = feature_only._cmdistill_style_loss(
        student_map.detach().clone().requires_grad_(True),
        teacher_map.detach().clone(),
        fg,
        target_scores,
        None,
        None,
        student_scores.detach().clone(),
        teacher_scores,
        None,
        None,
        level_index=1,
        num_levels=3,
    )
    assert torch.isfinite(middle_level_feature)
    assert middle_level_feature.item() == 0.0

    relation_only = make_loss(
        cmdistill_feature_weight=0.0,
        cmdistill_relation_weight=1.0,
        cmdistill_logit_weight=0.0,
        cmdistill_max_tokens=n_tokens,
    )
    relation_middle = relation_only._cmdistill_style_loss(
        student_map.detach().clone().requires_grad_(True),
        teacher_map.detach().clone(),
        fg,
        target_scores,
        None,
        None,
        None,
        None,
        None,
        None,
        level_index=1,
        num_levels=3,
    )
    relation_last = relation_only._cmdistill_style_loss(
        student_map.detach().clone().requires_grad_(True),
        teacher_map.detach().clone(),
        fg,
        target_scores,
        None,
        None,
        None,
        None,
        None,
        None,
        level_index=2,
        num_levels=3,
    )
    assert torch.isfinite(relation_middle)
    assert relation_middle.item() == 0.0
    assert torch.isfinite(relation_last) and relation_last.item() > 0

    teacher_scores_with_candidate = teacher_scores.clone()
    teacher_scores_with_candidate[:, 20, :] = 8.0
    candidate_fg_only = fg
    candidate_with_teacher_conf = candidate_fg_only | (teacher_scores_with_candidate.sigmoid().amax(dim=-1) >= 0.5)
    assert candidate_with_teacher_conf.sum() > candidate_fg_only.sum()

    # OpenMMLab PKD normalizes each channel over N/H/W, then uses MSE/2.
    pkd_input = torch.randn(2, 5, 4, 3)
    c = pkd_input.shape[1]
    reference = pkd_input.permute(1, 0, 2, 3).reshape(c, -1)
    reference = (reference - reference.mean(dim=-1, keepdim=True)) / (
        reference.std(dim=-1, keepdim=True) + 1e-6
    )
    reference = reference.reshape(c, *pkd_input.shape[:1], *pkd_input.shape[2:]).permute(1, 0, 2, 3)
    assert torch.allclose(_pkd_channel_standardize_map(pkd_input), reference)


def check_profile_names():
    validate = TeacherStudentDecompositionKDNRRLTeacherUAuxLossHBB._validate_comparison_kd_profile
    assert validate("fgd") == "fgd"
    assert validate("ld") == "ld"
    assert validate("cmdistill") == "cmdistill"
    for legacy in ("hallucidet", "hallucidet_style"):
        try:
            validate(legacy)
        except ValueError:
            pass
        else:
            raise AssertionError(f"legacy profile name should fail: {legacy}")


def main():
    check_fgd()
    check_ld()
    check_cmdistill()
    check_profile_names()
    print("comparison loss smoke checks passed")


if __name__ == "__main__":
    main()
