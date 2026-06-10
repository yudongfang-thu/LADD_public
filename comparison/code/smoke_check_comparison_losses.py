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
):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from teacher_student_decomposition_kd_hbb.loss import (  # noqa: E402
    TeacherStudentDecompositionKDNRRLTeacherUAuxLossHBB,
)


def make_loss(**overrides):
    obj = TeacherStudentDecompositionKDNRRLTeacherUAuxLossHBB.__new__(
        TeacherStudentDecompositionKDNRRLTeacherUAuxLossHBB
    )
    defaults = {
        "fgd_alpha": 1.0,
        "fgd_beta": 0.5,
        "fgd_gamma": 1.0,
        "fgd_lambda": 0.0,
        "fgd_temperature": 0.5,
        "fgd_mask_mode": "gt_box",
        "fgd_bg_norm": True,
        "ld_temperature": 10.0,
        "ld_use_vlr": True,
        "ld_quality_power": 1.0,
        "ld_min_vlr_weight": 0.0,
        "ld_vlr_topk": 0,
        "ld_vlr_weight": 1.0,
        "ld_main_weight": 1.0,
        "ld_allow_empty_vlr": True,
        "_ld_warned_missing_teacher_scores": False,
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


def check_profile_names():
    validate = TeacherStudentDecompositionKDNRRLTeacherUAuxLossHBB._validate_comparison_kd_profile
    assert validate("hallucidet_style") == "hallucidet_style"
    try:
        validate("hallucidet")
    except ValueError:
        pass
    else:
        raise AssertionError("legacy hallucidet profile name should fail")


def main():
    check_fgd()
    check_ld()
    check_profile_names()
    print("comparison loss smoke checks passed")


if __name__ == "__main__":
    main()
