from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (
    REPO_ROOT / "shared",
    REPO_ROOT / "shared" / "yolo",
    REPO_ROOT / "ladd" / "code" / "src",
):
    sys.path.insert(0, str(root))

from teacher_student_decomposition_kd_hbb.trainer import (  # noqa: E402
    ManualPhaseTeacherStudentDecompositionKDNRRLTeacherUAuxTrainer,
)


def test_set_bn_stats_eval_does_not_change_requires_grad() -> None:
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
        torch.nn.BatchNorm2d(8),
        torch.nn.ReLU(),
    )
    bn = model[1]

    bn.weight.requires_grad_(False)
    bn.bias.requires_grad_(False)
    model.train()
    ManualPhaseTeacherStudentDecompositionKDNRRLTeacherUAuxTrainer._set_bn_stats_eval(model)
    assert bn.training is False
    assert bn.weight.requires_grad is False
    assert bn.bias.requires_grad is False

    bn.weight.requires_grad_(True)
    bn.bias.requires_grad_(True)
    model.train()
    ManualPhaseTeacherStudentDecompositionKDNRRLTeacherUAuxTrainer._set_bn_stats_eval(model)
    assert bn.training is False
    assert bn.weight.requires_grad is True
    assert bn.bias.requires_grad is True


def test_grad_log_preserves_default_clip_norm() -> None:
    for path in (
        REPO_ROOT / "ladd" / "code" / "src" / "teacher_student_decomposition_kd_hbb" / "trainer.py",
        REPO_ROOT
        / "ladd"
        / "code_versions"
        / "current_hbb"
        / "src"
        / "teacher_student_decomposition_kd_hbb"
        / "trainer.py",
    ):
        text = path.read_text(encoding="utf-8")
        assert "_DEFAULT_ULTRALYTICS_GRAD_CLIP_NORM = 10.0" in text
        assert "ULTRALYTICS_DEFAULT" in text
        assert "ladd_grad_clip_norm" in text
        assert "ladd_assert_phase_freeze" in text
