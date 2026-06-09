from __future__ import annotations

import sys
import unittest
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


class BNFreezeTest(unittest.TestCase):
    def test_set_bn_stats_eval_preserves_requires_grad(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(4),
        )
        bn = model[1]

        bn.weight.requires_grad_(False)
        bn.bias.requires_grad_(False)
        model.train()
        ManualPhaseTeacherStudentDecompositionKDNRRLTeacherUAuxTrainer._set_bn_stats_eval(model)
        self.assertFalse(bn.training)
        self.assertFalse(bn.weight.requires_grad)
        self.assertFalse(bn.bias.requires_grad)

        bn.weight.requires_grad_(True)
        bn.bias.requires_grad_(True)
        model.train()
        ManualPhaseTeacherStudentDecompositionKDNRRLTeacherUAuxTrainer._set_bn_stats_eval(model)
        self.assertFalse(bn.training)
        self.assertTrue(bn.weight.requires_grad)
        self.assertTrue(bn.bias.requires_grad)


if __name__ == "__main__":
    unittest.main()
