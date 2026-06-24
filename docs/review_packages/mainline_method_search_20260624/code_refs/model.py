from __future__ import annotations

import torch
import torch.nn as nn

from teacher_student_decomposition_kd.model import ConvNormAct, TeacherTaskHead
from teacher_student_decomposition_kd_hbb.base_hbb import TeacherStudentDecompositionKDModelHBB


class WeakTaskDecoder(nn.Module):
    """Lightweight decoder used before task/reconstruction heads."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = ConvNormAct(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TeacherDirectDecoder(WeakTaskDecoder):
    """Backward-compatible alias for older checkpoints serialized before the decoder rename."""


class TeacherPrivateAwareDecompositionBlock(nn.Module):
    """Teacher decomposition with optional reduced-capacity unlearnable branch."""

    def __init__(self, channels: int, use_mask: bool = False, unlearnable_hidden_ratio: float = 1.0):
        super().__init__()
        if unlearnable_hidden_ratio <= 0:
            raise ValueError(f"unlearnable_hidden_ratio must be > 0, got {unlearnable_hidden_ratio}")
        hidden = max(1, int(round(channels * unlearnable_hidden_ratio)))
        self.use_mask = use_mask
        self.unlearnable_hidden_ratio = float(unlearnable_hidden_ratio)
        self.pre = ConvNormAct(channels, channels, kernel_size=1)
        self.learnable_branch = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels, 1),
        )
        self.unlearnable_branch = nn.Sequential(
            ConvNormAct(channels, hidden, kernel_size=1),
            nn.Conv2d(hidden, hidden, 1),
            ConvNormAct(hidden, channels, kernel_size=1),
            nn.Conv2d(channels, channels, 1),
        )
        self.reconstruct = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=1),
            nn.Conv2d(channels, channels, 1),
        )
        if use_mask:
            self.mask_branch = nn.Sequential(ConvNormAct(channels, channels, kernel_size=1), nn.Conv2d(channels, 1, 1))
        else:
            self.mask_branch = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        h = self.pre(x)
        z_t = self.learnable_branch(h)
        u_t = self.unlearnable_branch(h)
        mask = torch.sigmoid(self.mask_branch(h)) if self.mask_branch is not None else None
        recon = self.reconstruct(torch.cat((z_t, u_t), dim=1))
        return z_t, u_t, mask, recon


class TeacherStudentDecompositionKDNRRLTeacherUAuxModelHBB(TeacherStudentDecompositionKDModelHBB):
    """LADD HBB model with teacher decomposition and student common-space split."""

    def __init__(
        self,
        cfg="yolo11n-obb.yaml",
        ch=3,
        nc=None,
        verbose=True,
        use_mask: bool = False,
        fusion_mode: str = "sum",
        student_detect_mode: str = "raw",
        unlearnable_hidden_ratio: float = 1.0,
        kd_calibration_mode: str = "none",
        student_branch_mode: str = "split",
        teacher_feature_mode: str = "decomposed",
    ):
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            use_mask=use_mask,
            fusion_mode=fusion_mode,
            student_detect_mode=student_detect_mode,
        )
        feat_dims = self._get_feat_dims()
        head = self.model[-1]
        num_classes = head.nc
        self.unlearnable_hidden_ratio = float(unlearnable_hidden_ratio)
        self.kd_calibration_mode = str(kd_calibration_mode)
        self.student_branch_mode = str(student_branch_mode)
        self.teacher_feature_mode = str(teacher_feature_mode)
        if self.student_branch_mode not in {"split", "raw", "single_proj"}:
            raise ValueError(
                f"student_branch_mode must be 'split', 'raw', or 'single_proj', got {student_branch_mode!r}."
            )
        if self.teacher_feature_mode not in {"decomposed", "raw", "projected_raw", "raw_weak_reach"}:
            raise ValueError(
                "teacher_feature_mode must be 'decomposed', 'raw', 'projected_raw', or 'raw_weak_reach', "
                f"got {teacher_feature_mode!r}."
            )
        self.teacher_decomposition = nn.ModuleList(
            TeacherPrivateAwareDecompositionBlock(
                c,
                use_mask=use_mask,
                unlearnable_hidden_ratio=unlearnable_hidden_ratio,
            )
            for c in feat_dims
        )
        self.teacher_decoder = nn.ModuleList(WeakTaskDecoder(c) for c in feat_dims)
        self.teacher_task_heads = nn.ModuleList(TeacherTaskHead(c, num_classes) for c in feat_dims)
        self.student_kd_calibration = nn.ModuleList(nn.Conv2d(c, c, kernel_size=1) for c in feat_dims)
        for layer in self.student_kd_calibration:
            nn.init.dirac_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
