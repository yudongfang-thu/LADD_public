from __future__ import annotations

import math

import torch
import torch.nn as nn

from teacher_student_decomposition_kd.model import ConvNormAct, TeacherTaskHead
from teacher_student_decomposition_kd_hbb.base_hbb import TeacherStudentDecompositionKDModelHBB


class WeakTaskDecoder(nn.Module):
    """Weak decoder used before lightweight auxiliary heads."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = ConvNormAct(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TeacherDirectDecoder(WeakTaskDecoder):
    """Backward-compatible alias for older checkpoints serialized before the decoder rename."""


class ResidualForegroundHead(nn.Module):
    """Very light 1-channel head so r_s only learns a weak foreground cue."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=3),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StudentSarAuxHead(nn.Module):
    """Path D head: r_s -> 1-channel SAR high-frequency prediction map.

    Used only when ``enable_r_sar_head=True``; otherwise this module is not
    instantiated and existing checkpoints remain bit-identical. The narrative
    role: r_s should encode SAR-private high-frequency structure (edges/speckle
    boundaries) that the RGB teacher cannot transfer.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=1),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TeacherReconDetectHead(nn.Module):
    """Single-level OBB proxy head that matches the training loss interface."""

    def __init__(self, channels: int, num_classes: int, reg_max: int = 16, num_extra: int = 1):
        super().__init__()
        self.nc = int(num_classes)
        self.reg_max = int(reg_max)
        self.ne = int(num_extra)

        box_hidden = max(16, channels // 4, self.reg_max * 4)
        cls_hidden = max(channels, min(self.nc, 100))
        angle_hidden = max(channels // 4, self.ne)

        self.box_head = nn.Sequential(
            ConvNormAct(channels, box_hidden, kernel_size=3),
            ConvNormAct(box_hidden, box_hidden, kernel_size=3),
            nn.Conv2d(box_hidden, 4 * self.reg_max, kernel_size=1),
        )
        self.cls_head = nn.Sequential(
            ConvNormAct(channels, cls_hidden, kernel_size=3),
            ConvNormAct(cls_hidden, cls_hidden, kernel_size=3),
            nn.Conv2d(cls_hidden, self.nc, kernel_size=1),
        )
        self.angle_head = nn.Sequential(
            ConvNormAct(channels, angle_hidden, kernel_size=3),
            ConvNormAct(angle_hidden, angle_hidden, kernel_size=3),
            nn.Conv2d(angle_hidden, self.ne, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bs = x.shape[0]
        boxes = self.box_head(x).view(bs, 4 * self.reg_max, -1)
        scores = self.cls_head(x).view(bs, self.nc, -1)
        angle = self.angle_head(x).view(bs, self.ne, -1)
        angle = (angle.sigmoid() - 0.25) * math.pi
        return {"boxes": boxes, "scores": scores, "angle": angle, "feats": [x]}


class StudentResidualProjBlock(nn.Module):
    """Scheme Y block: z_s = bottleneck projection(f_s), r_s = f_s - z_s (identity residual).

    Semantics:
        - z_s has a low-rank bottleneck so it cannot simply copy f_s
        - r_s = f_s - z_s carries whatever z_s can't represent, 0 params
        - student_rec_loss = ||z_s + r_s - f_s|| = 0 trivially (skip in loss)
        - KD uses z_s ↔ z_t; r_s has no direct loss by default
    """

    def __init__(self, channels: int, bottleneck_ratio: float = 0.25):
        super().__init__()
        if bottleneck_ratio <= 0:
            raise ValueError(f"bottleneck_ratio must be > 0, got {bottleneck_ratio}")
        bottleneck_ch = max(1, int(round(channels * bottleneck_ratio)))
        self.bottleneck_ratio = float(bottleneck_ratio)
        self.pre = ConvNormAct(channels, channels, kernel_size=1)
        self.z_proj = nn.Sequential(
            ConvNormAct(channels, bottleneck_ch, kernel_size=1),
            nn.Conv2d(bottleneck_ch, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.pre(x)
        z_s = self.z_proj(h)
        r_s = x - z_s
        fused = x  # = z_s + r_s trivially
        recon = x  # trivial reconstruction (student_rec_loss will be skipped in residual mode)
        return z_s, r_s, fused, recon


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


class TeacherResidualDecompositionBlock(nn.Module):
    """Teacher Scheme Y: z_t = bottleneck projection(f_t), u_t = f_t - z_t.

    Mirror of `StudentResidualProjBlock` on the teacher side. Replaces the
    explicit unlearnable branch + reconstruction layer with a residual identity.

    Properties:
        - z_t has a low-rank bottleneck so it cannot trivially copy f_t (anti-collapse).
        - u_t = x - z_t carries whatever z_t cannot represent (0 params).
        - recon = x is returned so t_rec_loss = ||recon - f_t|| = 0 by construction.
        - mask_branch is preserved (independent of decomposition shape).
    """

    def __init__(self, channels: int, use_mask: bool = False, bottleneck_ratio: float = 0.25):
        super().__init__()
        if bottleneck_ratio <= 0:
            raise ValueError(f"bottleneck_ratio must be > 0, got {bottleneck_ratio}")
        bottleneck_ch = max(1, int(round(channels * bottleneck_ratio)))
        self.bottleneck_ratio = float(bottleneck_ratio)
        # Kept for trainer compatibility (no-op in residual mode)
        self.unlearnable_hidden_ratio = 1.0
        self.use_mask = use_mask
        self.pre = ConvNormAct(channels, channels, kernel_size=1)
        self.z_proj = nn.Sequential(
            ConvNormAct(channels, bottleneck_ch, kernel_size=1),
            nn.Conv2d(bottleneck_ch, channels, kernel_size=1),
        )
        if use_mask:
            self.mask_branch = nn.Sequential(
                ConvNormAct(channels, channels, kernel_size=1),
                nn.Conv2d(channels, 1, 1),
            )
        else:
            self.mask_branch = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        h = self.pre(x)
        z_t = self.z_proj(h)
        u_t = x - z_t
        mask = torch.sigmoid(self.mask_branch(h)) if self.mask_branch is not None else None
        recon = x  # trivial: z_t + u_t == x (t_rec_loss is 0 by construction)
        return z_t, u_t, mask, recon


class TeacherStudentDecompositionKDNRRLTeacherUAuxModelHBB(TeacherStudentDecompositionKDModelHBB):
    """NRRL TSKD model with current residual-energy mainline plus teacher-side u_t controls."""

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
        student_z_bottleneck_ratio: float = 0.25,
        teacher_branch_mode: str = "decomposed",
        teacher_z_bottleneck_ratio: float = 0.25,
        enable_recon_task: bool = False,
        enable_r_obb_head: bool = False,
        enable_r_sar_head: bool = False,
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
        reg_max = getattr(head, "reg_max", 16)
        num_extra = getattr(head, "ne", 1)
        self.unlearnable_hidden_ratio = float(unlearnable_hidden_ratio)
        self.kd_calibration_mode = str(kd_calibration_mode)
        self.student_branch_mode = str(student_branch_mode)
        self.teacher_feature_mode = str(teacher_feature_mode)
        self.student_z_bottleneck_ratio = float(student_z_bottleneck_ratio)
        self.teacher_branch_mode = str(teacher_branch_mode)
        self.teacher_z_bottleneck_ratio = float(teacher_z_bottleneck_ratio)
        if self.teacher_branch_mode not in {"decomposed", "residual"}:
            raise ValueError(
                f"teacher_branch_mode must be 'decomposed' or 'residual', got {teacher_branch_mode!r}"
            )
        if self.student_branch_mode == "residual":
            self.student_split = nn.ModuleList(
                StudentResidualProjBlock(c, bottleneck_ratio=self.student_z_bottleneck_ratio)
                for c in feat_dims
            )
        if self.teacher_branch_mode == "residual":
            self.teacher_decomposition = nn.ModuleList(
                TeacherResidualDecompositionBlock(
                    c,
                    use_mask=use_mask,
                    bottleneck_ratio=self.teacher_z_bottleneck_ratio,
                )
                for c in feat_dims
            )
        else:
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
        self.student_r_aux_decoder = nn.ModuleList(WeakTaskDecoder(c) for c in feat_dims)
        self.student_r_fg_heads = nn.ModuleList(ResidualForegroundHead(c) for c in feat_dims)
        self.student_kd_calibration = nn.ModuleList(nn.Conv2d(c, c, kernel_size=1) for c in feat_dims)
        for layer in self.student_kd_calibration:
            nn.init.dirac_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        self._enable_recon_task = bool(enable_recon_task)
        if self._enable_recon_task:
            self.teacher_recon_decoder = nn.ModuleList(WeakTaskDecoder(c) for c in feat_dims)
            self.teacher_recon_task_heads = nn.ModuleList(
                TeacherReconDetectHead(c, num_classes, reg_max=reg_max, num_extra=num_extra) for c in feat_dims
            )

        # Q3-轻: independent OBB head reading r_s features. Trained only in C
        # (when student_split is unfrozen); contributes r_obb_loss to the total
        # loss with weight LAMBDA_R_OBB. Inference still goes through the main
        # OBB head (self.model[-1]); ensemble is opt-in for a future iteration.
        self._enable_r_obb_head = bool(enable_r_obb_head)
        if self._enable_r_obb_head:
            self.student_r_obb_heads = nn.ModuleList(
                TeacherReconDetectHead(c, num_classes, reg_max=reg_max, num_extra=num_extra)
                for c in feat_dims
            )

        # Path D: SAR-private high-frequency self-supervised head on r_s.
        # Only instantiated when ``enable_r_sar_head=True``; otherwise no
        # parameters are added and existing serialized checkpoints remain
        # exactly compatible. Trained jointly with the main loss in B/C only.
        self._enable_r_sar_head = bool(enable_r_sar_head)
        if self._enable_r_sar_head:
            self.student_r_sar_heads = nn.ModuleList(
                StudentSarAuxHead(c) for c in feat_dims
            )

    @property
    def recon_task_enabled(self) -> bool:
        return self._enable_recon_task and hasattr(self, "teacher_recon_task_heads")

    @property
    def r_obb_head_enabled(self) -> bool:
        return self._enable_r_obb_head and hasattr(self, "student_r_obb_heads")

    @property
    def r_sar_head_enabled(self) -> bool:
        return self._enable_r_sar_head and hasattr(self, "student_r_sar_heads")
