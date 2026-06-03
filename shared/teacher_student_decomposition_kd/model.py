from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.tasks import OBBModel
from ultralytics.utils.plotting import feature_visualization


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TeacherDecompositionBlock(nn.Module):
    """Teacher-side learnable/unlearnable decomposition block."""

    def __init__(self, channels: int, use_mask: bool = False):
        super().__init__()
        self.use_mask = use_mask
        self.pre = ConvNormAct(channels, channels, kernel_size=1)
        self.learnable_branch = nn.Sequential(ConvNormAct(channels, channels, kernel_size=1), nn.Conv2d(channels, channels, 1))
        self.unlearnable_branch = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=1),
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


class StudentMimicResidualBlock(nn.Module):
    """Student-side mimic/residual split block."""

    def __init__(self, channels: int, fusion_mode: str = "sum"):
        super().__init__()
        if fusion_mode not in {"sum", "concat"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.pre = ConvNormAct(channels, channels, kernel_size=1)
        self.mimic_branch = nn.Sequential(ConvNormAct(channels, channels, kernel_size=1), nn.Conv2d(channels, channels, 1))
        self.residual_branch = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels, 1),
        )
        self.reconstruct = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=1),
            nn.Conv2d(channels, channels, 1),
        )
        if fusion_mode == "concat":
            self.fusion = nn.Sequential(ConvNormAct(channels * 2, channels, kernel_size=1), nn.Conv2d(channels, channels, 1))
        else:
            self.fusion = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.pre(x)
        z_s = self.mimic_branch(h)
        r_s = self.residual_branch(h)
        if self.fusion is None:
            fused = z_s + r_s
        else:
            fused = self.fusion(torch.cat((z_s, r_s), dim=1))
        recon = self.reconstruct(torch.cat((z_s, r_s), dim=1))
        return z_s, r_s, fused, recon


class StudentReachabilityAdapter(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(ConvNormAct(channels, channels, kernel_size=1), nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TeacherTaskHead(nn.Module):
    """Lightweight task proxy head on top of teacher learnable features."""

    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=3),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TeacherStudentDecompositionKDModel(OBBModel):
    """YOLO11-OBB student with mimic/residual detection features and trainable teacher-side TDN blocks."""

    def __init__(
        self,
        cfg="yolo11n-obb.yaml",
        ch=3,
        nc=None,
        verbose=True,
        use_mask: bool = False,
        fusion_mode: str = "sum",
        student_detect_mode: str = "fused",
    ):
        self._tskd_modules_ready = False
        self.use_mask = use_mask
        self.fusion_mode = fusion_mode
        self.student_detect_mode = student_detect_mode
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        if student_detect_mode not in {"fused", "mimic", "raw", "recon"}:
            raise ValueError(f"Unsupported student_detect_mode: {student_detect_mode}")

        feat_dims = self._get_feat_dims()
        num_classes = self.model[-1].nc

        self.teacher_decomposition = nn.ModuleList(
            TeacherDecompositionBlock(c, use_mask=use_mask) for c in feat_dims
        )
        self.student_split = nn.ModuleList(StudentMimicResidualBlock(c, fusion_mode=fusion_mode) for c in feat_dims)
        self.student_reachability = nn.ModuleList(StudentReachabilityAdapter(c) for c in feat_dims)
        self.teacher_task_heads = nn.ModuleList(TeacherTaskHead(c, num_classes) for c in feat_dims)
        self._tskd_modules_ready = True

    def _get_feat_dims(self) -> list[int]:
        head = self.model[-1]
        feat_dims = []
        for box_head in head.cv2:
            first_block = box_head[0]
            in_dim = getattr(first_block, "conv", first_block).in_channels
            feat_dims.append(in_dim)
        return feat_dims

    def _select_detect_features(
        self,
        raw_feats: list[torch.Tensor],
        z_feats: list[torch.Tensor],
        fused_feats: list[torch.Tensor],
        recon_feats: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if self.student_detect_mode == "raw":
            return raw_feats
        if self.student_detect_mode == "mimic":
            return z_feats
        if self.student_detect_mode == "recon":
            return recon_feats
        return fused_feats

    @staticmethod
    def _attach_auxiliary_outputs(
        outputs,
        raw_feats: list[torch.Tensor],
        z_feats: list[torch.Tensor],
        r_feats: list[torch.Tensor],
        fused_feats: list[torch.Tensor],
        recon_feats: list[torch.Tensor],
        detect_feats: list[torch.Tensor],
    ):
        aux = {
            "student_base_feats": raw_feats,
            "z_s_feats": z_feats,
            "r_s_feats": r_feats,
            "student_fused_feats": fused_feats,
            "student_recon_feats": recon_feats,
            "student_detect_feats": detect_feats,
        }
        if isinstance(outputs, dict):
            outputs.update(aux)
            return outputs
        if isinstance(outputs, tuple) and len(outputs) >= 2 and isinstance(outputs[1], dict):
            outputs[1].update(aux)
        return outputs

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        if not self._tskd_modules_ready:
            return super()._predict_once(x, profile, visualize, embed)

        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)

            if m is self.model[-1] and isinstance(x, list):
                raw_feats = list(x)
                z_feats, r_feats, fused_feats, recon_feats = [], [], [], []
                for i, feat in enumerate(raw_feats):
                    z_s, r_s, fused, recon = self.student_split[i](feat)
                    z_feats.append(z_s)
                    r_feats.append(r_s)
                    fused_feats.append(fused)
                    recon_feats.append(recon)
                detect_feats = self._select_detect_features(raw_feats, z_feats, fused_feats, recon_feats)
                x = self._attach_auxiliary_outputs(
                    m(detect_feats), raw_feats, z_feats, r_feats, fused_feats, recon_feats, detect_feats
                )
            else:
                x = m(x)

            y.append(x if m.i in self.save else None)
            if visualize and isinstance(x, torch.Tensor):
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed and isinstance(x, torch.Tensor):
                pooled = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
                embeddings.append(pooled)
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x
