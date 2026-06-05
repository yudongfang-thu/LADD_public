from __future__ import annotations

import torch.nn as nn

from ultralytics.nn.tasks import OBBModel


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class D2ADOBBModel(OBBModel):
    """OBB model with light projection heads for D2AD-R."""

    def __init__(
        self,
        cfg="yolo11n-obb.yaml",
        ch=3,
        nc=None,
        verbose=True,
        d2ad_proj_dim: int = 64,
        teacher_target_mode: str = "raw",
    ):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.d2ad_proj_dim = d2ad_proj_dim
        self.teacher_target_mode = teacher_target_mode
        feat_dims = self._get_feat_dims()
        if teacher_target_mode == "raw":
            level_dims = feat_dims
            self.d2ad_teacher_inv = None
        elif teacher_target_mode == "projected":
            level_dims = [d2ad_proj_dim] * len(feat_dims)
            self.d2ad_teacher_inv = nn.ModuleList(MLPProjector(c, d2ad_proj_dim) for c in feat_dims)
        else:
            raise ValueError(f"Unsupported teacher_target_mode: {teacher_target_mode}")

        self.d2ad_level_dims = level_dims
        self.d2ad_student_inv = nn.ModuleList(MLPProjector(c, d) for c, d in zip(feat_dims, level_dims))
        self.d2ad_student_res = nn.ModuleList(MLPProjector(c, d) for c, d in zip(feat_dims, level_dims))
        self.d2ad_reconstruct = nn.ModuleList(nn.Linear(d * 2, c) for c, d in zip(feat_dims, level_dims))

    def _get_feat_dims(self) -> list[int]:
        head = self.model[-1]
        feat_dims = []
        for box_head in head.cv2:
            first_block = box_head[0]
            in_dim = getattr(first_block, "conv", first_block).in_channels
            feat_dims.append(in_dim)
        return feat_dims
