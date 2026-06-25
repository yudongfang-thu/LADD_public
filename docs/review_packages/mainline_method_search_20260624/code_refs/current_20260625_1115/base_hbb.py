"""HBB base model — same decomposition architecture as OBB version, 
but built on DetectionModel."""
from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel
from teacher_student_decomposition_kd.model import (
    ConvNormAct, TeacherTaskHead, StudentMimicResidualBlock,
    StudentReachabilityAdapter, TeacherDecompositionBlock,
)


class TeacherStudentDecompositionKDModelHBB(DetectionModel):
    """HBB student with teacher decomposition blocks."""

    def __init__(self, cfg="yolo11s.yaml", ch=3, nc=None, verbose=True,
                 use_mask: bool = False, fusion_mode: str = "sum",
                 student_detect_mode: str = "raw"):
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
            TeacherDecompositionBlock(c, use_mask=use_mask) for c in feat_dims)
        self.student_split = nn.ModuleList(
            StudentMimicResidualBlock(c, fusion_mode=fusion_mode) for c in feat_dims)
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

    def _select_detect_features(self, raw_feats, z_feats, fused_feats, recon_feats):
        if self.student_detect_mode == "raw": return raw_feats
        if self.student_detect_mode == "mimic": return z_feats
        if self.student_detect_mode == "recon": return recon_feats
        return fused_feats

    @staticmethod
    def _attach_auxiliary_outputs(outputs, raw_feats, z_feats, r_feats,
                                   fused_feats, recon_feats, detect_feats):
        aux = {"student_base_feats": raw_feats, "z_s_feats": z_feats,
               "r_s_feats": r_feats, "student_fused_feats": fused_feats,
               "student_recon_feats": recon_feats, "student_detect_feats": detect_feats}
        if isinstance(outputs, dict):
            outputs.update(aux); return outputs
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
                    z_feats.append(z_s); r_feats.append(r_s)
                    fused_feats.append(fused); recon_feats.append(recon)
                detect_feats = self._select_detect_features(raw_feats, z_feats, fused_feats, recon_feats)
                x = self._attach_auxiliary_outputs(
                    m(detect_feats), raw_feats, z_feats, r_feats, fused_feats, recon_feats, detect_feats)
            else:
                x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize and isinstance(x, torch.Tensor):
                from ultralytics.utils.plotting import feature_visualization
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed and isinstance(x, torch.Tensor):
                pooled = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
                embeddings.append(pooled)
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x
