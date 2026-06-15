# Copyright (c) OpenMMLab. All rights reserved.
"""Reference snapshot of OpenMMLab MMRazor PKDLoss.

Source URL:
https://github.com/open-mmlab/mmrazor/blob/main/mmrazor/models/losses/pkd_loss.py

This file is stored for CMDistill review context only. CMDistill is defined by
the CMDistill paper; PKD is referenced only for the tensor-level PCC feature
normalization detail.
"""

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmrazor.registry import MODELS
except Exception:  # pragma: no cover - reference file only
    MODELS = None


def _register_module(cls):
    if MODELS is None:
        return cls
    return MODELS.register_module()(cls)


@_register_module
class PKDLoss(nn.Module):
    """PyTorch version of `PKD: General Distillation Framework for Object
    Detectors via Pearson Correlation Coefficient.

    <https://arxiv.org/abs/2207.02039>`_.

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
        resize_stu (bool): If True, we'll down/up sample the features of the
            student model to the spatial size of those of the teacher model if
            their spatial sizes are different. And vice versa. Defaults to
            True.
    """

    def __init__(self, loss_weight=1.0, resize_stu=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances.

        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 4
        n, c, h, w = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(c, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(c, n, h, w).permute(1, 0, 2, 3)

    def forward(
        self,
        preds_S: Union[torch.Tensor, Tuple],
        preds_T: Union[torch.Tensor, Tuple],
    ) -> torch.Tensor:
        """Forward computation."""
        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T = (preds_S,), (preds_T,)

        loss = 0.0
        for pred_S, pred_T in zip(preds_S, preds_T):
            size_S, size_T = pred_S.shape[2:], pred_T.shape[2:]
            if size_S[0] != size_T[0]:
                if self.resize_stu:
                    pred_S = F.interpolate(pred_S, size_T, mode="bilinear")
                else:
                    pred_T = F.interpolate(pred_T, size_S, mode="bilinear")
            assert pred_S.shape == pred_T.shape

            norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)

            # First conduct feature normalization and then calculate the
            # MSE loss. Mathematically, it is equivalent to first calculating
            # the Pearson Correlation Coefficient (r) between two feature
            # vectors, and then using 1-r as the feature imitation loss.
            loss += F.mse_loss(norm_S, norm_T) / 2

        return loss * self.loss_weight
