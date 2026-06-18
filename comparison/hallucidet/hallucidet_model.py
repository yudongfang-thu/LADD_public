#!/usr/bin/env python3
"""
HalluciDet: Hallucination Network for Cross-Modal Detection
Based on WACV 2024 paper: "Hallucinating RGB Modality for Person Detection Through Privileged Information"

This implementation fixes the paper-facing YOLO adaptation to the final
official-style U-Net variant used by the comparison experiments:
1. segmentation_models_pytorch U-Net with ImageNet-pretrained encoder
2. Frozen RGB detector (privileged information)
3. Detection loss on hallucinated representations
4. Training only updates the hallucination network
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OfficialStyleHallucinationNetwork(nn.Module):
    """
    HalluciDet-paper-aligned hallucination network.

    The official repository builds a segmentation_models U-Net with an
    ImageNet-pretrained encoder and a sigmoid output head. This is the only
    active HalluciDet-YOLO hallucination network in the paper-facing code.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "official_unet requires segmentation_models_pytorch. "
                "Install it on the training server, e.g. `pip install segmentation-models-pytorch`."
            ) from exc

        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )
        self.output = nn.Sigmoid()
        self.input_channels = in_channels
        self.outputs_unit_range = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.net(x))


class HalluciDetModel(nn.Module):
    """
    Complete HalluciDet model combining hallucination network and frozen detector

    Args:
        hallucination_net: Hallucination network
        rgb_detector: Frozen RGB detector (YOLO11)
        normalize_input: Deprecated. Inputs are expected to use the same
            normalization as the training dataloader.
    """

    def __init__(
        self,
        hallucination_net: nn.Module,
        rgb_detector: nn.Module,
        normalize_input: bool = False,
        hallucination_input_mode: str = "replicate3",
    ):
        super().__init__()
        self.hallucination_net = hallucination_net
        self.rgb_detector = rgb_detector
        self.normalize_input = normalize_input
        self.hallucination_input_mode = hallucination_input_mode
        if self.hallucination_input_mode not in {"replicate3", "rgb"}:
            raise ValueError(
                "hallucination_input_mode must be one of: replicate3, rgb."
            )

        # Freeze RGB detector
        for param in self.rgb_detector.parameters():
            param.requires_grad = False
        self.rgb_detector.eval()
        self.names = getattr(self.rgb_detector, "names", None)
        self.stride = getattr(self.rgb_detector, "stride", None)
        self.args = getattr(self.rgb_detector, "args", None)

    @staticmethod
    def _to_single_channel(image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            return image
        if image.shape[1] != 3:
            raise RuntimeError(f"HalluciDet expects 1 or 3 input channels, got {image.shape[1]}.")
        return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]

    def prepare_hallucination_input(self, sar_image: torch.Tensor) -> torch.Tensor:
        """Normalize dataloader images and adapt channels for the hallucination net."""
        sar_image = sar_image.float()
        if sar_image.numel() and sar_image.max() > 1.5:
            sar_image = sar_image / 255.0

        if self.normalize_input:
            raise RuntimeError(
                "HalluciDetModel no longer supports per-batch min-max normalization. "
                "Normalize inputs with the dataloader path used for training."
            )

        if self.hallucination_input_mode == "replicate3":
            prepared = self._to_single_channel(sar_image).repeat(1, 3, 1, 1)
        else:
            if sar_image.shape[1] == 1:
                prepared = sar_image.repeat(1, 3, 1, 1)
            elif sar_image.shape[1] == 3:
                prepared = sar_image
            else:
                raise RuntimeError(f"HalluciDet expects 1 or 3 input channels, got {sar_image.shape[1]}.")

        expected = getattr(self.hallucination_net, "input_channels", None)
        if expected is not None and prepared.shape[1] != int(expected):
            raise RuntimeError(
                f"Hallucination input mode {self.hallucination_input_mode!r} produced "
                f"{prepared.shape[1]} channels, but the hallucination network expects {expected}."
            )
        return prepared

    def postprocess_hallucination_output(self, hallucinated: torch.Tensor) -> torch.Tensor:
        """Convert hallucination-net output to the detector's [0, 1] image range."""
        if getattr(self.hallucination_net, "outputs_unit_range", False):
            return hallucinated.clamp(0.0, 1.0)
        return ((hallucinated + 1.0) / 2.0).clamp(0.0, 1.0)

    def hallucinate(self, sar_image: torch.Tensor) -> torch.Tensor:
        prepared = self.prepare_hallucination_input(sar_image)
        hallucinated = self.hallucination_net(prepared)
        return self.postprocess_hallucination_output(hallucinated)

    def forward(self, sar_image: torch.Tensor, return_hallucinated: bool = False):
        """
        Forward pass

        Args:
            sar_image: Input SAR image [B, 1, H, W] or [B, 3, H, W]
            return_hallucinated: Whether to return hallucinated image

        Returns:
            If return_hallucinated=False: Detection outputs from frozen detector
            If return_hallucinated=True: (detections, hallucinated_image)
        """
        hallucinated = self.hallucinate(sar_image)

        # Detect with frozen RGB detector
        # Detector is in eval mode and weights are frozen, BUT we don't use no_grad()
        # because gradients need to flow through hallucinated to train the hallucination net
        detections = self.rgb_detector(hallucinated)

        if return_hallucinated:
            return detections, hallucinated
        return detections


def build_hallucidet(
    rgb_detector_path: str,
    hallucination_input_mode: str = "replicate3",
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    device: str = 'cuda'
) -> HalluciDetModel:
    """
    Build complete HalluciDet model

    Args:
        rgb_detector_path: Path to pre-trained RGB detector weights
        hallucination_input_mode: Input adapter, either replicate3 or rgb.
        encoder_name: segmentation_models_pytorch encoder name.
        encoder_weights: encoder weights, e.g. imagenet or None.
        device: Device to load model

    Returns:
        Complete HalluciDet model ready for training
    """
    from ultralytics import YOLO

    hallucination_net = OfficialStyleHallucinationNetwork(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        out_channels=3,
    )

    # Load frozen RGB detector
    rgb_detector = YOLO(rgb_detector_path).model
    rgb_detector.eval()

    # Build complete model
    model = HalluciDetModel(
        hallucination_net,
        rgb_detector,
        hallucination_input_mode=hallucination_input_mode,
    )
    model.to(device)

    return model


if __name__ == "__main__":
    print("Testing OfficialStyleHallucinationNetwork...")
    net = OfficialStyleHallucinationNetwork(encoder_weights=None)

    x = torch.randn(2, 3, 256, 256)
    y = net(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")

    print("\nOfficialStyleHallucinationNetwork test passed.")
