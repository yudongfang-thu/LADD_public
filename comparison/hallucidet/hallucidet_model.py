#!/usr/bin/env python3
"""
HalluciDet: Hallucination Network for Cross-Modal Detection
Based on WACV 2024 paper: "Hallucinating RGB Modality for Person Detection Through Privileged Information"

This implementation follows the paper's architecture:
1. U-Net-based hallucination network with attention blocks
2. Frozen RGB detector (privileged information)
3. Detection loss on hallucinated representations
4. Training only updates the hallucination network
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AttentionBlock(nn.Module):
    """Attention block for U-Net following paper Section 3"""

    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(x)
        return x * attention_weights


class ConvBlock(nn.Module):
    """Convolutional block for U-Net encoder/decoder"""

    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = AttentionBlock(out_channels) if use_attention else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.attention is not None:
            x = self.attention(x)
        return x


class UpConvBlock(nn.Module):
    """Upsampling block for U-Net decoder with skip connections"""

    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, use_attention=use_attention)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HallucinationNetwork(nn.Module):
    """
    U-Net-based Hallucination Network following paper Figure 2 and Section 3

    Architecture:
    - Encoder: 4 levels with conv blocks
    - Bottleneck: With attention
    - Decoder: 4 levels with upconv blocks and skip connections
    - Output: 3-channel pseudo-RGB representation

    Args:
        in_channels: Input channels (1 for grayscale SAR/IR)
        out_channels: Output channels (3 for RGB-like representation)
        base_channels: Base number of channels (default: 64)
        use_attention: Whether to use attention blocks in decoder (default: True)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        base_channels: int = 64,
        use_attention: bool = True
    ):
        super().__init__()

        # Encoder (contracting path)
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck with attention
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16, use_attention=True)

        # Decoder (expanding path)
        self.dec4 = UpConvBlock(base_channels * 16, base_channels * 8, use_attention=use_attention)
        self.dec3 = UpConvBlock(base_channels * 8, base_channels * 4, use_attention=use_attention)
        self.dec2 = UpConvBlock(base_channels * 4, base_channels * 2, use_attention=use_attention)
        self.dec1 = UpConvBlock(base_channels * 2, base_channels, use_attention=use_attention)

        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 1),
            nn.Tanh()  # Output in [-1, 1] range
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following best practices"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input SAR/IR image [B, 1, H, W]

        Returns:
            Hallucinated pseudo-RGB representation [B, 3, H, W]
        """
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Final output
        out = self.final(d1)

        return out


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
        hallucination_net: HallucinationNetwork,
        rgb_detector: nn.Module,
        normalize_input: bool = False
    ):
        super().__init__()
        self.hallucination_net = hallucination_net
        self.rgb_detector = rgb_detector
        self.normalize_input = normalize_input

        # Freeze RGB detector
        for param in self.rgb_detector.parameters():
            param.requires_grad = False
        self.rgb_detector.eval()
        self.names = getattr(self.rgb_detector, "names", None)
        self.stride = getattr(self.rgb_detector, "stride", None)
        self.args = getattr(self.rgb_detector, "args", None)

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
        # Ensure single-channel input
        sar_image = sar_image.float()
        if sar_image.numel() and sar_image.max() > 1.5:
            sar_image = sar_image / 255.0
        if sar_image.shape[1] == 3:
            # If input is 3-channel, convert to grayscale
            sar_image = 0.299 * sar_image[:, 0:1] + 0.587 * sar_image[:, 1:2] + 0.114 * sar_image[:, 2:3]

        if self.normalize_input:
            raise RuntimeError(
                "HalluciDetModel no longer supports per-batch min-max normalization. "
                "Normalize inputs with the dataloader path used for training."
            )

        # Hallucinate
        hallucinated = self.hallucination_net(sar_image)  # [-1, 1]

        # Convert to [0, 1] for detector
        hallucinated = (hallucinated + 1.0) / 2.0

        # Detect with frozen RGB detector
        # Detector is in eval mode and weights are frozen, BUT we don't use no_grad()
        # because gradients need to flow through hallucinated to train the hallucination net
        detections = self.rgb_detector(hallucinated)

        if return_hallucinated:
            return detections, hallucinated
        return detections


def build_hallucidet(
    rgb_detector_path: str,
    in_channels: int = 1,
    base_channels: int = 64,
    use_attention: bool = True,
    device: str = 'cuda'
) -> HalluciDetModel:
    """
    Build complete HalluciDet model

    Args:
        rgb_detector_path: Path to pre-trained RGB detector weights
        in_channels: Input channels (1 for SAR)
        base_channels: Base channels for U-Net
        use_attention: Use attention blocks
        device: Device to load model

    Returns:
        Complete HalluciDet model ready for training
    """
    from ultralytics import YOLO

    # Build hallucination network
    hallucination_net = HallucinationNetwork(
        in_channels=in_channels,
        out_channels=3,
        base_channels=base_channels,
        use_attention=use_attention
    )

    # Load frozen RGB detector
    rgb_detector = YOLO(rgb_detector_path).model
    rgb_detector.eval()

    # Build complete model
    model = HalluciDetModel(hallucination_net, rgb_detector)
    model.to(device)

    return model


if __name__ == "__main__":
    # Test the hallucination network
    print("Testing HallucinationNetwork...")
    net = HallucinationNetwork(in_channels=1, out_channels=3, base_channels=32)

    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    y = net(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")

    print("\nHallucinationNetwork test passed! ✅")
