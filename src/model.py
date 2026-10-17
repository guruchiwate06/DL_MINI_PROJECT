"""
model.py
--------
Simple Encoder-Decoder CNN (U-Net style) for binary oil-spill segmentation.
Accepts single-channel SAR images [B, 1, 256, 256] and outputs raw logits [B, 1, 256, 256].
"""

import torch
import torch.nn as nn


class SimpleSegmentationCNN(nn.Module):
    """
    A lightweight, transparent Encoder-Decoder CNN for binary segmentation.
    
    Structure:
      - Encoder: Extracts spatial features while downsampling resolution (256 -> 128 -> 64 -> 32).
      - Bottleneck: Learns high-level context at the lowest resolution (32x32).
      - Decoder: Upsamples feature maps back to the original resolution (32 -> 64 -> 128 -> 256)
                 with skip connections to preserve fine boundary details.
      - Final Head: 1x1 convolution mapping channels to 1 raw logit per pixel.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(SimpleSegmentationCNN, self).__init__()

        # --- ENCODER (Downsampling path) ---
        # Level 1: Input (B, 1, 256, 256) -> (B, 16, 256, 256)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256 -> 128

        # Level 2: Input (B, 16, 128, 128) -> (B, 32, 128, 128)
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128 -> 64

        # Level 3: Input (B, 32, 64, 64) -> (B, 64, 64, 64)
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64 -> 32

        # --- BOTTLENECK ---
        # Lowest resolution: (B, 64, 32, 32) -> (B, 128, 32, 32)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # --- DECODER (Upsampling path) ---
        # Level 3: Upsample (32 -> 64), concat with enc3 (128 + 64 = 192 channels) -> (B, 64, 64, 64)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Level 2: Upsample (64 -> 128), concat with enc2 (64 + 32 = 96 channels) -> (B, 32, 128, 128)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Level 1: Upsample (128 -> 256), concat with enc1 (32 + 16 = 48 channels) -> (B, 16, 256, 256)
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(16 + 16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # --- FINAL OUTPUT HEAD ---
        # 1x1 convolution maps 16 features to 1 raw logit per pixel without activation
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Encoder forward ---
        e1 = self.enc1(x)                # [B, 16, 256, 256]
        p1 = self.pool1(e1)              # [B, 16, 128, 128]

        e2 = self.enc2(p1)               # [B, 32, 128, 128]
        p2 = self.pool2(e2)              # [B, 32, 64, 64]

        e3 = self.enc3(p2)               # [B, 64, 64, 64]
        p3 = self.pool3(e3)              # [B, 64, 32, 32]

        # --- Bottleneck ---
        b = self.bottleneck(p3)          # [B, 128, 32, 32]

        # --- Decoder forward with skip connections ---
        d3 = self.upconv3(b)             # [B, 64, 64, 64]
        d3 = torch.cat([d3, e3], dim=1)  # [B, 128, 64, 64]
        d3 = self.dec3(d3)               # [B, 64, 64, 64]

        d2 = self.upconv2(d3)            # [B, 32, 128, 128]
        d2 = torch.cat([d2, e2], dim=1)  # [B, 64, 128, 128]
        d2 = self.dec2(d2)               # [B, 32, 128, 128]

        d1 = self.upconv1(d2)            # [B, 16, 256, 256]
        d1 = torch.cat([d1, e1], dim=1)  # [B, 32, 256, 256]
        d1 = self.dec1(d1)               # [B, 16, 256, 256]

        # --- Output logits ---
        logits = self.final_conv(d1)     # [B, 1, 256, 256]
        return logits


if __name__ == "__main__":
    import os
    import sys

    # Allow importing dataset from src when running model.py directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.dataset import get_dataloaders

    print("=== Testing SimpleSegmentationCNN Model ===")

    # 1. Initialize model
    model = SimpleSegmentationCNN(in_channels=1, out_channels=1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model successfully initialized. Total trainable parameters: {total_params:,}")

    # 2. Load a real batch from dataset pipeline
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    train_loader, _ = get_dataloaders(data_dir, batch_size=4)

    images, masks = next(iter(train_loader))
    print(f"\nInput images batch shape : {images.shape} (dtype: {images.dtype})")
    print(f"Target masks batch shape : {masks.shape} (dtype: {masks.dtype})")

    # 3. Perform forward pass
    logits = model(images)
    print(f"Output logits shape      : {logits.shape} (dtype: {logits.dtype})")

    # 4. Verify output validity
    assert logits.shape == masks.shape, f"Shape mismatch: {logits.shape} vs {masks.shape}"
    assert not torch.isnan(logits).any(), "Output contains NaN values!"
    assert not torch.isinf(logits).any(), "Output contains infinite values!"

    print(f"Logits value range       : min={logits.min().item():.3f}, max={logits.max().item():.3f}")
    print("\nAll model verification checks PASSED successfully!")
