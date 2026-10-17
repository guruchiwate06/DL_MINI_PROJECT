"""
unet_model.py
-------------
Standard U-Net architecture for binary oil-spill segmentation on SAR satellite imagery.
Follows the classic Ronneberger et al. (2015) Encoder-Decoder design with skip connections:
  - 4 Downsampling stages (Encoder) capturing multi-scale spatial context
  - 1 Bottleneck stage at 16x16 resolution
  - 4 Upsampling stages (Decoder) with skip connections restoring fine boundary contours
  - 1x1 Convolution head outputting raw unnormalized logits
"""

import os
import sys
import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DoubleConv(nn.Module):
    """
    Standard U-Net building block:
    [Conv2d -> BatchNorm2d -> ReLU] x 2
    Preserves spatial dimensions via padding=1.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    Classic 4-stage U-Net for binary segmentation.
    
    Input shape : (B, 1, 256, 256)
    Output shape: (B, 1, 256, 256) raw logits (unbounded real values)
    
    Structure:
      - Encoder:
          Level 1: 256x256 -> 32 channels
          Level 2: 128x128 -> 64 channels
          Level 3:  64x64  -> 128 channels
          Level 4:  32x32  -> 256 channels
      - Bottleneck:
          Lowest:   16x16  -> 512 channels
      - Decoder:
          Level 4:  32x32  (512 upsampled to 256 + 256 skip = 512 -> 256)
          Level 3:  64x64  (256 upsampled to 128 + 128 skip = 256 -> 128)
          Level 2: 128x128 (128 upsampled to 64  +  64 skip = 128 ->  64)
          Level 1: 256x256 (64  upsampled to 32  +  32 skip =  64 ->  32)
      - Final Head:
          1x1 conv (32 -> 1) without activation
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features=(32, 64, 128, 256)):
        super(UNet, self).__init__()
        
        # --- ENCODER (Contracting Path) ---
        self.enc1 = DoubleConv(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256 -> 128

        self.enc2 = DoubleConv(features[0], features[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128 -> 64

        self.enc3 = DoubleConv(features[1], features[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64 -> 32

        self.enc4 = DoubleConv(features[2], features[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32 -> 16

        # --- BOTTLENECK ---
        bottleneck_channels = features[3] * 2  # 512
        self.bottleneck = DoubleConv(features[3], bottleneck_channels)

        # --- DECODER (Expanding Path) ---
        self.upconv4 = nn.ConvTranspose2d(bottleneck_channels, features[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features[3] * 2, features[3])

        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features[2] * 2, features[2])

        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features[1] * 2, features[1])

        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features[0] * 2, features[0])

        # --- FINAL 1x1 CONVOLUTION HEAD ---
        # Outputs raw logits; no Sigmoid inside model (BCEWithLogitsLoss handles it)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder passes
        e1 = self.enc1(x)               # [B, 32, 256, 256]
        p1 = self.pool1(e1)             # [B, 32, 128, 128]

        e2 = self.enc2(p1)              # [B, 64, 128, 128]
        p2 = self.pool2(e2)             # [B, 64, 64, 64]

        e3 = self.enc3(p2)              # [B, 128, 64, 64]
        p3 = self.pool3(e3)             # [B, 128, 32, 32]

        e4 = self.enc4(p3)              # [B, 256, 32, 32]
        p4 = self.pool4(e4)             # [B, 256, 16, 16]

        # Bottleneck
        b = self.bottleneck(p4)         # [B, 512, 16, 16]

        # Decoder passes with skip connections (torch.cat along channel dim)
        d4 = self.upconv4(b)            # [B, 256, 32, 32]
        d4 = torch.cat([d4, e4], dim=1) # [B, 512, 32, 32]
        d4 = self.dec4(d4)              # [B, 256, 32, 32]

        d3 = self.upconv3(d4)           # [B, 128, 64, 64]
        d3 = torch.cat([d3, e3], dim=1) # [B, 256, 64, 64]
        d3 = self.dec3(d3)              # [B, 128, 64, 64]

        d2 = self.upconv2(d3)           # [B, 64, 128, 128]
        d2 = torch.cat([d2, e2], dim=1) # [B, 128, 128, 128]
        d2 = self.dec2(d2)              # [B, 64, 128, 128]

        d1 = self.upconv1(d2)           # [B, 32, 256, 256]
        d1 = torch.cat([d1, e1], dim=1) # [B, 64, 256, 256]
        d1 = self.dec1(d1)              # [B, 32, 256, 256]

        logits = self.final_conv(d1)     # [B, 1, 256, 256]
        return logits


if __name__ == "__main__":
    from src.dataset import get_dataloaders

    print("=== Testing UNet Model ===")

    # 1. Initialize U-Net model
    model = UNet(in_channels=1, out_channels=1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net successfully initialized. Total trainable parameters: {total_params:,}")

    # 2. Synthetic batch test
    dummy_input = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        dummy_out = model(dummy_input)
    print(f"Synthetic forward pass: Input {dummy_input.shape} -> Output {dummy_out.shape}")
    assert dummy_out.shape == (2, 1, 256, 256), f"Unexpected shape {dummy_out.shape}"

    # 3. Real batch test from dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    train_loader, _ = get_dataloaders(data_dir, batch_size=4)

    images, masks = next(iter(train_loader))
    print(f"\nReal dataset batch: Images {images.shape} | Masks {masks.shape}")

    # Compute forward pass and test BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    logits = model(images)
    loss = criterion(logits, masks)

    print(f"Output logits shape : {logits.shape}")
    print(f"Logits range        : [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"Initial loss value  : {loss.item():.4f}")

    assert logits.shape == (4, 1, 256, 256), "Shape mismatch on real batch!"
    assert not torch.isnan(loss), "Loss is NaN!"

    print("\nAll UNet model verification checks PASSED successfully!")
