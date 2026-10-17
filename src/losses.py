"""
losses.py
---------
Loss functions for binary oil-spill segmentation.
Includes:
  1. SoftDiceLoss: Directly optimizes the Dice overlap coefficient:
       Dice = (2 * |P ∩ Y| + smooth) / (|P| + |Y| + smooth)
       Loss = 1 - Dice
  2. BCEDiceLoss: Compound loss combining Binary Cross-Entropy and Soft Dice Loss:
       total_loss = bce_loss + dice_loss
"""

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Operates on raw unnormalized logits by applying torch.sigmoid.
    
    Formula:
      Dice = (2 * sum(P * Y) + smooth) / (sum(P) + sum(Y) + smooth)
      Loss = 1.0 - Dice
    
    The smoothing term (smooth = 1.0) ensures:
      1. Numerical stability (prevents division by zero when both P and Y are empty).
      2. Smooth gradients during backpropagation.
    """
    def __init__(self, smooth: float = 1.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert raw logits to probabilities in [0.0, 1.0]
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions per batch item: (B, 1, H, W) -> (B, H*W)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        # Compute intersection and union cardinality per batch item
        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        cardinality = torch.sum(probs_flat, dim=1) + torch.sum(targets_flat, dim=1)

        # Soft Dice score per item in batch
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Dice loss is 1 - Dice score (averaged over batch)
        dice_loss = 1.0 - dice_score
        return dice_loss.mean()


class BCEDiceLoss(nn.Module):
    """
    Compound Loss: BCEWithLogitsLoss + SoftDiceLoss
    
    Why combine them?
      - BCE Loss: Optimizes per-pixel classification accuracy independently,
        providing strong, smooth gradients for every pixel.
      - Soft Dice Loss: Directly optimizes global shape overlap and foreground-background
        balance, helping capture thin slicks and sharp boundaries.
    
    Formula:
      total_loss = bce_weight * BCE + dice_weight * DiceLoss
    """
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss, bce_loss, dice_loss


if __name__ == "__main__":
    print("=== Testing SoftDiceLoss and BCEDiceLoss ===")

    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0, smooth=1.0)

    # 1. Test identical prediction (perfect match)
    perfect_logits = torch.full((2, 1, 64, 64), 10.0)  # sigmoid(10) ≈ 0.9999
    targets_all_ones = torch.ones((2, 1, 64, 64))
    tot, bce, d = criterion(perfect_logits, targets_all_ones)
    print(f"Perfect match test: Total={tot.item():.4f} | BCE={bce.item():.4f} | DiceLoss={d.item():.4f}")
    assert d.item() < 0.01, "Dice loss should be near 0 on perfect match!"

    # 2. Test empty masks (safe division by zero test)
    empty_logits = torch.full((2, 1, 64, 64), -10.0)  # sigmoid(-10) ≈ 0.0000
    targets_empty = torch.zeros((2, 1, 64, 64))
    tot, bce, d = criterion(empty_logits, targets_empty)
    print(f"Empty masks test  : Total={tot.item():.4f} | BCE={bce.item():.4f} | DiceLoss={d.item():.4f}")
    assert not torch.isnan(tot), "Loss should not be NaN on empty masks!"

    # 3. Test backpropagation (finite gradients)
    test_logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    test_targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    tot, bce, d = criterion(test_logits, test_targets)
    tot.backward()
    assert test_logits.grad is not None, "Gradient should exist!"
    assert not torch.isnan(test_logits.grad).any(), "Gradient contains NaNs!"
    print("Backpropagation test: PASSED (clean finite gradients).")

    print("\nAll loss tests PASSED successfully!")
