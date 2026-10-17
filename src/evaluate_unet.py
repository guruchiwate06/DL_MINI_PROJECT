"""
evaluate_unet.py
----------------
Quantitative evaluation of the trained U-Net model on the validation split.
Computes global pixel-level metrics:
  - Dice Coefficient (F1 Score)
  - Intersection over Union (IoU / Jaccard Index)
  - Precision
  - Recall (Sensitivity)
"""

import os
import sys
import argparse

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.dataset import get_dataloaders
from src.unet_model import UNet


def evaluate_unet(checkpoint_path: str, data_root: str, batch_size: int = 16, threshold: float = 0.5, max_batches: int = None):
    """
    Loads saved U-Net checkpoint and calculates global segmentation metrics on the validation split.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for U-Net evaluation: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    # Initialize U-Net and restore weights
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        saved_epoch = checkpoint.get("epoch", "N/A")
        saved_val_loss = checkpoint.get("val_loss", "N/A")
        print(f"Loaded U-Net checkpoint from Epoch {saved_epoch} (Checkpoint Val Loss: {saved_val_loss:.4f})")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw U-Net model state dictionary.")

    model.eval()

    # Load validation data
    _, val_loader = get_dataloaders(data_root, batch_size=batch_size)
    total_samples = len(val_loader.dataset)
    print(f"Loaded validation split with {total_samples} samples.")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    evaluated_images = 0

    print("\nRunning U-Net evaluation on validation split...")

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            tp = torch.sum((preds == 1.0) & (masks == 1.0)).item()
            fp = torch.sum((preds == 1.0) & (masks == 0.0)).item()
            fn = torch.sum((preds == 0.0) & (masks == 1.0)).item()
            tn = torch.sum((preds == 0.0) & (masks == 0.0)).item()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            evaluated_images += images.size(0)

            total_b = max_batches if max_batches is not None else len(val_loader)
            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_b:
                print(f"  --> Progress: Batch [{batch_idx + 1:3d}/{total_b:3d}] ({evaluated_images:4d} images evaluated)")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn) if (2.0 * total_tp + total_fp + total_fn) > 0 else 0.0

    print("\n" + "=" * 50)
    print("   U-NET VALIDATION METRICS REPORT (NOT TEST SET)")
    print("=" * 50)
    print(f"Model Architecture      : U-Net (7.76M parameters)")
    print(f"Evaluated Images        : {evaluated_images} / {total_samples}")
    print(f"Decision Threshold      : {threshold}")
    print("-" * 50)
    print(f"True Positives (TP)     : {int(total_tp):,}")
    print(f"False Positives (FP)    : {int(total_fp):,}")
    print(f"False Negatives (FN)    : {int(total_fn):,}")
    print(f"True Negatives (TN)     : {int(total_tn):,}")
    print("-" * 50)
    print(f"Precision               : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall (Sensitivity)    : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"IoU (Jaccard Index)     : {iou:.4f} ({iou * 100:.2f}%)")
    print(f"Dice Coefficient (F1)   : {dice:.4f} ({dice * 100:.2f}%)")
    print("=" * 50)
    print("Note: These results represent validation performance because the")
    print("checkpoint was selected based on validation loss.")
    print("=" * 50 + "\n")

    return {
        "evaluated_images": evaluated_images,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate U-Net Oil Spill Segmentation Model")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/unet/best_unet.pth", help="Path to U-Net checkpoint")
    parser.add_argument("--data_root", type=str, default="data", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for positive class")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional limit on batches to evaluate")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cp_path = os.path.join(base_dir, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    d_root = os.path.join(base_dir, args.data_root) if not os.path.isabs(args.data_root) else args.data_root

    evaluate_unet(
        checkpoint_path=cp_path,
        data_root=d_root,
        batch_size=args.batch_size,
        threshold=args.threshold,
        max_batches=args.max_batches
    )
