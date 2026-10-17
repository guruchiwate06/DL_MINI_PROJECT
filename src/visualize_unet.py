"""
visualize_unet.py
-----------------
Generates qualitative 4-panel visual comparisons for U-Net predictions on the validation set.
Saves figures to outputs/visualizations/unet/ without touching existing CNN figures.

Panels per sample:
  1. Original SAR Input Image (Grayscale)
  2. Ground-Truth Oil Spill Mask (0 = Sea, 1 = Oil)
  3. U-Net Predicted Binary Mask (with sample IoU and Dice)
  4. Error Comparison Overlay on SAR image:
     - Green : True Positive (TP - Correctly detected oil)
     - Red   : False Positive (FP - Incorrectly predicted oil / False alarm)
     - Yellow: False Negative (FN - Missed oil slick pixels)
"""

import os
import sys
import argparse
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import OilSpillDataset
from src.unet_model import UNet


def create_comparison_overlay(img_np: np.ndarray, pred_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """
    Creates an RGB overlay on the SAR background comparing prediction with ground truth:
      - Green  [0, 225, 0]   : True Positive (TP)
      - Red    [235, 45, 45] : False Positive (FP)
      - Yellow [240, 200, 20]: False Negative (FN)
    """
    # Base grayscale image in [0, 255]
    img_uint8 = (img_np * 255.0).astype(np.uint8)
    # Dim background slightly (85% intensity) so colored overlays stand out clearly
    dimmed_base = (img_uint8 * 0.85).astype(np.uint8)
    overlay = np.stack([dimmed_base, dimmed_base, dimmed_base], axis=-1)

    tp_mask = (pred_np == 1.0) & (mask_np == 1.0)
    fp_mask = (pred_np == 1.0) & (mask_np == 0.0)
    fn_mask = (pred_np == 0.0) & (mask_np == 1.0)

    # Color code regions
    overlay[tp_mask] = [0, 225, 0]     # Green = Correct Detection
    overlay[fp_mask] = [235, 45, 45]   # Red = False Alarm
    overlay[fn_mask] = [240, 200, 20]  # Yellow = Missed Oil

    return overlay


def run_unet_visualization(checkpoint_path: str, data_root: str, output_dir: str, sample_indices=None, threshold: float = 0.5):
    """
    Loads U-Net, performs inference on selected validation samples, and generates 4-panel comparison plots.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for U-Net visualization: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"U-Net checkpoint not found at: {checkpoint_path}")

    # Load model & weights
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "N/A")
        print(f"Loaded U-Net checkpoint from Epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw U-Net weights.")

    model.eval()

    # Load validation split
    val_images_dir = os.path.join(data_root, "images", "val")
    val_masks_dir = os.path.join(data_root, "masks", "val")
    dataset = OilSpillDataset(val_images_dir, val_masks_dir, is_train=False)

    os.makedirs(output_dir, exist_ok=True)

    # Curated mix of 10 diverse validation samples representing:
    #   - High accuracy / sharp boundaries (Dice > 80%)
    #   - Moderate / boundary over-expansion & diffuse edges (Dice 60% - 80%)
    #   - Challenging / partial detection & false negative look-alikes (Dice < 50%)
    if sample_indices is None:
        sample_indices = [800, 4, 1, 2, 598, 1265, 73, 50, 1254, 139]

    results_summary = []

    with torch.no_grad():
        for rank, idx in enumerate(sample_indices, 1):
            image_t, mask_t = dataset[idx]
            fname = dataset.filenames[idx]

            # Forward pass
            logits = model(image_t.unsqueeze(0).to(device))
            probs = torch.sigmoid(logits)
            pred_t = (probs >= threshold).float()

            img_np = image_t.squeeze(0).cpu().numpy()
            mask_np = mask_t.squeeze(0).cpu().numpy()
            pred_np = pred_t.squeeze().cpu().numpy()

            # Metric calculation
            tp = int(np.sum((pred_np == 1.0) & (mask_np == 1.0)))
            fp = int(np.sum((pred_np == 1.0) & (mask_np == 0.0)))
            fn = int(np.sum((pred_np == 0.0) & (mask_np == 1.0)))
            tn = int(np.sum((pred_np == 0.0) & (mask_np == 0.0)))

            denom_iou = tp + fp + fn
            iou = tp / denom_iou if denom_iou > 0 else 0.0

            denom_dice = 2 * tp + fp + fn
            dice = (2.0 * tp) / denom_dice if denom_dice > 0 else 0.0

            overlay = create_comparison_overlay(img_np, pred_np, mask_np)

            # Generate 1x4 subplot
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

            # 1. Original SAR Input
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title(f"1. SAR Input\n{fname} (idx={idx})", fontsize=10)
            axes[0].axis("off")

            # 2. Ground Truth Mask
            axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title(f"2. Ground Truth\nOil Pixels: {int(np.sum(mask_np)):,}", fontsize=10)
            axes[1].axis("off")

            # 3. U-Net Prediction
            axes[2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title(f"3. U-Net Prediction (Thresh=0.5)\nIoU: {iou:.4f} | Dice: {dice:.4f}", fontsize=10)
            axes[2].axis("off")

            # 4. Error Comparison Overlay
            axes[3].imshow(overlay)
            axes[3].set_title(f"4. Comparison Overlay\nGreen=TP | Red=FP | Yellow=FN", fontsize=10)
            axes[3].axis("off")

            plt.tight_layout()
            out_filename = f"sample_{rank:02d}_{os.path.splitext(fname)[0]}.png"
            out_path = os.path.join(output_dir, out_filename)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            record = {
                "rank": rank,
                "index": idx,
                "filename": fname,
                "iou": iou,
                "dice": dice,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "path": out_path
            }
            results_summary.append(record)
            print(f"  [{rank:02d}/10] Saved: {out_filename} | IoU: {iou:.4f} | Dice: {dice:.4f} | TP: {tp:5d} | FP: {fp:5d} | FN: {fn:5d}")

    print(f"\nSuccessfully saved all 10 U-Net visual comparisons to: {output_dir}")
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate U-Net Visual Comparisons")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/unet/best_unet.pth", help="U-Net checkpoint path")
    parser.add_argument("--data_root", type=str, default="data", help="Data directory root")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations/unet", help="Destination folder for PNGs")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision probability threshold")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cp_path = os.path.join(base_dir, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    d_root = os.path.join(base_dir, args.data_root) if not os.path.isabs(args.data_root) else args.data_root
    o_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    run_unet_visualization(
        checkpoint_path=cp_path,
        data_root=d_root,
        output_dir=o_dir,
        threshold=args.threshold
    )
