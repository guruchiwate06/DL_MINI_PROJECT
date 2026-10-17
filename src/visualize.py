"""
visualize.py
------------
Generates qualitative side-by-side visual comparisons of SAR input images,
ground-truth masks, predicted masks, and prediction overlays.
"""

import os
import sys
import argparse
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import OilSpillDataset
from src.model import SimpleSegmentationCNN


def visualize_predictions(checkpoint_path: str, data_root: str, output_dir: str, num_samples: int = 5, threshold: float = 0.5):
    """
    Loads model checkpoint, runs inference on validation samples, and saves 4-panel comparison plots.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for visualization: {device}")

    # 1. Verify checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 2. Restore model
    model = SimpleSegmentationCNN(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "N/A")
        print(f"Loaded checkpoint from Epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights.")

    model.eval()

    # 3. Load validation dataset directly
    val_images_dir = os.path.join(data_root, "images", "val")
    val_masks_dir = os.path.join(data_root, "masks", "val")
    dataset = OilSpillDataset(val_images_dir, val_masks_dir, is_train=False)

    total_samples = len(dataset)
    num_samples = min(num_samples, total_samples)
    print(f"Loaded validation dataset ({total_samples} samples). Generating {num_samples} visual comparisons...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    with torch.no_grad():
        for i in range(num_samples):
            image_t, mask_t = dataset[i]
            fname = dataset.filenames[i]

            # Model inference: add batch dimension [1, 1, 256, 256]
            input_tensor = image_t.unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            pred_t = (probs >= threshold).float()

            # Convert to numpy for plotting
            # image_t is in [0.0, 1.0], shape [1, 256, 256]
            img_np = image_t.squeeze(0).cpu().numpy()
            mask_np = mask_t.squeeze(0).cpu().numpy()
            pred_np = pred_t.squeeze().cpu().numpy()

            # Compute individual sample metrics
            tp = np.sum((pred_np == 1.0) & (mask_np == 1.0))
            fp = np.sum((pred_np == 1.0) & (mask_np == 0.0))
            fn = np.sum((pred_np == 0.0) & (mask_np == 1.0))

            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0

            # Create prediction overlay: grayscale background with red highlight on predicted oil
            img_uint8 = (img_np * 255.0).astype(np.uint8)
            overlay = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)
            overlay[pred_np == 1.0] = [235, 50, 50]  # Red for predicted oil

            # Plot 1x4 comparison
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            # Panel 1: Input SAR Image
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title(f"1. SAR Input ({fname})", fontsize=11)
            axes[0].axis("off")

            # Panel 2: Ground Truth Mask
            axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("2. Ground Truth (Oil=1)", fontsize=11)
            axes[1].axis("off")

            # Panel 3: Model Prediction
            axes[2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title(f"3. Prediction (Thresh={threshold})\nIoU: {iou:.2f} | Dice: {dice:.2f}", fontsize=11)
            axes[2].axis("off")

            # Panel 4: Overlay
            axes[3].imshow(overlay)
            axes[3].set_title("4. Overlay (Red=Pred Oil)", fontsize=11)
            axes[3].axis("off")

            plt.tight_layout()
            out_filename = f"sample_{i+1:02d}_{os.path.splitext(fname)[0]}.png"
            out_path = os.path.join(output_dir, out_filename)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            saved_files.append(out_path)
            print(f"  [{i+1}/{num_samples}] Saved visualization: {out_filename} (IoU: {iou:.4f}, Dice: {dice:.4f})")

    print(f"\nSuccessfully generated {len(saved_files)} visualizations in: {output_dir}")
    return saved_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Oil Spill Segmentation Predictions")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Path to checkpoint")
    parser.add_argument("--data_root", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations", help="Directory to save figures")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of validation samples to visualize")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for probability")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cp_path = os.path.join(base_dir, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    d_root = os.path.join(base_dir, args.data_root) if not os.path.isabs(args.data_root) else args.data_root
    o_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    visualize_predictions(
        checkpoint_path=cp_path,
        data_root=d_root,
        output_dir=o_dir,
        num_samples=args.num_samples,
        threshold=args.threshold
    )
