"""
train_unet_bce_dice.py
----------------------
Training and validation pipeline for the U-Net model using a combined loss function:
  total_loss = BCEWithLogitsLoss + SoftDiceLoss

Checkpoints and logs are saved separately in outputs/checkpoints/unet_bce_dice/
without overwriting or modifying the pure BCE U-Net checkpoint or CNN checkpoints.
"""

import os
import sys
import time
import argparse

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.optim import Adam

from src.dataset import get_dataloaders
from src.unet_model import UNet
from src.losses import BCEDiceLoss


def format_time(seconds: float) -> str:
    """Formats duration into human-readable string (e.g. 02m 15s or 45s)."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes:02d}m {secs:02d}s"
    else:
        return f"{secs:02d}s"


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs, print_freq=10, max_batches=None):
    """
    Executes one complete training epoch for U-Net with combined BCE + Soft Dice Loss.
    """
    model.train()
    running_total_loss = 0.0
    running_bce_loss = 0.0
    running_dice_loss = 0.0
    total_batches = 0
    total_b = max_batches if max_batches is not None else len(dataloader)
    epoch_start_time = time.time()

    for batch_idx, (images, masks) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        masks = masks.to(device)

        # 1. Clear gradients
        optimizer.zero_grad()

        # 2. Forward pass: compute raw logits
        logits = model(images)

        # 3. Calculate compound loss: BCE + Soft Dice
        total_loss, bce_loss, dice_loss = criterion(logits, masks)

        # 4. Backward pass
        total_loss.backward()

        # 5. Optimizer step
        optimizer.step()

        running_total_loss += total_loss.item()
        running_bce_loss += bce_loss.item()
        running_dice_loss += dice_loss.item()
        total_batches += 1

        is_final_batch = (batch_idx + 1) == total_b
        if (batch_idx + 1) % print_freq == 0 or is_final_batch:
            pct = ((batch_idx + 1) / total_b) * 100.0
            elapsed = time.time() - epoch_start_time
            curr_tot = running_total_loss / total_batches
            curr_bce = running_bce_loss / total_batches
            curr_d = running_dice_loss / total_batches
            print(
                f"  [Epoch {epoch:02d}/{total_epochs:02d}] "
                f"Batch {batch_idx + 1:3d}/{total_b:3d} ({pct:5.1f}%) | "
                f"Total: {curr_tot:.4f} (BCE: {curr_bce:.4f}, Dice: {curr_d:.4f}) | "
                f"Elapsed: {format_time(elapsed)}",
                flush=True
            )

    epoch_total = running_total_loss / total_batches if total_batches > 0 else 0.0
    epoch_bce = running_bce_loss / total_batches if total_batches > 0 else 0.0
    epoch_dice = running_dice_loss / total_batches if total_batches > 0 else 0.0
    return epoch_total, epoch_bce, epoch_dice


def validate(model, dataloader, criterion, device, threshold=0.5, max_batches=None):
    """
    Evaluates U-Net on validation data without updating weights.
    Computes validation total loss (BCE + Dice), component losses, and global Dice & IoU.
    """
    model.eval()
    running_total = 0.0
    running_bce = 0.0
    running_dice = 0.0
    total_batches = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            total_loss, bce_loss, dice_loss = criterion(logits, masks)

            running_total += total_loss.item()
            running_bce += bce_loss.item()
            running_dice += dice_loss.item()
            total_batches += 1

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            total_tp += torch.sum((preds == 1.0) & (masks == 1.0)).item()
            total_fp += torch.sum((preds == 1.0) & (masks == 0.0)).item()
            total_fn += torch.sum((preds == 0.0) & (masks == 1.0)).item()
            total_tn += torch.sum((preds == 0.0) & (masks == 0.0)).item()

    val_total = running_total / total_batches if total_batches > 0 else 0.0
    val_bce = running_bce / total_batches if total_batches > 0 else 0.0
    val_dice_loss = running_dice / total_batches if total_batches > 0 else 0.0

    denom_dice = 2.0 * total_tp + total_fp + total_fn
    val_dice = (2.0 * total_tp) / denom_dice if denom_dice > 0 else 0.0

    denom_iou = total_tp + total_fp + total_fn
    val_iou = total_tp / denom_iou if denom_iou > 0 else 0.0

    return val_total, val_bce, val_dice_loss, val_dice, val_iou


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    checkpoint_dir = os.path.join(base_dir, args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.smoke_test and args.checkpoint_name == "best_unet_bce_dice.pth":
        checkpoint_filename = "smoke_test_bce_dice.pth"
    else:
        checkpoint_filename = args.checkpoint_name
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    # 1. Load Data
    train_loader, val_loader = get_dataloaders(
        data_dir,
        batch_size=args.batch_size,
        augment=args.augment
    )

    # 2. Initialize Model
    model = UNet(in_channels=1, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. Compound Loss and Optimizer
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0, smooth=1.0)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epochs = 2 if args.smoke_test else args.epochs
    max_batches = 3 if args.smoke_test else None
    print_freq = 1 if args.smoke_test else args.print_freq

    # Startup Banner
    print("=" * 65, flush=True)
    print("     U-NET COMBINED LOSS (BCE + SOFT DICE) TRAINING", flush=True)
    print("=" * 65, flush=True)
    print(f"Model Architecture       : U-Net (4 Encoder-Decoder Levels + Bottleneck)", flush=True)
    print(f"Loss Function            : Combined BCEWithLogitsLoss + SoftDiceLoss (1:1)", flush=True)
    print(f"Device                   : {device}", flush=True)
    print(f"Trainable Parameters     : {total_params:,}", flush=True)
    print(f"Dataset Size             : Train = {len(train_loader.dataset):,} | Val = {len(val_loader.dataset):,}", flush=True)
    print(f"Batch Size               : {args.batch_size}", flush=True)
    print(f"Learning Rate            : {args.lr}", flush=True)
    print(f"Weight Decay             : {args.weight_decay}", flush=True)
    print(f"Data Augmentation        : {'ENABLED (Flips & 90-deg Rotations)' if args.augment else 'DISABLED'}", flush=True)
    print(f"Total Epochs             : {epochs}", flush=True)
    print(f"Early Stopping Patience  : {args.patience if args.patience > 0 else 'Disabled'}", flush=True)
    print(f"Checkpoint Target        : {checkpoint_path}", flush=True)
    if args.smoke_test:
        print("MODE                     : SMOKE TEST (2 epochs, 3 batches each)", flush=True)
    print("=" * 65 + "\n", flush=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_metrics = {}

    history = {
        "train_total_loss": [],
        "train_bce": [],
        "train_dice_loss": [],
        "val_total_loss": [],
        "val_bce": [],
        "val_dice_loss": [],
        "val_dice": [],
        "val_iou": []
    }

    overall_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_tot, train_bce, train_d = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            print_freq=print_freq,
            max_batches=max_batches
        )

        val_tot, val_bce, val_d, val_dice, val_iou = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
            max_batches=max_batches
        )

        epoch_duration = time.time() - epoch_start
        total_elapsed = time.time() - overall_start_time

        history["train_total_loss"].append(train_tot)
        history["train_bce"].append(train_bce)
        history["train_dice_loss"].append(train_d)
        history["val_total_loss"].append(val_tot)
        history["val_bce"].append(val_bce)
        history["val_dice_loss"].append(val_d)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        print(
            f"\nEpoch [{epoch:02d}/{epochs:02d}] Summary:\n"
            f"  Train Total Loss: {train_tot:.4f} (BCE: {train_bce:.4f}, DiceLoss: {train_d:.4f})\n"
            f"  Val Total Loss  : {val_tot:.4f} (BCE: {val_bce:.4f}, DiceLoss: {val_d:.4f})\n"
            f"  Val Dice Score  : {val_dice * 100:.2f}% | Val IoU: {val_iou * 100:.2f}%\n"
            f"  Epoch Duration  : {format_time(epoch_duration)} | Total Elapsed: {format_time(total_elapsed)}",
            flush=True
        )

        # Early stopping & checkpointing based on Validation Total Loss
        if val_tot < best_val_loss:
            best_val_loss = val_tot
            best_epoch = epoch
            best_metrics = {
                "epoch": epoch,
                "val_total_loss": val_tot,
                "val_bce": val_bce,
                "val_dice_loss": val_d,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "train_total_loss": train_tot,
                "train_bce": train_bce
            }
            patience_counter = 0

            torch.save({
                "model_type": "UNet",
                "loss_type": "BCE + SoftDiceLoss",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_bce,             # Preserved as standard BCE loss for cross-model comparability
                "val_total_loss": val_tot,       # Total compound loss
                "val_dice_loss": val_d,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "train_loss": train_bce,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "augmented": args.augment
            }, checkpoint_path)
            print(f"  --> [CHECKPOINT SAVED] Best val_total_loss improved to {val_tot:.4f} (BCE: {val_bce:.4f}) -> {checkpoint_path}\n", flush=True)
        else:
            patience_counter += 1
            print(f"  --> [NO IMPROVEMENT] Patience counter: {patience_counter}/{args.patience}\n", flush=True)
            if args.patience > 0 and patience_counter >= args.patience:
                print("=" * 65, flush=True)
                print(f"[EARLY STOPPING] Validation total loss did not improve for {args.patience} consecutive epochs.", flush=True)
                print(f"Stopping training at Epoch {epoch}.", flush=True)
                print("=" * 65 + "\n", flush=True)
                break

    total_training_time = time.time() - overall_start_time

    print("=" * 65, flush=True)
    print("      U-NET BCE + SOFT DICE TRAINING PIPELINE COMPLETED", flush=True)
    print("=" * 65, flush=True)
    if best_metrics:
        print(f"Best Epoch               : {best_metrics['epoch']}", flush=True)
        print(f"Best Val Total Loss      : {best_metrics['val_total_loss']:.4f}", flush=True)
        print(f"Validation BCE Loss      : {best_metrics['val_bce']:.4f}", flush=True)
        print(f"Validation Soft Dice Loss: {best_metrics['val_dice_loss']:.4f}", flush=True)
        print(f"Validation Dice (F1)     : {best_metrics['val_dice'] * 100:.2f}%", flush=True)
        print(f"Validation IoU           : {best_metrics['val_iou'] * 100:.2f}%", flush=True)
        print(f"Saved Checkpoint Path    : {checkpoint_path}", flush=True)
    print(f"Total Execution Time     : {format_time(total_training_time)}", flush=True)
    print("=" * 65 + "\n", flush=True)

    return history, best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net with BCE + Soft Dice Loss")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="L2 regularization (weight decay)")
    parser.add_argument("--augment", action="store_true", help="Enable training-only data augmentation")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for validation Dice/IoU")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints/unet_bce_dice", help="Directory to save weights")
    parser.add_argument("--checkpoint_name", type=str, default="best_unet_bce_dice.pth", help="Checkpoint filename")
    parser.add_argument("--print_freq", type=int, default=10, help="Batch interval for live terminal progress")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--smoke_test", action="store_true", help="Quick 2-epoch smoke test on 3 batches")

    args = parser.parse_args()
    run_training(args)
