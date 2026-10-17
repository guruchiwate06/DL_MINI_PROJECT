"""
train.py
--------
Training and validation pipeline for the Oil Spill Segmentation CNN.
Includes:
  - BCEWithLogitsLoss and Adam optimizer (with optional weight decay)
  - Configurable epochs, batch size, learning rate, and data augmentation
  - Live progress display every N batches with elapsed time
  - Validation loss, Dice, and IoU tracking after every epoch
  - Early stopping with configurable patience
  - Minimum validation loss checkpoint saving
"""

import os
import sys
import time
import argparse

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import Adam

from src.dataset import get_dataloaders
from src.model import SimpleSegmentationCNN


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
    Executes one complete training epoch with live progress reporting.
    Follows standard Deep Learning workflow:
      1. model.train()
      2. optimizer.zero_grad()
      3. Forward propagation
      4. Loss calculation
      5. Backward propagation (gradients)
      6. Optimizer weight update
    """
    model.train()
    running_loss = 0.0
    total_batches = 0
    total_b = max_batches if max_batches is not None else len(dataloader)
    epoch_start_time = time.time()

    for batch_idx, (images, masks) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        masks = masks.to(device)

        # 1. Clear old gradients from previous iteration
        optimizer.zero_grad()

        # 2. Forward propagation: compute raw logits
        logits = model(images)

        # 3. Calculate Binary Cross-Entropy loss on logits
        loss = criterion(logits, masks)

        # 4. Backpropagation: calculate gradients of loss w.r.t model weights
        loss.backward()

        # 5. Optimizer step: update weights using Adam formula
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

        # Periodic live progress feedback
        is_final_batch = (batch_idx + 1) == total_b
        if (batch_idx + 1) % print_freq == 0 or is_final_batch:
            pct = ((batch_idx + 1) / total_b) * 100.0
            elapsed = time.time() - epoch_start_time
            current_avg_loss = running_loss / total_batches
            print(
                f"  [Epoch {epoch:02d}/{total_epochs:02d}] "
                f"Batch {batch_idx + 1:3d}/{total_b:3d} ({pct:5.1f}%) | "
                f"Running Loss: {current_avg_loss:.4f} | "
                f"Elapsed: {format_time(elapsed)}",
                flush=True
            )

    epoch_loss = running_loss / total_batches if total_batches > 0 else 0.0
    return epoch_loss


def validate(model, dataloader, criterion, device, threshold=0.5, max_batches=None):
    """
    Evaluates the model on validation data without updating weights.
    Uses model.eval() and torch.no_grad().
    Computes validation loss, Dice score, and IoU.
    """
    model.eval()
    running_loss = 0.0
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

            # Forward pass only (no backprop)
            logits = model(images)
            loss = criterion(logits, masks)

            running_loss += loss.item()
            total_batches += 1

            # Convert logits to probabilities and binary predictions
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            total_tp += torch.sum((preds == 1.0) & (masks == 1.0)).item()
            total_fp += torch.sum((preds == 1.0) & (masks == 0.0)).item()
            total_fn += torch.sum((preds == 0.0) & (masks == 1.0)).item()
            total_tn += torch.sum((preds == 0.0) & (masks == 0.0)).item()

    val_loss = running_loss / total_batches if total_batches > 0 else 0.0
    
    # Global validation Dice and IoU
    denom_dice = 2.0 * total_tp + total_fp + total_fn
    val_dice = (2.0 * total_tp) / denom_dice if denom_dice > 0 else 0.0

    denom_iou = total_tp + total_fp + total_fn
    val_iou = total_tp / denom_iou if denom_iou > 0 else 0.0

    return val_loss, val_dice, val_iou


def run_training(args):
    # Set compute device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Set up data paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    checkpoint_dir = os.path.join(base_dir, args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resolve checkpoint filename (smoke test avoids overwriting production checkpoint)
    if args.smoke_test and args.checkpoint_name == "best_model.pth":
        checkpoint_filename = "smoke_test_model.pth"
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
    model = SimpleSegmentationCNN(in_channels=1, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. Loss function and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Setup configuration
    epochs = 2 if args.smoke_test else args.epochs
    max_batches = 3 if args.smoke_test else None
    print_freq = 1 if args.smoke_test else args.print_freq

    # Startup Banner
    print("=" * 65, flush=True)
    print("          OIL SPILL SEGMENTATION CNN TRAINING", flush=True)
    print("=" * 65, flush=True)
    print(f"Device                   : {device}", flush=True)
    print(f"Trainable Parameters     : {total_params:,}", flush=True)
    print(f"Dataset Size             : Train = {len(train_loader.dataset):,} | Val = {len(val_loader.dataset):,}", flush=True)
    print(f"Batch Size               : {args.batch_size}", flush=True)
    print(f"Learning Rate            : {args.lr}", flush=True)
    print(f"Weight Decay             : {args.weight_decay}", flush=True)
    print(f"Data Augmentation        : {'ENABLED (Flips & 90-deg Rotations)' if args.augment else 'DISABLED'}", flush=True)
    print(f"Total Epochs             : {epochs}", flush=True)
    print(f"Early Stopping Patience  : {args.patience if args.patience > 0 else 'Disabled'}", flush=True)
    print(f"Checkpoint Monitor       : {args.monitor}", flush=True)
    print(f"Loss Function            : BCEWithLogitsLoss", flush=True)
    print(f"Checkpoint Target        : {checkpoint_path}", flush=True)
    if args.smoke_test:
        print("MODE                     : SMOKE TEST (2 epochs, 3 batches each)", flush=True)
    print("=" * 65 + "\n", flush=True)

    # Tracking metrics
    best_score = float("inf") if args.monitor == "val_loss" else 0.0
    best_epoch = 0
    patience_counter = 0
    best_metrics = {}

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": []
    }

    overall_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training pass
        train_loss = train_one_epoch(
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

        # Validation pass (no weights updated)
        val_loss, val_dice, val_iou = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
            max_batches=max_batches
        )

        epoch_duration = time.time() - epoch_start
        total_elapsed = time.time() - overall_start_time

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        # End of epoch summary
        print(
            f"\nEpoch [{epoch:02d}/{epochs:02d}] Summary:\n"
            f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice * 100:.2f}% | Val IoU: {val_iou * 100:.2f}%\n"
            f"  Epoch Duration: {format_time(epoch_duration)} | Total Elapsed: {format_time(total_elapsed)}",
            flush=True
        )

        # Evaluate checkpoint improvement
        if args.monitor == "val_loss":
            improved = val_loss < best_score
            current_metric_val = val_loss
        else:  # val_dice
            improved = val_dice > best_score
            current_metric_val = val_dice

        if improved:
            best_score = current_metric_val
            best_epoch = epoch
            best_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou
            }
            patience_counter = 0

            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "augmented": args.augment
            }, checkpoint_path)
            print(f"  --> [CHECKPOINT SAVED] Best {args.monitor} improved to {best_score:.4f} -> {checkpoint_path}\n", flush=True)
        else:
            patience_counter += 1
            print(f"  --> [NO IMPROVEMENT] Patience counter: {patience_counter}/{args.patience}\n", flush=True)
            if args.patience > 0 and patience_counter >= args.patience:
                print("=" * 65, flush=True)
                print(f"[EARLY STOPPING] Validation {args.monitor} did not improve for {args.patience} consecutive epochs.", flush=True)
                print(f"Stopping training at Epoch {epoch}.", flush=True)
                print("=" * 65 + "\n", flush=True)
                break

    total_training_time = time.time() - overall_start_time

    # Training Completion Report
    print("=" * 65, flush=True)
    print("               TRAINING PIPELINE COMPLETED", flush=True)
    print("=" * 65, flush=True)
    if best_metrics:
        print(f"Best Epoch               : {best_metrics['epoch']}", flush=True)
        print(f"Best Validation Loss     : {best_metrics['val_loss']:.4f}", flush=True)
        print(f"Validation Dice (F1)     : {best_metrics['val_dice'] * 100:.2f}%", flush=True)
        print(f"Validation IoU           : {best_metrics['val_iou'] * 100:.2f}%", flush=True)
        print(f"Saved Checkpoint Path    : {checkpoint_path}", flush=True)
    else:
        print("No checkpoint was saved during this run.", flush=True)
    print(f"Total Execution Time     : {format_time(total_training_time)}", flush=True)
    print("=" * 65 + "\n", flush=True)

    return history, best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN for Oil Spill Segmentation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 regularization (weight decay)")
    parser.add_argument("--augment", action="store_true", help="Enable training-only data augmentation (flips & 90-deg rotations)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs without improvement; <=0 disables)")
    parser.add_argument("--monitor", type=str, default="val_loss", choices=["val_loss", "val_dice"], help="Metric to monitor for best checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for validation Dice/IoU")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints", help="Directory to save weights")
    parser.add_argument("--checkpoint_name", type=str, default="best_model.pth", help="Checkpoint filename")
    parser.add_argument("--print_freq", type=int, default=10, help="Batch interval for live terminal progress")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--smoke_test", action="store_true", help="Quick 2-epoch smoke test on 3 batches")

    args = parser.parse_args()
    run_training(args)
