"""
dataset.py
----------
PyTorch Dataset and DataLoader implementation for SAR Oil Spill Segmentation.
Loads single-channel SAR images and binary ground-truth masks.
"""

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class OilSpillDataset(Dataset):
    """
    Custom PyTorch Dataset for paired SAR images and segmentation masks.
    """
    def __init__(self, images_dir: str, masks_dir: str, is_train: bool = True, augment: bool = False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.is_train = is_train
        # Augmentation is strictly training-only
        self.augment = augment and is_train

        # Gather all valid PNG filenames
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
        mask_files = set([f for f in os.listdir(masks_dir) if f.endswith(".png")])

        # Match images and masks strictly by filename within this split
        self.filenames = [f for f in image_files if f in mask_files]

        if len(self.filenames) == 0:
            raise RuntimeError(f"No matching image-mask pairs found in {images_dir} and {masks_dir}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)

        # 1. Load SAR image as single-channel grayscale ('L' mode)
        # Decision: SAR images have identical R=G=B channels; 1 channel saves 66% RAM
        image_pil = Image.open(image_path).convert("L")
        image_np = np.array(image_pil, dtype=np.float32)

        # Normalize pixel values from [0, 255] to [0.0, 1.0]
        # Standard min-max normalization keeps input in a stable range for neural network layers
        image_np = image_np / 255.0

        # 2. Load mask and binarize
        # Decision: Convert to 'L' mode and threshold at 128 to clean up anti-aliased edges
        mask_pil = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_pil, dtype=np.float32)

        # Binary threshold: 0 = background/sea, 1 = oil spill
        mask_binary = (mask_np >= 128.0).astype(np.float32)

        # 3. Geometric Data Augmentation (Training split only)
        # SAR satellite images look straight down (nadir/off-nadir); oil slicks and sea
        # surface have no canonical orientation. Flips and 90-degree rotations are label-preserving.
        # Both image and mask MUST undergo the identical geometric transformation.
        if self.augment:
            # Random horizontal flip (p = 0.5)
            if np.random.rand() > 0.5:
                image_np = np.fliplr(image_np).copy()
                mask_binary = np.fliplr(mask_binary).copy()

            # Random vertical flip (p = 0.5)
            if np.random.rand() > 0.5:
                image_np = np.flipud(image_np).copy()
                mask_binary = np.flipud(mask_binary).copy()

            # Random 90-degree rotations (0, 90, 180, or 270 degrees)
            k = int(np.random.randint(0, 4))
            if k > 0:
                image_np = np.rot90(image_np, k).copy()
                mask_binary = np.rot90(mask_binary, k).copy()

        # Convert to tensors: add channel dimension -> shape (1, 256, 256)
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)

        return image_tensor, mask_tensor


def get_dataloaders(data_root: str, batch_size: int = 16, num_workers: int = 0, augment: bool = False):
    """
    Creates and returns train and validation DataLoaders.
    Augmentation is only applied to the training DataLoader if augment=True.
    """
    train_images_dir = os.path.join(data_root, "images", "train")
    train_masks_dir = os.path.join(data_root, "masks", "train")
    val_images_dir = os.path.join(data_root, "images", "val")
    val_masks_dir = os.path.join(data_root, "masks", "val")

    train_dataset = OilSpillDataset(train_images_dir, train_masks_dir, is_train=True, augment=augment)
    val_dataset = OilSpillDataset(val_images_dir, val_masks_dir, is_train=False, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Shuffle training data to break batch ordering bias
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # Do not shuffle validation data for reproducible evaluation
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test script for Stage 1 dataset validation
    print("=== Testing OilSpillDataset and DataLoader ===")
    
    # Path configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=16)
    
    print(f"Train samples count : {len(train_loader.dataset)}")
    print(f"Val samples count   : {len(val_loader.dataset)}")
    print(f"Train batches count : {len(train_loader)}")
    print(f"Val batches count   : {len(val_loader)}")

    # Fetch a single batch from train_loader
    images_batch, masks_batch = next(iter(train_loader))

    print("\n--- Batch Verification ---")
    print(f"Images batch shape : {images_batch.shape} (Expected: [16, 1, 256, 256])")
    print(f"Images dtype       : {images_batch.dtype} (Expected: torch.float32)")
    print(f"Images min/max     : {images_batch.min().item():.3f} / {images_batch.max().item():.3f}")
    
    print(f"\nMasks batch shape  : {masks_batch.shape} (Expected: [16, 1, 256, 256])")
    print(f"Masks dtype        : {masks_batch.dtype} (Expected: torch.float32)")
    print(f"Unique mask values : {torch.unique(masks_batch).tolist()} (Expected: [0.0, 1.0])")
    
    # Assertions for correctness
    assert images_batch.shape == (16, 1, 256, 256), "Image shape mismatch!"
    assert masks_batch.shape == (16, 1, 256, 256), "Mask shape mismatch!"
    assert set(torch.unique(masks_batch).tolist()).issubset({0.0, 1.0}), "Mask contains non-binary values!"
    assert images_batch.min() >= 0.0 and images_batch.max() <= 1.0, "Image normalization outside [0, 1]!"

    print("\nAll dataset loading checks PASSED successfully!")
