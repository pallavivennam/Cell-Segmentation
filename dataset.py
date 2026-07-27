"""
Data loading & preprocessing for the Kaggle Data Science Bowl 2018
nuclei/cell segmentation dataset.

Each sample lives in its own folder:
    stage1_train/<image_id>/images/<image_id>.png
    stage1_train/<image_id>/masks/*.png   (one file per individual nucleus)

This module:
  1. Discovers all sample folders.
  2. Loads the RGB image and merges the per-instance masks into a single
     binary semantic mask (foreground = any nucleus, background = 0).
  3. Applies preprocessing: resize, normalization, and (for training)
     light data augmentation (random flip, rotation, brightness/contrast).
  4. Exposes a PyTorch Dataset + DataLoader helpers.
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


def list_sample_ids(data_root: str):
    data_root = Path(data_root)
    ids = [p.name for p in data_root.iterdir() if p.is_dir()]
    return sorted(ids)


def load_image(data_root: str, image_id: str) -> np.ndarray:
    img_dir = Path(data_root) / image_id / "images"
    img_path = next(img_dir.glob("*.png"))
    img = Image.open(img_path).convert("RGB")
    return np.array(img)


def load_merged_mask(data_root: str, image_id: str, shape) -> np.ndarray:
    mask_dir = Path(data_root) / image_id / "masks"
    merged = np.zeros(shape[:2], dtype=np.uint8)
    if mask_dir.exists():
        for mask_path in mask_dir.glob("*.png"):
            m = np.array(Image.open(mask_path).convert("L"))
            merged = np.maximum(merged, (m > 0).astype(np.uint8))
    return merged


class NucleiDataset(Dataset):

    def __init__(self, data_root, image_ids, img_size=128, augment=False):
        self.data_root = data_root
        self.image_ids = image_ids
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.image_ids)

    def _resize(self, img_arr, mask_arr):
        size = (self.img_size, self.img_size)
        img = Image.fromarray(img_arr).resize(size, Image.BILINEAR)
        mask = Image.fromarray(mask_arr * 255).resize(size, Image.NEAREST)
        return np.array(img), (np.array(mask) > 127).astype(np.uint8)

    def _augment(self, img, mask):
        # Random horizontal flip
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        # Random vertical flip
        if random.random() < 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        # Random 90-degree rotation
        k = random.choice([0, 1, 2, 3])
        if k:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()
        # Random brightness/contrast jitter (image only)
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return img, mask

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_arr = load_image(self.data_root, image_id)
        mask_arr = load_merged_mask(self.data_root, image_id, img_arr.shape)

        img_arr, mask_arr = self._resize(img_arr, mask_arr)

        if self.augment:
            img_arr, mask_arr = self._augment(img_arr, mask_arr)

        img_t = torch.from_numpy(img_arr).float().permute(2, 0, 1) / 255.0
        mask_t = torch.from_numpy(mask_arr).float().unsqueeze(0)

        return img_t, mask_t


def train_val_split(image_ids, val_fraction=0.15, seed=42):
    ids = list(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids


def make_dataloaders(data_root, img_size=128, batch_size=16,
                      val_fraction=0.15, num_workers=2, seed=42,
                      max_samples=None):

    ids = list_sample_ids(data_root)
    if max_samples is not None:
        random.Random(seed).shuffle(ids)
        ids = ids[:max_samples]

    train_ids, val_ids = train_val_split(ids, val_fraction, seed)

    train_ds = NucleiDataset(data_root, train_ids, img_size=img_size, augment=True)
    val_ds = NucleiDataset(data_root, val_ids, img_size=img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    return train_loader, val_loader, train_ids, val_ids
