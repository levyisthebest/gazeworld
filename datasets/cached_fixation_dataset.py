"""
Cached Fixation Dataset - loads preprocessed fixation features from npz.
Eliminates the 18s/sample CSV parsing bottleneck.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CachedFixationDataset(Dataset):
    def __init__(self, cache_path, img_size=224, val_split=0.1, split="train"):
        data = np.load(cache_path, allow_pickle=True)
        self.img_paths = data['img_paths']
        self.patch_ids = data['patch_ids']
        self.coords = data['coords']
        self.meta = data['meta']
        self.masks = data['masks']

        N = len(self.img_paths)
        rng = np.random.RandomState(42)
        idx = rng.permutation(N)
        n_val = max(1, int(N * val_split))
        if split == "val":
            self.indices = idx[:n_val]
        else:
            self.indices = idx[n_val:]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        img_path = str(self.img_paths[i])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), 0)
        img = self.transform(img)

        return {
            "image": img,
            "fix_patch_ids": torch.tensor(self.patch_ids[i], dtype=torch.long),
            "fix_coords": torch.tensor(self.coords[i], dtype=torch.float32),
            "fix_meta": torch.tensor(self.meta[i], dtype=torch.float32),
            "fix_mask": torch.tensor(self.masks[i], dtype=torch.bool),
        }


def collate_cached(batch):
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "fix_patch_ids": torch.stack([b["fix_patch_ids"] for b in batch]),
        "fix_coords": torch.stack([b["fix_coords"] for b in batch]),
        "fix_meta": torch.stack([b["fix_meta"] for b in batch]),
        "fix_mask": torch.stack([b["fix_mask"] for b in batch]),
    }
