"""
GazeSearch dataset for scanpath prediction.

Each sample: (image, task_name, scanpath_7pts, task_id)
  - image: CXR (224×224)
  - task_name: disease to search for (e.g. "lung opacity")
  - scanpath: 7 fixation points [(x,y)] — center prepended as point 0
  - task_id: integer index (0-12) for the 13 disease categories
"""

import os
import json
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

TASKS = [
    "lung opacity", "pneumonia", "cardiomegaly", "support devices",
    "pleural effusion", "atelectasis", "edema", "enlarged cardiomediastinum",
    "lung lesion", "pleural other", "fracture", "consolidation", "pneumothorax",
]
TASK2ID = {t: i for i, t in enumerate(TASKS)}

IMG_CACHE_PATH = "/home/myid/yl80817/SemanticGazeJEPA/outputs/image_path_cache.pkl"
_IMG_CACHE = None


def _get_img_cache():
    global _IMG_CACHE
    if _IMG_CACHE is None:
        with open(IMG_CACHE_PATH, 'rb') as f:
            _IMG_CACHE = pickle.load(f)
    return _IMG_CACHE


def _find_image(name, fallback_dir=None):
    """Find CXR image via precomputed cache."""
    cache = _get_img_cache()
    if name in cache:
        return cache[name]
    if fallback_dir:
        p = os.path.join(fallback_dir, name)
        if os.path.exists(p):
            return p
    return None


class GazeSearchDataset(Dataset):
    def __init__(self, json_path, split="train", img_size=224,
                 fallback_img_dir=None, max_fixations=7,
                 first_fix_real=False):
        with open(json_path) as f:
            all_data = json.load(f)

        self.records = [d for d in all_data if d["split"] == split]
        self.img_size = img_size
        self.max_fix = max_fixations
        self.fallback_img_dir = fallback_img_dir
        self.first_fix_real = first_fix_real

        # Build image path cache
        self._img_cache = {}

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        name = rec["name"]
        task = rec["task"]
        task_id = TASK2ID.get(task, 0)

        # Load image
        if name not in self._img_cache:
            self._img_cache[name] = _find_image(name, self.fallback_img_dir)
        img_path = self._img_cache[name]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), 0)
        img = self.transform(img)

        # Build scanpath. Legacy (v1-v4): prepend center point → 7 points total.
        # v5 (first_fix_real=True): use real first fixation as the given conditioning point,
        # no center prepend — matches GazeSearch paper protocol.
        if self.first_fix_real:
            xs = rec["X"][:self.max_fix]
            ys = rec["Y"][:self.max_fix]
            ts = rec["T"][:self.max_fix]
        else:
            cx, cy = self.img_size // 2, self.img_size // 2
            xs = [cx] + rec["X"][:self.max_fix - 1]
            ys = [cy] + rec["Y"][:self.max_fix - 1]
            ts = [0.0] + rec["T"][:self.max_fix - 1]

        T = len(xs)
        scanpath = np.zeros((self.max_fix, 2), dtype=np.float32)
        durations = np.zeros(self.max_fix, dtype=np.float32)
        mask = np.zeros(self.max_fix, dtype=bool)

        for i in range(T):
            scanpath[i] = [xs[i] / self.img_size, ys[i] / self.img_size]  # normalize to [0,1]
            durations[i] = ts[i]
            mask[i] = True

        return {
            "image": img,
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "scanpath": torch.tensor(scanpath, dtype=torch.float32),  # (7, 2) normalized
            "durations": torch.tensor(durations, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "name": name,
            "task_name": task,
            "condition": rec["condition"],
        }


def collate_gazesearch(batch):
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "task_ids": torch.stack([b["task_id"] for b in batch]),
        "scanpaths": torch.stack([b["scanpath"] for b in batch]),
        "durations": torch.stack([b["durations"] for b in batch]),
        "masks": torch.stack([b["mask"] for b in batch]),
        "names": [b["name"] for b in batch],
        "task_names": [b["task_name"] for b in batch],
    }
