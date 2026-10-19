"""
Fixation Sequence Dataset for Fixation-AR-JEPA.

Loads from:
  - tables_csv_parts/*.csv — raw gaze points (x_px, y_px, t) at ~1000Hz
  - images/ — CXR images (from CSV image_path or ChongMa images dir)

Processing pipeline:
  1. Load raw gaze CSV (~40k points per session)
  2. Cluster into fixations (points within spatial+temporal threshold)
  3. Downsample to N key fixations
  4. Map each fixation (x,y) to nearest ViT patch index (14×14 grid)
  5. Compute metadata: duration, saccade amplitude, time offset
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def _cluster_fixations(x, y, t, spatial_thresh=50, temporal_thresh=0.1):
    """
    Cluster raw gaze points into fixations.
    A new fixation starts when the gaze moves > spatial_thresh pixels
    or there's a temporal gap > temporal_thresh seconds.

    Returns list of (cx, cy, start_t, end_t, n_points)
    """
    if len(x) == 0:
        return []

    fixations = []
    cx_sum, cy_sum, count = x[0], y[0], 1
    start_t = t[0]

    for i in range(1, len(x)):
        dx = x[i] - cx_sum / count
        dy = y[i] - cy_sum / count
        dt = t[i] - t[i - 1]
        dist = np.sqrt(dx ** 2 + dy ** 2)

        if dist > spatial_thresh or dt > temporal_thresh:
            # Save current fixation
            fixations.append((
                cx_sum / count,
                cy_sum / count,
                start_t,
                t[i - 1],
                count,
            ))
            # Start new fixation
            cx_sum, cy_sum, count = x[i], y[i], 1
            start_t = t[i]
        else:
            cx_sum += x[i]
            cy_sum += y[i]
            count += 1

    # Last fixation
    fixations.append((cx_sum / count, cy_sum / count, start_t, t[-1], count))
    return fixations


def _coord_to_patch_id(x_norm, y_norm, grid_size=14):
    """Map normalized (x,y) in [0,1] to patch index (0 to grid_size²-1)."""
    col = int(np.clip(x_norm * grid_size, 0, grid_size - 1))
    row = int(np.clip(y_norm * grid_size, 0, grid_size - 1))
    return row * grid_size + col


class FixationSequenceDataset(Dataset):
    """
    Each sample:
      - image:         (1, 224, 224) or (3, 224, 224)
      - fix_patch_ids: (T,) long — patch index per fixation
      - fix_coords:    (T, 2) float — normalized (x, y) in [0, 1]
      - fix_meta:      (T, 3) float — (duration, saccade_amp, time_offset)
      - fix_mask:      (T,) bool — True for valid
    """

    def __init__(self, csv_dir, images_dir=None, fallback_img_dir=None,
                 max_fixations=64, grid_size=14, img_size=224,
                 spatial_thresh=50, min_fixations=5,
                 is_train=True, val_split=0.1, split="train"):
        self.max_fixations = max_fixations
        self.grid_size = grid_size
        self.spatial_thresh = spatial_thresh
        self.min_fixations = min_fixations
        self.images_dir = images_dir
        self.fallback_img_dir = fallback_img_dir

        # Find all CSV files
        csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

        # Validate: check which CSVs have accessible images
        valid = []
        for f in csv_files:
            try:
                df_head = pd.read_csv(f, nrows=1)
                img_path = self._find_image(df_head)
                if img_path and os.path.exists(img_path):
                    valid.append(f)
            except Exception:
                pass

        # Train/val split
        np.random.RandomState(42).shuffle(valid)
        n_val = max(1, int(len(valid) * val_split))
        if split == "val":
            self.csv_files = valid[:n_val]
        else:
            self.csv_files = valid[n_val:]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _find_image(self, df_head):
        """Find image path from CSV row."""
        stem = df_head.image_stem.iloc[0]
        # Try original path from CSV
        orig_path = df_head.image_path.iloc[0]
        if os.path.exists(orig_path):
            return orig_path
        # Try images_dir
        if self.images_dir:
            p = os.path.join(self.images_dir, f"{stem}.jpg")
            if os.path.exists(p):
                return p
        # Try fallback
        if self.fallback_img_dir:
            p = os.path.join(self.fallback_img_dir, f"{stem}.jpg")
            if os.path.exists(p):
                return p
        return orig_path  # return anyway, may exist

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        csv_path = self.csv_files[idx]
        df = pd.read_csv(csv_path)

        # ---- Load image ----
        img_path = self._find_image(df)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback: black image
            img = Image.new("RGB", (224, 224), 0)
        img_w, img_h = img.size
        img = self.transform(img)

        # ---- Process gaze points → fixations ----
        x = df.x_px.values.astype(np.float32)
        y = df.y_px.values.astype(np.float32)
        t = df.t.values.astype(np.float32)

        # Remove invalid points
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
        x, y, t = x[valid], y[valid], t[valid]

        # Sort by time
        order = np.argsort(t)
        x, y, t = x[order], y[order], t[order]

        # Cluster into fixations
        fixations = _cluster_fixations(x, y, t, self.spatial_thresh)

        if len(fixations) < self.min_fixations:
            # Fallback: use raw points subsampled
            step = max(1, len(x) // self.max_fixations)
            fixations = [
                (x[i], y[i], t[i], t[min(i + step, len(t) - 1)], 1)
                for i in range(0, len(x), step)
            ]

        # ---- Build fixation tensors ----
        T = min(len(fixations), self.max_fixations)
        if len(fixations) > self.max_fixations:
            # Uniform subsample
            indices = np.linspace(0, len(fixations) - 1, T, dtype=int)
            fixations = [fixations[i] for i in indices]

        patch_ids = np.zeros(T, dtype=np.int64)
        coords = np.zeros((T, 2), dtype=np.float32)
        meta = np.zeros((T, 3), dtype=np.float32)
        mask = np.ones(T, dtype=bool)

        total_time = max(fixations[-1][3] - fixations[0][2], 1e-3) if fixations else 1.0

        for i, (cx, cy, st, et, n) in enumerate(fixations[:T]):
            # Normalize coordinates to [0, 1]
            x_norm = np.clip(cx / max(img_w, 1), 0, 1)
            y_norm = np.clip(cy / max(img_h, 1), 0, 1)

            patch_ids[i] = _coord_to_patch_id(x_norm, y_norm, self.grid_size)
            coords[i] = [x_norm, y_norm]

            duration = et - st
            if i > 0:
                prev_cx, prev_cy = fixations[i - 1][0], fixations[i - 1][1]
                saccade = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                saccade_norm = saccade / max(img_w, img_h)
            else:
                saccade_norm = 0.0

            meta[i] = [
                duration / 5.0,                        # normalize duration
                saccade_norm,                          # normalize saccade amplitude
                (st - fixations[0][2]) / total_time,   # relative time offset [0, 1]
            ]

        return {
            "image": img,
            "fix_patch_ids": torch.tensor(patch_ids, dtype=torch.long),
            "fix_coords": torch.tensor(coords, dtype=torch.float32),
            "fix_meta": torch.tensor(meta, dtype=torch.float32),
            "fix_mask": torch.tensor(mask, dtype=torch.bool),
        }


def collate_fixation(batch):
    """Pad fixation sequences to max length in batch."""
    imgs = torch.stack([b["image"] for b in batch])

    max_T = max(b["fix_patch_ids"].shape[0] for b in batch)
    B = len(batch)

    patch_ids = torch.zeros(B, max_T, dtype=torch.long)
    coords = torch.zeros(B, max_T, 2)
    meta = torch.zeros(B, max_T, 3)
    mask = torch.zeros(B, max_T, dtype=torch.bool)

    for i, b in enumerate(batch):
        T = b["fix_patch_ids"].shape[0]
        patch_ids[i, :T] = b["fix_patch_ids"]
        coords[i, :T] = b["fix_coords"]
        meta[i, :T] = b["fix_meta"]
        mask[i, :T] = b["fix_mask"]

    return {
        "images": imgs,
        "fix_patch_ids": patch_ids,
        "fix_coords": coords,
        "fix_meta": meta,
        "fix_mask": mask,
    }
