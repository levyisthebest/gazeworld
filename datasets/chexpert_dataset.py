"""
CheXpert dataset.

Handles both:
  - Stage 1 I-JEPA pretraining (returns images only)
  - Downstream linear probe evaluation (returns images + multi-label targets)

Label columns (14 classes, consistent with EGMA):
  No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity,
  Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax,
  Pleural Effusion, Pleural Other, Fracture, Support Devices
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from datasets.transforms import get_jepa_transform, get_chexpert_eval_transform


LABEL_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]


class CheXpertDataset(Dataset):
    """
    Args:
        csv_path:    path to labels CSV (has Path column + 14 label columns)
        img_root:    root directory that prefixes the Path column entries
                     Set to "" if Path column already contains absolute paths.
        transform:   torchvision transform (if None, default is used)
        mode:        "pretrain" — returns image only (for I-JEPA)
                     "finetune" — returns (image, label_tensor)
        label_policy:"zeros"  — replace NaN/-1 with 0
                     "ones"   — replace NaN with 0, -1 (uncertain) with 1
        frontal_only: if True, keep only frontal views
    """

    def __init__(self, csv_path, img_root="", transform=None,
                 mode="pretrain", label_policy="zeros", frontal_only=True,
                 img_size=224):
        self.img_root = img_root
        self.mode = mode

        df = pd.read_csv(csv_path)

        # Keep only frontal views (filter out lateral)
        if frontal_only and "Frontal/Lateral" in df.columns:
            df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)

        # Fill label columns
        for col in LABEL_COLS:
            if col not in df.columns:
                df[col] = 0.0

        if label_policy == "zeros":
            df[LABEL_COLS] = df[LABEL_COLS].fillna(0.0).replace(-1.0, 0.0)
        elif label_policy == "ones":
            df[LABEL_COLS] = df[LABEL_COLS].fillna(0.0).replace(-1.0, 1.0)

        self.df = df.reset_index(drop=True)
        self.transform = transform or (
            get_jepa_transform(img_size=img_size, is_train=(mode == "pretrain"))
            if mode == "pretrain"
            else get_chexpert_eval_transform(img_size=img_size)
        )

    def __len__(self):
        return len(self.df)

    def _load_image(self, row) -> Image.Image:
        path_col = "Path" if "Path" in row.index else row.index[0]
        rel_path = row[path_col]
        if self.img_root:
            full_path = os.path.join(self.img_root, rel_path)
            # Handle CheXpert CSV prefix mismatch:
            # CSV may have "CheXpert-v1.0-small/train/..." but files are in "train/..."
            if not os.path.exists(full_path):
                # Strip version prefixes (CheXpert-v1.0-small/, CheXpert-v1.0/)
                for prefix in ["CheXpert-v1.0-small/", "CheXpert-v1.0/"]:
                    if prefix in rel_path:
                        alt = os.path.join(self.img_root, rel_path.replace(prefix, ""))
                        if os.path.exists(alt):
                            full_path = alt
                            break
            if not os.path.exists(full_path):
                # Handle valid/ <-> val/ directory name mismatch
                for src, dst in [("/valid/", "/val/"), ("/val/", "/valid/")]:
                    if src in full_path:
                        alt = full_path.replace(src, dst)
                        if os.path.exists(alt):
                            full_path = alt
                            break
        else:
            full_path = rel_path
        img = Image.open(full_path).convert("L")   # grayscale
        return self._pad_square(img)

    @staticmethod
    def _pad_square(img: Image.Image, fill=0) -> Image.Image:
        w, h = img.size
        size = max(w, h)
        padded = Image.new("L", (size, size), fill)
        padded.paste(img, ((size - w) // 2, (size - h) // 2))
        return padded

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_image(row)
        img = self.transform(img)

        if self.mode == "pretrain":
            return img

        labels = row[LABEL_COLS].values.astype(np.float32)
        return img, torch.tensor(labels)
