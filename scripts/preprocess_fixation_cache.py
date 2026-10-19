#!/usr/bin/env python3
"""
Preprocess all fixation CSVs once and save to a single .npz cache.
This eliminates the 18s/sample CSV parsing bottleneck during training.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets.fixation_dataset import _cluster_fixations, _coord_to_patch_id

CSV_DIR = "/home/myid/yl80817/mimic_eye_all_organized/tables_csv_parts"
IMAGES_DIR = "/home/myid/yl80817/MIMIC_EYE/ChongMa_Gaze_Mimic/images"
OUT_PATH = "/home/myid/yl80817/FixationARJEPA/outputs/fixation_cache.npz"

MAX_FIXATIONS = 96
GRID_SIZE = 14  # we'll remap to 7x7 in dataset


def preprocess():
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    print(f"Total CSVs: {len(csv_files)}")

    valid_records = []
    for f in tqdm(csv_files):
        try:
            df = pd.read_csv(f)
            stem = df.image_stem.iloc[0]
            orig_path = df.image_path.iloc[0]

            # Find image path
            img_path = None
            if os.path.exists(orig_path):
                img_path = orig_path
            else:
                p = os.path.join(IMAGES_DIR, f"{stem}.jpg")
                if os.path.exists(p):
                    img_path = p
            if img_path is None:
                continue

            # Process gaze points
            x = df.x_px.values.astype(np.float32)
            y = df.y_px.values.astype(np.float32)
            t = df.t.values.astype(np.float32)

            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
            x, y, t = x[valid], y[valid], t[valid]

            order = np.argsort(t)
            x, y, t = x[order], y[order], t[order]

            fixations = _cluster_fixations(x, y, t, spatial_thresh=50)
            if len(fixations) < 5:
                continue

            # Get image dimensions for normalization
            from PIL import Image
            img = Image.open(img_path)
            img_w, img_h = img.size

            # Subsample to MAX_FIXATIONS
            T = min(len(fixations), MAX_FIXATIONS)
            if len(fixations) > MAX_FIXATIONS:
                indices = np.linspace(0, len(fixations) - 1, T, dtype=int)
                fixations = [fixations[i] for i in indices]

            patch_ids = np.zeros(T, dtype=np.int64)
            coords = np.zeros((T, 2), dtype=np.float32)
            meta = np.zeros((T, 3), dtype=np.float32)

            total_time = max(fixations[-1][3] - fixations[0][2], 1e-3)

            for i, (cx, cy, st, et, n) in enumerate(fixations[:T]):
                x_norm = np.clip(cx / max(img_w, 1), 0, 1)
                y_norm = np.clip(cy / max(img_h, 1), 0, 1)
                patch_ids[i] = _coord_to_patch_id(x_norm, y_norm, GRID_SIZE)
                coords[i] = [x_norm, y_norm]

                duration = et - st
                if i > 0:
                    prev_cx, prev_cy = fixations[i - 1][0], fixations[i - 1][1]
                    saccade = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                    saccade_norm = saccade / max(img_w, img_h)
                else:
                    saccade_norm = 0.0
                meta[i] = [duration / 5.0, saccade_norm,
                          (st - fixations[0][2]) / total_time]

            # Pad to MAX_FIXATIONS
            patch_ids_full = np.zeros(MAX_FIXATIONS, dtype=np.int64)
            coords_full = np.zeros((MAX_FIXATIONS, 2), dtype=np.float32)
            meta_full = np.zeros((MAX_FIXATIONS, 3), dtype=np.float32)
            mask = np.zeros(MAX_FIXATIONS, dtype=bool)
            patch_ids_full[:T] = patch_ids
            coords_full[:T] = coords
            meta_full[:T] = meta
            mask[:T] = True

            valid_records.append({
                'img_path': img_path,
                'patch_ids': patch_ids_full,
                'coords': coords_full,
                'meta': meta_full,
                'mask': mask,
            })
        except Exception as e:
            continue

    print(f"Valid records: {len(valid_records)}")

    # Save as npz
    np.savez(
        OUT_PATH,
        img_paths=np.array([r['img_path'] for r in valid_records]),
        patch_ids=np.stack([r['patch_ids'] for r in valid_records]),
        coords=np.stack([r['coords'] for r in valid_records]),
        meta=np.stack([r['meta'] for r in valid_records]),
        masks=np.stack([r['mask'] for r in valid_records]),
    )
    print(f"Saved to {OUT_PATH}")

if __name__ == "__main__":
    preprocess()
