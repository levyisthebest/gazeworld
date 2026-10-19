#!/usr/bin/env python3
"""
v5 scanpath training: CheXNet backbone + 14x14 grid + real-first-fix conditioning.

Usage:
  CUDA_VISIBLE_DEVICES=5 python scripts/train_scanpath_v5.py --out_dir outputs/scanpath_v5
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time, logging, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from sgj_models.scanpath_predictor_v5 import ScanpathPredictorV5, scanpath_loss_v5
from data.gazesearch_dataset import GazeSearchDataset, collate_gazesearch

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

DATA_JSON = "/home/myid/yl80817/GazeSearch/data/finding_visual_search_coco_format_train_test_filtered_max_6_split_train_valid_test_2024-07-22.json"
FALLBACK_IMG = "/home/myid/yl80817/MIMIC_EYE/ChongMa_Gaze_Mimic/images"


def euclidean_distance(pred, gt):
    return np.sqrt(((pred - gt) ** 2).sum(axis=-1)).mean()


def scanmatch_score(pred, gt, threshold=0.15):
    dists = np.sqrt(((pred - gt) ** 2).sum(axis=-1))
    return (dists < threshold).mean()


def evaluate_scanpaths(model, loader, device, num_steps=7):
    model.eval()
    all_dists, all_matches = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            task_ids = batch["task_ids"].to(device)
            gt = batch["scanpaths"].numpy()
            masks = batch["masks"].numpy()
            first_fix = batch["scanpaths"][:, 0, :].to(device)
            pred_paths, _ = model.generate(imgs, task_ids, first_fix, num_steps=num_steps)
            pred = pred_paths.cpu().numpy()
            for b in range(len(imgs)):
                n = masks[b].sum()
                if n < 2: continue
                # Skip index 0 (given) — measure predicted steps 1..n-1
                p = pred[b, 1:n]; g = gt[b, 1:n]
                if len(p) == 0: continue
                all_dists.append(euclidean_distance(p, g))
                all_matches.append(scanmatch_score(p, g, threshold=0.15))
    mean_dist = float(np.mean(all_dists)) if all_dists else 999.0
    mean_match = float(np.mean(all_matches)) if all_matches else 0.0
    return mean_dist, mean_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (for 5-seed error bars).")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=448)
    args = parser.parse_args()

    # --- seeding for 5-seed error bars (was absent: training was nondeterministic) ---
    import random as _random
    _random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    log.info(f"v5: CheXNet + 14x14 + real_first_fix, lr={args.lr}, bs={args.batch_size}")

    train_ds = GazeSearchDataset(DATA_JSON, split="train", img_size=args.img_size,
                                  fallback_img_dir=FALLBACK_IMG, first_fix_real=True)
    val_ds = GazeSearchDataset(DATA_JSON, split="valid", img_size=args.img_size,
                                fallback_img_dir=FALLBACK_IMG, first_fix_real=True)
    test_ds = GazeSearchDataset(DATA_JSON, split="test", img_size=args.img_size,
                                 fallback_img_dir=FALLBACK_IMG, first_fix_real=True)
    log.info(f"GazeSearch train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True,
                              collate_fn=collate_gazesearch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0,
                             collate_fn=collate_gazesearch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0,
                              collate_fn=collate_gazesearch)

    model = ScanpathPredictorV5(
        embed_dim=512, context_depth=6, context_heads=8, head_dim=64,
        num_tasks=13, grid_size=14, max_steps=7, freeze_backbone=True,
    ).to(device)
    model.backbone.eval()

    trainable = [p for p in model.parameters() if p.requires_grad]
    log.info(f"Trainable: {sum(p.numel() for p in trainable)/1e6:.1f}M")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_val_dist = 999.0

    for epoch in range(args.epochs):
        model.train()
        model.backbone.eval()  # always frozen + eval
        ep_loss = ep_spatial = ep_dur = 0.0
        t0 = time.time()

        for batch in train_loader:
            imgs = batch["images"].to(device)
            task_ids = batch["task_ids"].to(device)
            scanpaths = batch["scanpaths"].to(device)
            durations = batch["durations"].to(device)
            masks = batch["masks"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                sp_logits, dur_preds, term_preds = model(imgs, task_ids, scanpaths, masks)
                ld = scanpath_loss_v5(sp_logits, dur_preds, term_preds,
                                      scanpaths, durations, masks, grid_size=14)
                loss = ld["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item()
            ep_spatial += ld["l_spatial"].item()
            ep_dur += ld["l_dur"].item()

        scheduler.step()
        n = max(len(train_loader), 1)
        log.info(f"[E{epoch:03d}] loss={ep_loss/n:.4f} spatial={ep_spatial/n:.4f} "
                 f"dur={ep_dur/n:.4f} time={time.time()-t0:.1f}s")

        if (epoch + 1) % 3 == 0 or epoch == args.epochs - 1:
            val_dist, val_match = evaluate_scanpaths(model, val_loader, device)
            test_dist, test_match = evaluate_scanpaths(model, test_loader, device)
            log.info(f"[E{epoch:03d}] val_dist={val_dist:.4f} val_match={val_match:.4f} "
                     f"test_dist={test_dist:.4f} test_match={test_match:.4f}")

            if val_dist < best_val_dist:
                best_val_dist = val_dist
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "val_dist": val_dist, "test_dist": test_dist,
                            "variant": "v5_chexnet_14x14"},
                           os.path.join(args.out_dir, "best_ckpt.pth"))
                log.info(f"  *** New best val_dist={val_dist:.4f} ***")

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"best_val_dist": best_val_dist}, f, indent=2)


if __name__ == "__main__":
    main()
