#!/usr/bin/env python3
"""V4b Stage-1: AR-JEPA + Spatial-Completion on MIMIC-EYE radiologist gaze.

Recipe (paper):
  Backbone: MedCLIP Swin-ViT (frozen during Stage-1)
  Loss: smooth_L1(LN(ẑ_{t+1}), z_{t+1})  +  λ · smooth_L1(LN(r̂_p), LN(ema_patch_p))
        for p ∈ {unvisited patches}, λ = 1.0
  Optim: AdamW(lr=3e-4, wd=0.04), cosine EMA momentum 0.998 → 1.0
  15 epochs, bs=32, AMP

Default --freeze_swin is ON (the paper setting). Pass --unfreeze_swin to
reproduce the unfrozen ablation that exhibits target-collapse.
"""
import sys, os, argparse, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from models.fixation_ar_jepa_v4b import build_v4b
from datasets.cached_fixation_dataset import CachedFixationDataset, collate_cached

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)
DEVICE = torch.device("cuda")


def cosine_ema(base, final, total):
    return np.array([final + 0.5 * (base - final) * (1 + np.cos(np.pi * i / total))
                     for i in range(total)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--completion_weight", type=float, default=1.0,
                     help="Weight on spatial-completion loss; 0 = pure AR ablation")
    ap.add_argument("--ar_weight", type=float, default=1.0,
                     help="Weight on gaze AR loss; 0 = pure spatial-completion ablation")
    ap.add_argument("--unfreeze_swin", action="store_true")
    ap.add_argument("--random_init", action="store_true",
                     help="Random-init the Swin backbone (train from scratch) instead "
                          "of loading MedCLIP weights. Use with --unfreeze_swin.")
    ap.add_argument("--img_size", type=int, default=224,
                     help="Input resolution. 224 (paper) -> 7x7 grid, 448 -> 14x14.")
    ap.add_argument("--cache",
        default="/home/myid/yl80817/FixationARJEPA/outputs/fixation_cache.npz")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log.info(f"=== V4b Stage-1 | epochs={args.epochs} bs={args.batch_size} | "
             f"λ_ar={args.ar_weight} λ_sp={args.completion_weight} | "
             f"freeze_swin={not args.unfreeze_swin} ===")

    cfg = {"model": {"unfreeze_swin": args.unfreeze_swin,
                      "completion_weight": args.completion_weight,
                      "img_size": args.img_size,
                      "random_init": args.random_init}}
    model = build_v4b(cfg).to(DEVICE)

    train_ds = CachedFixationDataset(args.cache, img_size=args.img_size, split="train")
    log.info(f"MIMIC-EYE train: {len(train_ds)}")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True,
                        collate_fn=collate_cached)

    trainable = [p for p in model.parameters() if p.requires_grad]
    log.info(f"Trainable: {sum(p.numel() for p in trainable)/1e6:.1f}M")
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler()

    steps_per_epoch = len(loader)
    ema_sched = cosine_ema(0.998, 1.0, args.epochs * steps_per_epoch)

    log_f = open(os.path.join(args.out_dir, "train.log"), "w")
    for epoch in range(args.epochs):
        model.train()
        ep_loss = ep_lar = ep_lsp = 0.0
        t0 = time.time()
        for step, batch in enumerate(loader):
            gs = epoch * steps_per_epoch + step
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            with autocast():
                out = model(batch["images"], batch["fix_patch_ids"],
                             batch["fix_coords"], batch["fix_meta"], batch["fix_mask"])
                # Recompute losses here so we can re-weight them per ablation
                ar = F.layer_norm(out['ar_preds'], [out['ar_preds'].shape[-1]])
                l_ar = F.smooth_l1_loss(ar, out['ar_targets'])
                sp = F.layer_norm(out['spat_preds'], [out['spat_preds'].shape[-1]])
                l_sp = F.smooth_l1_loss(sp, out['spat_targets'])
                weighted_loss = args.ar_weight * l_ar + args.completion_weight * l_sp
                losses = {"loss": weighted_loss, "l_ar": l_ar.detach(), "l_sp": l_sp.detach()}
            scaler.scale(weighted_loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optim); scaler.update()
            if gs < len(ema_sched):
                model.update_target(ema_sched[gs])
            ep_loss += losses["loss"].item()
            ep_lar += losses["l_ar"].item()
            ep_lsp += losses["l_sp"].item()
        msg = (f"[E{epoch:03d}] loss={ep_loss/steps_per_epoch:.4f} "
               f"l_ar={ep_lar/steps_per_epoch:.4f} "
               f"l_sp={ep_lsp/steps_per_epoch:.4f} "
               f"time={time.time()-t0:.1f}s")
        log.info(msg); log_f.write(msg + "\n"); log_f.flush()

    log_f.close()
    torch.save({"model": model.state_dict(),
                 "epochs": args.epochs,
                 "completion_weight": args.completion_weight,
                 "frozen_swin": not args.unfreeze_swin},
                os.path.join(args.out_dir, "stage1_ckpt.pth"))
    log.info(f"Saved -> {args.out_dir}/stage1_ckpt.pth")


if __name__ == "__main__":
    main()
