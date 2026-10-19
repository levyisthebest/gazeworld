#!/usr/bin/env python3
"""V4b Stage-2: full fine-tune on CheXpert 14-class classification.

Recipe: head_lr=1e-4, backbone_lr=1e-5, AdamW(wd=1e-4), cosine LR, AMP, 20 ep.
Eval every 2 epochs on val + test; checkpoint best-val.

Reproduces the headline V4b CheXpert result reported in the paper
(0.8761 mean AUROC on the 668-image official test split).
"""
import sys, os, json, argparse, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

from models.fixation_ar_jepa_v4b import build_v4b
from datasets.chexpert_dataset import CheXpertDataset, LABEL_COLS
from datasets.transforms import get_chexpert_eval_transform

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)
DEVICE = torch.device("cuda")
CHEXPERT = "/home/myid/yl80817/datasets/chexlocalize/chexlocalize/CheXpert"


class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes=14, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x): return self.net(x)


class V4bClassifier(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    def forward(self, x):
        return self.head(self.backbone.get_features(x))


@torch.no_grad()
def evaluate_auroc(model, loader):
    model.eval()
    s, l = [], []
    for imgs, lbl in loader:
        imgs = imgs.to(DEVICE)
        with autocast():
            logits = model(imgs)
        s.append(torch.sigmoid(logits).cpu().float().numpy())
        l.append(lbl.numpy())
    scores, labels = np.concatenate(s), np.concatenate(l)
    per = {}
    for i, name in enumerate(LABEL_COLS):
        y = labels[:, i]
        if 0 < y.sum() < len(y):
            try: per[name] = float(roc_auc_score(y, scores[:, i]))
            except Exception: pass
    return float(np.mean(list(per.values()))) if per else 0.0, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--label_frac", type=float, default=1.0)
    ap.add_argument("--img_size", type=int, default=224,
                     help="Must match the resolution used in Stage-1.")
    ap.add_argument("--seed", type=int, default=42,
                     help="Controls head init, data shuffling, label subsample "
                          "(for multi-seed error bars).")
    args = ap.parse_args()

    # Full seeding for reproducible per-seed runs (head init + DataLoader shuffle
    # + label-fraction subsample all derive from this).
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    log.info(f"=== V4b Stage-2 (CheXpert SFT) | ckpt={args.stage1_ckpt} | seed={args.seed} ===")

    ck = torch.load(args.stage1_ckpt, map_location="cpu")
    state = ck.get("model", ck)
    # Match the Stage-1 checkpoint's temporal_pe length (some ckpts use max_fixations=145).
    max_fix = state["fix_embedder.temporal_pe"].shape[1] if "fix_embedder.temporal_pe" in state else 96
    backbone = build_v4b({"model": {"unfreeze_swin": True, "img_size": args.img_size,
                                     "max_fixations": max_fix}})
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    log.info(f"Stage-1 load: missing={len(missing)} unexpected={len(unexpected)}")
    backbone = backbone.to(DEVICE)

    feat_dim = backbone.get_features(
        torch.randn(1, 3, args.img_size, args.img_size, device=DEVICE)).shape[-1]
    log.info(f"Feature dim: {feat_dim}")
    head = ClassificationHead(feat_dim, num_classes=14, hidden_dim=512)
    model = V4bClassifier(backbone, head).to(DEVICE)

    eval_tf = get_chexpert_eval_transform(img_size=args.img_size)
    chex_tr = CheXpertDataset(f"{CHEXPERT}/train.csv", CHEXPERT,
                                mode="finetune", transform=eval_tf)
    if args.label_frac < 1.0:
        n = int(len(chex_tr) * args.label_frac)
        idx = np.random.RandomState(args.seed).choice(len(chex_tr), n, replace=False)
        chex_tr = Subset(chex_tr, idx)
    chex_va = CheXpertDataset(f"{CHEXPERT}/val_labels.csv", CHEXPERT,
                                mode="finetune", transform=eval_tf)
    chex_te = CheXpertDataset(f"{CHEXPERT}/test_labels.csv", CHEXPERT,
                                mode="finetune", transform=eval_tf)
    log.info(f"CheXpert tr={len(chex_tr)} va={len(chex_va)} te={len(chex_te)}")

    tr = DataLoader(chex_tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)
    va = DataLoader(chex_va, batch_size=128, num_workers=2)
    te = DataLoader(chex_te, batch_size=128, num_workers=2)

    optim = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.head.parameters(),     "lr": args.lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-7)
    scaler = GradScaler()

    best_va, best_te = 0.0, 0.0
    log_f = open(os.path.join(args.out_dir, "stage2.log"), "w")
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        t0 = time.time()
        for imgs, lbl in tr:
            imgs, lbl = imgs.to(DEVICE), lbl.to(DEVICE)
            optim.zero_grad(set_to_none=True)
            with autocast():
                logits = model(imgs)
                loss = F.binary_cross_entropy_with_logits(logits, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()
            ep_loss += loss.item()
        sched.step()
        msg = f"[E{ep:03d}] loss={ep_loss/max(len(tr),1):.4f} time={time.time()-t0:.1f}s"
        log.info(msg); log_f.write(msg + "\n"); log_f.flush()

        if (ep + 1) % 2 == 0 or ep == args.epochs - 1:
            va_a, _ = evaluate_auroc(model, va)
            te_a, te_per = evaluate_auroc(model, te)
            msg = f"[E{ep:03d}] val_AUROC={va_a:.4f} test_AUROC={te_a:.4f}"
            log.info(msg); log_f.write(msg + "\n"); log_f.flush()
            if va_a > best_va:
                best_va, best_te = va_a, te_a
                torch.save({"epoch": ep, "val": va_a, "test": te_a, "per": te_per,
                             "model": model.state_dict()},
                            os.path.join(args.out_dir, "best_ft_ckpt.pth"))
                log_f.write(f"  *** new best: val={va_a:.4f} test={te_a:.4f} ***\n"); log_f.flush()

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"stage1_ckpt": args.stage1_ckpt,
                     "best_val": best_va, "best_test": best_te}, f, indent=2)
    log_f.close()
    log.info(f"DONE  best_val={best_va:.4f}  best_test={best_te:.4f}")


if __name__ == "__main__":
    main()
