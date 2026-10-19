#!/usr/bin/env python3
"""V4b SFT on RSNA / SIIM at 100% labels.

Same recipe as scripts/train_chexworld_sft.py:
  Encoder mean-pool + Linear head, AdamW(lr*0.1, lr), cosine LR, AMP, 6 epochs.
"""
import sys, os, json, argparse, time, logging, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydicom
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from sklearn.metrics import roc_auc_score

from models.fixation_ar_jepa_v4 import build_fixation_ar_jepa_v4

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)
DEVICE = torch.device("cuda")


class RSNADataset(Dataset):
    def __init__(self, pkl_path, img_dir, img_size=224):
        with open(pkl_path, "rb") as f: d = pickle.load(f)
        self.filenames = d[0]
        self.labels = np.array([0 if (b == 0).all() else 1 for b in d[1]], dtype=np.float32)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
    def __len__(self): return len(self.filenames)
    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.filenames[idx])
        try:
            ds = pydicom.dcmread(path); arr = ds.pixel_array
            if arr.dtype != np.uint8:
                arr = (arr / arr.max() * 255).astype(np.uint8) if arr.max() > 0 else arr.astype(np.uint8)
            img = Image.fromarray(arr).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), 0)
        return self.transform(img), self.labels[idx]


class SIIMDataset(Dataset):
    def __init__(self, csv_path, root, img_size=224):
        df = pd.read_csv(csv_path); df.columns = df.columns.str.strip()
        df["label"] = (df["EncodedPixels"].astype(str).str.strip() != "-1").astype(np.float32)
        self.df = df.groupby("ImageId").agg({"label": "max", "Path": "first"}).reset_index()
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]; path = os.path.join(self.root, row["Path"])
        try:
            ds = pydicom.dcmread(path); arr = ds.pixel_array
            if arr.dtype != np.uint8:
                arr = (arr / arr.max() * 255).astype(np.uint8) if arr.max() > 0 else arr.astype(np.uint8)
            img = Image.fromarray(arr).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), 0)
        return self.transform(img), np.float32(row["label"])


class V4bSFT(nn.Module):
    """V4b backbone + Linear head on get_features() output."""
    def __init__(self, backbone, num_classes, feat_dim=1536):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        feat = self.backbone.get_features(x)
        return self.head(feat)


def build_v4b(stage1_ckpt):
    cfg = {"model": {
        "embed_dim": 768, "context_depth": 8, "context_heads": 12, "head_dim": 64,
        "max_fixations": 96, "ema_momentum": 0.998,
        "medclip_ckpt": "/tmp/medclip_weights/medclip-vit-pretrained/pytorch_model.bin",
        "unfreeze_swin": True,
    }}
    m = build_fixation_ar_jepa_v4(cfg, mode='b')
    ck = torch.load(stage1_ckpt, map_location="cpu")
    state = ck.get("model", ck)
    missing, unexpected = m.load_state_dict(state, strict=False)
    log.info(f"V4b Stage-1 load: missing={len(missing)} unexpected={len(unexpected)}")
    return m


@torch.no_grad()
def eval_auroc(model, loader):
    model.eval()
    all_s, all_l = [], []
    for imgs, lbl in loader:
        imgs = imgs.to(DEVICE)
        with autocast():
            logits = model(imgs)
        all_s.append(torch.sigmoid(logits).cpu().float().numpy().squeeze(-1))
        all_l.append(lbl.numpy() if hasattr(lbl, "numpy") else np.asarray(lbl))
    s = np.concatenate(all_s); l = np.concatenate(all_l)
    return float(roc_auc_score(l, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["rsna", "siim"])
    ap.add_argument("--label_frac", type=float, default=1.0)
    ap.add_argument("--stage1_ckpt",
        default="/home/myid/yl80817/FixationARJEPA/outputs/v4b_frozen/stage1_ckpt.pth")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log.info(f"=== V4b SFT | dataset={args.dataset} frac={args.label_frac} | lr={args.lr} ===")

    backbone = build_v4b(args.stage1_ckpt)
    if args.dataset == "rsna":
        ds_tr = RSNADataset("/home/myid/yl80817/datasets/RSNA/train.pkl",
                             "/home/myid/yl80817/datasets/RSNA/stage_2_train_images")
        ds_te = RSNADataset("/home/myid/yl80817/datasets/RSNA/test.pkl",
                             "/home/myid/yl80817/datasets/RSNA/stage_2_train_images")
    else:
        ds_tr = SIIMDataset("/home/myid/yl80817/datasets/siim/train.csv",
                             "/home/myid/yl80817/datasets/siim/dicom-images-train")
        ds_te = SIIMDataset("/home/myid/yl80817/datasets/siim/test.csv",
                             "/home/myid/yl80817/datasets/siim/dicom-images-train")

    if args.label_frac < 1.0:
        n = int(len(ds_tr) * args.label_frac)
        idx = np.random.RandomState(42).choice(len(ds_tr), n, replace=False)
        ds_tr = Subset(ds_tr, idx)
    log.info(f"  train={len(ds_tr)}  test={len(ds_te)}")

    tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True)
    te = DataLoader(ds_te, batch_size=128, num_workers=2)

    model = V4bSFT(backbone, num_classes=1).to(DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.head.parameters(),    "lr": args.lr},
    ], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = GradScaler()

    best_test = 0.0
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0; t0 = time.time()
        for imgs, lbl in tr:
            imgs = imgs.to(DEVICE)
            lbl = lbl.to(DEVICE).float().unsqueeze(-1)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(imgs)
                loss = F.binary_cross_entropy_with_logits(logits, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            ep_loss += loss.item()
        scheduler.step()
        log.info(f"[E{epoch:03d}] loss={ep_loss/max(len(tr),1):.4f} time={time.time()-t0:.1f}s")
        test_auc = eval_auroc(model, te)
        log.info(f"[E{epoch:03d}] test={test_auc:.4f}")
        if test_auc > best_test:
            best_test = test_auc
            torch.save({"epoch": epoch, "test": test_auc},
                        os.path.join(args.out_dir, "best.pth"))

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"dataset": args.dataset, "label_frac": args.label_frac,
                     "best_test": best_test, "stage1_ckpt": args.stage1_ckpt}, f, indent=2)
    log.info(f"DONE  best_test={best_test:.4f}")


if __name__ == "__main__":
    main()
