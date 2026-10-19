#!/usr/bin/env python3
"""V4b linear probe on CheXpert / RSNA / SIIM at 1% / 10% / 100%.

Loads V4b Stage-1 backbone (frozen by default), extracts features via
backbone.get_features (concat of [readout, mean-pool patch] = 1536d),
then trains LogisticRegression for each label fraction.

Mirrors scripts/benchmark_chexworld.py exactly.
"""
import sys, os, json, argparse, logging, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import pydicom
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from models.fixation_ar_jepa_v4 import build_fixation_ar_jepa_v4
from datasets.chexpert_dataset import CheXpertDataset, LABEL_COLS
from datasets.transforms import get_chexpert_eval_transform

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)
DEVICE = torch.device("cuda")
CHEXPERT = "/home/myid/yl80817/datasets/chexlocalize/chexlocalize/CheXpert"


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
    m = m.to(DEVICE).eval()
    @torch.no_grad()
    def fn(x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        return m.get_features(x)  # (B, 1536)
    return fn


@torch.no_grad()
def extract(fn, loader):
    feats, labels = [], []
    for imgs, lbl in loader:
        imgs = imgs.to(DEVICE)
        with autocast():
            f = fn(imgs)
        feats.append(f.cpu().float().numpy()); labels.append(np.array(lbl))
    return np.concatenate(feats), np.concatenate(labels)


def probe_binary(tr_f, tr_l, te_f, te_l, frac, seed=42):
    n = int(len(tr_f) * frac)
    idx = np.random.RandomState(seed).choice(len(tr_f), n, replace=False)
    X, y = tr_f[idx], tr_l[idx]
    sc = StandardScaler(); X = sc.fit_transform(X); Xte = sc.transform(te_f)
    if y.sum() == 0 or y.sum() == len(y): return 0.0
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X, y)
    return float(roc_auc_score(te_l, clf.predict_proba(Xte)[:, 1]))


def probe_chexpert(tr_f, tr_l, te_f, te_l, frac, seed=42):
    n = int(len(tr_f) * frac)
    idx = np.random.RandomState(seed).choice(len(tr_f), n, replace=False)
    X, y = tr_f[idx], tr_l[idx]
    sc = StandardScaler(); X = sc.fit_transform(X); Xte = sc.transform(te_f)
    per = {}
    for i, name in enumerate(LABEL_COLS):
        yt, ye = y[:, i], te_l[:, i]
        if yt.sum() == 0 or ye.sum() == 0 or ye.sum() == len(ye): continue
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        try:
            clf.fit(X, yt)
            per[name] = float(roc_auc_score(ye, clf.predict_proba(Xte)[:, 1]))
        except Exception: pass
    return float(np.mean(list(per.values()))) if per else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_ckpt",
        default="/home/myid/yl80817/FixationARJEPA/outputs/v4b_frozen/stage1_ckpt.pth")
    ap.add_argument("--out", default="/home/myid/yl80817/FixationARJEPA/outputs/v4b_linear_probe.json")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42],
                     help="Probe subsample seeds; report mean±std over them.")
    args = ap.parse_args()

    fn = build_v4b(args.stage1_ckpt)

    eval_tf = get_chexpert_eval_transform(img_size=224)
    chex_tr = CheXpertDataset(f"{CHEXPERT}/train.csv", CHEXPERT, mode="finetune", transform=eval_tf)
    chex_te = CheXpertDataset(f"{CHEXPERT}/test_labels.csv", CHEXPERT, mode="finetune", transform=eval_tf)
    log.info(f"CheXpert tr={len(chex_tr)} te={len(chex_te)}")

    rsna_tr = RSNADataset("/home/myid/yl80817/datasets/RSNA/train.pkl",
                           "/home/myid/yl80817/datasets/RSNA/stage_2_train_images")
    rsna_te = RSNADataset("/home/myid/yl80817/datasets/RSNA/test.pkl",
                           "/home/myid/yl80817/datasets/RSNA/stage_2_train_images")
    siim_tr = SIIMDataset("/home/myid/yl80817/datasets/siim/train.csv",
                           "/home/myid/yl80817/datasets/siim/dicom-images-train")
    siim_te = SIIMDataset("/home/myid/yl80817/datasets/siim/test.csv",
                           "/home/myid/yl80817/datasets/siim/dicom-images-train")

    # extract features once per dataset, then probe across seeds
    log.info("[CheXpert] extracting...")
    c_tr_f, c_tr_l = extract(fn, DataLoader(chex_tr, batch_size=64, num_workers=4))
    c_te_f, c_te_l = extract(fn, DataLoader(chex_te, batch_size=64, num_workers=2))
    log.info("[RSNA] extracting...")
    r_tr_f, r_tr_l = extract(fn, DataLoader(rsna_tr, batch_size=64, num_workers=4))
    r_te_f, r_te_l = extract(fn, DataLoader(rsna_te, batch_size=64, num_workers=2))
    log.info("[SIIM] extracting...")
    s_tr_f, s_tr_l = extract(fn, DataLoader(siim_tr, batch_size=64, num_workers=4))
    s_te_f, s_te_l = extract(fn, DataLoader(siim_te, batch_size=64, num_workers=2))

    per_seed = {}
    for sd in args.seeds:
        row = {}
        for nm, fr in [("1%", 0.01), ("10%", 0.10), ("100%", 1.0)]:
            row[f"chex_{nm}"] = probe_chexpert(c_tr_f, c_tr_l, c_te_f, c_te_l, fr, seed=sd)
            row[f"rsna_{nm}"] = probe_binary(r_tr_f, r_tr_l, r_te_f, r_te_l, fr, seed=sd)
            row[f"siim_{nm}"] = probe_binary(s_tr_f, s_tr_l, s_te_f, s_te_l, fr, seed=sd)
        per_seed[str(sd)] = row
        log.info(f"  [seed={sd}] chex={row['chex_1%']:.4f}/{row['chex_10%']:.4f}/{row['chex_100%']:.4f} "
                 f"rsna={row['rsna_1%']:.4f}/{row['rsna_10%']:.4f}/{row['rsna_100%']:.4f} "
                 f"siim={row['siim_1%']:.4f}/{row['siim_10%']:.4f}/{row['siim_100%']:.4f}")

    summ = {}
    for k in next(iter(per_seed.values())):
        arr = np.array([per_seed[s][k] for s in per_seed])
        summ[k] = {"mean": float(arr.mean()), "std": float(arr.std())}
    with open(args.out, "w") as f: json.dump({"V4b": per_seed, "V4b_summary": summ}, f, indent=2)
    log.info(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
