#!/usr/bin/env python3
"""Full scanpath evaluation for v5medclip (MedCLIP@448 14x14 + real first fix)."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from sgj_models.scanpath_predictor_v5medclip import ScanpathPredictorV5MedCLIP
from data.gazesearch_dataset import GazeSearchDataset, collate_gazesearch
from evaluation.scanpath_metrics import evaluate_scanpath_metrics

DATA_JSON = "/home/myid/yl80817/GazeSearch/data/finding_visual_search_coco_format_train_test_filtered_max_6_split_train_valid_test_2024-07-22.json"
FALLBACK_IMG = "/home/myid/yl80817/MIMIC_EYE/ChongMa_Gaze_Mimic/images"


@torch.no_grad()
def generate_all(model, loader, device, num_steps=7):
    model.eval()
    preds, gts, masks = [], [], []
    for batch in loader:
        imgs = batch["images"].to(device)
        tids = batch["task_ids"].to(device)
        first_fix = batch["scanpaths"][:, 0, :].to(device)
        gt = batch["scanpaths"].numpy()
        msk = batch["masks"].numpy()
        pred, _ = model.generate(imgs, tids, first_fix, num_steps=num_steps)
        preds.append(pred.cpu().numpy()); gts.append(gt); masks.append(msk)
    return (np.concatenate(preds), np.concatenate(gts), np.concatenate(masks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/home/myid/yl80817/SemanticGazeJEPA/outputs/scanpath_v5medclip/best_ckpt.pth")
    parser.add_argument("--img_size", type=int, default=448)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = GazeSearchDataset(DATA_JSON, split="test", img_size=args.img_size,
                                 fallback_img_dir=FALLBACK_IMG, first_fix_real=True)
    print(f"Test samples: {len(test_ds)}")
    loader = DataLoader(test_ds, batch_size=16, num_workers=0,
                        collate_fn=collate_gazesearch)

    model = ScanpathPredictorV5MedCLIP(
        embed_dim=512, context_depth=6, context_heads=8, head_dim=64,
        num_tasks=13, grid_size=14, max_steps=7, freeze_backbone=True,
        v3_ckpt=None,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.backbone.eval()
    print(f"Loaded {args.ckpt} (epoch={ckpt.get('epoch','?')})")

    print("Generating scanpaths...")
    preds, gts, masks = generate_all(model, loader, device)
    print("Computing metrics...")
    metrics = evaluate_scanpath_metrics(preds, gts, masks, im_w=224, im_h=224)

    print("\n" + "="*60)
    print(f"  v5medclip Scanpath Metrics (Test set, {len(test_ds)} samples)")
    print("="*60)
    print(f"  ScanMatch     \u2191 : {metrics['scanmatch']:.4f}")
    print("  MultiMatch:")
    for k in ['mm_shape','mm_length','mm_direction','mm_position','mm_duration']:
        print(f"    {k[3:]:9s} \u2191 : {metrics[k]:.4f}")
    print(f"  SED           \u2193 : {metrics['sed']:.4f}")
    print(f"  STDE          \u2191 : {metrics['stde']:.4f}")

    out = os.path.join(os.path.dirname(args.ckpt), "metrics_full.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
