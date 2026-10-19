"""
v5 scanpath predictor: CheXNet (DenseNet-121 on ChestX-ray14) backbone, 14x14 grid,
real-first-fixation conditioning.

Changes vs v1-v4 (MedCLIP 7x7, center prepend):
  - CheXNet backbone @ 448 -> 14x14x1024 patch map (196 patches)
  - Dataset no longer prepends center; fix_0 is the real human first fixation
  - spatial_head: 196-way classification
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fixation_ar_jepa import RMSNorm, CausalTransformerBlock


class CheXNetBackbone(nn.Module):
    """CheXNet (DenseNet-121 / NIH ChestX-ray14) -> 14x14x1024 at 448 input."""
    def __init__(self, weights="densenet121-res224-nih"):
        super().__init__()
        import torchxrayvision as xrv
        model = xrv.models.DenseNet(weights=weights)
        self.features = model.features
        self.embed_dim = 1024
        self.grid_size = 14
        self.num_patches = 14 * 14
        self.input_size = 448

    def forward(self, x):
        # xrv DenseNet expects 1-channel input in roughly [-1024, 1024]. Our images
        # are 3-channel ImageNet-normalized in [-1, 1]. Collapse channels + rescale.
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode='bilinear', align_corners=False)
        x = x * 1024.0  # approx rescale to xrv-expected range
        feat = self.features(x)  # (B, 1024, 14, 14)
        return feat.flatten(2).transpose(1, 2)  # (B, 196, 1024)


class ScanpathPredictorV5(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        context_depth=6,
        context_heads=8,
        head_dim=64,
        num_tasks=13,
        grid_size=14,
        max_steps=7,
        freeze_backbone=True,
    ):
        super().__init__()

        self.backbone = CheXNetBackbone()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.max_steps = max_steps
        self.embed_dim = embed_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.task_embed = nn.Embedding(num_tasks + 1, embed_dim)

        self.fix_proj = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        self.img_proj = nn.Linear(self.backbone.embed_dim, embed_dim)

        self.temporal_pe = nn.Parameter(torch.zeros(1, max_steps + 1, embed_dim))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

        self.cross_attn = nn.MultiheadAttention(embed_dim, context_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)

        self.blocks = nn.ModuleList([
            CausalTransformerBlock(embed_dim, context_heads, head_dim)
            for _ in range(context_depth)
        ])
        self.norm = RMSNorm(embed_dim)

        self.spatial_head = nn.Linear(embed_dim, self.num_patches)
        self.duration_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
        )
        self.term_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
        )

    def encode_image(self, imgs):
        with torch.no_grad():
            patches = self.backbone(imgs)  # (B, 196, 1024)
        return self.img_proj(patches)      # (B, 196, D)

    def forward(self, imgs, task_ids, scanpaths, masks):
        """
        scanpaths: (B, T, 2) normalized coords, scanpaths[:, 0, :] is the real first fix (given).
        Predicts steps 1..T-1 from inputs 0..T-2.
        """
        B, T = scanpaths.shape[:2]
        img_feats = self.encode_image(imgs)

        task_tok = self.task_embed(task_ids).unsqueeze(1)  # (B, 1, D)
        fix_embeds = self.fix_proj(scanpaths)               # (B, T, D)

        x = torch.cat([task_tok, fix_embeds[:, :-1, :]], dim=1)  # (B, T, D)
        x = x + self.temporal_pe[:, :T, :]

        x_cross, _ = self.cross_attn(self.cross_norm(x), img_feats, img_feats)
        x = x + x_cross

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        h = x[:, 1:, :]  # positions corresponding to predictions of fix_1..fix_{T-1}

        return self.spatial_head(h), self.duration_head(h), self.term_head(h)

    @torch.no_grad()
    def generate(self, imgs, task_ids, first_fix, num_steps=7):
        """
        first_fix: (B, 2) real human first fixation (normalized).
        Predicts num_steps - 1 subsequent fixations autoregressively.
        """
        B = imgs.shape[0]
        device = imgs.device
        img_feats = self.encode_image(imgs)

        scanpath = torch.zeros(B, num_steps, 2, device=device)
        scanpath[:, 0, :] = first_fix
        durations = torch.zeros(B, num_steps, device=device)

        for t in range(1, num_steps):
            task_tok = self.task_embed(task_ids).unsqueeze(1)
            fix_embeds = self.fix_proj(scanpath[:, :t, :])
            x = torch.cat([task_tok, fix_embeds], dim=1)
            x = x + self.temporal_pe[:, :t + 1, :]

            x_cross, _ = self.cross_attn(self.cross_norm(x), img_feats, img_feats)
            x = x + x_cross
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

            h = x[:, -1, :]
            logits = self.spatial_head(h)
            patch_idx = logits.argmax(dim=-1)

            row = patch_idx // self.grid_size
            col = patch_idx % self.grid_size
            scanpath[:, t, 0] = (col.float() + 0.5) / self.grid_size
            scanpath[:, t, 1] = (row.float() + 0.5) / self.grid_size
            durations[:, t] = self.duration_head(h).squeeze(-1).clamp(0, 5)

        return scanpath, durations


def scanpath_loss_v5(spatial_logits, dur_preds, term_preds,
                     gt_scanpaths, gt_durations, masks, grid_size=14):
    B = spatial_logits.shape[0]
    num_patches = grid_size * grid_size

    gt_coords = gt_scanpaths[:, 1:, :]
    gt_col = (gt_coords[:, :, 0] * grid_size).long().clamp(0, grid_size - 1)
    gt_row = (gt_coords[:, :, 1] * grid_size).long().clamp(0, grid_size - 1)
    gt_patch = gt_row * grid_size + gt_col  # (B, T-1)

    valid = masks[:, 1:].float()

    l_spatial = F.cross_entropy(
        spatial_logits.reshape(-1, num_patches), gt_patch.reshape(-1), reduction='none'
    ).reshape(B, -1)
    l_spatial = (l_spatial * valid).sum() / valid.sum().clamp(min=1)

    gt_dur = gt_durations[:, 1:].unsqueeze(-1)
    l_dur = (F.l1_loss(dur_preds, gt_dur, reduction='none').squeeze(-1) * valid).sum() / valid.sum().clamp(min=1)

    term_gt = torch.zeros_like(term_preds)
    l_term = F.binary_cross_entropy_with_logits(
        term_preds.squeeze(-1), term_gt.squeeze(-1), reduction='none'
    )
    l_term = (l_term * valid).sum() / valid.sum().clamp(min=1)

    total = l_spatial + 0.1 * l_dur + 0.1 * l_term
    return {"loss": total, "l_spatial": l_spatial.detach(),
            "l_dur": l_dur.detach(), "l_term": l_term.detach()}
