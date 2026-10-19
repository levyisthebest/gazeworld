"""
v5 ResNet variant: ResNet50 (Mask R-CNN COCO-pretrained backbone) @ 448 -> 14x14x2048.
Matches LogitGaze-Med "Res" configuration.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

from models.fixation_ar_jepa import RMSNorm, CausalTransformerBlock


class ResNet50CocoBackbone(nn.Module):
    """ResNet50 from Mask R-CNN (COCO pretrained) -> 14x14x2048 at 448 input."""
    def __init__(self):
        super().__init__()
        mr = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
        # body is IntermediateLayerGetter returning stage feature maps
        self.body = mr.backbone.body
        # Use layer4 output (stride 32) for 14x14 at 448 input
        self.embed_dim = 2048
        self.grid_size = 14
        self.num_patches = 14 * 14
        self.input_size = 448
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # Dataset gives us 3-channel images normalized with mean=0.5, std=0.5 (i.e. in [-1,1]).
        # Convert back to [0,1] then re-normalize to ImageNet stats (what ResNet expects).
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode='bilinear', align_corners=False)
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        out = self.body(x)  # dict with keys '0','1','2','3' (stage1-4)
        feat = out["3"]     # (B, 2048, 14, 14) at 448 input
        return feat.flatten(2).transpose(1, 2).contiguous()  # (B, 196, 2048)


class ScanpathPredictorV5Res(nn.Module):
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
        self.backbone = ResNet50CocoBackbone()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.max_steps = max_steps
        self.embed_dim = embed_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.task_embed = nn.Embedding(num_tasks + 1, embed_dim)
        self.fix_proj = nn.Sequential(
            nn.Linear(2, embed_dim // 2), nn.GELU(),
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
            patches = self.backbone(imgs)  # (B, 196, 2048)
        return self.img_proj(patches)      # (B, 196, D)

    def forward(self, imgs, task_ids, scanpaths, masks):
        B, T = scanpaths.shape[:2]
        img_feats = self.encode_image(imgs)
        task_tok = self.task_embed(task_ids).unsqueeze(1)
        fix_embeds = self.fix_proj(scanpaths)
        x = torch.cat([task_tok, fix_embeds[:, :-1, :]], dim=1)
        x = x + self.temporal_pe[:, :T, :]
        x_cross, _ = self.cross_attn(self.cross_norm(x), img_feats, img_feats)
        x = x + x_cross
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        h = x[:, 1:, :]
        return self.spatial_head(h), self.duration_head(h), self.term_head(h)

    @torch.no_grad()
    def generate(self, imgs, task_ids, first_fix, num_steps=7):
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
