"""v5 with MedCLIP Swin-T backbone (from V3 ckpt) @ 448 -> 14x14x768.

Same decoder head as v5 (causal transformer + 196-way spatial head + duration head).
This is the scanpath branch of "Option A" — the MedCLIP backbone is shared with V3's
classification branch to enable joint classification + scanpath modelling.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fixation_ar_jepa import RMSNorm, CausalTransformerBlock


class MedCLIPSwinAt448(nn.Module):
    """MedCLIP Swin-T at 448 input -> 14x14x768 patch map.

    Can optionally load V3 (v3a_finetune_100pct) backbone weights — which are
    MedCLIP-initialized, then AR-JEPA + CheXpert-SFT fine-tuned.
    """
    def __init__(self, v3_ckpt=None,
                 medclip_ckpt="/tmp/medclip_weights/medclip-vit-pretrained/pytorch_model.bin"):
        super().__init__()
        from transformers import SwinModel
        self.model = SwinModel.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224")
        if v3_ckpt is not None and os.path.exists(v3_ckpt):
            # v3a's best_ft_ckpt stores classifier wrapper weights under
            # "backbone.vit.model.<...>". Extract and map to plain SwinModel keys.
            ck = torch.load(v3_ckpt, map_location="cpu")
            state = ck["model"]
            swin_state = {k.replace("backbone.vit.model.", ""): v
                          for k, v in state.items()
                          if k.startswith("backbone.vit.model.")}
            missing, unexpected = self.model.load_state_dict(swin_state, strict=False)
            print(f"[MedCLIPSwinAt448] V3 load: missing={len(missing)} unexpected={len(unexpected)}")
        else:
            # Fall back to plain MedCLIP weights
            from medclip import MedCLIPModel, MedCLIPVisionModelViT
            full = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
            st = torch.load(medclip_ckpt, map_location="cpu")
            for k in list(st.keys()):
                if "position_ids" in k: del st[k]
            full.load_state_dict(st, strict=False)
            swin_state = {k.replace("vision_model.model.", ""): v
                          for k, v in full.state_dict().items()
                          if k.startswith("vision_model.model.")}
            self.model.load_state_dict(swin_state, strict=False)
            del full
            print("[MedCLIPSwinAt448] loaded plain MedCLIP weights")

        self.embed_dim = 768
        self.grid_size = 14
        self.num_patches = 196
        self.input_size = 448

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        return self.model(x).last_hidden_state   # (B, 196, 768)


class ScanpathPredictorV5MedCLIP(nn.Module):
    """Same decoder as v5, with MedCLIP@448 backbone (14x14x768)."""
    def __init__(
        self,
        embed_dim=512, context_depth=6, context_heads=8, head_dim=64,
        num_tasks=13, grid_size=14, max_steps=7, freeze_backbone=True,
        v3_ckpt=None,
    ):
        super().__init__()
        self.backbone = MedCLIPSwinAt448(v3_ckpt=v3_ckpt)
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
            patches = self.backbone(imgs)
        return self.img_proj(patches)

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
