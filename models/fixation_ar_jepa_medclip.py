"""
Fixation-AR-JEPA with MedCLIP's pretrained Swin-T as backbone.

MedCLIP pretrained on MIMIC-CXR with image-text contrastive learning.
Internal structure: HuggingFace SwinModel (same as EGMA), different weights.

MedCLIP achieved the best linear probe scores on CheXpert in our benchmark:
  1%: 0.7465  10%: 0.8288  100%: 0.8558
(beating EGMA's 0.7121 / 0.7979 / 0.8545)
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .fixation_ar_jepa import (
    RMSNorm, CausalTransformerBlock, FixationEmbedder,
)


class MedCLIPSwinEncoder(nn.Module):
    """Wraps MedCLIP's SwinModel (HuggingFace), loads MedCLIP pretrained weights."""

    def __init__(self, medclip_ckpt='/tmp/medclip_weights/medclip-vit-pretrained/pytorch_model.bin',
                 img_size=224, random_init=False):
        super().__init__()
        from medclip import MedCLIPModel, MedCLIPVisionModelViT

        # Load full MedCLIP, extract just the inner Swin
        full_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        state = torch.load(medclip_ckpt, map_location='cpu')
        for k in list(state.keys()):
            if 'position_ids' in k:
                del state[k]
        full_model.load_state_dict(state, strict=False)

        # Extract just the inner SwinModel (HuggingFace style)
        self.model = full_model.vision_model.model
        del full_model  # free memory

        # Backbone-sensitivity: "trained from scratch" = SAME Swin-Tiny architecture
        # but re-initialised randomly (discards the MedCLIP pretrained weights), so
        # the gaze objective must learn the encoder end-to-end from scratch. This
        # isolates the contribution of the pretrained init exactly.
        if random_init:
            from transformers import SwinModel
            self.model = SwinModel(self.model.config)

        # Swin-Tiny downsamples the input by 32x. The grid is therefore
        # img_size // 32 (7 @ 224, 14 @ 448). Swin's windowed attention with
        # relative position bias is resolution-independent, so the same
        # pretrained weights run at any size divisible by 32.
        self.embed_dim = 768
        self.img_size = img_size
        self.grid_size = img_size // 32
        self.num_patches = self.grid_size ** 2

    def forward_features(self, x):
        """Returns (B, num_patches, 768)."""
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        out = self.model(x)
        return out.last_hidden_state  # (B, num_patches, 768)

    def parameters(self, recurse=True):
        return self.model.parameters(recurse)


class FixationARJEPA_MedCLIP(nn.Module):
    """Fixation-AR-JEPA with MedCLIP Swin-T backbone."""

    def __init__(
        self,
        embed_dim=768,
        context_depth=8,
        context_heads=12,
        head_dim=64,
        max_fixations=96,
        ema_momentum=0.998,
        medclip_ckpt='/tmp/medclip_weights/medclip-vit-pretrained/pytorch_model.bin',
        unfreeze_swin=True,
        use_infonce=False,
        nce_temperature=0.07,
    ):
        super().__init__()

        # ---- MedCLIP Swin-T backbone ----
        self.vit = MedCLIPSwinEncoder(medclip_ckpt)
        self.vit_dim = self.vit.embed_dim  # 768
        self.num_patches = self.vit.num_patches  # 49
        self.grid_size = self.vit.grid_size  # 7
        self.unfreeze_vit_top_n = 0  # for compatibility with train_v2

        if not unfreeze_swin:
            for p in self.vit.parameters():
                p.requires_grad = False

        # ---- Fixation embedder (7×7 grid) ----
        self.fix_embedder = FixationEmbedder(
            vit_dim=self.vit_dim, embed_dim=embed_dim,
            max_fixations=max_fixations, grid_size=self.grid_size,
        )

        # ---- Readout token ----
        self.readout_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.readout_token, std=0.02)

        # ---- Causal context encoder ----
        self.context_blocks = nn.ModuleList([
            CausalTransformerBlock(embed_dim, context_heads, head_dim)
            for _ in range(context_depth)
        ])
        self.context_norm = RMSNorm(embed_dim)

        # ---- Predictor ----
        self.predictor = nn.Sequential(
            RMSNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # ---- Eval projection ----
        self.eval_proj = nn.Sequential(
            RMSNorm(embed_dim),
            nn.Linear(embed_dim, self.vit_dim),
        )

        # ---- EMA target ----
        self.target_embedder = copy.deepcopy(self.fix_embedder)
        self.target_blocks = copy.deepcopy(self.context_blocks)
        self.target_norm = copy.deepcopy(self.context_norm)
        for p in list(self.target_embedder.parameters()) + \
                 list(self.target_blocks.parameters()) + \
                 list(self.target_norm.parameters()):
            p.requires_grad = False

        self.embed_dim = embed_dim
        self.ema_momentum = ema_momentum
        self.use_infonce = use_infonce
        self.nce_temperature = nce_temperature

    def _encode_image(self, imgs):
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        return self.vit.forward_features(imgs)  # (B, 49, 768)

    def _remap_patch_ids(self, fix_patch_ids, fix_coords):
        """Remap any patch IDs to 7x7 grid using normalized coords."""
        x_norm = fix_coords[:, :, 0].clamp(0, 0.999)
        y_norm = fix_coords[:, :, 1].clamp(0, 0.999)
        col = (x_norm * self.grid_size).long()
        row = (y_norm * self.grid_size).long()
        return (row * self.grid_size + col).clamp(0, self.num_patches - 1)

    def forward(self, imgs, fix_patch_ids, fix_coords, fix_meta, fix_mask):
        B, T = fix_patch_ids.shape
        patch_tokens = self._encode_image(imgs)
        fix_ids_7 = self._remap_patch_ids(fix_patch_ids, fix_coords)

        ctx_embeds = self.fix_embedder(
            patch_tokens, fix_ids_7, fix_coords, fix_meta, fix_mask
        )
        x = ctx_embeds
        for blk in self.context_blocks:
            x = blk(x)
        x = self.context_norm(x)

        preds = self.predictor(x[:, :-1, :])

        with torch.no_grad():
            tgt_embeds = self.target_embedder(
                patch_tokens.detach(), fix_ids_7, fix_coords, fix_meta, fix_mask
            )
            y = tgt_embeds
            for blk in self.target_blocks:
                y = blk(y)
            y = self.target_norm(y)
            targets = y[:, 1:, :]
            targets = F.layer_norm(targets, [targets.shape[-1]])

        valid = fix_mask[:, 1:].unsqueeze(-1).float()
        preds = preds * valid
        targets = targets * valid
        return preds, targets

    def compute_loss(self, preds, targets):
        preds_norm = F.layer_norm(preds, [preds.shape[-1]])
        l_pred = F.smooth_l1_loss(preds_norm, targets)
        result = {"loss": l_pred, "l_pred": l_pred.detach()}

        if self.use_infonce:
            pred_pool = F.normalize(preds_norm.mean(dim=1), dim=-1)
            tgt_pool = F.normalize(targets.mean(dim=1), dim=-1)
            logits = (pred_pool @ tgt_pool.T) / self.nce_temperature
            labels = torch.arange(logits.shape[0], device=logits.device)
            l_nce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            result["loss"] = l_pred + l_nce
            result["l_nce"] = l_nce.detach()
        return result

    @torch.no_grad()
    def update_target(self, momentum=None):
        m = momentum if momentum is not None else self.ema_momentum
        for pairs in [
            (self.fix_embedder.parameters(), self.target_embedder.parameters()),
            (self.context_blocks.parameters(), self.target_blocks.parameters()),
            (self.context_norm.parameters(), self.target_norm.parameters()),
        ]:
            for p_ctx, p_tgt in zip(*pairs):
                p_tgt.data.mul_(m).add_((1 - m) * p_ctx.data)

    def get_features(self, imgs):
        patch_tokens = self._encode_image(imgs)
        B = patch_tokens.shape[0]
        G = self.grid_size

        T_scan = self.num_patches
        fix_ids = torch.arange(T_scan, device=imgs.device).unsqueeze(0).expand(B, -1)
        fix_coords = torch.zeros(B, T_scan, 2, device=imgs.device)
        for i in range(T_scan):
            fix_coords[:, i, 0] = (i % G) / G
            fix_coords[:, i, 1] = (i // G) / G
        fix_meta = torch.zeros(B, T_scan, 3, device=imgs.device)
        fix_meta[:, :, 2] = torch.linspace(0, 1, T_scan, device=imgs.device)
        fix_mask = torch.ones(B, T_scan, dtype=torch.bool, device=imgs.device)

        ctx_embeds = self.fix_embedder(patch_tokens, fix_ids, fix_coords, fix_meta, fix_mask)
        readout = self.readout_token.expand(B, -1, -1)
        x = torch.cat([readout, ctx_embeds], dim=1)
        for blk in self.context_blocks:
            x = blk(x)
        x = self.context_norm(x)

        feat = self.eval_proj(x[:, 0, :])
        vit_feat = patch_tokens.mean(dim=1)
        return torch.cat([feat, vit_feat], dim=-1)


def build_fixation_ar_jepa_medclip(cfg):
    m = cfg["model"]
    return FixationARJEPA_MedCLIP(
        embed_dim=m.get("embed_dim", 768),
        context_depth=m.get("context_depth", 8),
        context_heads=m.get("context_heads", 12),
        head_dim=m.get("head_dim", 64),
        max_fixations=m.get("max_fixations", 96),
        ema_momentum=m.get("ema_momentum", 0.998),
        medclip_ckpt=m.get("medclip_ckpt", '/tmp/medclip_weights/medclip-vit-pretrained/pytorch_model.bin'),
        unfreeze_swin=m.get("unfreeze_swin", True),
        use_infonce=m.get("use_infonce", False),
        nce_temperature=m.get("nce_temperature", 0.07),
    )
