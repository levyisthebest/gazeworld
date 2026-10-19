"""
Fixation-AR-JEPA: Autoregressive JEPA on Radiologist Fixation Sequences.

Inspired by AIM (ICML 2024) but replaces raster-scan ordering with
radiologist eye-tracking order — capturing diagnostic reasoning as
sequential predictions in latent space.

Architecture:
  CXR → Frozen ViT-L (ImageNet pretrained, 1024d) → 196 patch tokens
  Fixation CSV → downsample to N keypoints → map to nearest patches
  Each fixation → extract ViT patch embedding at that location

  Causal Context Encoder (8-layer Llama-style Transformer):
    [fix₁, fix₂, ..., fixₜ] → hidden states h₁..hₜ   (causal mask)

  Predictor Head:
    hₜ → ẑ_{t+1}  (predict next fixation's embedding)

  Target Encoder (EMA of Context Encoder):
    fix_{t+1} → z_{t+1}  (prediction target)

  Loss = Σ_t SmoothL1(ẑ_{t+1}, z_{t+1})   — pure JEPA, no contrastive

Inference: use context encoder's CLS/pooled features for classification.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from functools import partial


# ---------------------------------------------------------------------------
# Llama-style building blocks (RMSNorm + RoPE + SwiGLU + GQA)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    c = cos[:x.shape[-2], :].unsqueeze(0).unsqueeze(0)
    s = sin[:x.shape[-2], :].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)  # round to 256
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal mask and RoPE."""

    def __init__(self, dim, num_heads=16, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.q = nn.Linear(dim, inner, bias=False)
        self.k = nn.Linear(dim, inner, bias=False)
        self.v = nn.Linear(dim, inner, bias=False)
        self.o = nn.Linear(inner, dim, bias=False)
        self.rope = RotaryEmbedding(head_dim)
        self.scale = head_dim ** -0.5

    def forward(self, x):
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(N, device=x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Causal mask: each position can only attend to previous positions
        causal = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        attn.masked_fill_(causal, float('-inf'))
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.o(out)


class CausalTransformerBlock(nn.Module):
    """Llama-style: RMSNorm → Causal Self-Attn → RMSNorm → SwiGLU"""

    def __init__(self, dim, num_heads=16, head_dim=64):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, head_dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Fixation Embedding: map (x,y,t) to patch index → ViT patch embedding
# ---------------------------------------------------------------------------

class FixationEmbedder(nn.Module):
    """
    Convert raw fixation coordinates to embeddings by:
    1. Map (x, y) to nearest ViT patch index
    2. Extract that patch's embedding from frozen ViT
    3. Add spatial PE (from normalized x,y) + temporal PE
    4. Project through learnable layer
    """

    def __init__(self, vit_dim=1024, embed_dim=768, max_fixations=128, grid_size=14):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        # Project ViT patch dim to our working dim
        self.patch_proj = nn.Linear(vit_dim, embed_dim)

        # Spatial PE: MLP on normalized (x, y)
        self.spatial_pe = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Temporal PE: learnable
        self.temporal_pe = nn.Parameter(torch.zeros(1, max_fixations, embed_dim))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

        # Fixation metadata embedding (duration, saccade amplitude)
        self.meta_proj = nn.Sequential(
            nn.Linear(3, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim),
        )

    def forward(self, patch_tokens, fix_patch_ids, fix_coords, fix_meta, fix_mask):
        """
        Args:
            patch_tokens:  (B, 196, D_vit) — frozen ViT patch embeddings
            fix_patch_ids: (B, T) long — patch index (0-195) for each fixation
            fix_coords:    (B, T, 2) float — normalized (x, y) in [0, 1]
            fix_meta:      (B, T, 3) float — (duration, saccade_amp, time_offset)
            fix_mask:      (B, T) bool — True for valid fixations

        Returns:
            (B, T, D) — fixation embeddings
        """
        B, T = fix_patch_ids.shape
        D_vit = patch_tokens.shape[-1]

        # Gather ViT patch embeddings at fixation positions
        idx = fix_patch_ids.clamp(0, patch_tokens.shape[1] - 1)
        idx = idx.unsqueeze(-1).expand(-1, -1, D_vit)
        fix_patches = torch.gather(patch_tokens, 1, idx)  # (B, T, D_vit)

        # Project to working dim
        x = self.patch_proj(fix_patches)  # (B, T, D)

        # Add spatial PE
        x = x + self.spatial_pe(fix_coords)

        # Add temporal PE
        x = x + self.temporal_pe[:, :T, :]

        # Add metadata
        x = x + self.meta_proj(fix_meta)

        # Zero out padding
        x = x * fix_mask.unsqueeze(-1).float()

        return x


# ---------------------------------------------------------------------------
# Full Fixation-AR-JEPA Model
# ---------------------------------------------------------------------------

class FixationARJEPA(nn.Module):
    """
    Autoregressive JEPA on radiologist fixation sequences.

    Training:
      fix₁..fixₜ → context encoder (causal) → predict fix_{t+1} embedding
      target: EMA encoder(fix_{t+1})

    Inference (no gaze needed):
      Image → ViT → CLS + patch_mean → downstream classifier
    """

    def __init__(
        self,
        vit_model="vit_large_patch16_224",
        vit_pretrained=True,
        embed_dim=768,
        context_depth=8,
        context_heads=12,
        head_dim=64,
        max_fixations=128,
        ema_momentum=0.996,
    ):
        super().__init__()

        # ---- Frozen ViT image encoder ----
        self.vit = timm.create_model(vit_model, pretrained=vit_pretrained, num_classes=0)
        self.vit_dim = self.vit.embed_dim  # 1024 for ViT-L
        self.num_patches = self.vit.patch_embed.num_patches  # 196
        self.grid_size = int(self.num_patches ** 0.5)  # 14
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        # ---- Fixation embedder ----
        self.fix_embedder = FixationEmbedder(
            vit_dim=self.vit_dim,
            embed_dim=embed_dim,
            max_fixations=max_fixations,
            grid_size=self.grid_size,
        )

        # ---- Context encoder: causal Transformer (trainable) ----
        self.context_blocks = nn.ModuleList([
            CausalTransformerBlock(embed_dim, context_heads, head_dim)
            for _ in range(context_depth)
        ])
        self.context_norm = RMSNorm(embed_dim)

        # ---- Predictor head: predict next fixation embedding ----
        self.predictor = nn.Sequential(
            RMSNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # ---- Target encoder (EMA of fix_embedder + context_blocks) ----
        self.target_embedder = copy.deepcopy(self.fix_embedder)
        self.target_blocks = copy.deepcopy(self.context_blocks)
        self.target_norm = copy.deepcopy(self.context_norm)
        for p in list(self.target_embedder.parameters()) + \
                 list(self.target_blocks.parameters()) + \
                 list(self.target_norm.parameters()):
            p.requires_grad = False

        self.embed_dim = embed_dim
        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def _encode_image(self, imgs):
        """Frozen ViT → patch tokens."""
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        features = self.vit.forward_features(imgs)  # (B, N+1, D) or (B, N, D)
        if features.shape[1] == self.num_patches + 1:
            return features[:, 1:, :]  # strip CLS → (B, 196, D_vit)
        return features

    def forward(self, imgs, fix_patch_ids, fix_coords, fix_meta, fix_mask):
        """
        Training forward: autoregressive next-fixation prediction.

        Args:
            imgs:          (B, 1, 224, 224)
            fix_patch_ids: (B, T) — patch index per fixation
            fix_coords:    (B, T, 2) — normalized (x,y)
            fix_meta:      (B, T, 3) — (duration, saccade_amp, time_offset)
            fix_mask:      (B, T) — valid mask

        Returns:
            preds:   (B, T-1, D) — predicted embeddings for fix₂..fixₜ
            targets: (B, T-1, D) — EMA target embeddings for fix₂..fixₜ
        """
        B, T = fix_patch_ids.shape

        # 1. Extract ViT patch tokens (frozen)
        patch_tokens = self._encode_image(imgs)  # (B, 196, D_vit)

        # 2. Context path: embed fixations → causal transformer → predictor
        ctx_embeds = self.fix_embedder(
            patch_tokens, fix_patch_ids, fix_coords, fix_meta, fix_mask
        )  # (B, T, D)

        x = ctx_embeds
        for blk in self.context_blocks:
            x = blk(x)
        x = self.context_norm(x)  # (B, T, D)

        # Predict next fixation: hₜ → ẑ_{t+1}
        preds = self.predictor(x[:, :-1, :])  # (B, T-1, D) — predict from fix₁..T-1

        # 3. Target path: EMA embedder + EMA transformer (no causal mask for targets)
        with torch.no_grad():
            tgt_embeds = self.target_embedder(
                patch_tokens, fix_patch_ids, fix_coords, fix_meta, fix_mask
            )
            y = tgt_embeds
            for blk in self.target_blocks:
                y = blk(y)
            y = self.target_norm(y)
            targets = y[:, 1:, :]  # (B, T-1, D) — targets are fix₂..T
            targets = F.layer_norm(targets, [targets.shape[-1]])

        # Mask out padding positions
        valid = fix_mask[:, 1:].unsqueeze(-1).float()  # (B, T-1, 1)
        preds = preds * valid
        targets = targets * valid

        return preds, targets

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
        """For downstream eval — no fixation data needed."""
        patch_tokens = self._encode_image(imgs)  # (B, 196, D_vit)
        cls_feat = patch_tokens.mean(dim=1)       # (B, D_vit) global avg pool
        return cls_feat


def build_fixation_ar_jepa(cfg):
    m = cfg["model"]
    return FixationARJEPA(
        vit_model=m.get("vit_model", "vit_large_patch16_224"),
        vit_pretrained=m.get("vit_pretrained", True),
        embed_dim=m.get("embed_dim", 768),
        context_depth=m.get("context_depth", 8),
        context_heads=m.get("context_heads", 12),
        head_dim=m.get("head_dim", 64),
        max_fixations=m.get("max_fixations", 128),
        ema_momentum=m.get("ema_momentum", 0.996),
    )
